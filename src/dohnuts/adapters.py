"""Backbone-specific loading, image encoding, and language adaptation."""

import hashlib
import os
from collections import OrderedDict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor

from dohnuts.execution import language_forward
from dohnuts.recipe import IMAGE_PIXELS


class Qwen35Adapter:
    """The Qwen3.5 text/image adapter, using BF16 and merged inference."""

    name = "qwen3.5"
    base_model = "Qwen/Qwen3.5-0.8B"
    max_input_tokens = 4096
    marker = "<|fim_suffix|>"
    image_prefix = "<|vision_start|><|image_pad|><|vision_end|>\n"

    def processor(self, checkpoint):
        return AutoProcessor.from_pretrained(checkpoint, local_files_only=True)

    def load(self, checkpoint):
        from dohnuts.kernels import (
            enable_fusion,
            enable_linear_patch_embedding,
            enable_triton_convolution,
        )

        # The desktop shares this GPU. Bound the caching allocator so a sequence
        # of evaluation shapes cannot retain nearly all VRAM and crash the compositor.
        torch.cuda.set_per_process_memory_fraction(0.8)
        if torch.version.hip:
            os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
            enable_triton_convolution()
        backbone = AutoModel.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        ).to("cuda")
        backbone.requires_grad_(False)
        backbone.config.use_cache = False
        enable_linear_patch_embedding(backbone.visual)
        enable_fusion(backbone)
        backbone._dohnuts_image_cache = OrderedDict()
        return backbone

    def hidden_size(self, backbone):
        return backbone.config.text_config.hidden_size

    def adapt_language(self, backbone, *, training):
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        backbone.language_model = get_peft_model(backbone.language_model, config)
        if training:
            backbone.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        backbone.visual.requires_grad_(False)

    def merge(self, backbone):
        backbone.language_model = backbone.language_model.merge_and_unload(safe_merge=True)

    def freeze_vision(self, backbone):
        backbone.visual.eval()

    def batch_inputs(self, processor, texts, images):
        kwargs = {"text": texts, "padding": True, "return_tensors": "pt", "truncation": False}
        if images:
            kwargs.update(
                images=images,
                images_kwargs={
                    "size": {"shortest_edge": IMAGE_PIXELS, "longest_edge": IMAGE_PIXELS}
                },
            )
        inputs = dict(processor(**kwargs))
        if images:
            self.image_metadata(inputs, images)
        return inputs

    def image_metadata(self, inputs, images):
        inputs["image_keys"] = tuple(
            hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()
            for image in images
        )
        inputs["image_patches"] = inputs["image_grid_thw"].prod(-1).tolist()

    def shared_image_inputs(self, processor, texts, image):
        inputs = dict(
            processor.image_processor(
                images=[image],
                return_tensors="pt",
                size={"shortest_edge": IMAGE_PIXELS, "longest_edge": IMAGE_PIXELS},
            )
        )
        replacement = processor.replace_image_token(inputs, 0)
        texts = [text.replace(processor.image_token, replacement, 1) for text in texts]
        tokens = dict(
            processor.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=False,
                return_token_type_ids=False,
            )
        )
        tokens["mm_token_type_ids"] = torch.tensor(
            processor.create_mm_token_type_ids(tokens["input_ids"]), dtype=torch.long
        )
        inputs.update(tokens)
        self.image_metadata(inputs, [image])
        inputs["image_grid_thw"] = inputs["image_grid_thw"].expand(len(texts), -1)
        inputs["shared_image"] = True
        return inputs

    def forward(self, backbone, inputs):
        input_ids = inputs["input_ids"]
        embeddings = backbone.get_input_embeddings()(input_ids)
        position_ids = None
        if "pixel_values" in inputs:
            features = self.image_features(backbone, inputs)
            if inputs.get("shared_image"):
                features = features.repeat(input_ids.shape[0], 1)
            features = features.to(embeddings.dtype)
            image_mask, _ = backbone.get_placeholder_mask(
                input_ids, inputs_embeds=embeddings, image_features=features
            )
            embeddings = embeddings.masked_scatter(image_mask, features)
            position_ids = backbone.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=inputs["image_grid_thw"],
                attention_mask=inputs["attention_mask"],
                inputs_embeds=embeddings,
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
        cut = inputs.get("prefix_length", 0)
        hidden = language_forward(
            backbone.language_model, embeddings, inputs["attention_mask"], position_ids, cut
        )
        return hidden, cut

    def image_features(self, backbone, inputs):
        """The frozen encoder has a model-local 128 MiB LRU; no language state persists."""
        cache = backbone._dohnuts_image_cache
        pixels = inputs["pixel_values"].split(inputs["image_patches"])
        missing = {}
        for i, key in enumerate(inputs["image_keys"]):
            if key not in cache:
                missing.setdefault(key, i)
        if missing:
            indices = list(missing.values())
            with torch.inference_mode(False), torch.no_grad():
                features = backbone.get_image_features(
                    torch.cat([pixels[i] for i in indices]),
                    inputs["image_grid_thw"][indices],
                    return_dict=True,
                ).pooler_output
            cache.update(zip(missing, features, strict=True))
        result = torch.cat([cache[key] for key in inputs["image_keys"]])
        for key in inputs["image_keys"]:
            cache.move_to_end(key)
        size = sum(t.numel() * t.element_size() for t in cache.values())
        while size > 128 * 2**20:
            _, value = cache.popitem(last=False)
            size -= value.numel() * value.element_size()
        return result
