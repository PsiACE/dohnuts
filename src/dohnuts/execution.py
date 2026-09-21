"""Request-scoped, differentiable prefix sharing for the Qwen hybrid cache."""

import torch
from transformers.cache_utils import DynamicCache


def plan_prefix(inputs, positions):
    """Plan on CPU, before asynchronous transfer; candidates stay in each branch."""
    ids = inputs["input_ids"]
    cut = 0
    images = inputs.get("image_keys", ())
    if len(ids) > 1 and len(set(images)) <= 1:
        common = (ids == ids[:1]).all(0) & inputs["attention_mask"].bool().all(0)
        different = (~common).nonzero()
        length = int(different[0]) if len(different) else common.numel()
        cut = min(length, int(positions[:, 0].min()), ids.shape[1] - 2) // 64 * 64
    inputs["prefix_length"] = cut


class PrefixCache(DynamicCache):
    """Replace state tensors instead of overwriting values needed by backward."""

    def update_conv_state(
        self, conv_states, layer_idx, state_idx=0, conv_kernel_size=None, **kwargs
    ):
        if conv_kernel_size is None:
            raise ValueError("PrefixCache requires conv_kernel_size")
        layer = self.layers[layer_idx]
        layer.device, layer.dtype = conv_states.device, conv_states.dtype
        if layer.has_previous_state[state_idx]:
            conv_states = torch.cat([layer.conv_states[state_idx], conv_states], dim=-1)
        layer.conv_kernel_size[state_idx] = conv_kernel_size
        layer.conv_states[state_idx] = conv_states[..., -conv_kernel_size:]
        layer.has_previous_state[state_idx] = True
        layer.is_conv_states_initialized[state_idx] = True
        return conv_states

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx=0, **kwargs):
        layer = self.layers[layer_idx]
        layer.recurrent_states[state_idx] = recurrent_states
        layer.is_recurrent_states_initialized[state_idx] = True
        return recurrent_states


def language_forward(language, embeddings, attention_mask, position_ids, cut):
    if not cut:
        return language(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    def shared_forward():
        # A mutable cache cannot survive a decoder-layer checkpoint replay.
        # Replay the complete request instead, constructing a fresh cache each time.
        layers = [m for m in language.modules() if getattr(m, "gradient_checkpointing", False)]
        for layer in layers:
            layer.gradient_checkpointing = False
        try:
            cache = PrefixCache(config=language.config)
            language(
                inputs_embeds=embeddings[:1, :cut],
                attention_mask=attention_mask[:1, :cut],
                position_ids=None if position_ids is None else position_ids[:, :1, :cut],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            # Transformers expects LongTensor; factories return Tensor with int64 dtype.
            indices = torch.zeros(len(embeddings), dtype=torch.long, device=embeddings.device)
            cache.reorder_cache(indices)  # ty: ignore[invalid-argument-type]
            return language(
                inputs_embeds=embeddings[:, cut:],
                attention_mask=attention_mask,
                position_ids=None if position_ids is None else position_ids[:, :, cut:],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
        finally:
            for layer in layers:
                layer.gradient_checkpointing = True

    if torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint

        return checkpoint(shared_forward, use_reentrant=False)
    return shared_forward()
