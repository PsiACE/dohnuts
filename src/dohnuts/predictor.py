"""Candidate selection, truth estimates, and ordered scores over shared inputs."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image

from dohnuts.adapters import Qwen35Adapter
from dohnuts.execution import plan_prefix
from dohnuts.model import DecisionModel, marker_positions
from dohnuts.recipe import BASE_MODEL, IMAGE_PIXELS

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def options_for(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        criteria = criteria or {}
        if not isinstance(criteria, Mapping):
            raise ValueError("noul criteria must map false/true to descriptions")
        labels = ["false", "true"]
        options = [
            "false: " + render(criteria.get("false", "no, the statement does not hold")),
            "true: " + render(criteria.get("true", "yes, the statement holds")),
        ]
    elif kind == "choice":
        if isinstance(criteria, list):
            labels = [str(value) for value in criteria]
            options = [render(value) for value in criteria]
        elif isinstance(criteria, Mapping):
            labels = list(criteria)
            options = [
                str(key) if value is None or value == "" else f"{key}: {render(value)}"
                for key, value in criteria.items()
            ]
        else:
            raise ValueError("choice requires a candidate list or mapping")
        if len(set(labels)) != len(labels):
            raise ValueError("Candidate labels must be unique")
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("score requires an ordered list of level descriptions")
        labels = [str(i) for i in range(len(criteria))]
        options = [f"level {i}: {render(value)}" for i, value in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported decision type: {kind}")
    if not 2 <= len(options) <= 128:
        raise ValueError("Each question requires 2–128 candidates")
    return labels, options


def render_question(state_text, question, *, has_image=False, adapter=None):
    """One prompt template shared by training, evaluation, and serving."""
    adapter = adapter or Qwen35Adapter()
    marker = adapter.marker
    labels, options = options_for(question)
    content = (
        f"State: {state_text}\n{question['type']} question: "
        f"{question.get('instructions', '')}\nOptions:\n"
    )
    if marker in content or any(marker in option for option in options):
        raise ValueError("Input contains the reserved candidate marker")
    content += "".join(f"- {option}{marker}" for option in options)
    if has_image:
        content = adapter.image_prefix + content
    return content, labels


class Predictor:
    metadata: dict[str, Any]

    def __init__(self, model: DecisionModel):
        self.model = model
        self.image_pixels = IMAGE_PIXELS
        self.max_length = model.adapter.max_input_tokens
        self.temperatures = [1.0, 1.0, 1.0]

    @classmethod
    def from_checkpoint(cls, directory, *, revision=None, base_model=None, adapter=None):
        """Load a local export or Hub repository and merge its calibrated decision model."""
        source = directory
        directory = Path(source).expanduser()
        if not directory.is_dir():
            if isinstance(source, Path) or str(source).startswith(("/", ".", "~")):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
            directory = Path(
                snapshot_download(
                    str(source),
                    revision=revision,
                    allow_patterns=["dohnuts.json", "adapter.safetensors", "LICENSE"],
                )
            )
        config = json.loads((directory / "dohnuts.json").read_text())
        if base_model is None:
            base_model = Path(config.get("base_path", BASE_MODEL)).expanduser()
            if not base_model.is_dir():
                base_model = snapshot_download(
                    config["base_model"], revision=config["base_revision"]
                )
        base_model = Path(base_model).expanduser()
        model = DecisionModel(base_model, adapter=adapter)
        model.enable_lora(checkpointing=False)
        config = model.load_adapter(directory)
        model.merge()
        predictor = cls(model)
        predictor.metadata = config
        predictor.temperatures = [config["temperatures"][kind] for kind in QTYPES]
        return predictor

    def prepare(self, state, questions):
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        image = None
        if isinstance(state, Mapping):
            image = state.get("image")
            if "images" in state:
                raise ValueError("Pass one decoded PIL image via state['image']")
            state_text = render({key: value for key, value in state.items() if key != "image"})
        else:
            state_text = render(state)
        if image is not None and not isinstance(image, Image.Image):
            raise ValueError("Decode the image as a PIL image before passing it in state['image']")
        texts, metadata = [], []
        for qid, question in questions.items():
            content, labels = render_question(
                state_text, question, has_image=image is not None, adapter=self.model.adapter
            )
            texts.append(content)
            metadata.append((qid, question["type"], labels))
        if image is not None:
            inputs = self.model.adapter.shared_image_inputs(self.model.processor, texts, image)
        else:
            inputs = self.model.adapter.batch_inputs(self.model.processor, texts, [])
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                "Input exceeds the token budget; no question or candidate was truncated"
            )
        maximum = max(len(labels) for _, _, labels in metadata)
        positions = torch.zeros(len(texts), maximum, dtype=torch.long)
        mask = torch.zeros_like(positions, dtype=torch.bool)
        for row, (_, _, labels) in enumerate(metadata):
            positions[row, : len(labels)] = marker_positions(
                inputs["input_ids"][row : row + 1],
                self.model.marker_id,
                len(labels),
            )[0]
            mask[row, : len(labels)] = True
        plan_prefix(inputs, positions)
        return inputs, positions, mask, metadata

    @torch.inference_mode()
    def predict(self, state, questions):
        self.model.eval()
        inputs, positions, mask, metadata = self.prepare(state, questions)
        token_count = int(inputs["attention_mask"].sum())
        has_image = "pixel_values" in inputs
        inputs = {
            key: value.to("cuda") if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        positions = positions.to("cuda")
        logits = self.model(inputs, positions).cpu().masked_fill(~mask, -torch.inf)
        answers = {}
        for row, (qid, kind, labels) in enumerate(metadata):
            temperature = self.temperatures[QTYPES[kind]]
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Calibration temperatures must be positive and finite")
            probabilities = (logits[row, : len(labels)] / temperature).softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Model produced a non-finite distribution")
            values = probabilities.tolist()
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in values)
            answer = {
                "type": kind,
                "confidence": max(0.0, min(1.0, 1 - entropy / math.log(len(labels)))),
            }
            if kind == "noul":
                answer["noul"] = values[1]
                answer["confidence"] = max(values)
            else:
                answer["probabilities"] = dict(zip(labels, values, strict=True))
                if kind == "choice":
                    answer["choice"] = labels[int(probabilities.argmax())]
                else:
                    answer["score"] = sum(i * p for i, p in enumerate(values))
            answers[qid] = answer
        return {
            "model": "dohnuts",
            "answers": answers,
            "usage": {"input_tokens": token_count, "images": int(has_image)},
        }
