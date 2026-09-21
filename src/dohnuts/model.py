"""Parallel candidate scoring over shared text/image computation."""

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from dohnuts.adapters import Qwen35Adapter
from dohnuts.recipe import IMAGE_PIXELS, MAX_LENGTH


def model_revision(directory):
    """Identify a downloaded Hub snapshot or a locally pinned training base."""
    directory = Path(directory)
    revision = directory / "revision.txt"
    if revision.is_file():
        return revision.read_text().strip()
    if directory.parent.name == "snapshots" and len(directory.name) == 40:
        return directory.name
    raise ValueError("The base model must be a pinned Hub snapshot or contain revision.txt")


class DecisionModel(nn.Module):
    def __init__(self, checkpoint: str | Path, *, adapter=None):
        super().__init__()
        self.adapter = adapter or Qwen35Adapter()
        self.base_path = Path(checkpoint)
        self.processor = self.adapter.processor(checkpoint)
        self.backbone = self.adapter.load(checkpoint)
        self.head = nn.Linear(
            self.adapter.hidden_size(self.backbone),
            1,
            bias=False,
            device="cuda",
            dtype=torch.float32,
        )
        nn.init.normal_(self.head.weight, std=0.01)
        self.marker_id = self.processor.tokenizer.convert_tokens_to_ids(self.adapter.marker)
        if self.processor.tokenizer.encode(self.adapter.marker, add_special_tokens=False) != [
            self.marker_id
        ]:
            raise ValueError("The candidate marker must be a single reserved token")

    def forward(self, inputs: dict[str, Tensor], positions: Tensor) -> Tensor:
        hidden, offset = self.adapter.forward(self.backbone, inputs)
        return self.score_hidden(hidden, (positions - offset).clamp_min(0))

    def score_hidden(self, hidden: Tensor, positions: Tensor) -> Tensor:
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        return self.head(hidden[rows, positions].float()).squeeze(-1)

    def enable_lora(self, *, checkpointing=True):
        self.adapter.adapt_language(self.backbone, training=checkpointing)

    def load_adapter(self, directory):
        """Load the same verified unmerged weights for learning or deployment."""
        from safetensors.torch import load_file

        directory = Path(directory)
        metadata = json.loads((directory / "dohnuts.json").read_text())
        weights = directory / "adapter.safetensors"
        if (
            metadata.get("format_version") != 1
            or metadata.get("adapter", "qwen3.5") != self.adapter.name
            or metadata.get("base_revision") != model_revision(self.base_path)
            or (metadata["image_pixels"], metadata["max_length"], metadata["lora_rank"])
            != (IMAGE_PIXELS, MAX_LENGTH, 8)
            or hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["weights_sha256"]
        ):
            raise ValueError(
                "Checkpoint differs from the pinned model, training recipe, or checksum"
            )
        state = load_file(str(weights))
        parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if state.keys() != parameters.keys() or any(
            state[name].shape != parameter.shape for name, parameter in parameters.items()
        ):
            raise ValueError("Checkpoint trainable parameters differ")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(state[name])
        return metadata

    def merge(self):
        self.adapter.merge(self.backbone)
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.freeze_vision(self.backbone)
        return self


def marker_positions(input_ids: Tensor, marker_id: int, candidates: int) -> Tensor:
    positions = []
    for row in input_ids:
        found = (row == marker_id).nonzero(as_tuple=True)[0]
        if found.numel() != candidates:
            raise ValueError(f"Expected {candidates} candidate markers, found {found.numel()}")
        positions.append(found)
    return torch.stack(positions)
