"""Deterministic sampling and shared train/serve prompt collation."""

import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from dohnuts.adapters import Qwen35Adapter
from dohnuts.execution import plan_prefix
from dohnuts.model import marker_positions
from dohnuts.predictor import render, render_question
from dohnuts.recipe import MAX_LENGTH


def stable_key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load_records(path, limit_per_dataset=None):
    groups = defaultdict(list)
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["dataset"]].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: stable_key("doh-sample-v1:" + r["id"]))
        if limit_per_dataset:
            groups[key] = groups[key][:limit_per_dataset]
    return dict(groups)


class DecisionCollator:
    def __init__(self, model_path, *, adapter=None):
        self.adapter = adapter or Qwen35Adapter()
        self.processor = self.adapter.processor(model_path)
        self.marker_id = self.processor.tokenizer.convert_tokens_to_ids(self.adapter.marker)
        self.max_length = MAX_LENGTH

    def __call__(self, records, *, permutation_seed=None):
        texts, images, targets, types = [], [], [], []
        rng = random.Random(permutation_seed)
        for record in records:
            question = copy.deepcopy(record["question"])
            target = record["target"][:]
            # Ordinal levels and binary semantic order must remain fixed.
            if permutation_seed is not None and question["type"] == "choice":
                criteria = question["criteria"]
                order = list(range(len(target)))
                rng.shuffle(order)
                if isinstance(criteria, dict):
                    keys = list(criteria)
                    question["criteria"] = {keys[i]: criteria[keys[i]] for i in order}
                else:
                    question["criteria"] = [criteria[i] for i in order]
                target = [target[i] for i in order]
            has_image = bool(record.get("image"))
            state = render(record["state"])
            text, _ = render_question(state, question, has_image=has_image, adapter=self.adapter)
            texts.append(text)
            if has_image:
                with Image.open(record["image"]) as image:
                    images.append(image.convert("RGB"))
            targets.append(target)
            types.append(question["type"])
        inputs = self.adapter.batch_inputs(self.processor, texts, images)
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError(f"Token budget exceeded: {[r['id'] for r in records]}")
        maximum = max(map(len, targets))
        positions = torch.zeros(len(records), maximum, dtype=torch.long)
        mask = torch.zeros_like(positions, dtype=torch.bool)
        target_tensor = torch.zeros_like(positions, dtype=torch.float32)
        for row, target in enumerate(targets):
            positions[row, : len(target)] = marker_positions(
                inputs["input_ids"][row : row + 1], self.marker_id, len(target)
            )[0]
            mask[row, : len(target)] = True
            target_tensor[row, : len(target)] = torch.tensor(target)
        plan_prefix(inputs, positions)
        return inputs, positions, mask, target_tensor, torch.tensor([t == "score" for t in types])


def to_gpu(batch):
    inputs, positions, mask, target, ordinal = batch
    inputs = {
        k: v.to("cuda", dtype=torch.bfloat16 if k == "pixel_values" else v.dtype, non_blocking=True)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }
    return inputs, *(v.to("cuda", non_blocking=True) for v in (positions, mask, target, ordinal))


def prefetch_batches(loader):
    """Overlap one pinned input transfer with computation, preserving input order.

    Each loader item is (collated tensors, metadata...). An event makes the
    consumer wait only for its own batch, not the following transfer. Recording
    the consumer stream keeps transferred storage alive through asynchronous use.
    """
    iterator = iter(loader)
    transfer = torch.cuda.Stream()

    def enqueue(item):
        batch, *metadata = item
        with torch.cuda.stream(transfer):
            tensors = to_gpu(batch)
            ready = torch.cuda.Event()
            ready.record()
        return tensors, metadata, ready, batch

    item = next(iterator, None)
    if item is None:
        return
    pending = enqueue(item)
    try:
        while pending is not None:
            tensors, metadata, ready, _source = pending
            consumer = torch.cuda.current_stream()
            consumer.wait_event(ready)
            inputs, *other = tensors
            for tensor in [*inputs.values(), *other]:
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(consumer)
            item = next(iterator, None)
            pending = None if item is None else enqueue(item)
            yield tensors, *metadata
    finally:
        # Also covers a consumer stopping before exhausting the loader.
        transfer.synchronize()


class EvaluationBatches(torch.utils.data.Dataset):
    """Prepare independent evaluation batches in CPU workers, in stable order."""

    def __init__(self, groups, collator, batch_size):
        self.collator = collator
        self.batches = []
        for _, rows in sorted(groups.items()):
            rows = sorted(
                rows, key=lambda r: len(json.dumps(r["state"])) + len(json.dumps(r["question"]))
            )
            self.batches.extend(
                rows[start : start + batch_size] for start in range(0, len(rows), batch_size)
            )

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        rows = self.batches[index]
        return self.collator(rows), rows


class TrainingBatches(torch.utils.data.Dataset):
    """Index determines the exact examples and permutations, including after resume."""

    def __init__(self, groups, collator, *, seed, batch_size, steps, accumulation, start_step=0):
        self.groups, self.collator = groups, collator
        self.keys = sorted(groups)
        self.seed, self.batch_size = seed, batch_size
        self.steps, self.accumulation, self.start_step = steps, accumulation, start_step

    def __len__(self):
        return (self.steps - self.start_step) * self.accumulation

    def __getitem__(self, index):
        index += self.start_step * self.accumulation
        rng = random.Random(self.seed * 1000000007 + index)
        key = self.keys[rng.randrange(len(self.keys))]
        rows = [rng.choice(self.groups[key]) for _ in range(self.batch_size)]
        return (
            self.collator(rows, permutation_seed=rng.randrange(2**63)),
            key,
            [r["id"] for r in rows],
        )
