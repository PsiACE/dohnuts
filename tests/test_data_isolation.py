"""Different source IDs must not let identical screenshots cross data splits."""

import importlib.util
from collections import Counter
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).parents[1] / "scripts/prepare_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_reencoded_screenshots_join_before_split_priority(tmp_path):
    image = Image.new("RGB", (16, 16), "red")
    image.save(tmp_path / "train.png", compress_level=0)
    image.save(tmp_path / "test.png", compress_level=9)
    assert (tmp_path / "train.png").read_bytes() != (tmp_path / "test.png").read_bytes()
    records = [
        prepare.example(
            "screenqa_choice",
            split,
            "source-id:" + split,
            split,
            {},
            prepare.choice("What color?", ["red", "blue"]),
            0,
            image=str(tmp_path / f"{split}.png"),
        )
        for split in ["train", "test"]
    ]
    kept = list(prepare.isolate(records, Counter()))
    assert [r["id"] for r in kept] == ["screenqa_choice:test"]
    assert kept[0]["split"] == "test"
