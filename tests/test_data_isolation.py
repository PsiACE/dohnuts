"""Different source IDs must not let identical screenshots cross data splits."""

import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location("prepare_data", Path("scripts/prepare_data.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SplitIsolationTests(unittest.TestCase):
    def test_reencoded_screenshots_join_before_split_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (16, 16), "red")
            image.save(root / "train.png", compress_level=0)
            image.save(root / "test.png", compress_level=9)
            self.assertNotEqual((root / "train.png").read_bytes(), (root / "test.png").read_bytes())
            records = [
                prepare.example(
                    "screenqa_choice",
                    split,
                    "source-id:" + split,
                    split,
                    {},
                    prepare.choice("What color?", ["red", "blue"]),
                    0,
                    image=str(root / f"{split}.png"),
                )
                for split in ["train", "test"]
            ]
            kept = list(prepare.isolate(records, Counter()))
            self.assertEqual([r["id"] for r in kept], ["screenqa_choice:test"])
            self.assertEqual(kept[0]["split"], "test")


if __name__ == "__main__":
    unittest.main()
