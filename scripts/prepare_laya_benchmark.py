"""Freeze the public suites behind Laya's comparison chart using its original builders."""

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
from huggingface_hub import HfApi, hf_hub_download

UPSTREAM = Path(".cache/upstream/laya-research")
REVISION = "28d43add7e47ce502489c9433310d55276c64e0f"
OUTPUT = Path("data/benchmarks/laya")
RAW = Path("data/raw/laya-chart")
SOURCE = UPSTREAM / "research/scripts"
ORIGINAL_LOAD = datasets.load_dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Laya research source revision changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "data-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "upstream_revision": REVISION,
            "datasets_version": importlib.metadata.version("datasets"),
            "repositories": {},
            "files": {},
        }
    )

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    @cache
    def pinned_load(repo, name=None, *, split):
        print(json.dumps({"dataset": repo, "config": name, "split": split}), flush=True)
        if repo == "PolyAI/banking77":
            path = Path("data/raw/banking77/test.csv")
            names = json.loads(Path("data/raw/banking77/categories.json").read_text())
            data = ORIGINAL_LOAD("csv", data_files=str(path), split="train")
            data = data.map(lambda row: {"label": names.index(row["category"])})
            data = data.cast_column("label", datasets.ClassLabel(names=names))
            manifest["files"][str(path)] = digest(path)
            save_manifest()
            return data
        if repo == "google/boolq":
            path = Path("data/raw/boolq/val.jsonl")
            manifest["files"][str(path)] = digest(path)
            save_manifest()
            return ORIGINAL_LOAD("json", data_files=str(path), split="train").rename_column(
                "label", "answer"
            )
        if repo not in manifest["repositories"]:
            info = HfApi().dataset_info(repo)
            manifest["repositories"][repo] = {
                "revision": info.sha,
                "files": [s.rfilename for s in info.siblings],
                "configs": info.card_data.to_dict().get("configs", []) if info.card_data else [],
            }
            save_manifest()
        source = manifest["repositories"][repo]
        configs = source["configs"]
        config = next((c for c in configs if c["config_name"] == (name or "default")), None)
        if config:
            patterns = next(row["path"] for row in config["data_files"] if row["split"] == split)
            patterns = [patterns] if isinstance(patterns, str) else patterns
        elif repo == "Tobi-Bueck/customer-support-tickets":
            patterns = ["*.csv"]
        elif repo == "zefang-liu/phishing-email-dataset":
            patterns = ["Phishing_Email.csv"]
        else:
            patterns = [f"{name or 'data'}/{split}-*.parquet", f"{split}.jsonl"]
        files = sorted(f for f in source["files"] if any(fnmatch.fnmatch(f, p) for p in patterns))
        if not files:
            raise ValueError(f"No data files for {repo}/{name}/{split}")
        paths = []
        for filename in files:
            path = Path(
                hf_hub_download(
                    repo,
                    filename,
                    repo_type="dataset",
                    revision=source["revision"],
                    local_dir=RAW / repo,
                )
            )
            paths.append(str(path))
            manifest["files"][str(path)] = digest(path)
        save_manifest()
        format_name = (
            "parquet"
            if files[0].endswith(".parquet")
            else "csv"
            if files[0].endswith(".csv")
            else "json"
        )
        return ORIGINAL_LOAD(format_name, data_files=paths, split="train")

    # Keep upstream question construction intact; replace only its data access
    # with pinned, hashed local data files. No model or training code is executed.
    datasets.load_dataset = pinned_load
    sys.path[:0] = [str(SOURCE.resolve()), str(UPSTREAM.resolve())]
    import bench_apps
    import bench_local

    bench_apps.build()
    if len(bench_apps.SUITES) != 10:
        raise RuntimeError("Not all ten application suites were built; inspect dataset errors")
    suites = {}
    for name, suite in bench_apps.SUITES.items():
        suites["apps/" + name] = {
            **suite,
            "gold": [
                {next(iter(case[1])): {"idx": gold}}
                for case, gold in zip(suite["cases"], suite["gold"], strict=True)
            ],
        }
    notebook = json.loads((SOURCE / "laya_benchmark_colab.ipynb").read_text())
    namespace = {}
    for index in [10, 12, 14, 16]:
        exec(  # noqa: S102 — reviewed data-builder cells from the pinned upstream notebook
            compile(
                "".join(notebook["cells"][index]["source"]), f"laya-notebook-cell-{index}", "exec"
            ),
            namespace,
        )
    if len(namespace["SUITES"]) != 50:
        raise RuntimeError("Not all fifty Colab suites were built; inspect dataset errors")
    suites.update({"colab/" + name: value for name, value in namespace["SUITES"].items()})
    reference = json.loads((UPSTREAM / "research/results/cpu_51_language_sweep.json").read_text())
    languages = reference["part_a"]["config"]["languages"]
    massive = bench_local.build_massive(languages, 100)
    if len(massive) != 51:
        raise RuntimeError("Expected all 51 MASSIVE languages")
    for language, (cases, gold, labels) in massive.items():
        suites["massive51/" + language] = {
            "cases": cases,
            "gold": [{"intent": {"idx": value}} for value in gold],
            "meta": {"language": language, "label_space": labels, "n_options": 20},
        }
    suite_path = OUTPUT / "suites.json"
    suite_path.write_text(json.dumps(suites, ensure_ascii=False) + "\n")
    manifest["suite_sha256"] = digest(suite_path)
    manifest["builder_sha256"] = {
        str(p): digest(p)
        for p in [
            SOURCE / "bench_apps.py",
            SOURCE / "bench_local.py",
            SOURCE / "laya_benchmark_colab.ipynb",
        ]
    }
    manifest["counts"] = {name: sum(len(qs) for _, qs in s["cases"]) for name, s in suites.items()}
    manifest["scope"] = [
        "Original upstream question builders, seed 13, case order, candidate order and state shortening.",
        "Dataset snapshots are pinned here. Upstream results lack dataset revisions and per-example hashes, so historical byte identity cannot be independently established.",
        "Upstream in_training/held_out metadata describes Laya, not Dohnuts; this is additional evaluation, with no checkpoint or temperature selection.",
    ]
    save_manifest()
    print(
        json.dumps(
            {
                "suites": len(suites),
                "decisions": sum(manifest["counts"].values()),
                "sha256": manifest["suite_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
