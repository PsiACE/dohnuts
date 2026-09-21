"""Convert public annotations to decision rows and audit grouped split isolation.

No labels, rationales, scene graphs, or annotator text are put in model state.
Run only after downloads finish. The unbounded manifests retain all eligible rows;
training caps and monitoring subsets belong to the separately frozen run recipe.
"""

import argparse
import csv
import email
import hashlib
import io
import json
import mailbox
import re
import subprocess
import tarfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from email import policy
from html.parser import HTMLParser
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from dohnuts.predictor import render, render_question
from dohnuts.recipe import BASE_MODEL, DATA, IMAGE_PIXELS, MAX_LENGTH
from dohnuts.training_data import load_records

BUSINESS_RAW = Path("data/raw/enrichment")

RAW = Path("data/raw")
OUT = Path("data/processed/converted")
IMAGE_CACHE = {}
SPLIT_ORDER = {"train": 0, "calibration": 1, "dev": 2, "test": 3}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    return " ".join(value.casefold().split())


def train_split(group, *, has_dev=False):
    bucket = int(digest("doh-split-2026:" + group)[:8], 16) % 100
    if bucket < 5:
        return "calibration"
    if not has_dev and bucket < 10:
        return "dev"
    return "train"


def rows(path):
    with Path(path).open() as handle:
        yield from (json.loads(line) for line in handle)


def parquet(repo, prefix):
    for path in sorted((RAW / "hf" / repo).glob(prefix + "*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()


def image_asset(value):
    if not value or not value.get("bytes"):
        raise ValueError("Missing embedded image")
    byte_hash = hashlib.sha256(value["bytes"]).hexdigest()
    if byte_hash in IMAGE_CACHE:
        return IMAGE_CACHE[byte_hash]
    with Image.open(io.BytesIO(value["bytes"])) as image:
        image = image.convert("RGB")
        key = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        path = OUT / "images" / (key + ".image")
        if not path.exists():
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(value["bytes"])
    IMAGE_CACHE[byte_hash] = (str(path), "pixels:" + key)
    return IMAGE_CACHE[byte_hash]


def example(dataset, uid, group, split, state, question, target, *, image=None, aliases=()):
    if isinstance(target, int):
        k = 2 if question["type"] == "noul" else len(question["criteria"])
        target = [float(i == target) for i in range(k)]
    total = sum(target)
    if not target or min(target) < 0 or abs(total - 1) > 1e-4:
        raise ValueError(f"Invalid target: {dataset}/{uid}")
    target = [v / total for v in target]
    return {
        "id": f"{dataset}:{uid}",
        "dataset": dataset,
        "group": group,
        "aliases": list(aliases),
        "split": split,
        "state": state,
        "image": image,
        "question": question,
        "target": target,
    }


def choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions):
    return {"type": "noul", "instructions": instructions}


def text_sources():
    massive = {lang: list(rows(RAW / "massive" / f"{lang}.jsonl")) for lang in ["en-US", "zh-CN"]}
    intents = sorted({r["intent"] for r in massive["en-US"] if r["partition"] == "train"})
    assert len(intents) == 60
    criteria = {label: label.replace("_", " ") for label in intents}
    for lang, data in massive.items():
        for r in data:
            group = "massive:" + r["id"]
            split = (
                train_split(group, has_dev=True)
                if r["partition"] == "train"
                else {"dev": "dev", "test": "test"}[r["partition"]]
            )
            yield example(
                "massive_" + lang,
                r["id"],
                group,
                split,
                r["utt"],
                choice("Which intent best describes this utterance?", criteria),
                intents.index(r["intent"]),
                aliases=["utterance:" + digest(normalized(r["utt"]))],
            )
    for split in ["train", "val"]:
        for i, r in enumerate(rows(RAW / "boolq" / f"{split}.jsonl")):
            group = "passage:" + digest(normalized(r["passage"]))
            yield example(
                "boolq",
                f"{split}:{i}",
                group,
                train_split(group) if split == "train" else "test",
                r["passage"],
                noul(r["question"]),
                int(r["label"]),
            )
    configs = [
        (
            "ag_news",
            "fancyzhx/ag_news",
            "data/",
            ["World", "Sports", "Business", "Science and Technology"],
            "What is the news topic?",
        ),
        (
            "emotion",
            "dair-ai/emotion",
            "split/",
            ["sadness", "joy", "love", "anger", "fear", "surprise"],
            "What emotion is expressed?",
        ),
    ]
    for name, repo, folder, labels, instruction in configs:
        for split in ["train", "validation", "test"]:
            for i, r in enumerate(parquet(repo, folder + split)):
                group = "utterance:" + digest(normalized(r["text"]))
                part = (
                    train_split(group, has_dev=name == "emotion")
                    if split == "train"
                    else {"validation": "dev", "test": "test"}[split]
                )
                yield example(
                    name,
                    f"{split}:{i}",
                    group,
                    part,
                    r["text"],
                    choice(instruction, labels),
                    r["label"],
                )
    labels = json.loads((RAW / "banking77/categories.json").read_text())
    for split in ["train", "test"]:
        with (RAW / "banking77" / f"{split}.csv").open() as handle:
            for i, r in enumerate(csv.DictReader(handle)):
                group = "utterance:" + digest(normalized(r["text"]))
                yield example(
                    "banking77",
                    f"{split}:{i}",
                    group,
                    train_split(group) if split == "train" else "test",
                    r["text"],
                    choice(
                        "Which banking intent best describes this request?",
                        {x: x.replace("_", " ") for x in labels},
                    ),
                    labels.index(r["category"]),
                )
    for lang in ["en", "zh"]:
        for split in ["train", "validation", "test"]:
            for i, r in enumerate(parquet("facebook/xnli", f"{lang}/{split}")):
                group = f"xnli:{split}:{i}"  # aligned translation rows share a partition
                yield example(
                    "xnli_" + lang,
                    f"{split}:{i}",
                    group,
                    train_split(group, has_dev=True)
                    if split == "train"
                    else {"validation": "dev", "test": "test"}[split],
                    {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                    choice(
                        "What is the relation of the hypothesis to the premise?",
                        ["entailment", "neutral", "contradiction"],
                    ),
                    r["label"],
                    aliases=["premise:" + digest(normalized(r["premise"]))],
                )
    for split in ["train", "test"]:
        for r in parquet("LocalLLaMA/typed-decisions", f"all/{split}"):
            state, questions, gold = (json.loads(r[k]) for k in ["state", "questions", "gold"])
            group = "typed:" + digest(json.dumps(state, sort_keys=True))
            part = train_split(group) if split == "train" else "test"
            for qid, q in questions.items():
                labels = (
                    ["false", "true"]
                    if q["type"] == "noul"
                    else (
                        list(q["criteria"])
                        if q["type"] == "choice"
                        else [str(i) for i in range(len(q["criteria"]))]
                    )
                )
                yield example(
                    "typed_decisions",
                    r["id"] + ":" + qid,
                    group,
                    part,
                    state,
                    q,
                    [gold[qid]["probabilities"][label] for label in labels],
                )


def vision_sources():
    aok_ids = {}
    for path in (RAW / "aokvqa").glob("aokvqa_v1p0_*.json"):
        for row in json.loads(path.read_text()):
            aok_ids[row["question_id"]] = row["image_id"]
    for name, repo in [
        ("aokvqa", "HuggingFaceM4/A-OKVQA"),
        ("scienceqa", "derek-thomas/ScienceQA"),
    ]:
        for split in ["train", "validation", "test"]:
            if name == "aokvqa" and split == "test":
                continue  # labels withheld; official validation is our final set
            for i, r in enumerate(parquet(repo, "data/" + split)):
                if r.get("image") is None:
                    continue  # match Laya Vision's image subset
                choices = r["choices"]
                if len(set(choices)) != len(choices):
                    continue
                path, group = image_asset(r["image"])
                official = (
                    "test"
                    if (name == "aokvqa" and split == "validation") or split == "test"
                    else "dev"
                )
                part = (
                    train_split(group, has_dev=name == "scienceqa")
                    if split == "train"
                    else official
                )
                state = {"context": r["hint"]} if name == "scienceqa" and r.get("hint") else {}
                aliases = ["coco:" + str(aok_ids[r["question_id"]])] if name == "aokvqa" else []
                yield example(
                    name,
                    str(r.get("question_id", f"{split}:{i}")),
                    group,
                    part,
                    state,
                    choice(r["question"], choices),
                    int(r["correct_choice_idx"] if name == "aokvqa" else r["answer"]),
                    image=path,
                    aliases=aliases,
                )
    for i, r in enumerate(parquet("lmms-lab-encoder/VQAv2", "data/validation-")):
        if r.get("answer_type") != "yes/no":
            continue
        votes = [a["answer"] if isinstance(a, dict) else a for a in r["answers"]]
        votes = [v for v in votes if v in ("yes", "no")]
        if not votes:
            continue
        path, pixels = image_asset(r["image"])
        group = "coco:" + str(r["image_id"])
        bucket = int(digest("doh-vqa:" + group)[:8], 16) % 100
        part = (
            "test" if bucket < 10 else ("dev" if bucket < 15 else train_split(group, has_dev=True))
        )
        p = votes.count("yes") / len(votes)
        yield example(
            "vqav2_yesno",
            str(r["question_id"]),
            group,
            part,
            {},
            noul(r["question"]),
            [1 - p, p],
            image=path,
            aliases=[pixels],
        )


def clevr_sources():
    domains = {
        "query_color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
        "query_shape": ["cube", "sphere", "cylinder"],
        "query_size": ["small", "large"],
        "query_material": ["rubber", "metal"],
        "count": [str(i) for i in range(11)],
        "exist": ["no", "yes"],
    }
    root = RAW / "clevr/CLEVR_v1.0"
    for split in ["train", "val"]:
        data = json.loads((root / "questions" / f"CLEVR_{split}_questions.json").read_text())
        for r in data["questions"]:
            op = r["program"][-1]["function"]
            if op not in domains:
                continue
            group = "clevr:" + r["image_filename"]
            labels = domains[op]
            q = noul(r["question"]) if op == "exist" else choice(r["question"], labels)
            if op == "count":
                q["type"] = "score"
            yield example(
                "clevr_" + ("attribute" if op.startswith("query") else op),
                str(r["question_index"]) + ":" + split,
                group,
                train_split(group) if split == "train" else "test",
                {},
                q,
                labels.index(str(r["answer"])),
                image=str(root / "images" / split / r["image_filename"]),
            )


def screen_sources(audit):
    root = RAW / "rico/combined"
    for split in ["train", "validation", "test"]:
        data = json.loads(
            Path(f".cache/upstream/screen-qa/answers_and_bboxes/{split}.json").read_text()
        )
        cache = {}
        for i, r in enumerate(data):
            reason = None
            image_id = r["image_id"]
            if image_id not in cache:
                try:
                    tree = json.loads((root / f"{image_id}.json").read_text())["activity"]["root"]
                    nodes = []

                    def visit(node, destination):
                        if not isinstance(node, dict):
                            raise TypeError("Malformed view hierarchy node")
                        destination.append(node)
                        for child in node.get("children", []):
                            visit(child, destination)

                    visit(tree, nodes)
                    _, _, width, height = tree["bounds"]
                    # All visible leaves with valid in-screen bounds. No answer-dependent pruning.
                    candidates, seen = {}, set()
                    for j, node in enumerate(nodes):
                        bounds = tuple(node.get("bounds", []))
                        if (
                            len(bounds) != 4
                            or node.get("children")
                            or node.get("visibility", "visible") != "visible"
                        ):
                            continue
                        l, t, rr, b = bounds
                        if not (0 <= l < rr <= width and 0 <= t < b <= height) or bounds in seen:
                            continue
                        seen.add(bounds)
                        candidates[j] = [
                            round(l / width, 4),
                            round(t / height, 4),
                            round(rr / width, 4),
                            round(b / height, 4),
                        ]
                    cache[image_id] = candidates
                except (OSError, KeyError, ValueError, TypeError):
                    cache[image_id] = None
            candidates = cache[image_id]
            answers = r["ground_truth"]
            if candidates is None or not (root / f"{image_id}.jpg").exists():
                reason = "missing_assets"
            elif not 2 <= len(candidates) <= 128:
                reason = "candidate_count"
            elif not answers or any(len(a["ui_elements"]) != 1 for a in answers):
                reason = "multi_or_empty"
            else:
                indices = {a["ui_elements"][0]["vh_index"] for a in answers}
                if len(indices) != 1:
                    reason = "disagreement"
                elif next(iter(indices)) not in candidates:
                    reason = "unmapped_region"
            if reason:
                audit[f"screenqa:{split}:{reason}"] += 1
                continue
            index = next(iter(indices))
            keys = list(candidates)
            criteria = {
                f"r{key}": "box [left, top, right, bottom] = " + str(value)
                for key, value in candidates.items()
            }
            group = f"rico:{image_id}"
            part = (
                train_split(group, has_dev=True)
                if split == "train"
                else {"validation": "dev", "test": "test"}[split]
            )
            base = example(
                "screenqa_choice",
                f"{split}:{i}",
                group,
                part,
                {},
                choice(r["question"], criteria),
                keys.index(index),
                image=str(root / f"{image_id}.jpg"),
            )
            yield base
            negative_keys = [k for k in keys if k != index]
            negative = negative_keys[int(digest(base["id"])[:8], 16) % len(negative_keys)]
            for key, label in [(index, 1), (negative, 0)]:
                yield example(
                    "screenqa_noul",
                    f"{split}:{i}:r{key}",
                    group,
                    part,
                    {"candidate": criteria[f"r{key}"]},
                    noul("Does this candidate region answer the question: " + r["question"]),
                    label,
                    image=base["image"],
                )


def isolate(data, audit):
    """Union group aliases before resolving conflicts; higher-priority split wins.

    Lower-priority examples are excluded, never moved into the test set. Image
    aliases join COCO datasets; normalized passages/utterances join duplicates.
    """
    # Source IDs are not sufficient: Rico contains identical screenshots under
    # different IDs. Join exact bytes everywhere, and decoded pixels for Rico.
    image_aliases = {}
    for r in data:
        path = r.get("image")
        if not path:
            continue
        if path not in image_aliases:
            with Path(path).open("rb") as stream:
                aliases = ["image-bytes:" + hashlib.file_digest(stream, "sha256").hexdigest()]
            if r["dataset"].startswith("screenqa_"):
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    aliases.append(
                        "pixels:"
                        + hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    )
            image_aliases[path] = aliases
        r["aliases"] = list(dict.fromkeys([*r["aliases"], *image_aliases[path]]))
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for r in data:
        for alias in r["aliases"]:
            a, b = find(r["group"]), find(alias)
            if a != b:
                parents[b] = a
    priority = defaultdict(int)
    for r in data:
        group = find(r["group"])
        priority[group] = max(priority[group], SPLIT_ORDER[r["split"]])
    seen = set()
    for r in data:
        r["group"] = find(r["group"])
        if SPLIT_ORDER[r["split"]] != priority[r["group"]]:
            audit[f"{r['dataset']}:{r['split']}:cross_split_group"] += 1
            continue
        key = digest(
            json.dumps([r["dataset"], r["group"], r["state"], r["question"]], sort_keys=True)
        )
        if key in seen:
            audit[f"{r['dataset']}:{r['split']}:duplicate_input"] += 1
            continue
        seen.add(key)
        yield r


def download_file(item):
    path = Path(item["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != item["bytes"]:
        partial = path.with_suffix(path.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "-fLsS",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--output",
                str(partial),
                item["url"],
            ],
            check=True,
        )
        if partial.stat().st_size != item["bytes"]:
            raise ValueError(f"Wrong download length: {partial}")
        partial.replace(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if item.get("sha256") and digest != item["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    print(json.dumps({**item, "sha256": digest}), flush=True)


def download_sources():
    manifests = [
        "upstream-downloads.sha256.json",
        "original-downloads.json",
        "enrichment-downloads.json",
    ]
    items = [
        item
        for name in manifests
        for item in json.loads((Path("data/manifests") / name).read_text())
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download_file, items))


def extract_archives():
    with tarfile.open("data/raw/massive/massive-1.1.tar.gz") as archive:
        for lang in ["en-US", "zh-CN"]:
            Path(f"data/raw/massive/{lang}.jsonl").write_bytes(
                archive.extractfile(f"1.1/data/{lang}.jsonl").read()
            )
    with zipfile.ZipFile("data/raw/boolq/BoolQ.zip") as archive:
        for split in ["train", "val"]:
            Path(f"data/raw/boolq/{split}.jsonl").write_bytes(archive.read(f"BoolQ/{split}.jsonl"))
    for folder, filename in [("rico", "unique_uis.tar.gz"), ("aokvqa", "aokvqa_v1p0.tar.gz")]:
        with tarfile.open(Path("data/raw") / folder / filename) as archive:
            archive.extractall(Path("data/raw") / folder, filter="data")
    with zipfile.ZipFile("data/raw/clevr/CLEVR_v1.0.zip") as archive:
        for name in archive.namelist():
            if name.startswith(
                ("CLEVR_v1.0/questions/", "CLEVR_v1.0/images/train/", "CLEVR_v1.0/images/val/")
            ):
                archive.extract(name, "data/raw/clevr")


def fingerprint(text):
    return digest(" ".join(re.findall(r"\w+", text.casefold())))


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def blocked_text(base):
    """Protect fixed benchmark inputs and the training mixture's existing groups."""
    blocked = set(
        json.loads(Path("data/manifests/evaluation-text-sha256.json").read_text())["hashes"]
    )
    trained = load_records(base / "train.jsonl", 6000)
    trained_ids = {row["id"] for group in trained.values() for row in group}
    for split in SPLIT_ORDER:
        for row in rows(base / f"{split}.jsonl"):
            if split != "train" or row["id"] in trained_ids:
                for value in strings(row["state"]):
                    if len(value.strip()) >= 80:
                        blocked.add(fingerprint(value))
    return blocked


def policy_rows():
    with zipfile.ZipFile(BUSINESS_RAW / "sharc.zip") as archive:
        for split in ["train", "dev", "test"]:
            rows = json.loads(archive.read(f"sharc1-official/json/sharc_{split}.json"))
            for row in rows:
                # One source page can yield several snippets and conversation trees.
                group = "sharc-page:" + digest(row["source_url"] or normalized(row["snippet"]))
                part = train_split(group, has_dev=True) if split == "train" else split
                answer = row["answer"].casefold()
                label = (
                    ["yes", "no", "irrelevant"].index(answer)
                    if answer in ["yes", "no", "irrelevant"]
                    else 3
                )
                yield example(
                    "sharc",
                    row["utterance_id"],
                    group,
                    part,
                    {key: row[key] for key in ["snippet", "question", "scenario", "history"]},
                    choice(
                        "Given these rules and the conversation, what should the agent do?",
                        {
                            "yes": "The rules support answering yes.",
                            "no": "The rules support answering no.",
                            "irrelevant": "The rules do not address this question.",
                            "ask_follow_up": "Request missing information before deciding.",
                        },
                    ),
                    label,
                    aliases=["sharc-snippet:" + fingerprint(row["snippet"])],
                )
    with zipfile.ZipFile(BUSINESS_RAW / "contract.zip") as archive:
        for split in ["train", "dev", "test"]:
            data = json.loads(archive.read(f"contract-nli/{split}.json"))
            for document in data["documents"]:
                group = "contract:" + fingerprint(document["text"])
                part = train_split(group, has_dev=True) if split == "train" else split
                for key, annotation in document["annotation_sets"][0]["annotations"].items():
                    yield example(
                        "contract_nli",
                        f"{document['id']}:{key}",
                        group,
                        part,
                        {"contract": document["text"], "claim": data["labels"][key]["hypothesis"]},
                        choice(
                            "How does the full contract relate to the claim?",
                            {
                                "Entailment": "The contract supports the claim.",
                                "Contradiction": "The contract contradicts the claim.",
                                "NotMentioned": "The contract does not establish either conclusion.",
                            },
                        ),
                        ["Entailment", "Contradiction", "NotMentioned"].index(annotation["choice"]),
                    )


def relevance_rows():
    for split in ["train", "validation", "test"]:
        for i, row in enumerate(
            pq.read_table(BUSINESS_RAW / f"wikiqa-{split}.parquet").to_pylist()
        ):
            group = "wikiqa-query:" + fingerprint(row["question"])
            part = (
                train_split(group, has_dev=True)
                if split == "train"
                else {"validation": "dev", "test": "test"}[split]
            )
            yield example(
                "wikiqa",
                f"{split}:{i}",
                group,
                part,
                {"query": row["question"], "passage": row["answer"]},
                noul("Does this passage answer the query?"),
                int(row["label"]),
                aliases=["wikiqa-document:" + fingerprint(row["document_title"])],
            )
    # Freeze whole queries before joining product text. The small official task
    # contains challenging, manually judged customer queries in three locales.
    frame = pq.read_table(BUSINESS_RAW / "esci-examples.parquet").to_pandas()
    frame = frame[frame.small_version == 1].copy()
    queries = {}
    for locale, query, split in frame[["product_locale", "query", "split"]].itertuples(
        index=False, name=None
    ):
        key = (locale, query)
        part = (
            train_split("esci:" + locale + ":" + normalized(query)) if split == "train" else "test"
        )
        # Test takes precedence if upstream query strings span both partitions.
        if queries.get(key) != "test":
            queries[key] = part
    selected, counts = set(), Counter()
    for key in sorted(queries, key=lambda key: digest(json.dumps(key))):
        part = queries[key]
        # Fixed query counts, not label-aware sampling or benchmark tuning.
        cap = 450 if part == "train" else 60
        if counts[(key[0], part)] < cap:
            selected.add(key)
            counts[(key[0], part)] += 1
    frame = frame[
        [key in selected for key in zip(frame.product_locale, frame["query"], strict=True)]
    ]
    needed = set(zip(frame.product_locale, frame.product_id, strict=True))
    products = {}
    for batch in pq.ParquetFile(BUSINESS_RAW / "esci-products.parquet").iter_batches(
        batch_size=4096
    ):
        for row in batch.to_pylist():
            key = (row["product_locale"], row["product_id"])
            if key in needed:
                products[key] = {
                    k.removeprefix("product_"): v or ""
                    for k, v in row.items()
                    if k not in ["product_id", "product_locale"]
                }
    for row in frame.to_dict("records"):
        key = (row["product_locale"], row["query"])
        yield example(
            "esci_" + key[0],
            str(row["example_id"]),
            "esci-query:" + digest(json.dumps(key)),
            queries[key],
            {"query": row["query"], "product": products[(key[0], row["product_id"])]},
            choice(
                "How well does this product match the customer's search?",
                {
                    "E": "Exact match: satisfies the search requirements.",
                    "S": "Substitute: a useful alternative that does not meet every requirement.",
                    "C": "Complement: useful together with the requested product.",
                    "I": "Irrelevant to the search.",
                },
            ),
            ["E", "S", "C", "I"].index(row["esci_label"]),
        )


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ["script", "style"]:
            self.hidden += 1
        if tag in ["br", "p", "div", "tr", "li"]:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ["script", "style"]:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def message_body(message):
    message = email.message_from_bytes(message.as_bytes(), policy=policy.default)
    body = message.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    payload = body.get_payload(decode=True)
    if payload is None:
        return ""
    charset = body.get_content_charset() or "utf-8"
    try:
        text = payload.decode(charset, errors="replace")
    except LookupError:
        text = payload.decode("utf-8", errors="replace")
    if body.get_content_type() == "text/html":
        parser = VisibleText()
        parser.feed(text)
        text = " ".join(parser.parts)
    return " ".join(text.split())


def mail_rows():
    for path in sorted(BUSINESS_RAW.glob("*.tar.bz2")):
        spam = "_spam" in path.name
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or not re.match(r"\d+\.", Path(member.name).name):
                    continue
                raw = archive.extractfile(member).read()
                body = message_body(email.message_from_bytes(raw))
                if not body:
                    continue
                group = "mail:" + fingerprint(body)
                # A family signature suppresses URL/number variants across splits.
                family = "mail-family:" + fingerprint(re.sub(r"https?://\S+|\b\d+\b", "X", body))
                bucket = int(digest(family)[:8], 16) % 100
                part = (
                    "test"
                    if bucket < 15
                    else "dev"
                    if bucket < 25
                    else "calibration"
                    if bucket < 35
                    else "train"
                )
                uid = path.stem + ":" + member.name
                yield example(
                    "mail_spam",
                    uid,
                    group,
                    part,
                    {"email": body},
                    noul("Is this email unsolicited spam or bulk marketing?"),
                    int(spam),
                    aliases=[family],
                )
                if not spam:
                    yield example(
                        "mail_phishing",
                        uid,
                        group,
                        part,
                        {"email": body},
                        noul(
                            "Is this email a phishing or scam attempt to steal money, credentials, or personal data?"
                        ),
                        0,
                        aliases=[family],
                    )
    for year, part in [(2023, "train"), (2024, "dev"), (2025, "test")]:
        for i, message in enumerate(mailbox.mbox(BUSINESS_RAW / f"phishing-{year}", create=False)):
            body = message_body(message)
            if not body:
                continue
            family = "mail-family:" + fingerprint(re.sub(r"https?://\S+|\b\d+\b", "X", body))
            split = train_split(family, has_dev=True) if part == "train" else part
            yield example(
                "mail_phishing",
                f"nazario:{year}:{i}",
                "mail:" + fingerprint(body),
                split,
                {"email": body},
                noul(
                    "Is this email a phishing or scam attempt to steal money, credentials, or personal data?"
                ),
                1,
                aliases=[family],
            )
    with zipfile.ZipFile(BUSINESS_RAW / "sms.zip") as archive:
        for i, line in enumerate(archive.read("SMSSpamCollection").decode().splitlines()):
            label, text = line.split("\t", 1)
            group = "sms:" + fingerprint(text)
            bucket = int(digest(group)[:8], 16) % 100
            part = (
                "test"
                if bucket < 15
                else "dev"
                if bucket < 25
                else "calibration"
                if bucket < 35
                else "train"
            )
            yield example(
                "sms_spam",
                str(i),
                group,
                part,
                text,
                noul("Is this message unsolicited spam?"),
                int(label == "spam"),
            )


def prepare_original(output):
    global OUT
    OUT = output
    sources = ["text", "vision", "clevr", "screen"]
    OUT.mkdir(parents=True, exist_ok=True)
    audit = Counter()
    generators = {
        "text": text_sources,
        "vision": vision_sources,
        "clevr": clevr_sources,
        "screen": lambda: screen_sources(audit),
    }
    data = []
    for source in sources:
        records = list(generators[source]())
        data.extend(records)
        print(source, len(records), flush=True)
    counts = Counter()
    handles = {split: (OUT / f"{split}.jsonl").open("w") for split in SPLIT_ORDER}
    try:
        for r in isolate(data, audit):
            handles[r["split"]].write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[(r["dataset"], r["split"])] += 1
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "sources": sources,
        "counts": [{"dataset": d, "split": s, "n": n} for (d, s), n in sorted(counts.items())],
        "exclusions": dict(sorted(audit.items())),
        "split_seed": "doh-split-2026",
        "schema_version": 1,
        "image_isolation": "exact bytes and decoded Rico pixels",
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def filter_data(source, output, model):
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    image_tokens = {}
    counts, excluded = Counter(), Counter()
    sample_hashes = {}
    with (output / "excluded.jsonl").open("w") as exclusions:
        for split in ["train", "dev", "calibration", "test"]:
            with (
                (source / f"{split}.jsonl").open() as input_stream,
                (output / f"{split}.jsonl").open("w") as out,
            ):
                batch = []

                def process(records, split=split):
                    prompts = [
                        render_question(
                            render(r["state"]), r["question"], has_image=bool(r["image"])
                        )[0]
                        for r in records
                    ]
                    tokens = processor.tokenizer(prompts, truncation=False, padding=False)[
                        "input_ids"
                    ]
                    for r, ids in zip(records, tokens, strict=True):
                        length = len(ids)
                        if r["image"]:
                            path = r["image"]
                            if path not in image_tokens:
                                with Image.open(path) as image:
                                    w, h = smart_resize(
                                        image.height,
                                        image.width,
                                        factor=factor,
                                        min_pixels=IMAGE_PIXELS,
                                        max_pixels=IMAGE_PIXELS,
                                    )[::-1]
                                image_tokens[path] = (h // factor) * (w // factor)
                            length += image_tokens[path] - 1
                        if length > MAX_LENGTH:
                            excluded[(r["dataset"], split)] += 1
                            exclusions.write(
                                json.dumps(
                                    {
                                        "id": r["id"],
                                        "split": split,
                                        "reason": "token_budget",
                                        "tokens": length,
                                    }
                                )
                                + "\n"
                            )
                            continue
                        r["tokens"] = length
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        counts[(r["dataset"], split)] += 1

                for line in input_stream:
                    batch.append(json.loads(line))
                    if len(batch) == 512:
                        process(batch)
                        batch = []
                if batch:
                    process(batch)
            with (output / f"{split}.jsonl").open("rb") as stream:
                sample_hashes[split] = hashlib.file_digest(stream, "sha256").hexdigest()
            print(split, "complete", flush=True)
    manifest = {
        "input": str(source),
        "max_length": MAX_LENGTH,
        "image_pixels": IMAGE_PIXELS,
        "resize_factor": factor,
        "counts": [{"dataset": d, "split": s, "n": n} for (d, s), n in sorted(counts.items())],
        "exclusions": [
            {"dataset": d, "split": s, "n": n, "reason": "token_budget"}
            for (d, s), n in sorted(excluded.items())
        ],
        "sha256": sample_hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def prepare_mixture(base, output, model):
    output.mkdir(parents=True, exist_ok=True)
    blocked = blocked_text(base)
    audit = Counter()
    rows = []
    for builder in [policy_rows, relevance_rows, mail_rows]:
        rows.extend(builder())
        print(json.dumps({"source": builder.__name__, "cumulative_rows": len(rows)}), flush=True)
    for row in rows:
        # Laya's application protocol cuts bodies at 3,000 characters.
        if any(
            fingerprint(variant) in blocked
            for text in strings(row["state"])
            for variant in [text, text[:3000]]
            if len(variant.strip()) >= 80
        ):
            row["benchmark_overlap"] = True
    # Propagate aliases before excluding a group, including duplicate documents.
    isolated = list(isolate(rows, audit))
    # isolate mutates all rows to their union roots, including dropped rows.
    rejected_groups = {row["group"] for row in rows if row.pop("benchmark_overlap", False)}
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    counts, classes = Counter(), defaultdict(Counter)
    handles = {
        split: (output / f"{split}.jsonl").open("w")
        for split in ["train", "dev", "calibration", "test"]
    }
    try:
        for start in range(0, len(isolated), 256):
            batch = isolated[start : start + 256]
            tokens = tokenizer(
                [render_question(render(row["state"]), row["question"])[0] for row in batch],
                truncation=False,
            )["input_ids"]
            for row, ids in zip(batch, tokens, strict=True):
                if row["group"] in rejected_groups:
                    audit[row["dataset"] + ":frozen_benchmark_overlap"] += 1
                    continue
                if len(ids) > MAX_LENGTH:
                    audit[row["dataset"] + ":token_budget"] += 1
                    continue
                row["tokens"] = len(ids)
                handles[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[(row["dataset"], row["split"])] += 1
                classes[row["dataset"] + ":" + row["split"]][str(row["target"].index(1.0))] += 1
    finally:
        for handle in handles.values():
            handle.close()
    new_counts = counts.copy()
    original = load_records(base / "train.jsonl", 6000)
    replay_ids = {row["id"] for group in original.values() for row in group}
    for split in handles:
        with (
            (base / f"{split}.jsonl").open() as source,
            (output / f"{split}.jsonl").open("a") as out,
        ):
            for line in source:
                row = json.loads(line)
                if split != "train" or row["id"] in replay_ids:
                    out.write(line)
                    counts[(row["dataset"], split)] += 1
    report = {
        "counts": [{"dataset": d, "split": s, "n": n} for (d, s), n in sorted(counts.items())],
        "new_counts": [
            {"dataset": d, "split": s, "n": n} for (d, s), n in sorted(new_counts.items())
        ],
        "class_counts": classes,
        "exclusions": dict(sorted(audit.items())),
        "max_length": MAX_LENGTH,
        "benchmark_guard": "Laya/JevBench frozen inputs, sampled training inputs and original heldout inputs; text leaves >=80 characters, punctuation-insensitive exact match including 3000-character prefixes; source/alias groups isolated. Not semantic near-duplicate proof.",
        "scope": "The 26-group training mixture; fixed original training pool and heldout partitions, public business-task groups, and no benchmark examples as training data.",
    }
    report["sha256"] = {}
    for split in handles:
        with (output / f"{split}.jsonl").open("rb") as stream:
            report["sha256"][split] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=BASE_MODEL)
    args = parser.parse_args()
    download_sources()
    extract_archives()
    sources = args.output / "sources"
    base = args.output / "base"
    prepare_original(sources)
    filter_data(sources, base, args.model)
    prepare_mixture(base, args.output, args.model)


if __name__ == "__main__":
    main()
