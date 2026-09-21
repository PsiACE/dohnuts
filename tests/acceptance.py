"""Accept an exported checkpoint through prediction, reload, and the real Bub SDK."""

import argparse
import ast
import asyncio
import gc
import json
import os
import shlex
from pathlib import Path

import torch
from PIL import Image

from dohnuts.bub_agent import BUB_REVISION, create_agent
from dohnuts.predictor import Predictor


def check_response(response, questions):
    assert response["model"] == "dohnuts", "Responses must identify the model family"
    answers = response["answers"]
    assert set(answers) == set(questions), "Every submitted question must have an answer"
    for key, question in questions.items():
        answer = answers[key]
        assert answer["type"] == question["type"]
        assert 0 <= answer["confidence"] <= 1
        if question["type"] == "noul":
            assert 0 <= answer["noul"] <= 1
        else:
            probabilities = answer["probabilities"]
            assert all(0 <= value <= 1 for value in probabilities.values())
            assert abs(sum(probabilities.values()) - 1) < 1e-5
            if question["type"] == "choice":
                assert set(probabilities) == set(question["criteria"])
                assert answer["choice"] == max(probabilities, key=probabilities.get)
            else:
                assert 0 <= answer["score"] <= len(question["criteria"]) - 1


async def verify(args):
    os.environ["BUB_HOME"] = str((args.output / "bub-home").resolve())
    torch.set_num_threads(8)
    predictor = Predictor.from_checkpoint(args.checkpoint)
    metadata = predictor.metadata
    state = {"message": "I was charged twice for invoice 4411. Please refund the duplicate."}
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle the request?",
            "criteria": ["billing", "technical support", "sales"],
        },
        "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
        "charges": {
            "type": "score",
            "instructions": "How many charges does the customer report?",
            "criteria": ["zero", "one", "two", "three or more"],
        },
    }
    expected = predictor.predict(state, questions)
    check_response(expected, questions)
    image_path = args.output / "red-square.png"
    Image.new("RGB", (96, 96), "red").save(image_path)
    image_questions = {
        "color": {
            "type": "choice",
            "instructions": "What color is the image?",
            "criteria": ["red", "blue", "green"],
        },
        "red": {"type": "noul", "instructions": "Is the image red?"},
        "count": {
            "type": "score",
            "instructions": "How many colors fill the image?",
            "criteria": ["zero", "one", "two", "three or more"],
        },
    }
    with Image.open(image_path) as image:
        image_response = predictor.predict({"image": image.convert("RGB")}, image_questions)
    check_response(image_response, image_questions)
    shared_state = {"message": "Context note. " * 100 + state["message"]}
    shared_response = predictor.predict(shared_state, questions)
    check_response(shared_response, questions)
    predictor.predict({"message": "Unrelated request. " * 100}, questions)
    assert predictor.predict(shared_state, questions) == shared_response, (
        "Another request changed the answer"
    )
    predictor.predict({"image": Image.new("RGB", (96, 96), "blue")}, image_questions)
    with Image.open(image_path) as image:
        repeated = predictor.predict({"image": image.convert("RGB")}, image_questions)
    assert repeated == image_response, "Another image request changed the answer"
    long_questions = {"route": questions["route"]}
    long_response = predictor.predict(
        {"message": "Context note. " * 800 + state["message"]}, long_questions
    )
    check_response(long_response, long_questions)
    assert long_response["usage"]["input_tokens"] > 2048, "Exercise input beyond training length"
    try:
        predictor.predict("word " * (predictor.max_length + 1), questions)
    except ValueError:
        pass
    else:
        raise AssertionError("An oversized request must be rejected without silent truncation")

    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    predictor = Predictor.from_checkpoint(args.checkpoint)
    assert predictor.predict(state, questions) == expected, "Reload changed the decision response"
    framework, agent = create_agent(predictor, Path.cwd(), args.output / "tapes")
    responses = []
    async with framework.running():
        for label, request_state, request_questions, reference in [
            ("text", state, questions, expected),
            ("image", {}, image_questions, image_response),
        ]:
            command = ",dohnuts.decide state=" + shlex.quote(json.dumps(request_state))
            command += " questions=" + shlex.quote(json.dumps(request_questions))
            if label == "image":
                command += " image_path=" + shlex.quote(str(image_path.resolve()))
            stream = await agent.run_stream(session_id="acceptance-" + label, prompt=command)
            events = [event async for event in stream]
            actual = ast.literal_eval(events[-1].data["text"])
            assert actual == reference, "Bub changed the decision response"
            responses.append(actual)
    report = {
        "version": metadata["version"],
        "model_id": metadata["model_id"],
        "weights_sha256": metadata["weights_sha256"],
        "checkpoint": str(args.checkpoint),
        "acceptance_pass_rate": 1.0,
        "accepted": [
            "text and image decisions",
            "all three decision types",
            "candidate distributions",
            "input beyond the training token budget",
            "oversized request rejection",
            "checkpoint reload",
            "independent text requests",
            "independent image requests",
            "Bub decision-tool dispatch",
        ],
        "bub_dispatch_equal_direct_prediction": True,
        "input_limit": predictor.max_length,
        "long_input_tokens": long_response["usage"]["input_tokens"],
        "bub_revision": BUB_REVISION,
        "scope": "interface acceptance; semantic quality is measured on held-out datasets",
        "responses": responses,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint directory or Hub ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(verify(args))


if __name__ == "__main__":
    main()
