"""Bub main SDK composition for agents that call the Dohnuts batch decision tool."""

import asyncio
import json
import threading
from pathlib import Path

from PIL import Image

BUB_REVISION = "9bf70488e22a44eeb875a9d55d9a7d352b82e381"


def create_agent(predictor, workspace: Path, tape_directory: Path):
    """Return (framework, SDK agent); caller owns framework.running() lifetime.

    Bub's model/provider configuration remains the application's responsibility.
    This instance exposes the decision tool only. All questions in one tool call
    share state and are submitted together to the non-generative decision model.
    """
    from bub import BubFramework, hookimpl
    from bub.builtin import Agent
    from bub.builtin.hook_impl import BuiltinImpl
    from bub.store import FileTapeStore
    from bub.tools import Tool

    workspace = workspace.resolve()
    lock = threading.Lock()

    def predict(state, questions, image_path):
        if image_path is not None:
            path = (workspace / image_path).resolve()
            if not path.is_relative_to(workspace):
                raise ValueError("Image must be inside the configured workspace")
            with Image.open(path) as image:
                state = {**state, "image": image.convert("RGB")}
        with lock:
            return predictor.predict(state, questions)

    async def decide(
        state: dict | str, questions: dict | str, image_path: str | None = None
    ) -> dict:
        """Return candidate selections, truth estimates, or ordered scores for shared state.

        questions maps IDs to {type, instructions, criteria}. Choice criteria map
        candidate labels to descriptions; score criteria list levels in ascending
        order; noul optionally describes false and true. Supply all independent
        questions together. Returns model probabilities; it does not execute the selected actions.
        image_path optionally names an image relative to this application's workspace.
        """
        state = json.loads(state) if isinstance(state, str) else state
        questions = json.loads(questions) if isinstance(questions, str) else questions
        if not isinstance(state, dict) or not isinstance(questions, dict):
            raise TypeError("state and questions must be JSON objects")
        return await asyncio.to_thread(predict, state, questions, image_path)

    class DecisionPrompt(BuiltinImpl):
        @hookimpl
        def system_prompt(self, prompt, state):
            return (
                "Use dohnuts_decide to evaluate decisions. Batch independent questions over "
                "the same state into one tool call. Preserve returned probabilities and "
                "candidate meanings in your response. State any uncertainty. "
                "A decision distribution alone does not authorize external actions."
            )

    framework = BubFramework(config_file=workspace / "bub.yml")
    framework.workspace = workspace
    framework.plugin_manager.register(DecisionPrompt(framework), name="dohnuts_prompt")
    agent = Agent(
        framework,
        tools=[Tool.from_callable(decide, name="dohnuts.decide")],
        skill_dirs=[],
        tape_store=FileTapeStore(tape_directory),
    )
    return framework, agent
