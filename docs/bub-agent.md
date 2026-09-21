# Agent integration with Bub main

Dohnuts provides one decision tool through Bub. The `agent` extra pins Bub main at
`9bf70488e22a44eeb875a9d55d9a7d352b82e381`. Integration uses its
[Python SDK](https://github.com/bubbuild/bub/blob/9bf70488e22a44eeb875a9d55d9a7d352b82e381/website/src/content/docs/docs/build/sdk.md)
to provide one batch decision tool and persistent session tapes.

Follow the [runtime and checkpoint setup](inference.md) first.

```python
from pathlib import Path

from dohnuts.bub_agent import create_agent
from dohnuts.predictor import Predictor

predictor = Predictor.from_checkpoint("PsiACE/Dohnuts-0.1.0-0.8B")
framework, agent = create_agent(
    predictor, workspace=Path.cwd(), tape_directory=Path("runs/agent-tapes")
)

async def decide():
    command = ',dohnuts.decide state=\'{"message":"Please refund this invoice."}\' '
    command += 'questions=\'{"refund":{"type":"noul","instructions":"Is a refund requested?"}}\''
    async with framework.running():
        stream = await agent.run_stream(session_id="billing", prompt=command)
        return [event async for event in stream]
```

The tool is named `dohnuts.decide`; its model-facing alias is `dohnuts_decide`. Supply
independent questions together against one state. Its optional `image_path`
argument names an image within the application's workspace. The returned
probabilities retain the predictor's meanings. Calls are serialized per predictor
instance, while each call batches its questions.

An application may configure Bub's planner/provider and call `run_stream` with a
natural-language prompt. The application owns the framework lifetime and decides
how to act on the returned distributions.

The explicit comma-command calls the decision tool without an external planner.
Planner quality and provider cost depend on the application's provider settings.
