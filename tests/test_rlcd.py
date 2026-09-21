import json
from pathlib import Path

import pytest
import torch

from dohnuts.rlcd import RLCDConfig, distribution_rewards, rlcd_loss

CASES = json.loads((Path(__file__).parent / "fixtures/laya-rlcd.json").read_text())["cases"]


@pytest.mark.parametrize(
    "case", CASES, ids=lambda case: f"seed-{case['seed']}-ce-{case['ce_weight']}"
)
def test_loss_reward_and_gradient_match_pinned_laya(case):
    """Golden values were evaluated by the actual upstream Python functions."""
    logits = torch.tensor(case["logits"], requires_grad=True)
    torch.manual_seed(case["seed"])
    loss, metrics = rlcd_loss(
        logits,
        torch.tensor(case["targets"]),
        mask=torch.tensor(case["mask"]),
        ordinal=torch.tensor(case["ordinal"]),
        config=RLCDConfig(ce_weight=case["ce_weight"]),
    )
    loss.backward()
    torch.testing.assert_close(loss.detach(), torch.tensor(case["loss"]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(metrics["reward_mean"], torch.tensor(case["reward_mean"]))
    torch.testing.assert_close(logits.grad, torch.tensor(case["gradient"]), atol=1e-6, rtol=1e-5)


def test_ordinal_reward_respects_distance():
    config = RLCDConfig()
    actions = torch.tensor([[[10.0, -10, -10, -10]], [[-10.0, -10, 10, -10]]])
    rewards = distribution_rewards(
        actions,
        torch.tensor([3]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        config,
    )
    assert rewards[1, 0].item() > rewards[0, 0].item()
