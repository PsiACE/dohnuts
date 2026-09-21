"""Joint RLCD and cross-entropy loss aligned with pinned Laya references.

See docs/rlcd.md for upstream revisions, reduction semantics, and limitations.
"""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class RLCDConfig:
    """The two RLCD controls; the scoring rule and estimator are fixed."""

    samples: int = field(default=4, init=False)
    sigma: float = 0.3
    log_weight: float = field(default=1.0, init=False)
    spherical_weight: float = field(default=0.75, init=False)
    ordinal_weight: float = field(default=1.0, init=False)
    ce_weight: float = 1.0
    log_floor: float = field(default=-9.21, init=False)

    def __post_init__(self):
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        if not math.isfinite(self.ce_weight) or self.ce_weight < 0:
            raise ValueError("ce_weight must be nonnegative and finite")


def distribution_rewards(
    actions: Tensor,
    targets: Tensor,
    mask: Tensor,
    ordinal: Tensor,
    config: RLCDConfig,
) -> Tensor:
    """Score [G,B,K] sampled logits; padded candidates have zero probability.

    Counts use semantic ascending order. RPS excludes the final (always-one)
    cumulative probability and is divided by K-1, using each row's actual K.
    """
    log_probs = actions.float().masked_fill(~mask, -torch.inf).log_softmax(-1)
    probabilities = log_probs.exp()
    if targets.ndim == 1:
        targets = torch.nn.functional.one_hot(targets, actions.shape[-1]).float()
    log_score = probabilities.clamp_min(1e-12).log().clamp_min(config.log_floor)
    log_score = (log_score * targets).sum(-1)
    target_probability = (probabilities * targets).sum(-1)
    spherical = target_probability / probabilities.norm(dim=-1).clamp_min(1e-9)
    reward = config.log_weight * log_score + config.spherical_weight * spherical
    levels = torch.arange(actions.shape[-1], device=actions.device)
    target_cdf = targets.cumsum(-1)
    counts = mask.sum(-1)
    cdf_mask = levels[None, :] < (counts - 1)[:, None]
    rps = ((probabilities.cumsum(-1) - target_cdf).square() * cdf_mask).sum(-1)
    rps = rps / (counts - 1).clamp_min(1)
    return reward - config.ordinal_weight * rps * ordinal


def rlcd_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Tensor | None = None,
    ordinal: Tensor | None = None,
    config: RLCDConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """REINFORCE estimate; G perturbations never repeat the backbone forward.

    Actions AND rewards are detached. Without detaching sampled actions, the
    location parameter cancels in the Gaussian log density and its gradient
    vanishes. Match Laya's per-question group-mean baseline, then normalize with
    ONE sample standard deviation across G and B. This practical normalization,
    reward clipping, and auxiliary CE do not guarantee calibrated probabilities.
    """
    if config is None:
        config = RLCDConfig()
    if logits.ndim != 2 or targets.shape not in (logits.shape[:1], logits.shape):
        raise ValueError("Expected logits [B,K] and targets [B] or distributions [B,K]")
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    if ordinal is None:
        ordinal = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    mean = logits.float()
    noise = torch.randn((config.samples, *mean.shape), device=mean.device) * config.sigma * mask
    noise = (noise - noise.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)) * mask
    actions = (mean.detach()[None] + noise).detach()
    with torch.no_grad():
        rewards = distribution_rewards(actions, targets, mask, ordinal, config)
        advantages = rewards - rewards.mean(0, keepdim=True)
        advantages = advantages / (advantages.std() + 1e-6)
    # Gaussian normalizing constants are independent of trainable parameters.
    log_density = -0.5 * ((actions - mean[None]) / config.sigma).square()
    log_density = log_density.masked_fill(~mask, 0).sum(-1)
    policy_loss = -(advantages * log_density).mean()
    ce_loss = -mean.masked_fill(~mask, -torch.inf).log_softmax(-1)
    if targets.ndim == 1:
        ce_loss = ce_loss.gather(-1, targets[:, None]).mean()
    else:
        ce_loss = (ce_loss.masked_fill(~mask, 0) * targets).sum(-1).mean()
    loss = policy_loss + config.ce_weight * ce_loss
    return loss, {
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std(unbiased=False),
        "advantage_rms": advantages.square().mean().sqrt(),
        "policy_loss": policy_loss.detach(),
        "ce_loss": ce_loss.detach(),
    }
