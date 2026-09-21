"""The prefix cache preserves the Transformers cache API and autograd graph."""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from dohnuts.execution import PrefixCache


@pytest.fixture
def prefix_cache():
    config = Qwen3_5TextConfig(num_hidden_layers=1, layer_types=["linear_attention"])
    return PrefixCache(config=config)


def test_keyword_updates_preserve_states_and_gradients(prefix_cache):
    prefix = torch.randn(1, 2, 6, requires_grad=True)
    suffix = torch.randn(1, 2, 2, requires_grad=True)
    prefix_cache.update_conv_state(conv_states=prefix, layer_idx=0, conv_kernel_size=4)
    previous = prefix_cache.layers[0].conv_states[0]
    updated = prefix_cache.update_conv_state(conv_states=suffix, layer_idx=0, conv_kernel_size=4)
    torch.testing.assert_close(updated, torch.cat([prefix[..., -4:], suffix], dim=-1))
    torch.testing.assert_close(previous, prefix[..., -4:])
    torch.testing.assert_close(prefix_cache.layers[0].conv_states[0], updated[..., -4:])

    recurrent = torch.randn(1, 2, 2, requires_grad=True)
    state = recurrent.square()
    returned = prefix_cache.update_recurrent_state(recurrent_states=state, layer_idx=0)
    assert returned is state
    assert prefix_cache.layers[0].recurrent_states[0] is state
    (updated.sum() + returned.sum()).backward()
    expected = torch.ones_like(prefix)
    expected[..., :2] = 0
    torch.testing.assert_close(prefix.grad, expected)
    torch.testing.assert_close(suffix.grad, torch.ones_like(suffix))
    torch.testing.assert_close(recurrent.grad, 2 * recurrent.detach())


def test_missing_kernel_size_fails_before_mutating_cache(prefix_cache):
    with pytest.raises(ValueError, match="conv_kernel_size"):
        prefix_cache.update_conv_state(conv_states=torch.ones(1, 2, 4), layer_idx=0)
    assert not prefix_cache.layers[0].is_conv_states_initialized[0]
