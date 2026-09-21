"""Fused differentiable Qwen operations shared by training and inference."""

from types import MethodType


def fused_rms_norm(module, hidden):
    from fla.modules.layernorm import rms_norm

    return rms_norm(hidden, module._dohnuts_norm_weight, None, eps=module.eps)


def fused_gated_norm(module, hidden, gate):
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated(hidden, gate, module.weight, None, eps=module.variance_epsilon)


def fused_mlp(module, hidden):
    from fla.modules.activations import swiglu

    # Keep the projection modules in the graph: they own the trainable LoRA.
    return module.down_proj(swiglu(module.gate_proj(hidden), module.up_proj(hidden)))


def enable_fusion(backbone):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    functions = {
        qwen.Qwen3_5RMSNorm: fused_rms_norm,
        qwen.Qwen3_5RMSNormGated: fused_gated_norm,
        qwen.Qwen3_5MLP: fused_mlp,
    }
    for module in backbone.modules():
        if isinstance(module, qwen.Qwen3_5RMSNorm):
            module.register_buffer(
                "_dohnuts_norm_weight", module.weight.detach().float() + 1, persistent=False
            )
        if type(module) in functions:
            module.forward = MethodType(functions[type(module)], module)


def linear_patch_embedding(module, hidden_states):
    """The patch Conv3d has exactly one output per independent input patch.

    Flattening its kernel is the same affine map, without a convolution solver
    search for every total patch count. No weights or tokens are added/removed.
    """
    from torch.nn import functional

    weight = module.proj.weight
    patches = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    return functional.linear(patches, weight.flatten(1), module.proj.bias)


def enable_linear_patch_embedding(visual):
    module = visual.patch_embed
    projection = module.proj
    if (
        projection.kernel_size != projection.stride
        or projection.groups != 1
        or any(projection.padding)
        or projection.dilation != (1, 1, 1)
    ):
        raise ValueError("Linear patch embedding requires non-overlapping independent patches")
    module.forward = MethodType(linear_patch_embedding, module)


def triton_causal_conv1d(hidden_states, weight, bias=None, activation=None, **kwargs):
    from fla.modules.conv import causal_conv1d

    output, _ = causal_conv1d(
        hidden_states.transpose(1, 2),
        weight=weight,
        bias=bias,
        activation=activation,
        backend="triton",
        output_final_state=False,
        cu_seqlens=kwargs.get("cu_seq_lens_q"),
    )
    return output.transpose(1, 2)


def enable_triton_convolution():
    """Explicit process-local override; applies only to full-sequence prefill."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    modeling_qwen3_5.causal_conv1d_fn = triton_causal_conv1d
