# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_mhc_kernels import is_cutile_available, is_triton_available
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig

_FUSED_BACKENDS = ["auto", "native"]
if is_triton_available():
    _FUSED_BACKENDS.append("triton")
if is_cutile_available():
    _FUSED_BACKENDS.append("cutile")


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        layernorm_epsilon=1e-5,
        enable_mhc_connections=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _reference_mappings(layer, residual, eps, eps_inside_sqrt, dtype):
    """Unfused FP32 compute_mappings arithmetic for a [s, b, 4 * 32] residual."""
    x = residual.float()
    rms = (
        torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        if eps_inside_sqrt
        else (x.norm(dim=-1, keepdim=True) / 128**0.5 + eps).reciprocal()
    )
    scales = torch.cat(
        [layer.alpha_pre.expand(4), layer.alpha_post.expand(4), layer.alpha_res.expand(16)]
    )
    logits = (x @ layer.mapping_proj.weight.T) * rms * scales + layer.bias
    pre = (logits[..., :4].sigmoid() + 1e-6).to(dtype)
    post = (2 * logits[..., 4:8].sigmoid()).to(dtype)
    comb = logits[..., 8:].reshape(*residual.shape[:2], 4, 4).softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(layer.sinkhorn_iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return pre, post, comb.to(dtype)


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)


@pytest.mark.parametrize("use_fused_mhc", [False, True])
@pytest.mark.parametrize("fp32_mixing", [False, True])
@pytest.mark.parametrize("input_scale", [1.0, 0.001])
def test_mhc_precision(use_fused_mhc, fp32_mixing, input_scale):
    torch.manual_seed(42)
    config = _config(
        use_fused_mhc=use_fused_mhc,
        mhc_norm_eps_inside_sqrt=fp32_mixing,
        mhc_keep_mappings_in_fp32=fp32_mixing,
    )
    layer = HyperConnectionModule(config, layer_number=1).cuda()
    expected_eps = config.layernorm_epsilon if fp32_mixing else 1e-6
    assert layer.norm_eps == expected_eps
    with torch.no_grad():
        layer.mapping_proj.weight.normal_(std=0.2)
        layer.bias.normal_()
    residual = (torch.randn(16, 2, 128, device="cuda") * input_scale).to(torch.bfloat16)
    residual.requires_grad_()
    output = torch.randn(16, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pre, post, comb = layer.compute_mappings(residual)
    expected_dtype = torch.float32 if fp32_mixing else torch.bfloat16
    assert pre.dtype == post.dtype == comb.dtype == expected_dtype

    expected_pre, expected_post, expected_comb = _reference_mappings(
        layer, residual, expected_eps, fp32_mixing, expected_dtype
    )
    # The fused mapping projection runs on TF32 tensor cores; the unfused path is FP32.
    if not use_fused_mhc:
        mapping_tol = {}
    elif expected_dtype == torch.float32:
        mapping_tol = {"rtol": 2e-3, "atol": 2e-4}
    else:
        mapping_tol = {"rtol": 1.6e-2, "atol": 1e-3}
    torch.testing.assert_close(pre, expected_pre, **mapping_tol)
    torch.testing.assert_close(post, expected_post, **mapping_tol)
    torch.testing.assert_close(comb, expected_comb, **mapping_tol)

    aggregated = layer.aggregate(residual, pre)
    assert aggregated.dtype == residual.dtype
    streams = residual.view(16, 2, 4, 32)
    expected_aggregate = (streams.float() * pre.float().unsqueeze(-1)).sum(2).to(residual.dtype)
    torch.testing.assert_close(aggregated, expected_aggregate, rtol=0.02, atol=0.01)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp32_mixing):
        actual = layer.fused_h_res_h_post_bda(
            comb, residual, post, (output, None), 0.0, True, False
        )
    # The mixed streams re-enter the residual stream in its own dtype.
    assert actual.dtype == residual.dtype
    expected_mix = torch.einsum("...ij,...ih->...jh", comb, streams.to(expected_dtype))
    expected_output = (
        expected_mix.float() + post.float().unsqueeze(-1) * output.float().unsqueeze(2)
    ).to(residual.dtype)
    torch.testing.assert_close(actual.view_as(streams), expected_output, rtol=0.02, atol=0.01)
    (actual.float().square().mean() + aggregated.float().square().mean()).backward()
    for tensor in (residual, output, layer.mapping_proj.weight, layer.bias):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("backend", _FUSED_BACKENDS)
def test_mhc_fused_matches_unfused_with_precision_flags(backend):
    """Fused mHC honours both precision flags and matches the unfused module end to end."""
    torch.manual_seed(7)
    flags = dict(mhc_norm_eps_inside_sqrt=True, mhc_keep_mappings_in_fp32=True)
    unfused = HyperConnectionModule(_config(**flags), layer_number=1).cuda()
    fused = HyperConnectionModule(
        _config(**flags, use_fused_mhc=True, mhc_fused_backend=backend), layer_number=1
    ).cuda()
    # Same fused module with the epsilon outside the square root: on small rows the two
    # norm definitions differ by several x, which shows the flag reaches the fused kernels.
    fused_eps_outside = HyperConnectionModule(
        _config(mhc_keep_mappings_in_fp32=True, use_fused_mhc=True, mhc_fused_backend=backend),
        layer_number=1,
    ).cuda()
    with torch.no_grad():
        unfused.mapping_proj.weight.normal_(std=0.2)
        unfused.bias.normal_()
        for alpha in (unfused.alpha_pre, unfused.alpha_post, unfused.alpha_res):
            alpha.fill_(1.0)
    fused.load_state_dict(unfused.state_dict())
    fused_eps_outside.load_state_dict(unfused.state_dict())

    # Rows with mean(x**2) close to layernorm_epsilon = 1e-5.
    residual = (torch.randn(16, 2, 128, device="cuda") * 3e-3).to(torch.bfloat16)
    output = torch.randn(16, 2, 32, device="cuda", dtype=torch.bfloat16)

    def run(module):
        x = residual.clone().requires_grad_()
        y = output.clone().requires_grad_()
        aggregated, h_res, h_post = module(x)
        out = module.fused_h_res_h_post_bda(h_res, x, h_post, (y, None), 0.0, True, False)
        (out.float().square().mean() + aggregated.float().square().mean()).backward()
        grads = {name: p.grad for name, p in module.named_parameters()}
        return aggregated, h_res, h_post, out, x.grad, y.grad, grads

    expected = run(unfused)
    actual = run(fused)

    aggregated, h_res, h_post, out, x_grad, y_grad, grads = actual
    assert h_res.dtype == h_post.dtype == torch.float32
    assert aggregated.dtype == out.dtype == x_grad.dtype == torch.bfloat16
    # Mapping projection on TF32 tensor cores for the cuTile kernel, FP32 otherwise.
    torch.testing.assert_close(h_res, expected[1], rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(h_post, expected[2], rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(aggregated, expected[0], rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(out, expected[3], rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(x_grad, expected[4], rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(y_grad, expected[5], rtol=2e-2, atol=1e-3)
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), name
        assert _cosine(grad, expected[6][name]) > 0.999, name
        torch.testing.assert_close(grad, expected[6][name], rtol=2e-2, atol=1e-4, msg=name)

    _, h_res_outside, h_post_outside, *_ = run(fused_eps_outside)
    assert not torch.allclose(h_post_outside, h_post, rtol=1e-2, atol=1e-2)
    assert not torch.allclose(h_res_outside, h_res, rtol=1e-2, atol=1e-2)
