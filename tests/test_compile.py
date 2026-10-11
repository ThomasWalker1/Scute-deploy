"""The admission compiler, on a tiny Llama geometry: every method's normal form is the
function PEFT computes."""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from scute.compile import PREFIX, SHARED_FILE, base_shapes, compile_adapter

CFG = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4,
           num_key_value_heads=2, num_hidden_layers=2)
SH = base_shapes(CFG)


def _mod(layer, name):
    return f"model.layers.{layer}.{'self_attn' if name[0] in 'qkvo' else 'mlp'}.{name}"


def _write(tmp, cfg, tensors):
    os.makedirs(tmp, exist_ok=True)
    json.dump(cfg, open(os.path.join(tmp, "adapter_config.json"), "w"))
    save_file(tensors, os.path.join(tmp, "adapter_model.safetensors"))
    return tmp


def test_shapes():
    assert SH["k_proj"] == (64, 32) and SH["down_proj"] == (128, 64)


def test_ia3_feedforward_moves_to_up(tmp_path):
    g = torch.Generator().manual_seed(0)
    t = {}
    for layer in range(2):
        t[f"{PREFIX}{_mod(layer, 'k_proj')}.ia3_l"] = torch.rand(32, 1, generator=g)
        t[f"{PREFIX}{_mod(layer, 'down_proj')}.ia3_l"] = torch.rand(1, 128, generator=g)
    src = _write(str(tmp_path / "src"), dict(peft_type="IA3", target_modules=["k_proj", "down_proj"],
                                           feedforward_modules=["down_proj"]), t)
    c = compile_adapter(src, str(tmp_path / "out"), CFG)
    out = load_file(os.path.join(c.out_dir, "adapter_model.safetensors"))
    for layer in range(2):
        up = f"{PREFIX}{_mod(layer, 'up_proj')}.scute_s"
        assert torch.equal(out[up], t[f"{PREFIX}{_mod(layer, 'down_proj')}.ia3_l"].flatten())
        assert f"{PREFIX}{_mod(layer, 'down_proj')}.scute_s" not in out
        assert out[f"{PREFIX}{_mod(layer, 'up_proj')}.lora_B.weight"].abs().sum() == 0
    # the identity behind the mapping: act(gate) * (up * l) == (act(gate) * up) * l
    x = torch.randn(5, 64)
    wg, wu, l = torch.randn(128, 64), torch.randn(128, 64), torch.rand(128)
    lhs = (torch.nn.functional.silu(x @ wg.T) * (x @ wu.T)) * l
    rhs = torch.nn.functional.silu(x @ wg.T) * ((x @ wu.T) * l)
    torch.testing.assert_close(lhs, rhs)


def test_vera_vectors_and_shared(tmp_path):
    g = torch.Generator().manual_seed(1)
    r = 8
    va, vb = torch.randn(r, 128, generator=g), torch.randn(128, r, generator=g)
    t = {f"{PREFIX}vera_A": va, f"{PREFIX}vera_B": vb}
    for name in ("q_proj", "down_proj"):
        d_out = SH[name][1]
        t[f"{PREFIX}{_mod(0, name)}.vera_lambda_d"] = torch.randn(r, generator=g)
        t[f"{PREFIX}{_mod(0, name)}.vera_lambda_b"] = torch.randn(d_out, generator=g)
    src = _write(str(tmp_path / "src"), dict(peft_type="VERA", r=r, target_modules=["q_proj"]), t)
    c = compile_adapter(src, str(tmp_path / "out"), CFG)
    out = load_file(os.path.join(c.out_dir, "adapter_model.safetensors"))
    sh = load_file(os.path.join(c.out_dir, SHARED_FILE))
    assert torch.equal(sh["vera_A"], va) and torch.equal(sh["vera_B"], vb) and c.shared_key
    # normal form reproduces PEFT's lambda_b * ((lambda_d * x A^T) B^T) for down_proj
    m = f"{PREFIX}{_mod(0, 'down_proj')}"
    d_in, d_out = SH["down_proj"]
    x = torch.randn(3, d_in)
    peft = t[m + ".vera_lambda_b"] * ((t[m + ".vera_lambda_d"] * (x @ va[:, :d_in].T)) @ vb[:d_out].T)
    nf = ((x @ sh["vera_A"][:, :d_in].T) * out[m + ".scute_d"]) @ sh["vera_B"][:d_out].T * out[m + ".scute_e"]
    torch.testing.assert_close(nf, peft)


@pytest.mark.parametrize("neumann", [True, False])
def test_oft_blocks_match_peft(tmp_path, neumann):
    peft = pytest.importorskip("peft")
    from peft import OFTConfig, get_peft_model

    torch.manual_seed(0)
    lin = torch.nn.Sequential()
    lin.add_module("q_proj", torch.nn.Linear(64, 64, bias=False))
    model = get_peft_model(lin, OFTConfig(r=0, oft_block_size=16, target_modules=["q_proj"],
                                          use_cayley_neumann=neumann))
    for n, p in model.named_parameters():
        if ".oft_R." in n:
            with torch.no_grad():
                p.copy_(0.05 * torch.randn_like(p))
    model.save_pretrained(str(tmp_path / "src"))
    # the saved module path is "q_proj"; give it a Llama path for the compiler
    st = load_file(str(tmp_path / "src" / "adapter_model.safetensors"))
    key = next(k for k in st if "oft_R" in k)
    renamed = {f"{PREFIX}{_mod(0, 'q_proj')}.oft_R.weight": st[key]}
    cfg = json.load(open(tmp_path / "src" / "adapter_config.json"))
    src = _write(str(tmp_path / "src2"), cfg, renamed)
    c = compile_adapter(src, str(tmp_path / "out"), CFG)
    out = load_file(os.path.join(c.out_dir, "adapter_model.safetensors"))
    r_blocks = out[f"{PREFIX}{_mod(0, 'q_proj')}.scute_R"]
    x = torch.randn(4, 64)
    with torch.no_grad():
        want = model(x)
    w = lin.q_proj.base_layer.weight if hasattr(lin.q_proj, "base_layer") else lin.q_proj.weight
    xr = torch.einsum("njk,jkc->njc", x.view(4, 4, 16), r_blocks).reshape(4, 64)
    torch.testing.assert_close(xr @ w.T, want, atol=1e-4, rtol=1e-4)
