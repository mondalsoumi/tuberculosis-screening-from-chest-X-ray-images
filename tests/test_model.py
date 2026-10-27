# -*- coding: utf-8 -*-
"""tests/test_model.py — shape, backward, spatial-size, and model_report.md."""
import sys, os
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model    import LiteTBNet
from baselines import build_swin_tiny, build_inception_v3


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def litetb_eca():
    m = LiteTBNet(use_eca=True); m.eval(); return m

@pytest.fixture(scope="module")
def litetb_no_eca():
    m = LiteTBNet(use_eca=False); m.eval(); return m

@pytest.fixture(scope="module")
def swin():
    m = build_swin_tiny(pretrained=False); m.eval(); return m

@pytest.fixture(scope="module")
def incv3():
    m = build_inception_v3(pretrained=False); m.eval(); return m


# ── output shape (B=2) ───────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name,fix", [
    ("LiteTBNet(eca=True)",  "litetb_eca"),
    ("LiteTBNet(eca=False)", "litetb_no_eca"),
    ("Swin-T",               "swin"),
    ("InceptionV3",          "incv3"),
])
def test_output_shape(model_name, fix, request):
    model = request.getfixturevalue(fix)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1), f"{model_name}: expected (2,1), got {out.shape}"


# ── forward + backward (B=4) ─────────────────────────────────────────────────

@pytest.mark.parametrize("model_name,fix", [
    ("LiteTBNet(eca=True)",  "litetb_eca"),
    ("LiteTBNet(eca=False)", "litetb_no_eca"),
    ("Swin-T",               "swin"),
    ("InceptionV3",          "incv3"),
])
def test_backward(model_name, fix, request):
    model = request.getfixturevalue(fix)
    model.train()
    x = torch.randn(4, 3, 224, 224, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()
    model.eval()


# ── spatial sizes after each stage (LiteTBNet only) ──────────────────────────

def test_spatial_sizes(litetb_eca):
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        stem, s1, s2, s3, s4 = litetb_eca.feature_maps(x)
    assert stem.shape[-1] == 112, f"stem: {stem.shape}"
    assert s1.shape[-1]   ==  56, f"stage1: {s1.shape}"
    assert s2.shape[-1]   ==  28, f"stage2: {s2.shape}"
    assert s3.shape[-1]   ==  14, f"stage3: {s3.shape}"
    assert s4.shape[-1]   ==  14, f"stage4: {s4.shape}"


# ── model_report.md ───────────────────────────────────────────────────────────

def _count_params(m):
    return sum(p.numel() for p in m.parameters())

def _mb(m):
    return _count_params(m) * 4 / 1e6   # fp32 bytes → MB

def _gmacs(m):
    try:
        from thop import profile
        x = torch.randn(1, 3, 224, 224)
        macs, _ = profile(m, inputs=(x,), verbose=False)
        return macs / 1e9
    except Exception as e:
        return f"N/A ({e})"

def _eca_params(m):
    from model import ECA
    return sum(p.numel() for mod in m.modules()
               if isinstance(mod, ECA) for p in mod.parameters())

def test_write_model_report(litetb_eca, litetb_no_eca, swin, incv3):
    models = [
        ("LiteTBNet (ECA=True)",  litetb_eca),
        ("LiteTBNet (ECA=False)", litetb_no_eca),
        ("Swin-Tiny",             swin),
        ("InceptionV3",           incv3),
    ]

    rows = []
    for name, m in models:
        p      = _count_params(m)
        mb     = _mb(m)
        gmacs  = _gmacs(m)
        eca_p  = _eca_params(m) if "LiteTBNet" in name else "–"
        gmacs_s = f"{gmacs:.3f}" if isinstance(gmacs, float) else gmacs
        rows.append((name, f"{p:,}", f"{mb:.2f}", gmacs_s, str(eca_p)))

    os.makedirs(str(ROOT / "outputs"), exist_ok=True)
    lines = [
        "# Model Report\n",
        "> GMACs counted with [thop](https://github.com/Lyken17/pytorch-OpCounter) "
        "(MACs = multiply-accumulate ops; GMACs = 10⁹ MACs).\n\n",
        "| Model | Params | FP32 MB | GMACs @ 224 | ECA params |\n",
        "|---|---|---|---|---|\n",
    ]
    for r in rows:
        lines.append(f"| {' | '.join(r)} |\n")

    with open(str(ROOT / "outputs" / "model_report.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)

    assert (ROOT / "outputs" / "model_report.md").exists()
