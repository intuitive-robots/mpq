"""The layer reproduces, bit for bit, the outputs of the version that produced the paper's numbers.

`data/reference_outputs.pt` was recorded with that version (before its parameters were named after the
paper's terms) on chunks of 20, 16 and 8 steps: half decoded library segments under a random yaw with small
noise (mostly unclipped), half random (clipped), under every configuration the paper evaluates.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from mpq import MPQ

ROOT = Path(__file__).resolve().parents[1]
LIB = str(ROOT / "libraries" / "libero90_yaw_k256_g5.pt")
REF = torch.load(ROOT / "tests" / "data" / "reference_outputs.pt", weights_only=False)

CONFIGS = {
    "tolerance_bound": {},
    "amplitude_bound": {"amplitude_bound": 0.25, "budget": 0, "clip_gripper": True},
    "verbatim": {"amplitude_bound": 0, "budget": 0},
    "passthrough": {"amplitude_bound": float("inf")},
    "fit_only": {"fit_only": True},
    "forbid": {"forbid": list(range(0, 256, 3))},
    "euclidean": {"use_gram": False},
    "yaw_amplitude": {"canonicalization": "yaw_amplitude"},
}


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("horizon", [20, 16, 8])
def test_matches_the_paper_layer(name, horizon):
    layer = MPQ(LIB, **CONFIGS[name])
    out = layer.quantize(REF["inputs"][horizon])
    want = REF["outputs"][(name, horizon)]
    assert np.array_equal(out, want["out"])
    assert layer.trace == want["trace"]
    assert layer.cap_ratio == want["cap_ratio"] and layer.caps == want["caps"] and layer.ratios == want["ratios"]
