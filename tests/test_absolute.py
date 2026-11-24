"""The absolute-command wrapper: exact conversion, and the layer's invariances carried over to absolute poses.

    pytest tests/
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mpq import (
    AbsoluteConfig,
    AbsoluteMPQ,
    UniformBSpline,
    absolute_config_path,
    demo_to_deltas,
    from_deltas,
    libero_scales,
    load_library,
    to_deltas,
    tolerances_path,
)
from mpq.absolute import LIBERO90_POS_STD, LIBERO90_ROT_STD, from_wxyz, quat_exp, quat_mul, to_wxyz
from mpq.canonical import rotate_z

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "libraries" / "libero90_yaw_k256_g5.pt"
SCALES = (0.01, 0.1)    # metres / radians per delta unit, about those of a 20 Hz arm


@pytest.fixture(scope="module")
def library() -> torch.Tensor:
    return load_library(str(LIB))


def random_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    """Unit wxyz quaternions around the gripper-down pose (pi about x), where axis-angle wraps."""
    return quat_mul(quat_exp(rng.normal(size=(n, 3)) * 0.3), np.array([0.0, 1.0, 0.0, 0.0]))


def absolute_chunks(library: torch.Tensor, cfg: AbsoluteConfig, n: int, T: int = 16, seed: int = 0):
    """Library segments at random yaw plus noise, integrated from random anchors: `[n, T, 8]`, anchors `[n, 7]`."""
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(library), (n,), generator=g)
    yaw = (torch.rand(n, generator=g) * 2 - 1) * np.pi
    coeffs = rotate_z(library[idx], yaw) + 0.05 * torch.randn(n, *library.shape[1:], generator=g)
    deltas = UniformBSpline().decode(coeffs, T).numpy()
    anchors = np.concatenate([rng.uniform(-0.5, 0.5, (n, 3)), from_wxyz(random_quats(rng, n), cfg.quat_order)], -1)
    return from_deltas(deltas, anchors, cfg), anchors


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_round_trip_is_exact(order):
    cfg = AbsoluteConfig(order, *SCALES, gripper_open=0.08, gripper_closed=0.0)
    rng = np.random.default_rng(0)
    q = random_quats(rng, 5 * 12).reshape(5, 12, 4)
    chunk = np.concatenate([rng.normal(size=(5, 12, 3)), from_wxyz(q, order), rng.uniform(0, 0.08, (5, 12, 1))], -1)
    anchor = chunk[:, 0, :7] + np.concatenate([np.full((5, 3), 0.01), np.zeros((5, 4))], -1)
    back = from_deltas(to_deltas(chunk, anchor, cfg), anchor, cfg)
    same_sign = np.sign((to_wxyz(back[..., 3:7], order) * q).sum(-1, keepdims=True))
    assert np.allclose(back[..., :3], chunk[..., :3], atol=1e-9)
    assert np.allclose(back[..., 3:7] * same_sign, chunk[..., 3:7], atol=1e-9)
    assert np.allclose(back[..., 7], chunk[..., 7], atol=1e-9)


def test_quaternion_sign_does_not_matter():
    """q and -q are the same orientation; a sign flip between steps must not read as a 2 pi rotation."""
    cfg = AbsoluteConfig("xyzw", *SCALES)
    rng = np.random.default_rng(1)
    chunk = np.concatenate([rng.normal(size=(10, 3)), from_wxyz(random_quats(rng, 10), "xyzw"), np.ones((10, 1))], -1)
    flipped = chunk.copy()
    flipped[::2, 3:7] *= -1
    assert np.allclose(to_deltas(chunk, chunk[0, :7], cfg), to_deltas(flipped, chunk[0, :7], cfg))


def test_gripper_maps_to_libero_range():
    cfg = AbsoluteConfig("wxyz", *SCALES, gripper_open=0.08, gripper_closed=0.0)
    chunk = np.zeros((3, 8))
    chunk[:, 3] = 1.0
    chunk[:, 7] = [0.08, 0.04, 0.0]
    assert np.allclose(to_deltas(chunk, chunk[0, :7], cfg)[:, 6], [-1, 0, 1])
    out = np.zeros((2, 7))
    out[:, 6] = [-1.7, 1.3]                                   # a residual past the ends is clipped to the gripper range
    assert np.allclose(from_deltas(out, chunk[0, :7], cfg)[:, 7], [0.08, 0.0])


def test_demo_starts_with_a_zero_delta():
    cfg = AbsoluteConfig("wxyz", *SCALES)
    rng = np.random.default_rng(2)
    demo = np.concatenate([rng.normal(size=(30, 3)), random_quats(rng, 30), np.ones((30, 1))], -1)
    d = demo_to_deltas(demo, cfg)
    assert d.shape == (30, 7) and np.allclose(d[0, :6], 0)
    assert np.allclose(from_deltas(d, demo[0, :7], cfg)[:, :3], demo[:, :3], atol=1e-5)


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_unbounded_layer_returns_the_chunk(library, order):
    """With no bound the executed chunk is the policy's, up to the spline fit, starting at the anchor."""
    cfg = AbsoluteConfig(order, *SCALES)
    chunks, anchors = absolute_chunks(library, cfg, 32)
    out = AbsoluteMPQ(library, cfg, amplitude_bound=float("inf"), budget=0.0).quantize(chunks, anchors)
    assert out.shape == chunks.shape
    assert np.abs(out[..., :3] - chunks[..., :3]).max() < 5e-3                  # metres
    dots = np.abs((to_wxyz(out[..., 3:7], order) * to_wxyz(chunks[..., 3:7], order)).sum(-1))
    assert np.degrees(2 * np.arccos(np.clip(dots, -1, 1))).max() < 1.0          # degrees


def test_absolute_layer_is_translation_and_yaw_equivariant(library):
    """Moving the scene (any translation, any yaw about a vertical axis) moves the executed chunk with it."""
    cfg = AbsoluteConfig("xyzw", *SCALES)
    layer = AbsoluteMPQ(str(LIB), cfg)
    chunks, anchors = absolute_chunks(library, cfg, 128, seed=3)
    out = layer.quantize(chunks, anchors)
    i0 = np.array(layer.mpq.trace)

    theta, centre, shift = 0.9, np.array([0.3, -0.2, 0.0]), np.array([0.1, 0.4, -0.05])
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    qz = quat_exp(np.array([0.0, 0.0, theta]))

    def move(x):
        y = x.copy()
        y[..., :3] = (x[..., :3] - centre) @ Rz.T + centre + shift
        y[..., 3:7] = from_wxyz(quat_mul(qz, to_wxyz(x[..., 3:7], "xyzw")), "xyzw")
        return y

    layer.mpq.reset_telemetry()
    moved = layer.quantize(move(chunks), move(anchors))
    same = i0 == np.array(layer.mpq.trace)
    assert same.mean() > 0.98                                   # ties at near-still segments may flip
    expect = move(out)
    assert np.allclose(moved[same][..., :3], expect[same][..., :3], atol=1e-4)
    dots = np.abs((moved[same][..., 3:7] * expect[same][..., 3:7]).sum(-1))
    assert dots.min() > 1 - 1e-6


def test_config_sidecar_is_read(tmp_path, library):
    lib = tmp_path / "robot.pt"
    shutil.copy(LIB, lib)
    shutil.copy(tolerances_path(str(LIB)), tolerances_path(str(lib)))
    with pytest.raises(FileNotFoundError):
        AbsoluteMPQ(str(lib))
    AbsoluteConfig("wxyz", *SCALES, gripper_open=0.08, gripper_closed=0.0).save(absolute_config_path(str(lib)))
    layer = AbsoluteMPQ(str(lib))
    assert layer.config == AbsoluteConfig("wxyz", *SCALES, 0.08, 0.0)
    chunks, anchors = absolute_chunks(library, layer.config, 1)
    assert layer.quantize(chunks[0].astype(np.float32), anchors[0]).dtype == np.float32
    with pytest.raises(ValueError):
        AbsoluteConfig("xyz", *SCALES)


def test_build_library_from_absolute_demos(tmp_path, library):
    """`build_library.py --abs-demos` end to end: library, tolerances and conversion sidecar, loadable as one layer."""
    cfg = AbsoluteConfig("xyzw", *SCALES, gripper_open=0.08, gripper_closed=0.0)
    chunks, anchors = absolute_chunks(library, cfg, 12, T=60, seed=4)
    demos = tmp_path / "demos"
    demos.mkdir()
    for i, demo in enumerate(chunks):
        np.save(demos / f"demo_{i}.npy", demo.astype(np.float32))
    out = tmp_path / "robot.pt"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_library.py"), "--abs-demos", str(demos),
                    "--quat-order", "xyzw", "--gripper-open", "0.08", "--gripper-closed", "0", "--k", "8",
                    "--device", "cpu", "--out", str(out)], check=True, cwd=ROOT, capture_output=True,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])})
    layer = AbsoluteMPQ(str(out))
    pos_scale, rot_scale = libero_scales(list(chunks.astype(np.float32)), "xyzw")
    assert layer.config == AbsoluteConfig("xyzw", pos_scale, rot_scale, 0.08, 0.0) and layer.mpq.K == 8
    executed = layer.quantize(chunks[0, :16], anchors[0])
    assert executed.shape == (16, 8) and np.isfinite(executed).all()


def test_libero_scales_match_libero_spread(library):
    """Demonstrations converted with `libero_scales` have LIBERO-90's per-step spread, whatever units they came in."""
    chunks, _ = absolute_chunks(library, AbsoluteConfig("wxyz", 0.003, 0.02), 12, T=60, seed=5)
    pos_scale, rot_scale = libero_scales(list(chunks), "wxyz")
    d = np.concatenate([demo_to_deltas(c, AbsoluteConfig("wxyz", pos_scale, rot_scale)) for c in chunks])
    assert np.isclose(d[:, :3].std(), LIBERO90_POS_STD, rtol=1e-3)
    assert np.isclose(d[:, 3:6].std(), LIBERO90_ROT_STD, rtol=1e-3)
