"""Properties of the execution layer, checked on the shipped LIBERO-90 library.

    pytest tests/
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mpq import (
    GRIPPER_DIM,
    MPQ,
    UniformBSpline,
    build_library,
    canonicalize,
    clip_residual,
    cluster_tolerances,
    load_library,
    load_tolerances,
    norm_M,
    place,
    residual_cap,
    segment_windows,
    tolerances_path,
    whiten,
)
from mpq.canonical import rotate_z
from mpq.library import assign

LIB = Path(__file__).resolve().parents[1] / "libraries" / "libero90_yaw_k256_g5.pt"


@pytest.fixture(scope="module")
def library() -> torch.Tensor:
    return load_library(str(LIB))


def realistic_chunks(library: torch.Tensor, n: int, T: int = 20, noise: float = 0.05, seed: int = 0) -> torch.Tensor:
    """Library segments at random yaw plus smooth noise: chunks shaped like a policy's output."""
    g = torch.Generator().manual_seed(seed)
    sp = UniformBSpline()
    idx = torch.randint(len(library), (n,), generator=g)
    yaw = (torch.rand(n, generator=g) * 2 - 1) * math.pi
    coeffs = rotate_z(library[idx], yaw) + noise * torch.randn(n, *library.shape[1:], generator=g)
    return sp.decode(coeffs, T)


def test_library_file(library):
    assert library.shape == (256, 10, 7)
    assert library.dtype == torch.float32
    assert torch.isfinite(library).all()


@pytest.mark.parametrize("amplitude_bound", [0.0, 0.1, 0.25, 0.5])
def test_residual_respects_the_amplitude_bound(library, amplitude_bound):
    """The executed chunk is the placed segment plus a residual with ||Delta||_M <= amplitude_bound ||segment||_M."""
    layer = MPQ(library, amplitude_bound=amplitude_bound, budget=0.0)
    coeffs = layer.spline.fit(realistic_chunks(library, 512))
    q, idx, ratio = layer.quantize_coeffs(coeffs, 20)
    _, tf = canonicalize(coeffs, layer.spline.collocation(20), layer.L, "yaw")
    placed = place(library[idx], tf.yaw, torch.ones(len(idx)))
    resid = norm_M(q - placed, layer.L)
    assert (resid <= amplitude_bound * norm_M(placed, layer.L) * (1 + 1e-4) + 1e-6).all()
    if amplitude_bound > 0:                        # unclipped chunks keep their full residual
        free = ratio <= amplitude_bound
        assert torch.allclose(q[free], coeffs[free], atol=1e-5)


def test_unbounded_cap_returns_the_fitted_chunk(library):
    chunks = realistic_chunks(library, 64)
    out = MPQ(library, amplitude_bound=float("inf"), budget=0.0).quantize(chunks)
    fit = MPQ(library, fit_only=True).quantize(chunks)
    assert torch.allclose(out, fit, atol=1e-5)


def test_zero_cap_executes_a_library_segment(library):
    """With every cap term zero the executed chunk, canonicalized, is exactly its library entry."""
    layer = MPQ(library, amplitude_bound=0.0, budget=0.0)
    coeffs = layer.spline.fit(realistic_chunks(library, 64))
    q, idx, _ = layer.quantize_coeffs(coeffs, 20)
    back, _ = canonicalize(q, layer.spline.collocation(20), layer.L, "yaw")
    assert torch.allclose(back, library[idx], atol=1e-4)


def test_assignment_is_yaw_invariant(library):
    """Rotating a chunk about gravity changes neither its index nor the executed chunk's canonical form."""
    layer = MPQ(str(LIB))
    coeffs = layer.spline.fit(realistic_chunks(library, 128, noise=0.02))
    q0, i0, _ = layer.quantize_coeffs(coeffs, 20)
    turn = torch.full((len(coeffs),), 0.7)
    q1, i1, _ = layer.quantize_coeffs(rotate_z(coeffs, turn), 20)
    assert (i0 == i1).float().mean() > 0.98            # ties at degenerate (near-still) segments may flip
    same = i0 == i1
    assert torch.allclose(rotate_z(q0[same], turn[same]), q1[same], atol=1e-4)


def test_gripper_channel_is_not_rotated(library):
    turned = rotate_z(library, torch.full((len(library),), 1.3))
    assert torch.equal(turned[..., 6], library[..., 6])


@pytest.mark.parametrize("T", [5, 8])
def test_short_chunks_are_interpolated_exactly(library, T):
    """Chunks shorter than B (VQ-BeT's 5, OpenVLA-OFT's 8 steps) are fit by exact minimum-norm interpolation."""
    chunk = torch.randn(3, T, 7) * 0.3
    out = MPQ(library, fit_only=True).quantize(chunk)
    assert torch.allclose(out, chunk, atol=1e-4)


def test_numpy_in_numpy_out(library):
    chunk = realistic_chunks(library, 1)[0].numpy()
    out = MPQ(str(LIB)).quantize(chunk)
    assert isinstance(out, np.ndarray) and out.shape == chunk.shape and out.dtype == chunk.dtype


def test_forbidden_entries_are_never_assigned(library):
    layer = MPQ(str(LIB), forbid=range(0, 256, 2))
    layer.quantize(realistic_chunks(library, 256))
    assert all(i % 2 == 1 for i in layer.trace)


def test_build_library_projects_onto_real_segments():
    """Every prototype is the canonical form of an actual demonstration window."""
    rng = np.random.default_rng(0)
    demo = np.cumsum(rng.normal(size=(400, 7)).astype(np.float32) * 0.05, axis=0)
    windows, _ = segment_windows(demo, horizon=20, stride=5)
    lib = build_library(windows, k=8, seed=0)
    sp = UniformBSpline()
    canonical, _ = canonicalize(sp.fit(torch.as_tensor(windows)), sp.collocation(20), sp.gram_cholesky(20), "yaw")
    d = (lib[:, None] - canonical[None]).abs().flatten(2).amax(-1).min(1).values
    assert lib.shape == (8, 10, 7) and (d < 1e-6).all()


def test_build_library_is_deterministic():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(300, 20, 7)).astype(np.float32) * 0.1
    a = build_library(windows, k=16, seed=3)
    b = build_library(windows, k=16, seed=3)
    assert torch.equal(a, b)


def test_nearest_entry_is_selected(library):
    """The executed index is the nearest library entry in the whitened (Gram) metric."""
    layer = MPQ(str(LIB))
    coeffs = layer.spline.fit(realistic_chunks(library, 64))
    canonical, _ = canonicalize(coeffs, layer.spline.collocation(20), layer.L, "yaw")
    _, idx, _ = layer.quantize_coeffs(coeffs, 20)
    Z, bank = whiten(canonical, layer.L, 5.0), whiten(library, layer.L, 5.0)
    assert torch.equal(idx, assign(Z, bank))


# --- the deviation bound: cap = amplitude_bound ||P||_M + budget tolerance_k ---------------------------------------


def test_default_bound_reads_the_tolerances_sidecar(library):
    """The default layer is the paper's tolerance bound with the gripper passed through."""
    layer = MPQ(str(LIB))
    assert layer.amplitude_bound == 0.0 and layer.budget == 0.5 and not layer.clip_gripper
    assert torch.equal(layer.tolerance, load_tolerances(tolerances_path(str(LIB)), 0.75))
    assert layer.tolerance.shape == (256,) and (layer.tolerance > 0).all()
    with pytest.raises(ValueError):
        MPQ(library)                                   # a bare tensor has no sidecar: ask for tolerances or budget=0


def test_tolerance_bound_is_respected_and_records_the_cap_ratio(library):
    layer = MPQ(str(LIB))
    coeffs = layer.spline.fit(realistic_chunks(library, 512, noise=0.2))
    q, idx, _ = layer.quantize_coeffs(coeffs, 20)
    _, tf = canonicalize(coeffs, layer.spline.collocation(20), layer.L, "yaw")
    placed = place(library[idx], tf.yaw, torch.ones(len(idx)))
    cap = 0.5 * layer.tolerance[idx]
    assert (norm_M(q - placed, layer.L) <= cap * (1 + 1e-4) + 1e-6).all()
    layer.quantize(layer.spline.decode(coeffs, 20))
    cap_ratio = torch.tensor(layer.cap_ratio)
    assert torch.allclose(cap_ratio, norm_M(coeffs - placed, layer.L) / cap, rtol=1e-3, atol=1e-4)
    assert 0 < (cap_ratio > 1).float().mean() < 1                 # some chunks clipped, some not


def test_tolerance_is_relatively_larger_for_small_prototypes(library):
    """The demonstrations spread inversely to prototype amplitude: tolerance_k / ||P_k||_M falls with ||P_k||_M."""
    r = load_tolerances(tolerances_path(str(LIB)), 0.75)
    n = norm_M(library, UniformBSpline().gram_cholesky(20))
    small, large = n <= n.quantile(0.25), n >= n.quantile(0.75)
    assert (r[small] / n[small]).mean() > 2 * (r[large] / n[large]).mean()


def test_free_gripper_passes_the_gripper_residual_through():
    sp = UniformBSpline()
    L = sp.gram_cholesky(20)
    g = torch.Generator().manual_seed(0)
    ref, res = torch.randn(8, sp.num_basis, 7, generator=g), 3.0 * torch.randn(8, sp.num_basis, 7, generator=g)
    free, _ = clip_residual(res, ref, L, 0.25, clip_gripper=False)
    tight, _ = clip_residual(res, ref, L, 0.25, clip_gripper=True)
    assert torch.equal(free[:, :, GRIPPER_DIM], res[:, :, GRIPPER_DIM])
    assert torch.equal(free[:, :, :6], tight[:, :, :6])
    assert not torch.equal(tight[:, :, GRIPPER_DIM], res[:, :, GRIPPER_DIM])


def test_cap_terms_add():
    sp = UniformBSpline()
    L = sp.gram_cholesky(20)
    ref = torch.randn(4, sp.num_basis, 7)
    tol = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cap = residual_cap(ref, L, 0.25, tolerance=tol, budget=0.5)
    assert torch.allclose(cap, 0.25 * norm_M(ref, L) + 0.5 * tol)
    assert torch.isinf(residual_cap(ref, L, float("inf"), tolerance=tol, budget=0.5)).all()


def test_cluster_tolerances_from_windows():
    rng = np.random.default_rng(0)
    demo = np.cumsum(rng.normal(size=(600, 7)).astype(np.float32) * 0.05, axis=0)
    windows, _ = segment_windows(demo, horizon=20, stride=5)
    lib = build_library(windows, k=8, seed=0)
    tol = cluster_tolerances(windows, lib)
    assert tol["tolerance"].shape == (8, 5) and int(tol["counts"].sum()) == len(windows)
    assert (tol["tolerance"][:, 1:] >= tol["tolerance"][:, :-1] - 1e-6).all()   # quantiles are monotone
    assert (tol["rms"] > 0).all() and torch.isfinite(tol["tolerance"]).all()
