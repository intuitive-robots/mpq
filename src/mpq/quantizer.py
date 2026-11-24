"""The execution layer: a predicted chunk -> a library segment plus a bounded correction.

    quantize(a) = decode( place(P_k, g) + clip( fit(a) - place(P_k, g) ) )

with `k` the nearest library entry to the chunk's canonical control points in
the Gram metric, `g` the chunk's own frame (yaw), and the correction clipped so
that `||Delta||_M <= cap_k`, where

    cap_k = budget * tolerance_k  +  amplitude_bound * ||place(P_k, g)||_M .

`tolerance_k` (tau_k in the paper) is a quantile of the Gram distances from the
demonstration windows of prototype `k`'s cluster to it (`library.cluster_tolerances`,
stored next to the library as `<library>.tolerances.pt`). `budget` is beta in the
paper. The defaults are the paper's tolerance bound, `budget = 0.5` at the 0.75
quantile with `amplitude_bound = 0`, and the gripper channel of the correction
passes through unclipped. The amplitude bound the paper compares against is
`amplitude_bound = 0.25, budget = 0`; with `clip_gripper = True` it is the
amplitude-bound configuration of the paper's Appendix B. `amplitude_bound = budget = 0` executes the
segment verbatim; `amplitude_bound = inf` returns the policy's own chunk (after the
spline fit). Nothing is learned; the library is built once per demonstration set.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import torch

from mpq.canonical import GEOM_DIMS, canonicalize, norm_M, place, whiten
from mpq.library import assign, load_library, load_tolerances, tolerances_path
from mpq.spline import UniformBSpline


def residual_cap(reference: torch.Tensor, L: torch.Tensor, amplitude_bound: float,
                 tolerance: torch.Tensor | None = None, budget: float = 0.0) -> torch.Tensor:
    """Per-chunk cap on `||Delta||_M`: `amplitude_bound * ||reference||_M + budget * tolerance`, `[n]`.

    `tolerance` is the tolerance of each chunk's prototype (`[n]`, already gathered by index).
    With `amplitude_bound = inf` the cap is infinite regardless of the other terms.
    """
    ref = norm_M(reference, L).clamp_min(1e-9)
    if amplitude_bound == float("inf"):
        return torch.full_like(ref, float("inf"))
    cap = amplitude_bound * ref
    if budget and tolerance is not None:
        cap = cap + budget * tolerance.to(cap)
    return cap


def bound_residual(residual: torch.Tensor, reference: torch.Tensor, L: torch.Tensor, cap: torch.Tensor,
                   clip_gripper: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale each residual `[n, B, d]` so `||residual||_M <= cap` (`[n]`).

    Returns the clipped residual and the pre-clip Gram norm `||residual||_M` per chunk. With
    `clip_gripper=False` only the geometric channels are scaled: the gripper channel is outside
    the norm, so the policy's gripper timing then passes through untouched instead of being
    attenuated by a geometric budget.
    """
    have = norm_M(residual, L).clamp_min(1e-9)
    if bool(torch.isinf(cap).all()):
        return residual, have
    scale = torch.clamp(cap / have, max=1.0)
    if bool((cap == 0).all()):
        scale = torch.zeros_like(scale)
    if clip_gripper:
        return residual * scale[:, None, None], have
    out = residual.clone()
    out[:, :, GEOM_DIMS] = residual[:, :, GEOM_DIMS] * scale[:, None, None]
    return out, have


def clip_residual(residual: torch.Tensor, reference: torch.Tensor, L: torch.Tensor,
                  amplitude_bound: float, tolerance: torch.Tensor | None = None, budget: float = 0.0,
                  clip_gripper: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale each residual `[n, B, d]` so `||residual||_M <= amplitude_bound * ||reference||_M + budget * tolerance`.

    Returns the clipped residual and the pre-clip ratio `||residual||_M / ||reference||_M`
    per chunk. Under the amplitude bound (`budget = 0`) a chunk is clipped iff that ratio
    exceeds `amplitude_bound`; see `residual_cap` for the tolerance term.
    """
    have = norm_M(residual, L).clamp_min(1e-9)
    ref = norm_M(reference, L).clamp_min(1e-9)
    ratio = have / ref
    if amplitude_bound == float("inf"):
        return residual, ratio
    if amplitude_bound == 0.0 and not (budget and tolerance is not None):
        return torch.zeros_like(residual), ratio
    cap = residual_cap(reference, L, amplitude_bound, tolerance, budget)
    out, _ = bound_residual(residual, reference, L, cap, clip_gripper)
    return out, ratio


class MPQ:
    """Quantize predicted action chunks onto a library of demonstration segments.

    Parameters
    ----------
    library:
        `[K, B, d]` canonical control points, or a path to such a file.
    budget:
        Multiple of the prototype's tolerance that the correction may use (beta in the paper, 0.5).
        Needs `tolerance`, or a library given by path with a `<library>.tolerances.pt` sidecar
        (`library.cluster_tolerances`). 0 switches the tolerance term off.
    tolerance:
        `[K]` tolerance per prototype, a path to a tolerances file, or `None` to read the sidecar.
    tolerance_quantile:
        Which stored quantile of the within-cluster distances is the tolerance (0.75 in the paper).
    amplitude_bound:
        Amplitude term of the cap, as a fraction of the placed segment's Gram norm. 0 in the paper's
        tolerance bound; `amplitude_bound=0.25, budget=0` is the amplitude bound it compares against.
    clip_gripper:
        `False` (paper) leaves the gripper channel untouched, so the policy decides when the gripper
        closes. `True` scales the whole residual by the geometric clip factor, as the
        amplitude-bound configuration of the paper's Appendix B does.
    gripper_weight:
        Weight of the gripper channel in the assignment metric (`w_g`, 5).
    metric_horizon:
        Chunk length defining the Gram metric (the library's horizon, 20).
    canonicalization:
        Frame the chunks are compared in: `"yaw"` (paper default), `"yaw_amplitude"` (yaw and amplitude
        divided out), or `"none"`. Must match the library's.
    fit_only:
        Control arm: spline fit and decode, no library.
    use_gram:
        `False` assigns in Euclidean coefficient space (ablation).
    forbid:
        Library indices never assigned (deployment-time editing).

    Telemetry, one entry per quantized chunk: `trace` (executed library index), `cap_ratio`
    (pre-clip `||Delta||_M / cap`, above 1 iff the chunk was clipped), `caps` (the cap) and `ratios`
    (pre-clip `||Delta||_M / ||placed||_M`).
    """

    def __init__(self, library: torch.Tensor | str, budget: float = 0.5,
                 tolerance: torch.Tensor | str | None = None, tolerance_quantile: float = 0.75,
                 amplitude_bound: float = 0.0, clip_gripper: bool = False, gripper_weight: float = 5.0,
                 metric_horizon: int = 20, canonicalization: str = "yaw", fit_only: bool = False,
                 use_gram: bool = True, forbid: Iterable[int] = (), device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.library = (load_library(library, device=str(self.device)) if isinstance(library, str)
                        else library.to(self.device))
        self.amplitude_bound = float(amplitude_bound)
        self.budget = float(budget)
        self.clip_gripper = bool(clip_gripper)
        self.tolerance: torch.Tensor | None = None
        if self.budget and not fit_only and self.amplitude_bound != float("inf"):   # arms that never clip need none
            if tolerance is None:
                if not isinstance(library, str):
                    raise ValueError("budget > 0 needs `tolerance` (a [K] tensor or a tolerances file), or a library "
                                     "given by path with its <library>.tolerances.pt sidecar; pass budget=0 for the "
                                     "amplitude-only bound")
                tolerance = tolerances_path(library)
                if not os.path.exists(tolerance):
                    raise FileNotFoundError(f"{tolerance}: no tolerances sidecar for {library}; build it with "
                                            "scripts/build_library.py (or mpq.cluster_tolerances), or pass budget=0")
            if isinstance(tolerance, str):
                tolerance = load_tolerances(tolerance, tolerance_quantile, device=str(self.device))
            self.tolerance = tolerance.to(self.device)
            if self.tolerance.shape != (len(self.library),):
                raise ValueError(f"tolerance has shape {tuple(self.tolerance.shape)}, library has K={len(self.library)}")
        self.gripper_weight = float(gripper_weight)
        self.canonicalization = canonicalization
        self.fit_only = fit_only
        self.use_gram = use_gram
        self.spline = UniformBSpline(num_basis=int(self.library.shape[1]), device=self.device)
        self.L = self.spline.gram_cholesky(metric_horizon)
        forbid = {int(i) for i in forbid}
        self.keep = torch.tensor([k for k in range(len(self.library)) if k not in forbid], device=self.device)
        self.bank = whiten(self.library[self.keep], self.L, self.gripper_weight, use_gram=use_gram)
        self.trace: list[int] = []        # library index executed per chunk
        self.ratios: list[float] = []     # pre-clip ||Delta||_M / ||placed||_M per chunk
        self.cap_ratio: list[float] = []  # pre-clip ||Delta||_M / cap per chunk (> 1 iff the chunk was clipped)
        self.caps: list[float] = []       # the cap itself per chunk (action units)

    @property
    def K(self) -> int:
        return int(self.library.shape[0])

    @property
    def relative_bound(self) -> bool:
        """True under the amplitude bound alone (then `ratios > amplitude_bound` marks a clip)."""
        return not self.budget

    def reset_telemetry(self) -> None:
        self.trace, self.ratios, self.cap_ratio, self.caps = [], [], [], []

    @torch.no_grad()
    def quantize_coeffs(self, coeffs: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Control points `[n, B, d]` of `horizon`-step chunks -> (quantized control points, index, ratio)."""
        coeffs = coeffs.to(dtype=self.spline.dtype, device=self.device)
        N = self.spline.collocation(horizon)
        canonical, tf = canonicalize(coeffs, N, self.L, self.canonicalization)
        idx = self.keep[assign(whiten(canonical, self.L, self.gripper_weight, use_gram=self.use_gram), self.bank)]
        n = len(idx)
        yaw = tf.yaw if tf.yaw is not None else torch.zeros(n, device=self.device, dtype=coeffs.dtype)
        scale = tf.scale if tf.scale is not None else torch.ones(n, device=self.device, dtype=coeffs.dtype)
        placed = place(self.library[idx].to(coeffs.dtype), yaw, scale)
        # The tolerance is measured in the canonical frame; a canonicalization that divides out amplitude places the
        # prototype rescaled by `scale`, so the cap rescales with it (scale = 1 under yaw).
        tol = self.tolerance[idx] * scale if self.tolerance is not None else None
        resid, ratio = clip_residual(coeffs - placed, placed, self.L, self.amplitude_bound,
                                     tolerance=tol, budget=self.budget, clip_gripper=self.clip_gripper)
        # The cap ratio is against the actual cap (``ratio`` is against ``||placed||_M``, for the amplitude bound).
        self._last_cap = residual_cap(placed, self.L, self.amplitude_bound, tol, self.budget)
        self._last_have = norm_M(coeffs - placed, self.L)
        return placed + resid, idx, ratio

    @torch.no_grad()
    def quantize(self, chunks: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Action chunks `[n, T, d]` (or one chunk `[T, d]`) -> same shape and type.

        Records the executed library index, the pre-clip ratio and the cap ratio of every chunk in
        `trace` / `ratios` / `cap_ratio` (`cap_ratio > 1` marks a clipped chunk).
        """
        single = chunks.ndim == 2
        x = torch.as_tensor(chunks, dtype=self.spline.dtype, device=self.device)
        if single:
            x = x[None]
        T = x.shape[1]
        coeffs = self.spline.fit(x)
        if self.fit_only:
            out = self.spline.decode(coeffs, T)
        else:
            q, idx, ratio = self.quantize_coeffs(coeffs, T)
            self.trace.extend(int(i) for i in idx)
            self.ratios.extend(ratio.tolist())
            self.caps.extend(self._last_cap.tolist())
            self.cap_ratio.extend((self._last_have / self._last_cap.clamp_min(1e-9)).tolist())
            out = self.spline.decode(q, T)
        if single:
            out = out[0]
        if isinstance(chunks, np.ndarray):
            return out.cpu().numpy().astype(chunks.dtype)
        return out.to(chunks.dtype)
