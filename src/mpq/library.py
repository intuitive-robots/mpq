"""Library construction: demonstration segments -> K executable prototypes.

Segments are sliding windows of demonstration actions, fitted to control points,
canonicalized, and clustered with k-means in the whitened (Gram) space. Every
centroid is then projected onto the nearest real segment, so each library entry
is a verbatim slice of a demonstration and is executable as such. A library is a
plain tensor `[K, num_basis, num_dof]` of canonical control points.
"""

from __future__ import annotations

import numpy as np
import torch

from mpq.canonical import GEOM_DIMS, canonicalize, norm_M, unwhiten, whiten
from mpq.spline import UniformBSpline


def kmeans(Z: torch.Tensor, k: int, iters: int = 40, seed: int = 0,
           tol: float = 1e-5) -> torch.Tensor:
    """k-means with k-means++ seeding on whitened vectors. Centroids `[k, dim]`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(Z)
    first = int(torch.randint(n, (1,), generator=g).item())
    centroids = [Z[first]]
    d2 = ((Z - centroids[0]) ** 2).sum(-1)
    for _ in range(k - 1):
        probs = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        nxt = int(torch.multinomial(probs, 1, generator=g).item())
        centroids.append(Z[nxt])
        d2 = torch.minimum(d2, ((Z - centroids[-1]) ** 2).sum(-1))
    C = torch.stack(centroids)
    prev = None
    for _ in range(iters):
        a = assign(Z, C)
        counts = torch.zeros(k, device=Z.device).index_add_(0, a, torch.ones(len(Z), device=Z.device))
        sums = torch.zeros_like(C).index_add_(0, a, Z)
        alive = counts > 0
        C = torch.where(alive[:, None], sums / counts.clamp_min(1).unsqueeze(-1), C)
        inertia = float(((Z - C[a]) ** 2).sum(-1).mean())
        if prev is not None and abs(prev - inertia) < tol * max(prev, 1e-12):
            break
        prev = inertia
    return C


def assign(Z: torch.Tensor, C: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Nearest-centroid index of each row of `Z`, `[n]`."""
    out = []
    for i in range(0, len(Z), chunk):
        out.append(torch.cdist(Z[i:i + chunk], C).argmin(-1))
    return torch.cat(out)


def project_medoids(Z: torch.Tensor, C: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Index of the real segment nearest to each centroid, `[k]`."""
    out = []
    for i in range(0, len(C), chunk):
        out.append(torch.cdist(C[i:i + chunk], Z).argmin(-1))
    return torch.cat(out)


def segment_windows(actions: np.ndarray, horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping windows `[n, horizon, d]` of one demonstration and their start steps."""
    n = actions.shape[0]
    if n < horizon:
        return np.empty((0, horizon, actions.shape[1]), dtype=actions.dtype), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, n - horizon + 1, stride)
    return actions[starts[:, None] + np.arange(horizon)[None, :]], starts


def build_library(windows: np.ndarray | torch.Tensor, k: int, gripper_weight: float = 5.0,
                  canonicalization: str = "yaw", seed: int = 0, project: bool = True, use_gram: bool = True,
                  spline: UniformBSpline | None = None, device: str = "cpu",
                  return_labels: bool = False):
    """Cluster demonstration windows `[n, T, d]` into a library `[k, B, d]`.

    `project=False` keeps the k-means centroids (synthetic averages) instead of
    real segments; `use_gram=False` clusters raw control points (the Euclidean
    ablation). Both are ablations; the default library is projected, Gram,
    `canonicalization="yaw"`, `gripper_weight=5`.
    """
    spline = spline or UniformBSpline(device=device)
    T = windows.shape[1]
    coeffs = spline.fit(torch.as_tensor(np.asarray(windows) if not isinstance(windows, torch.Tensor) else windows))
    N, L = spline.collocation(T), spline.gram_cholesky(T)
    canonical, _ = canonicalize(coeffs, N, L, canonicalization)
    Z = whiten(canonical, L, gripper_weight, use_gram=use_gram)
    C = kmeans(Z, k, seed=seed)
    if project:
        medoids = project_medoids(Z, C)
        lib = canonical[medoids]
    else:
        lib = unwhiten(C, L, gripper_weight, use_gram=use_gram)
    if return_labels:
        return lib, assign(Z, whiten(lib, L, gripper_weight, use_gram=use_gram))
    return lib


TOLERANCE_QUANTILES = (0.5, 0.75, 0.9, 0.95, 1.0)


def cluster_tolerances(windows: np.ndarray | torch.Tensor, lib: torch.Tensor, gripper_weight: float = 5.0,
                       canonicalization: str = "yaw", use_gram: bool = True, spline: UniformBSpline | None = None,
                       quantiles: tuple[float, ...] = TOLERANCE_QUANTILES, chunk: int = 65536) -> dict:
    """Spread of the demonstration windows around the prototype each is assigned to.

    Every window `[n, T, d]` is fit, canonicalized and assigned to its nearest library entry exactly as
    at execution time; its distance to that entry is then taken in the *geometric* Gram metric
    (`norm_M`, the metric the deviation bound is stated in, gripper excluded). Per prototype the
    requested `quantiles` of those distances are returned as `tolerance` `[K, len(quantiles)]`, with
    `counts` `[K]` windows per cluster and `rms` `[K]`. A prototype with at most one assigned window (an
    empty cluster, or a library that keeps every segment as its own prototype) has no spread of its own
    and gets the distance to its nearest other prototype in every column.
    """
    spline = spline or UniformBSpline(device=lib.device)
    lib = lib.to(spline.dtype)
    T = windows.shape[1]
    L = spline.gram_cholesky(T)
    N = spline.collocation(T)
    bank = whiten(lib, L, gripper_weight, use_gram=use_gram)
    K = len(lib)
    labels, dists = [], []
    for i in range(0, len(windows), chunk):
        w = windows[i:i + chunk]
        coeffs = spline.fit(torch.as_tensor(np.asarray(w) if not isinstance(w, torch.Tensor) else w,
                                            device=lib.device))
        canonical, _ = canonicalize(coeffs, N, L, canonicalization)
        a = assign(whiten(canonical, L, gripper_weight, use_gram=use_gram), bank)
        labels.append(a)
        dists.append(norm_M(canonical - lib[a], L))
    labels, dists = torch.cat(labels).cpu(), torch.cat(dists).cpu()
    q = torch.tensor(quantiles, dtype=dists.dtype)
    tolerance = torch.full((K, len(quantiles)), float("nan"), dtype=dists.dtype)
    rms = torch.full((K,), float("nan"), dtype=dists.dtype)
    counts = torch.bincount(labels, minlength=K)
    for k in range(K):
        d = dists[labels == k]
        if len(d):
            tolerance[k] = torch.quantile(d, q)
            rms[k] = d.pow(2).mean().sqrt()
    # A prototype that stands for at most one window (an empty cluster, or a library that keeps every segment) has no
    # within-cluster spread; its tolerance is the distance to the nearest other prototype, the scale at which the library
    # itself resolves motions there.
    lone = counts <= 1
    if bool(lone.any()):
        geom = torch.einsum("ab,kbd->kad", L.T, lib[:, :, GEOM_DIMS]).flatten(1)   # Gram-whitened geometric block, [K, B*6]
        nearest = torch.full((K,), float("inf"), dtype=dists.dtype)
        idx_lone = torch.nonzero(lone).flatten()
        for i in range(0, len(idx_lone), 4096):
            sel = idx_lone[i:i + 4096]
            d = torch.cdist(geom[sel], geom)                                        # [n, K]
            d[torch.arange(len(sel)), sel] = float("inf")
            nearest[sel] = d.min(1).values.cpu().to(dists.dtype)
        tolerance[lone] = nearest[lone][:, None].expand(-1, len(quantiles))
        rms[lone] = nearest[lone]
    return {"quantiles": list(quantiles), "tolerance": tolerance, "rms": rms, "counts": counts,
            "gripper_weight": gripper_weight, "canonicalization": canonicalization, "use_gram": use_gram, "horizon": T}


def tolerances_path(library_path: str) -> str:
    """Sidecar file holding `cluster_tolerances` of a library: `<library>.tolerances.pt` next to `<library>.pt`."""
    return library_path.removesuffix(".pt") + ".tolerances.pt"


def save_tolerances(tolerances: dict, path: str) -> None:
    torch.save({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in tolerances.items()}, path)


def load_tolerances(path: str, quantile: float = 0.75, device: str = "cpu") -> torch.Tensor:
    """The `[K]` tolerance column of a tolerances file at `quantile` (must be one of the stored quantiles)."""
    d = torch.load(path, map_location=device)
    qs = [float(x) for x in d["quantiles"]]
    if quantile not in qs:
        raise ValueError(f"{path}: quantile {quantile} not stored; have {qs}")
    return d["tolerance"][:, qs.index(quantile)].to(device)


def load_library(path: str, device: str = "cpu") -> torch.Tensor:
    """A library file is a bare tensor `[K, B, d]` (`torch.save` of `build_library`'s output)."""
    lib = torch.load(path, map_location=device)
    if not isinstance(lib, torch.Tensor) or lib.ndim != 3:
        raise ValueError(f"{path}: expected a [K, B, d] tensor, got {type(lib).__name__}")
    return lib


def save_library(lib: torch.Tensor, path: str) -> None:
    torch.save(lib.detach().cpu(), path)
