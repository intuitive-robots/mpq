"""Canonical frame of a chunk, and the metric in which chunks are compared.

Actions are 7-D end-effector deltas in the robot base frame (3 position, 3
axis-angle rotation, 1 gripper), so translation invariance is already free. The
quotient taken by MPQ is yaw about base z (`canonicalization="yaw"`); dividing out
amplitude as well (`"yaw_amplitude"`) is kept as an ablation. Both act on control points directly
because rotation acts on the DOF axis and decoding on the time axis.

The frame is computed from the control points themselves, never from simulator
state, so it transfers to hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GRIPPER_DIM = 6
POS_DIMS = slice(0, 3)
ROT_DIMS = slice(3, 6)
GEOM_DIMS = slice(0, 6)

CANONICALIZATIONS = ("none", "yaw", "yaw_amplitude")


@dataclass
class GroupTransform:
    """Group element taking each raw segment to its canonical form."""

    yaw: torch.Tensor | None = None            # [n], radians about base z
    scale: torch.Tensor | None = None          # [n], amplitude in the Gram metric
    yaw_degenerate: torch.Tensor | None = None  # [n] bool: PCA/identity fallback used


def net_displacement(coeffs: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """Net Cartesian displacement of each segment, `[n, 3]` (linear in `C`)."""
    return torch.einsum("b,nbd->nd", N.sum(0), coeffs[:, :, POS_DIMS])


def rotate_z(coeffs: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate position and orientation blocks about base z by `angle` `[n]`."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    R = torch.stack([torch.stack([cos, -sin, zero], -1),
                     torch.stack([sin, cos, zero], -1),
                     torch.stack([zero, zero, one], -1)], -2)
    out = coeffs.clone()
    for dims in (POS_DIMS, ROT_DIMS):     # axis-angle deltas rotate like vectors under a proper rotation
        out[:, :, dims] = torch.einsum("nij,nbj->nbi", R, coeffs[:, :, dims])
    return out


def yaw_canonicalize(coeffs: torch.Tensor, N: torch.Tensor,
                     eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate each segment so its horizontal net displacement points along +x.

    Only yaw is quotiented (gravity breaks the rest of SO(3)). Segments with a
    near-zero horizontal displacement use the principal horizontal axis of the
    path, and the identity if that is degenerate too.
    """
    d = net_displacement(coeffs, N)
    horiz = torch.linalg.norm(d[:, :2], dim=-1)
    degenerate = horiz < eps
    yaw = torch.atan2(d[:, 1], d[:, 0])
    if bool(degenerate.any()):
        path = torch.einsum("tb,nbd->ntd", N, coeffs[degenerate][:, :, POS_DIMS])[..., :2]
        path = path - path.mean(1, keepdim=True)
        _, _, Vh = torch.linalg.svd(path, full_matrices=False)
        fallback = torch.atan2(Vh[:, 0, 1], Vh[:, 0, 0])
        spread = torch.linalg.norm(path, dim=(1, 2))
        yaw = yaw.clone()
        yaw[degenerate] = torch.where(spread > eps, fallback, torch.zeros_like(fallback))
    return rotate_z(coeffs, -yaw), yaw, degenerate


def scale_canonicalize(coeffs: torch.Tensor, L: torch.Tensor,
                       eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide out the geometric amplitude `||C_geom||_M`; the gripper is untouched."""
    geom = torch.einsum("ab,nbd->nad", L.T, coeffs[:, :, GEOM_DIMS])
    scale = torch.linalg.norm(geom.reshape(len(coeffs), -1), dim=-1).clamp_min(eps)
    out = coeffs.clone()
    out[:, :, GEOM_DIMS] = coeffs[:, :, GEOM_DIMS] / scale[:, None, None]
    return out, scale


def canonicalize(coeffs: torch.Tensor, N: torch.Tensor, L: torch.Tensor,
                 canonicalization: str = "yaw") -> tuple[torch.Tensor, GroupTransform]:
    """Divide out the segment's frame: nothing (`"none"`), yaw about gravity, or yaw and amplitude."""
    if canonicalization not in CANONICALIZATIONS:
        raise ValueError(f"unknown canonicalization {canonicalization!r}; expected one of {CANONICALIZATIONS}")
    tf = GroupTransform()
    out = coeffs
    if canonicalization != "none":
        out, tf.yaw, tf.yaw_degenerate = yaw_canonicalize(out, N)
    if canonicalization == "yaw_amplitude":
        out, tf.scale = scale_canonicalize(out, L)
    return out, tf


def invert(canonical: torch.Tensor, tf: GroupTransform) -> torch.Tensor:
    """Canonical control points back into each segment's own frame."""
    out = canonical
    if tf.scale is not None:
        out = out.clone()
        out[:, :, GEOM_DIMS] = out[:, :, GEOM_DIMS] * tf.scale[:, None, None]
    if tf.yaw is not None:
        out = rotate_z(out, tf.yaw)
    return out


def place(canonical: torch.Tensor, yaw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Place library entries into the frame of the chunk they replace (`invert` with explicit args)."""
    scaled = canonical.clone()
    scaled[:, :, GEOM_DIMS] = scaled[:, :, GEOM_DIMS] * scale[:, None, None]
    return invert(scaled, GroupTransform(yaw=yaw))


def whiten(coeffs: torch.Tensor, L: torch.Tensor, gripper_weight: float,
           use_gram: bool = True) -> torch.Tensor:
    """Flatten control points so Euclidean distance is the trajectory metric.

    `||X||_M^2 = ||L^T X||_F^2`, so whitening by the Gram Cholesky factor lets
    nearest-neighbour search and k-means run on plain vectors while the
    distances they minimize are L2 distances between decoded trajectories. The
    gripper channel is weighted separately (`w_g`; 5 by default).
    `use_gram=False` is the Euclidean ablation.
    """
    out = torch.einsum("ab,nbd->nad", L.T, coeffs) if use_gram else coeffs.clone()
    out = out.clone()
    out[:, :, GRIPPER_DIM] *= gripper_weight
    return out.reshape(len(coeffs), -1)


def unwhiten(Z: torch.Tensor, L: torch.Tensor, gripper_weight: float,
             use_gram: bool = True) -> torch.Tensor:
    """Inverse of `whiten`: `[n, B * d]` back to control points `[n, B, d]`."""
    B = L.shape[0]
    out = Z.reshape(len(Z), B, -1).clone()
    out[:, :, GRIPPER_DIM] /= gripper_weight
    if use_gram:
        out = torch.linalg.solve_triangular(L.T.to(out.dtype), out, upper=True)
    return out


def norm_M(X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Gram norm of the geometric block of control points `[n, B, d]` -> `[n]`."""
    w = torch.einsum("ab,nbd->nad", L.T, X[:, :, GEOM_DIMS])
    return torch.linalg.norm(w.flatten(1), dim=-1)
