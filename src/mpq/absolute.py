"""MPQ on a robot stack that commands absolute end-effector poses.

The layer itself works on 7-D end-effector deltas in the base frame (see `canonical`).
An absolute stack is wrapped at the boundary, so the method and the library format
are unchanged:

    absolute chunk --to_deltas--> deltas --MPQ--> deltas --from_deltas--> absolute chunk

An absolute chunk is `[T, 8]`: position (3, metres), quaternion (4, in `quat_order`),
gripper command (1). A pose is the first 7 of those. The deltas between consecutive
commanded poses are taken in the base frame,

    d_t = (p_t - p_{t-1}) / pos_scale ,   w_t = log(q_t q_{t-1}^-1) / rot_scale ,

starting from an *anchor* pose: the last target sent to the robot (the measured pose
before the first chunk). Integrating them back from the same anchor reproduces the
chunk exactly, so the executed chunk always starts where the robot was told to be.
The scales put the deltas in the units where the layer's defaults (the gripper
weight, the bound, the canonicalization thresholds) were selected: `libero_scales`
chooses them so that the demonstrations' per-step deltas have the spread of
LIBERO-90's actions. The gripper command is absolute on both sides and is mapped linearly
from `[gripper_open, gripper_closed]` to LIBERO's `[-1, 1]`.

A library built from absolute demonstrations must use the same conversion; build it with
`scripts/build_library.py --abs-demos`, which writes it next to the library as
`<library>.absolute.json`, where `AbsoluteMPQ` reads it:

    layer = AbsoluteMPQ("libraries/my_robot.pt")
    anchor = robot.measured_pose()                     # [7] pos + quat, before the first chunk
    while True:
        chunk = policy(obs)                            # [T, 8] absolute
        executed = layer.quantize(chunk, anchor)       # [T, 8] absolute, same quaternion order
        for target in executed[:n_exec]:
            robot.send(target)
        anchor = executed[n_exec - 1, :7]              # the last target sent

Depends on numpy only (plus `MPQ`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from mpq.quantizer import MPQ

QUAT_ORDERS = ("wxyz", "xyzw")


# --- quaternions (internally wxyz) ---------------------------------------------------------------------------------


def to_wxyz(q: np.ndarray, order: str) -> np.ndarray:
    _check_order(order)
    return q if order == "wxyz" else q[..., [3, 0, 1, 2]]


def from_wxyz(q: np.ndarray, order: str) -> np.ndarray:
    _check_order(order)
    return q if order == "wxyz" else q[..., [1, 2, 3, 0]]


def _check_order(order: str) -> None:
    if order not in QUAT_ORDERS:
        raise ValueError(f"quat_order must be one of {QUAT_ORDERS}, got {order!r}")


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions `[..., 4]`."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion -> axis-angle `[..., 3]` of the shorter of the two rotations `q`, `-q`."""
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    k = np.where(s > 1e-9, angle / np.maximum(s, 1e-9), 2.0 / np.maximum(q[..., :1], 1e-9))
    return v * k


def quat_exp(w: np.ndarray) -> np.ndarray:
    """Axis-angle `[..., 3]` -> unit wxyz quaternion."""
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    k = np.where(theta > 1e-9, np.sin(theta / 2) / np.maximum(theta, 1e-9), 0.5)
    return np.concatenate([np.cos(theta / 2), w * k], -1)


def _normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# --- the conversion ------------------------------------------------------------------------------------------------


@dataclass
class AbsoluteConfig:
    """How absolute commands map to the layer's deltas. Build and deployment must use the same one."""

    quat_order: str                 # "wxyz" or "xyzw", as the stack reports it; no default on purpose
    pos_scale: float                # metres per delta unit (`libero_scales`)
    rot_scale: float                # radians per delta unit (`libero_scales`)
    gripper_open: float = -1.0      # the stack's gripper command for open  -> -1
    gripper_closed: float = 1.0     # the stack's gripper command for closed -> +1

    def __post_init__(self):
        _check_order(self.quat_order)
        if self.gripper_open == self.gripper_closed:
            raise ValueError("gripper_open and gripper_closed must differ")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> AbsoluteConfig:
        with open(path) as f:
            return cls(**json.load(f))


def absolute_config_path(library_path: str) -> str:
    """Sidecar holding the `AbsoluteConfig` of a library: `<library>.absolute.json`."""
    return library_path.removesuffix(".pt") + ".absolute.json"


# Per-step spread of LIBERO-90's actions that `libero_scales` matches, in the layer's units. The per-channel
# standard deviations are 0.30 / 0.36 / 0.40 (position) and 0.048 / 0.059 / 0.089 (rotation); these are the
# values the paper's real-robot library was built with.
LIBERO90_POS_STD = 0.30
LIBERO90_ROT_STD = 0.065


def libero_scales(demos: list[np.ndarray], quat_order: str) -> tuple[float, float]:
    """`(pos_scale, rot_scale)` that give absolute demonstrations `[n, 8]` the per-step spread of LIBERO-90.

    The standard deviation of the demonstrations' base-frame position (rotation) deltas, pooled over the three
    channels, divided by `LIBERO90_POS_STD` (`LIBERO90_ROT_STD`).
    """
    unit = AbsoluteConfig(quat_order, 1.0, 1.0)
    d = np.concatenate([demo_to_deltas(demo, unit) for demo in demos]).astype(np.float64)
    return float(d[:, :3].std() / LIBERO90_POS_STD), float(d[:, 3:6].std() / LIBERO90_ROT_STD)


def to_deltas(chunk: np.ndarray, anchor: np.ndarray, cfg: AbsoluteConfig) -> np.ndarray:
    """Absolute chunk `[..., T, 8]` and anchor pose `[..., 7]` -> layer actions `[..., T, 7]`."""
    chunk, anchor = np.asarray(chunk, np.float64), np.asarray(anchor, np.float64)
    pos = chunk[..., :3]
    q = _normalize(to_wxyz(chunk[..., 3:7], cfg.quat_order))
    prev_pos = np.concatenate([anchor[..., None, :3], pos[..., :-1, :]], -2)
    prev_q = np.concatenate([_normalize(to_wxyz(anchor[..., None, 3:7], cfg.quat_order)), q[..., :-1, :]], -2)
    d = (pos - prev_pos) / cfg.pos_scale
    w = quat_log(quat_mul(q, quat_conj(prev_q))) / cfg.rot_scale      # base frame: q_t = dq q_{t-1}
    g = 2.0 * (chunk[..., 7:8] - cfg.gripper_open) / (cfg.gripper_closed - cfg.gripper_open) - 1.0
    return np.concatenate([d, w, g], -1)


def from_deltas(actions: np.ndarray, anchor: np.ndarray, cfg: AbsoluteConfig) -> np.ndarray:
    """Layer actions `[..., T, 7]` integrated from the anchor pose `[..., 7]` -> absolute chunk `[..., T, 8]`.

    Quaternions come out in `cfg.quat_order`, on the hemisphere continuous with the anchor's, and the
    gripper command clipped to the range between `gripper_open` and `gripper_closed`.
    """
    actions, anchor = np.asarray(actions, np.float64), np.asarray(anchor, np.float64)
    pos = anchor[..., None, :3] + np.cumsum(actions[..., :3] * cfg.pos_scale, -2)
    dq = quat_exp(actions[..., 3:6] * cfg.rot_scale)
    q = _normalize(to_wxyz(anchor[..., 3:7], cfg.quat_order))
    quats = []
    for t in range(actions.shape[-2]):
        q = _normalize(quat_mul(dq[..., t, :], q))
        quats.append(q)
    quats = np.stack(quats, -2)
    g = np.clip(actions[..., 6:7], -1.0, 1.0)
    g = cfg.gripper_open + (g + 1.0) / 2.0 * (cfg.gripper_closed - cfg.gripper_open)
    return np.concatenate([pos, from_wxyz(quats, cfg.quat_order), g], -1)


def demo_to_deltas(poses: np.ndarray, cfg: AbsoluteConfig) -> np.ndarray:
    """One demonstration of commanded absolute targets `[n, 8]` -> layer actions `[n, 7]` for the library.

    The anchor of the first step is the first target itself, so its delta is zero.
    """
    poses = np.asarray(poses, np.float64)
    return to_deltas(poses, poses[0, :7], cfg).astype(np.float32)


# --- the layer -----------------------------------------------------------------------------------------------------


class AbsoluteMPQ:
    """`MPQ` for absolute end-effector commands: `quantize(chunk [T, 8], anchor [7]) -> [T, 8]`.

    `library` is a path (its `<library>.absolute.json` sidecar supplies the conversion unless `config` is
    given) or a tensor with an explicit `config`. Other keyword arguments go to `MPQ`; the wrapped layer and
    its telemetry (`trace`, `cap_ratio`, ...) are `self.mpq`.
    """

    def __init__(self, library, config: AbsoluteConfig | None = None, **mpq_kwargs):
        if config is None:
            if not isinstance(library, str):
                raise ValueError("a library given as a tensor needs an explicit `config`")
            path = absolute_config_path(library)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path}: no absolute-command config for {library}; build the library with "
                                        "scripts/build_library.py --abs-demos, or pass `config`")
            config = AbsoluteConfig.load(path)
        self.config = config
        self.mpq = MPQ(library, **mpq_kwargs)

    def quantize(self, chunk: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Absolute chunk `[T, 8]` (or `[n, T, 8]` with anchors `[n, 7]`) -> same shape and dtype."""
        chunk = np.asarray(chunk)
        deltas = to_deltas(chunk, anchor, self.config).astype(np.float32)
        out = from_deltas(self.mpq.quantize(deltas), anchor, self.config)
        return out.astype(chunk.dtype) if chunk.dtype.kind == "f" else out
