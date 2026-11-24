"""Build an MPQ library from LIBERO demonstrations.

Demonstrations are cut into 20-step windows at stride 5, fit to control points and
canonicalized. k-means in the Gram-whitened space picks K prototypes, each projected
onto its nearest real window. Defaults match `libraries/libero90_yaw_k256_g5.pt`.
Next to the library it writes `<out>.tolerances.pt`, the spread of the windows around each
prototype (`mpq.cluster_tolerances`), which the default deviation bound of `MPQ` reads.

    python scripts/build_library.py --demos data/libero_90 --out libraries/rebuilt.pt

`--windows` takes a `.npy` of windows `[n, 20, 7]` instead of LIBERO HDF5 files, for
demonstrations recorded on another robot.

`--abs-demos` takes a directory of `.npy` files, one demonstration each, `[n, 8]`: the
commanded absolute targets (position in metres, quaternion in `--quat-order`, gripper
command) at the policy's control rate. They are converted to base-frame deltas with
`mpq.absolute`, scaled to LIBERO-90's per-step spread (`mpq.libero_scales`) unless
`--pos-scale` / `--rot-scale` are given, and the conversion is written to
`<out>.absolute.json` for `AbsoluteMPQ`:

    python scripts/build_library.py --abs-demos data/my_robot --quat-order xyzw \
        --gripper-open 0 --gripper-closed 1 --out libraries/my_robot.pt
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from mpq import (
    AbsoluteConfig,
    absolute_config_path,
    build_library,
    cluster_tolerances,
    demo_to_deltas,
    libero_scales,
    save_library,
    save_tolerances,
    segment_windows,
    tolerances_path,
)


def load_windows(demo_dir: str, horizon: int, stride: int) -> np.ndarray:
    """Windows `[n, horizon, 7]` of every demonstration in a suite directory of LIBERO HDF5 files."""
    import h5py

    files = sorted(glob.glob(os.path.join(demo_dir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no hdf5 files under {demo_dir}")
    windows = []
    for path in files:
        with h5py.File(path, "r") as f:
            for key in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1])):
                actions = np.asarray(f["data"][key]["actions"][:], dtype=np.float32)
                w, _ = segment_windows(actions, horizon, stride)
                if len(w):
                    windows.append(w)
    return np.concatenate(windows)


def load_absolute_demos(demo_dir: str) -> list[np.ndarray]:
    """Absolute demonstrations `[n, 8]`, one `.npy` each."""
    files = sorted(glob.glob(os.path.join(demo_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no .npy files under {demo_dir}")
    demos = [np.load(path) for path in files]
    for path, demo in zip(files, demos, strict=True):
        if demo.ndim != 2 or demo.shape[1] != 8:
            raise ValueError(f"{path}: expected [n, 8] (position, quaternion, gripper), got {demo.shape}")
    return demos


def absolute_windows(demos: list[np.ndarray], cfg: AbsoluteConfig, horizon: int, stride: int) -> np.ndarray:
    """Windows `[n, horizon, 7]` of layer actions from absolute demonstrations `[n, 8]`."""
    windows = [segment_windows(demo_to_deltas(demo, cfg), horizon, stride)[0] for demo in demos]
    return np.concatenate([w for w in windows if len(w)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", default=None, help="directory of LIBERO *_demo.hdf5 files (one suite)")
    p.add_argument("--windows", default=None, help=".npy of action windows [n, horizon, 7] (any robot)")
    p.add_argument("--abs-demos", default=None, help="directory of absolute demonstrations [n, 8] (.npy, one per demo)")
    p.add_argument("--quat-order", choices=["wxyz", "xyzw"], help="quaternion order of --abs-demos")
    p.add_argument("--pos-scale", type=float, default=None,
                   help="--abs-demos: metres per delta unit (default: LIBERO-90's spread, mpq.libero_scales)")
    p.add_argument("--rot-scale", type=float, default=None,
                   help="--abs-demos: radians per delta unit (default: LIBERO-90's spread, mpq.libero_scales)")
    p.add_argument("--gripper-open", type=float, default=-1.0, help="--abs-demos: gripper command when open")
    p.add_argument("--gripper-closed", type=float, default=1.0, help="--abs-demos: gripper command when closed")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--gripper-weight", type=float, default=5.0)
    p.add_argument("--canonicalization", default="yaw", choices=["none", "yaw", "yaw_amplitude"])
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if sum(x is not None for x in (args.demos, args.windows, args.abs_demos)) != 1:
        p.error("give exactly one of --demos, --windows or --abs-demos")
    absolute = None
    if args.abs_demos:
        if args.quat_order is None:
            p.error("--abs-demos needs --quat-order, the order in which the stack reports quaternions")
        demos = load_absolute_demos(args.abs_demos)
        pos_scale, rot_scale = libero_scales(demos, args.quat_order)
        absolute = AbsoluteConfig(args.quat_order, args.pos_scale or pos_scale, args.rot_scale or rot_scale,
                                  args.gripper_open, args.gripper_closed)
        print(f"{len(demos)} demonstrations; pos_scale {absolute.pos_scale:.4g} m, rot_scale {absolute.rot_scale:.4g} rad")
        windows = absolute_windows(demos, absolute, args.horizon, args.stride)
    else:
        windows = (np.load(args.windows) if args.windows else load_windows(args.demos, args.horizon, args.stride))
    windows = torch.as_tensor(np.asarray(windows, dtype=np.float32), device=args.device)
    print(f"{len(windows):,} segments")
    lib = build_library(windows, args.k, gripper_weight=args.gripper_weight, canonicalization=args.canonicalization,
                        seed=args.seed, device=args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_library(lib, args.out)
    print(f"wrote {args.out}: {tuple(lib.shape)}")
    if absolute is not None:
        absolute.save(absolute_config_path(args.out))
        print(f"wrote {absolute_config_path(args.out)}: {absolute}")
    tol = cluster_tolerances(windows, lib.to(args.device), gripper_weight=args.gripper_weight,
                             canonicalization=args.canonicalization)
    save_tolerances(tol, tolerances_path(args.out))
    t = tol["tolerance"][:, tol["quantiles"].index(0.75)]
    print(f"wrote {tolerances_path(args.out)}: tolerance (q75) median {t.median():.3f}, "
          f"range [{t.min():.3f}, {t.max():.3f}]")


if __name__ == "__main__":
    main()
