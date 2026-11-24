"""Closed-loop LIBERO evaluation of the Diffusion Policy, bare or through MPQ.

Each predicted 16-step chunk is quantized whole (with --mpq) and its first 8 steps
are executed. Defaults follow the paper: LIBERO-90, the first 10 fixed initial
states per task, 600-step cap, 10-step DDIM.

    python scripts/eval_libero.py --checkpoint checkpoints/dp_s0/dp.pt [--mpq]

With --mpq the default bound is the paper's tolerance bound (--budget 0.5 on the
0.75-quantile tolerance, gripper passed through).
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import torch

ENV_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")   # live equivalents of the recorded cameras
_W: dict = {}


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    """uint8 [n_cam, H, W, 3] -> float [n_cam, 3, H, W] in [0, 1]. Identical to train_dp.py."""
    frames = np.ascontiguousarray(frames[:, ::-1])
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


def task_names(suite: str) -> list[str]:
    """Tasks in the order of the training task index: sorted by `<name>_demo.hdf5`, the demo file names.

    Sorting by `<name>.bddl` instead gives a different order for tasks whose names are
    prefixes of each other, which would feed the policy another task's one-hot.
    """
    from libero.libero import get_libero_path

    names = [os.path.basename(p)[:-len(".bddl")]
             for p in glob.glob(os.path.join(get_libero_path("bddl_files"), suite, "*.bddl"))]
    return sorted(names, key=lambda n: f"{n}_demo.hdf5")


def make_env(name: str, suite: str, image_size: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), suite, f"{name}.bddl"),
                              camera_heights=image_size, camera_widths=image_size, ignore_done=True, horizon=10_000)


def observe(obs: dict, task_idx: int, n_state: int) -> dict:
    """Policy inputs from a live observation. The state matches the demos' `robot_states`:
    gripper qpos (2), end-effector position (3) and quaternion (4), then the task one-hot."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    images = preprocess_frames(np.stack([obs[c] for c in ENV_CAMERAS]))
    state = np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
    onehot = np.zeros(n_state - len(state), np.float32)
    onehot[task_idx] = 1.0
    batch = {OBS_STATE: torch.from_numpy(np.concatenate([state, onehot]).astype(np.float32))[None, None]}
    for i in range(len(images)):
        batch[f"{OBS_IMAGES}.cam{i}"] = images[i][None]
    return batch


def _init(args) -> None:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.utils.constants import OBS_STATE

    torch.set_num_threads(1)
    meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = meta["cfg"]
    cfg.n_action_steps = cfg.horizon                 # return the full chunk; MPQ quantizes all of it
    cfg.noise_scheduler_type = "DDIM"                # DDIM is valid for the DDPM-trained weights
    cfg.num_inference_steps = args.inference_steps
    policy = DiffusionPolicy(cfg)
    policy.load_state_dict(meta["model"])
    _W.update(args=args, policy=policy.eval(), n_state=cfg.input_features[OBS_STATE].shape[0])
    if args.mpq:
        from mpq import MPQ
        _W["mpq"] = MPQ(args.library, budget=args.budget, tolerance_quantile=args.tolerance_quantile,
                        amplitude_bound=args.amplitude_bound, clip_gripper=args.clip_gripper,
                        gripper_weight=args.gripper_weight)


@torch.no_grad()
def _episode(env, init_state, task_idx: int) -> bool:
    args, policy = _W["args"], _W["policy"]
    obs = env.set_init_state(init_state)
    step = 0
    while step < args.max_steps:
        chunk = policy.predict_action_chunk(observe(obs, task_idx, _W["n_state"]))[0].numpy()   # [16, 7]
        if "mpq" in _W:
            chunk = _W["mpq"].quantize(chunk)
        for a in np.clip(chunk[:args.replan], -1, 1):
            obs, _, _, _ = env.step(a)
            step += 1
            if env.check_success():
                return True
            if step >= args.max_steps:
                return False
    return False


def _task(item: tuple[int, str]) -> tuple[str, float, dict | None]:
    task_idx, name = item
    args = _W["args"]
    from libero.libero import get_libero_path

    inits = torch.load(os.path.join(get_libero_path("init_states"), args.suite, f"{name}.pruned_init"),
                       weights_only=False)
    env = make_env(name, args.suite, args.image_size)
    env.reset()
    rate = float(np.mean([_episode(env, inits[e], task_idx) for e in range(args.episodes)]))
    env.close()
    clip = None
    if "mpq" in _W:                                  # how often the residual bound was active
        r, cap_ratio = np.asarray(_W["mpq"].ratios), np.asarray(_W["mpq"].cap_ratio)
        clip = {"clip_frac": float((cap_ratio > 1).mean()), "median_cap_ratio": float(np.median(cap_ratio)),
                "median_ratio": float(np.median(r))}
        _W["mpq"].reset_telemetry()
    return name, rate, clip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mpq", action="store_true", help="execute every chunk through MPQ")
    p.add_argument("--library", default=os.path.join(os.path.dirname(__file__), "..", "libraries",
                                                     "libero90_yaw_k256_g5.pt"))
    p.add_argument("--budget", type=float, default=0.5, help="multiple of the prototype's tolerance (beta)")
    p.add_argument("--tolerance-quantile", type=float, default=0.75,
                   help="quantile of the within-cluster distances used as tolerance")
    p.add_argument("--amplitude-bound", type=float, default=0.0,
                   help="amplitude term of the cap (the amplitude bound is --amplitude-bound 0.25 --budget 0)")
    p.add_argument("--clip-gripper", action="store_true",
                   help="scale the gripper residual with the geometric clip (the amplitude-bound configuration of the "
                        "paper's Appendix B); "
                        "default passes it through")
    p.add_argument("--gripper-weight", type=float, default=5.0)
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--episodes", type=int, default=10, help="initial states per task (the first N of the 50)")
    p.add_argument("--max-tasks", type=int, default=None, help="quick check: evaluate only the first N tasks")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--replan", type=int, default=8, help="actions executed per predicted chunk")
    p.add_argument("--inference-steps", type=int, default=10)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8, help="parallel simulator processes; results do not depend on it")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    tasks = list(enumerate(task_names(args.suite)))[:args.max_tasks]
    print(f"{len(tasks)} tasks x {args.episodes} episodes, {f'MPQ budget={args.budget:g} amplitude_bound={args.amplitude_bound:g}' if args.mpq else 'bare'}")
    results, clips = {}, {}
    with mp.get_context("spawn").Pool(args.workers, _init, (args,)) as pool:
        for name, rate, clip in pool.imap_unordered(_task, tasks):
            results[name] = rate
            if clip is not None:
                clips[name] = clip
            print(f"  {rate:5.0%}  [{len(results):>3d}/{len(tasks)}]  {name}", flush=True)

    per_task = np.array(sorted(results.values()))     # fixed order: the interval must not depend on worker timing
    boot = [np.random.default_rng(i).choice(per_task, len(per_task)).mean() for i in range(10_000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    tag = (f"mpq_budget{args.budget:g}q{args.tolerance_quantile:g}_amp{args.amplitude_bound:g}"
           + ("_clipgrip" if args.clip_gripper else "") if args.mpq else "bare")
    out = os.path.join(args.out, f"{args.suite}_{os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))}"
                                 f"_{tag}.json")
    os.makedirs(args.out, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"success_rate": float(per_task.mean()), "ci95_over_tasks": [float(lo), float(hi)],
                   "per_task": results, "clip": clips or None, "args": vars(args)}, f, indent=1)
    print(f"success {per_task.mean():.3f} [{lo:.3f}, {hi:.3f}] -> {out}")


if __name__ == "__main__":
    main()
