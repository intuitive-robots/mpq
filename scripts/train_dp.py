"""Train LeRobot's Diffusion Policy on LIBERO-90 with the paper's recipe.

ResNet-18 per camera, 1D-UNet over 16-step chunks, task one-hot appended to the
robot state. 100k steps at batch 64, AdamW 1e-4 (500-step warmup, cosine decay),
random 112x112 crops, DDPM with 100 steps. The EMA weights (decay 0.9995) are saved.
Inputs are not normalized. The paper's policies were trained the same way.

    python scripts/train_dp.py --demos data/libero_90 --out checkpoints/dp_s0 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import h5py
import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch.utils.data import DataLoader, Dataset

CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    """uint8 [n_cam, H, W, 3] -> float [n_cam, 3, H, W] in [0, 1]. Shared with eval_libero.py."""
    frames = np.ascontiguousarray(frames[:, ::-1])            # vertical flip, applied identically at eval
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


class LiberoChunks(Dataset):
    """One sample per (demo, timestep): both camera frames, robot state, task index and the next `horizon` actions.

    The task index is the file's position in sorted order, which `eval_libero.task_names` reproduces.
    """

    def __init__(self, demo_dir: str, horizon: int):
        self.files = sorted(glob.glob(os.path.join(demo_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"no hdf5 files under {demo_dir}")
        self.horizon = horizon
        self._handles: dict[int, h5py.File] = {}       # opened lazily per dataloader worker
        index = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                for key in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1])):
                    n = f["data"][key]["actions"].shape[0]
                    index.extend((fi, int(key.split("_")[1]), t) for t in range(max(0, n - horizon)))
        self.index = np.asarray(index, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        fi, di, t = (int(x) for x in self.index[i])
        if fi not in self._handles:
            self._handles[fi] = h5py.File(self.files[fi], "r")
        demo = self._handles[fi]["data"][f"demo_{di}"]
        return {
            "images": preprocess_frames(np.stack([demo["obs"][c][t] for c in CAMERAS])),
            "state": torch.from_numpy(np.asarray(demo["robot_states"][t], np.float32)),
            "task": torch.tensor(fi, dtype=torch.long),
            "actions": torch.from_numpy(np.asarray(demo["actions"][t:t + self.horizon], np.float32)),
        }


def to_batch(batch: dict, device: str, n_tasks: int) -> dict:
    imgs = batch["images"].to(device, non_blocking=True)                       # [B, cams, C, H, W]
    state = batch["state"].to(device, non_blocking=True)
    onehot = torch.nn.functional.one_hot(batch["task"].to(device), n_tasks).float()
    actions = batch["actions"].to(device, non_blocking=True)
    out = {
        OBS_STATE: torch.cat([state, onehot], dim=-1)[:, None],                 # [B, n_obs=1, D + n_tasks]
        ACTION: actions,
        "action_is_pad": torch.zeros(actions.shape[:2], dtype=torch.bool, device=device),
    }
    for i in range(imgs.shape[1]):
        out[f"{OBS_IMAGES}.cam{i}"] = imgs[:, i][:, None]                       # [B, 1, C, H, W]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", required=True, help="directory of LIBERO-90 *_demo.hdf5 files")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    ds = LiberoChunks(args.demos, args.horizon)
    n_val = max(1, len(ds) // 50)
    train_ds, _ = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val],
                                                generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    sample = ds[0]
    n_cams, c, h, w = sample["images"].shape
    n_tasks = len(ds.files)
    cfg = DiffusionConfig(
        n_obs_steps=1,
        horizon=args.horizon,
        n_action_steps=8,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (sample["state"].shape[0] + n_tasks,)),
            **{f"{OBS_IMAGES}.cam{i}": PolicyFeature(FeatureType.VISUAL, (c, h, w)) for i in range(n_cams)},
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        crop_shape=(112, 112),
    )
    policy = DiffusionPolicy(cfg).to(args.device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, total_iters=500),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps - 500)], [500])
    ema = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    print(f"{len(train_ds):,} chunks, {n_tasks} tasks, {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params")

    def save(step: int) -> None:
        torch.save({"model": ema, "cfg": cfg, "step": step}, os.path.join(args.out, "dp.pt"))

    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            loss, _ = policy.forward(to_batch(batch, args.device, n_tasks))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            with torch.no_grad():
                for k, v in policy.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.9995).add_(v, alpha=1 - 0.9995)
                    else:
                        ema[k].copy_(v)
            step += 1
            if step % 100 == 0:
                print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  ({step / (time.time() - t0):.1f} it/s)",
                      flush=True)
            if step % 10_000 == 0:
                save(step)
    save(step)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out}/dp.pt")


if __name__ == "__main__":
    main()
