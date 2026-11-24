# What Does a Skill Vocabulary Cost? A Closed-Loop Decomposition over Demonstrated Motions

[![Project page](https://img.shields.io/badge/Project-page-2563c9)](https://intuitive-robots.github.io/mpq)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](docs/Li2026_MPQ.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776ab)](pyproject.toml)

Maximilian Xiling Li, Nils Blank, Paul Mattes, Pankhuri Vanjani, Rudolf Lioutikov

Intuitive Robots Lab, Karlsruhe Institute of Technology (KIT), and Robotics Institute Germany (RIG)

![MPQ executes a policy's predicted chunk as its nearest movement prototype, corrected toward the prediction within a bound given by the prototype's demonstrations](docs/teaser/execution.gif#gh-light-mode-only)
![MPQ executes a policy's predicted chunk as its nearest movement prototype, corrected toward the prediction within a bound given by the prototype's demonstrations](docs/teaser/execution-dark.gif#gh-dark-mode-only)

Movement Prototype Quantization (MPQ) places a library of demonstrated motions after a
trained policy. It replaces each predicted action chunk with its nearest library element and
corrects that element toward the prediction within a bound. Nothing is learned. Across six
architectures, including a 7B vision-language-action model, the wrapped policy is
statistically equivalent to the original, so a reference policy executed through the library
is a baseline for any selector over the same library. This repository contains the layer, the
LIBERO-90 library of the paper, and the scripts to build a library and reproduce the paper's
Diffusion Policy row.

## Installation

```bash
git clone https://github.com/intuitive-robots/mpq && cd mpq
pip install -e .            # the layer: torch and numpy only, Python >= 3.12
pip install -e ".[test]" && pytest
```

## Quickstart

Run from the repository root: the libraries are files in `libraries/`, not part of the installed package.

```python
from mpq import MPQ

layer = MPQ("libraries/libero90_yaw_k256_g5.pt")   # reads the .tolerances.pt sidecar next to it
executed = layer.quantize(chunk)   # [T, 7] or [n, T, 7]: position, axis-angle, gripper deltas
layer.trace                        # library index of each executed chunk
layer.cap_ratio                    # ||correction|| / cap per chunk; > 1 means it was clipped
```

## How it works

MPQ fits each chunk to B-spline control points, rotates it into a canonical frame about the
gravity axis, and assigns it to the nearest library segment in the Gram metric, the L2
distance between decoded trajectories. The robot executes that segment, placed back in the
chunk's frame, plus the policy's correction clipped to a cap. The library is k-means over
demonstration segments, with each centroid projected onto a real segment (Section 3 of the
paper).

The cap is `budget * tolerance_k` (β τ_k in the paper). The tolerance of segment `k` is the
0.75-quantile of the distances from the demonstrations of its cluster to it, so a segment may
be adapted about as far as the demonstrations vary around it. The gripper channel passes
through unclipped.

| parameter | default | meaning |
|---|---|---|
| `budget` | `0.5` | multiple of the tolerance the correction may use (β) |
| `tolerance_quantile` | `0.75` | which stored quantile of the within-cluster distances is the tolerance |
| `amplitude_bound` | `0.0` | additional cap as a fraction of the segment's norm; `0.25` with `budget=0` is the amplitude bound the paper compares against |
| `clip_gripper` | `False` | `True` also scales the gripper correction, as the amplitude-bound configuration in Appendix B of the paper does |
| `forbid` | `()` | library indices never executed, for editing the library after training |

`MPQ(path, amplitude_bound=0, budget=0)` executes library segments verbatim, and
`amplitude_bound=float("inf")` returns the policy's chunk up to the spline fit.

## Using MPQ on your own robot

**End-effector deltas.** Record demonstrations as 7-D end-effector deltas (3 position,
3 axis-angle, 1 gripper) at the policy's control rate, cut them into windows `[n, 20, 7]`
(`mpq.segment_windows(actions, 20, 5)[0]`; it returns the windows and their start indices), and save them as one
`.npy`:

```bash
python scripts/build_library.py --windows my_windows.npy --out libraries/my_robot.pt
```

This writes the library and its `.tolerances.pt` sidecar, and `MPQ("libraries/my_robot.pt")`
reads both. The yaw frame and the metric assume the 7-D layout above, in units with roughly
the per-step spread of LIBERO-90's actions (`mpq.LIBERO90_POS_STD`, `mpq.LIBERO90_ROT_STD`).
The paper uses K=256 on every suite and on the real robot.

**Absolute poses.** `AbsoluteMPQ` converts at the boundary. An absolute chunk `[T, 8]`
(position in metres, quaternion, gripper command) becomes base-frame deltas between
consecutive targets, goes through the unchanged layer, and is integrated back from the same
anchor, so the executed chunk starts exactly there. The anchor is the last target sent to the
robot, or the measured pose before the first chunk. Save one `.npy` per demonstration, `[n, 8]`,
holding the *commanded* targets at the policy's rate, and build the library with the stack's
quaternion order and gripper commands:

```bash
python scripts/build_library.py --abs-demos data/my_robot --quat-order xyzw \
    --gripper-open 0 --gripper-closed 1 --out libraries/my_robot.pt
```

```python
from mpq import AbsoluteMPQ

layer = AbsoluteMPQ("libraries/my_robot.pt")       # also reads my_robot.absolute.json
anchor = robot.measured_pose()                     # [7]: position, quaternion
while running:
    executed = layer.quantize(policy(obs), anchor)   # [T, 8] -> [T, 8], same quaternion order
    for target in executed[:n_exec]:
        robot.send(target)
    anchor = executed[n_exec - 1, :7]
```

The deltas are rescaled so that the demonstrations have the per-step spread of LIBERO-90's
actions, where the defaults were selected (`mpq.libero_scales`).

## Reproducing the LIBERO-90 Diffusion Policy row

Table 1 reports 91.7 bare and 92.9 through MPQ for this policy, averaged over three training
seeds.

<details>
<summary>Install, data, library, training and evaluation</summary>

**Install.** This pulls in LeRobot and the official LIBERO (robosuite 1.4.0) at the LeRobot
commit used for the paper. One dependency builds with CMake, and CMake 4 needs the flag below.
On first import, LIBERO asks about a custom dataset folder. Answer `N`.

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e ".[libero]"
export MUJOCO_GL=egl
```

**Demonstrations** (63 GB):

```bash
python -c "from libero.libero.utils.download_utils import libero_dataset_download; \
libero_dataset_download(datasets='libero_90', download_dir='data', use_huggingface=True)"
```

**Library** (optional, about 2 min on a GPU). k-means is not bit-reproducible across hardware,
so a rebuilt library differs slightly from the shipped one at equal quality.

```bash
python scripts/build_library.py --demos data/libero_90 --out libraries/rebuilt.pt
```

**Policy** (about 2 h on an RTX 5090; repeat for seeds 1 and 2):

```bash
python scripts/train_dp.py --demos data/libero_90 --out checkpoints/dp_s0 --seed 0
```

**Evaluation** (900 rollouts per run on CPU workers):

```bash
python scripts/eval_libero.py --checkpoint checkpoints/dp_s0/dp.pt          # bare
python scripts/eval_libero.py --checkpoint checkpoints/dp_s0/dp.pt --mpq    # the paper's tolerance bound
python scripts/eval_libero.py --checkpoint checkpoints/dp_s0/dp.pt --mpq \
    --amplitude-bound 0.25 --budget 0 --clip-gripper                         # amplitude-bound configuration (Appendix B)
```

Each run writes the success rate and a 95% bootstrap interval over tasks to `results/`. The
protocol has 90 tasks with the first 10 fixed initial states each, a 600-step cap and 10-step
DDIM sampling. Each 16-step chunk is quantized whole and its first 8 steps are executed.
Repeated evaluations of one checkpoint vary by up to 3.5 points.

</details>

## Repository layout

| path | |
|---|---|
| `src/mpq/` | the layer (torch and numpy only) |
| `libraries/libero90_yaw_k256_g5.pt` | the paper's LIBERO-90 library: 256 segments of 20 steps |
| `libraries/libero90_yaw_k256_g5.tolerances.pt` | its tolerances (quantiles 0.5, 0.75, 0.9, 0.95, 1.0; the paper uses 0.75) |
| `scripts/build_library.py` | demonstrations → library + tolerances |
| `scripts/train_dp.py` | Diffusion Policy on LIBERO-90 |
| `scripts/eval_libero.py` | closed-loop evaluation, bare or through MPQ |
| `tests/` | properties of the layer, and bit-equality with the version that produced the paper's numbers |
| `docs/` | project page (GitHub Pages) and the paper PDF |

## Citation

```bibtex
@article{li2026skillvocabulary,
  title   = {What Does a Skill Vocabulary Cost? A Closed-Loop Decomposition over Demonstrated Motions},
  author  = {Li, Maximilian Xiling and Blank, Nils and Mattes, Paul and Vanjani, Pankhuri and Lioutikov, Rudolf},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgments

The B-spline chunk representation follows BEAST (Zhou et al., 2025), the action tokenizer whose basis and knot
configuration MPQ uses.

The work was funded by the German Research Foundation (DFG) – 448648559. The authors
gratefully acknowledge the support of the Robotics Institute Germany (RIG).

## License

MIT, see [LICENSE](LICENSE).
