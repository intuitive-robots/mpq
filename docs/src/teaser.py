"""Animated panels for the README and the project page, written to docs/teaser/{name}.{gif,mp4}.

    python docs/src/teaser.py library      # building the library
    python docs/src/teaser.py execution    # executing a chunk (README and page teaser)
    python docs/src/teaser.py costs        # Table 3: quantization, selection and execution per backbone
    python docs/src/teaser.py all          # each in a light and a dark version (--theme light|dark|both)

Everything drawn is real: the library is `libraries/libero90_yaw_k256_g5.pt`; the demonstration
segments, the example chunk and its cluster's demonstrations are LIBERO-90 windows stored in
`teaser_data.npz`; the executed path is what `MPQ` returns under the paper's default bound; the
costs are Table 3's numbers. Needs matplotlib and ffmpeg.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from mpq import MPQ, UniformBSpline, canonicalize, load_library
from mpq.canonical import norm_M, place

ROOT = Path(__file__).resolve().parents[2]
LIB = str(ROOT / "libraries" / "libero90_yaw_k256_g5.pt")
DATA = np.load(Path(__file__).with_name("teaser_data.npz"))
OUT = ROOT / "docs" / "teaser"

THEMES = {
    "light": {"SUFFIX": "", "BG": "#ffffff", "INK": "#1d1d1f", "GREY": "#8a8a90", "LIGHT": "#d9d9de",
              "BLUE": "#2f7cf6", "FAN": "#9dbff9", "RED": "#e63946", "DEMO_LABEL": "#4f7fd0",
              "CLUSTER_COLORS": ["#2f7cf6", "#0f9d8a", "#e08a1e", "#7b5ea7", "#d6457a"],
              "BAR_GREY": "#c9c9c9", "SEL_GREY": "#6e6e6e", "SEL_FILL": "#ededed", "EXEC_FILL": "#fce8ea",
              "QUANT_FILL": "#e3edfe", "SEL_TEXT": "#555555"},
    # the page's dark tokens; hues lifted in lightness so lines and labels keep their contrast on #141416
    "dark": {"SUFFIX": "-dark", "BG": "#141416", "INK": "#ececf0", "GREY": "#a0a0a8", "LIGHT": "#34343a",
             "BLUE": "#5b9bff", "FAN": "#2f4a78", "RED": "#ff5c6a", "DEMO_LABEL": "#8fb6ff",
             "CLUSTER_COLORS": ["#5b9bff", "#2ec4ae", "#f0a23a", "#a78bdb", "#f06a9b"],
             "BAR_GREY": "#3d3d44", "SEL_GREY": "#a0a0a8", "SEL_FILL": "#26262b", "EXEC_FILL": "#4a1f27",
             "QUANT_FILL": "#1f3254", "SEL_TEXT": "#b0b0b8"},
}


def use_theme(name: str) -> None:
    """Rebind the module's colours; every panel reads them at draw time."""
    g = globals()
    g.update(THEMES[name])
    g["LABEL_BOX"] = {"boxstyle": "round,pad=0.2", "fc": g["BG"], "ec": "none", "alpha": 0.85}


BG = INK = GREY = LIGHT = BLUE = FAN = RED = DEMO_LABEL = SUFFIX = ""
BAR_GREY = SEL_GREY = SEL_FILL = EXEC_FILL = QUANT_FILL = SEL_TEXT = ""
CLUSTER_COLORS: list[str] = []
LABEL_BOX: dict = {}
use_theme("light")




def use_noto() -> None:
    """Noto Sans from the system font cache (e.g. fonts-noto-core on Debian/Ubuntu, noto-fonts on Arch)."""
    try:
        font_manager.findfont(font_manager.FontProperties(family="Noto Sans"), fallback_to_default=False)
    except ValueError as err:
        raise SystemExit("Noto Sans not found; install it (fonts-noto-core / noto-fonts) and clear the "
                         "matplotlib font cache") from err
    plt.rcParams["font.family"] = "Noto Sans"
FPS = 20


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def phase(t: float, start: float, end: float) -> float:
    return ease((t - start) / (end - start))


def positions(actions: np.ndarray) -> np.ndarray:
    """End-effector path `[T+1, 3]` from per-step position deltas, starting at the origin."""
    return np.vstack([np.zeros(3), np.cumsum(actions[:, :3], axis=0)])


def view(x: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
    """Orthographic camera: rotate about gravity by `azimuth`, tilt by `elevation` (90 = from above)."""
    a, b = np.radians(azimuth), np.radians(elevation)
    u = x[:, 0] * np.cos(a) - x[:, 1] * np.sin(a)
    v = x[:, 0] * np.sin(a) + x[:, 1] * np.cos(a)
    return np.stack([u, v * np.sin(b) + x[:, 2] * np.cos(b)], axis=1)


def fit_box(paths: list[np.ndarray], box: tuple[float, float, float, float], pad: float = 0.08):
    """Uniform scale + offset mapping all `paths` (2-D) into `box` = (x0, y0, x1, y1), aspect kept."""
    allp = np.concatenate(paths)
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1e-9)
    bw, bh = (box[2] - box[0]) * (1 - 2 * pad), (box[3] - box[1]) * (1 - 2 * pad)
    s = min(bw / span[0], bh / span[1])
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return lambda p: (p - (lo + hi) / 2) * s + centre, s


def new_frame(w: int, h: int):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def encode(frames_dir: Path, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "%04d.png")
    palette = frames_dir / "palette.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", pattern,
                    "-vf", "palettegen=stats_mode=diff", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", pattern, "-i", str(palette),
                    "-lavfi", "paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-loop", "0",
                    str(OUT / f"{name}.gif")], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    str(OUT / f"{name}.mp4")], check=True)
    for ext in ("gif", "mp4"):
        print(f"wrote {OUT / f'{name}.{ext}'} ({(OUT / f'{name}.{ext}').stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------------------------- panel (b)

def caption(ax, step: int, text: str, alpha: float, y: float) -> None:
    """Step badge and caption at the top of the frame, where the reader starts."""
    ax.add_patch(plt.Circle((36, y), 13, color=BLUE, alpha=alpha, zorder=8))
    ax.text(36, y - 0.5, str(step), color="white", fontsize=12, ha="center", va="center", alpha=alpha, zorder=9,
            fontweight="bold")
    ax.text(60, y, text, color=INK, fontsize=16, va="center", alpha=alpha, zorder=9)


def execution() -> None:
    """A predicted chunk, its nearest prototype placed in the chunk's frame, the bound, the executed path."""
    w, h = 1040, 340
    chunk = DATA["chunk"]
    layer = MPQ(LIB)
    executed = layer.quantize(chunk)
    k = layer.trace[-1]
    placed = MPQ(LIB, budget=0, amplitude_bound=0).quantize(chunk)          # the prototype in the chunk's frame
    library = load_library(LIB)
    spline = layer.spline

    # The prototype's tolerance is the spread of its cluster's demonstrations: show them in the chunk's frame.
    N, Lg = spline.collocation(20), layer.L
    _, tf_chunk = canonicalize(spline.fit(torch.as_tensor(chunk[None])), N, Lg, "yaw")
    m_canon, _ = canonicalize(spline.fit(torch.as_tensor(DATA["members"])), N, Lg, "yaw")
    dist = norm_M(m_canon - library[k].to(m_canon.dtype), Lg)
    m_canon = m_canon[dist <= dist.quantile(0.75)]                           # the 75% that define the tolerance
    m_world = place(m_canon, tf_chunk.yaw.expand(len(m_canon)), torch.ones(len(m_canon)))
    members = [positions(m) for m in spline.decode(m_world, 20).numpy()]

    # Scene: world frame, camera looking across the chunk's direction of travel, slightly from above.
    c3, e3, p3 = positions(chunk), positions(executed), positions(placed)
    heading = np.degrees(np.arctan2(c3[-1, 1], c3[-1, 0]))
    az, el = -heading + 25.0, 22.0
    c2, e2, p2 = view(c3, az, el), view(e3, az, el), view(p3, az, el)
    m2 = [view(m, az, el) for m in members]
    scene, _ = fit_box([c2, e2, p2, *m2], (500, 16, 1030, 262), pad=0.03)

    # Library: canonical side view (travel direction x, height z) of well-used prototypes; the match top right.
    usage = DATA["usage"]
    shown = [int(i) for i in np.argsort(-usage) if int(i) != k][:11]
    shown.insert(3, k)
    dec = spline.decode(library.to(spline.dtype), 20).numpy()
    side = {i: positions(dec[i])[:, [0, 2]] for i in shown}
    cols, rows, cell, gap = 4, 3, (80, 68), 7
    boxes = {}
    for j, i in enumerate(shown):
        r, cidx = divmod(j, cols)
        x0, y0 = 24 + cidx * (cell[0] + gap), 24 + (rows - 1 - r) * (cell[1] + gap)
        boxes[i] = (x0, y0, x0 + cell[0], y0 + cell[1])
    allside = np.concatenate([side[i] - side[i].mean(0) for i in shown])
    s_lib = 0.8 * min(cell[0] / np.ptp(allside[:, 0]), cell[1] / np.ptp(allside[:, 1]))

    def in_box(path2, box):
        centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        return (path2 - path2.mean(0)) * s_lib + centre

    lib_paths = {i: in_box(side[i], boxes[i]) for i in shown}
    grid_top = boxes[k][3]
    query_box = (boxes[k][2] + 26, boxes[k][1], boxes[k][2] + 26 + cell[0], boxes[k][3])
    canon, _ = canonicalize(spline.fit(torch.as_tensor(chunk[None])), N, Lg, "yaw")
    query_path = in_box(positions(spline.decode(canon, 20)[0].numpy())[:, [0, 2]], query_box)

    C, E, Pp = scene(c2), scene(e2), scene(p2)
    M = [scene(m) for m in m2]
    fan_top = max((m for m in M), key=lambda m: m[:, 1].max())
    fan_label = fan_top[np.argmax(fan_top[:, 1])]

    steps = [
        (0.0, "A policy predicts an action chunk."),
        (2.4, "MPQ matches it to the nearest movement prototype in the library,"),
        (5.2, "and places that prototype in the chunk's frame."),
        (7.0, "The prototype's demonstrations bound how far it may be corrected."),
        (9.6, "The robot executes the prototype, corrected toward the prediction within that bound."),
    ]
    T_FADE, T_END = 13.4, 14.0
    tmp = Path(tempfile.mkdtemp())
    use_noto()
    for f in range(int(T_END * FPS)):
        t = f / FPS
        fig, ax = new_frame(w, h)
        out = 1 - phase(t, T_FADE, T_END - 0.1)

        for j, (t0, text) in enumerate(steps):
            t1 = steps[j + 1][0] if j + 1 < len(steps) else T_FADE
            a = phase(t, t0, t0 + 0.25) * (1 - phase(t, t1 - 0.2, t1)) if j + 1 < len(steps) else phase(t, t0, t0 + 0.25) * out
            if a > 0:
                caption(ax, j + 1, text, a, h - 24)

        # library
        ax.text(24, grid_top + 9, "Library: K = 256 movement prototypes", fontsize=12, color=GREY, alpha=out)
        match = phase(t, 4.0, 4.5)
        for i in shown:
            x0, y0, x1, y1 = boxes[i]
            hit = match if i == k else 0.0
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=BG, ec=BLUE if hit else LIGHT,
                                       lw=1.0 + 1.4 * hit, alpha=out))
            ax.plot(*lib_paths[i].T, color=BLUE if hit else GREY, lw=1.6 + 1.0 * hit, alpha=out, solid_capstyle="round")
        q_in = phase(t, 2.6, 3.6)
        q_out = 1 - phase(t, 6.2, 6.8)
        if q_in > 0 and q_out > 0:
            ax.plot(*((1 - q_in) * C + q_in * query_path).T, color=RED, lw=1.8, ls=(0, (3, 2)), alpha=0.9 * q_out)
            if q_in >= 1:
                ax.text((query_box[0] + query_box[2]) / 2, query_box[1] - 6, "chunk in\ncanonical frame", ha="center",
                        va="top", fontsize=9.5, color=RED, alpha=q_out)
        if match > 0 and q_out > 0:
            ymid = (boxes[k][1] + boxes[k][3]) / 2
            ax.annotate("", xy=(boxes[k][2] + 3, ymid), xytext=(query_box[0] - 3, ymid),
                        arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.5, "alpha": match * q_out})

        # scene
        draw_c = phase(t, 0.2, 1.8)
        if draw_c > 0:
            n = max(2, int(round(draw_c * (len(C) - 1))) + 1)
            ax.plot(*C[:n].T, color=RED, lw=2.4, ls=(0, (4, 2.5)), alpha=out, zorder=4)
            tip = C[n - 1]
            ax.text(min(tip[0] + 14, w - 150), tip[1], "policy's prediction", color=RED, fontsize=12, va="center",
                    alpha=out, bbox=LABEL_BOX, zorder=7)
        fly = phase(t, 5.4, 6.6)
        if fly > 0:
            fan = phase(t, 7.2, 8.2)
            if fan > 0:
                for m in M:
                    ax.plot(*m.T, color=FAN, lw=1.0, alpha=0.75 * fan * out, zorder=1)
                ax.text(*(fan_label + [0, 8]), "its demonstrations", color=DEMO_LABEL, fontsize=12, ha="center",
                        va="bottom", alpha=fan * out, bbox=LABEL_BOX, zorder=7)
            ax.plot(*((1 - fly) * lib_paths[k] + fly * Pp).T, color=BLUE, lw=2.6, alpha=out, zorder=3,
                    solid_capstyle="round")
            if fly >= 1:
                ax.text(*(Pp[-1] + [-12, 6]), "nearest prototype", color=BLUE, fontsize=12, ha="right", alpha=out,
                        bbox=LABEL_BOX, zorder=7)
        morph = phase(t, 9.8, 11.0)
        if morph > 0:
            ax.plot(*(Pp + morph * (E - Pp)).T, color=INK, lw=2.8, alpha=out, zorder=5, solid_capstyle="round")
            if morph >= 1:
                ax.text(*(E[-1] + [12, 10]), "executed", color=INK, fontsize=12, alpha=out, bbox=LABEL_BOX, zorder=7)
        ride = phase(t, 11.2, 12.6)
        ax.plot(*E[int(round(ride * (len(E) - 1)))], "o", color=INK, ms=8, alpha=out, zorder=6)

        fig.savefig(tmp / f"{f:04d}.png", facecolor=BG)
        plt.close(fig)
    encode(tmp, "execution" + SUFFIX)
    shutil.rmtree(tmp)


# --------------------------------------------------------------------------------------------- panel (a)



def rotate_xy(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate paths `[n, T, 3]` about gravity by `theta` `[n]` (radians)."""
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    out = x.copy()
    out[..., 0] = c * x[..., 0] - s * x[..., 1]
    out[..., 1] = s * x[..., 0] + c * x[..., 1]
    return out


def side(x: np.ndarray, elevation: float) -> np.ndarray:
    """Screen coordinates of canonical-frame paths `[..., 3]` seen with the travel direction to the right."""
    b = np.radians(elevation)
    return np.stack([x[..., 0], x[..., 1] * np.sin(b) + x[..., 2] * np.cos(b)], axis=-1)


def library() -> None:
    """Demonstration segments -> rotated about gravity -> clustered -> projected onto real segments -> library."""
    w, h = 1040, 340
    lib = load_library(LIB)
    spline = UniformBSpline()
    N, Lg = spline.collocation(20), spline.gram_cholesky(20)
    ks, groups = DATA["build_k"], DATA["build_members"]                    # [5], [5, 12, 20, 7]
    n_c, n_m = groups.shape[:2]
    windows = groups.reshape(-1, 20, 7)
    cluster = np.repeat(np.arange(n_c), n_m)
    raw = np.stack([positions(wi) for wi in windows])                      # [60, 21, 3] world frame
    _, tf = canonicalize(spline.fit(torch.as_tensor(windows)), N, Lg, "yaw")
    theta = -tf.yaw.numpy()                                                 # rotation into the canonical frame
    net = rotate_xy(raw, theta)[:, -1, :2]
    if np.median(net[:, 0]) < 0:                                            # place() rotates by +yaw: undo it
        theta = -theta
    canon = rotate_xy(raw, theta)
    protos = np.stack([positions(p) for p in spline.decode(lib[ks].to(spline.dtype), 20).numpy()])
    means = np.stack([canon[cluster == c].mean(0) for c in range(n_c)])

    # 1-3: one segment per cell of a shuffled grid, small scale.
    order = np.random.default_rng(3).permutation(len(raw))
    gcols, grows = 12, 5
    area = (26, 16, 1014, 286)
    cw, ch = (area[2] - area[0]) / gcols, (area[3] - area[1]) / grows
    s_grid = 0.9 * min(cw, ch) / np.percentile(np.linalg.norm(raw[:, -1], axis=1), 90)
    cells = np.array([[area[0] + (j % gcols + 0.5) * cw, area[1] + (grows - 1 - j // gcols + 0.5) * ch] for j in range(len(raw))])
    cell_of = cells[np.argsort(order)]

    # 4-5: one column per cluster, larger scale; starts placed so each cluster is centred in its column.
    col_w = (area[2] - area[0]) / n_c
    s_col = min(0.7 * col_w / max(np.ptp(side(canon[cluster == c], 0.0)[..., 0]) for c in range(n_c)),
                0.78 * (area[3] - area[1]) / max(np.ptp(side(canon[cluster == c], 0.0)[..., 1]) for c in range(n_c)))
    col_start = np.zeros((n_c, 2))
    for c in range(n_c):
        pts = side(np.concatenate([canon[cluster == c], protos[c][None]]), 0.0).reshape(-1, 2) * s_col
        centre = np.array([area[0] + (c + 0.5) * col_w, (area[1] + area[3]) / 2 - 8])
        col_start[c] = centre - (pts.min(0) + pts.max(0)) / 2

    # 6: library grid, as in the execution panel.
    usage = DATA["usage"]
    others = [int(i) for i in np.argsort(-usage) if int(i) not in set(ks.tolist())][:17]
    grid_ids = [int(k) for k in ks] + others
    gcell, ggap, g_cols = (80, 68), 7, 11
    gx0, gy0 = 26, 48
    grid_box = {}
    for j, i in enumerate(grid_ids):
        r, cidx = divmod(j, g_cols)
        x0, y0 = gx0 + cidx * (gcell[0] + ggap), gy0 + (1 - r) * (gcell[1] + ggap)
        grid_box[i] = (x0, y0, x0 + gcell[0], y0 + gcell[1])
    all_side = {i: side(positions(spline.decode(lib[i:i + 1].to(spline.dtype), 20)[0].numpy()), 0.0) for i in grid_ids}
    allc = np.concatenate([p - p.mean(0) for p in all_side.values()])
    s_lib = 0.8 * min(gcell[0] / np.ptp(allc[:, 0]), gcell[1] / np.ptp(allc[:, 1]))

    def in_cell(p2, box):
        centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        return (p2 - p2.mean(0)) * s_lib + centre

    steps = [
        (0.0, "Demonstrations are cut into short segments that point in every direction."),
        (2.8, "Each segment is rotated about gravity until it points the same way."),
        (5.4, "Seen from the side, their shapes differ."),
        (7.8, "k-means groups segments of similar shape, measured as distance between trajectories."),
        (10.6, "Each cluster's mean is replaced by its nearest real segment: a movement prototype."),
        (13.8, "Together they form the library, 256 movement prototypes for LIBERO-90."),
    ]
    T_FADE, T_END = 16.8, 17.4
    tmp = Path(tempfile.mkdtemp())
    use_noto()
    for f in range(int(T_END * FPS)):
        t = f / FPS
        fig, ax = new_frame(w, h)
        out = 1 - phase(t, T_FADE, T_END - 0.1)
        for j, (t0, text) in enumerate(steps):
            t1 = steps[j + 1][0] if j + 1 < len(steps) else T_FADE
            last = j + 1 == len(steps)
            a = phase(t, t0, t0 + 0.25) * (out if last else 1 - phase(t, t1 - 0.2, t1))
            if a > 0:
                caption(ax, j + 1, text, a, h - 24)

        draw = phase(t, 0.2, 1.4)
        rot = phase(t, 3.0, 4.6)
        tilt = phase(t, 5.6, 7.0)
        group = phase(t, 8.0, 9.6)
        mean_in = phase(t, 10.8, 11.4)
        snap = phase(t, 11.8, 12.8)
        to_lib = phase(t, 14.0, 15.2)
        fill = phase(t, 15.0, 16.0)

        paths = side(rotate_xy(raw, theta * rot), 90.0 - 90.0 * tilt)              # [60, 21, 2]
        scale = s_grid + group * (s_col - s_grid)
        start_grid = cell_of - 0.5 * s_grid * paths[:, -1]                          # centred in its cell as it turns
        starts = (1 - group) * start_grid + group * col_start[cluster]
        if to_lib < 1:
            fade_members = 1 - 0.75 * phase(t, 11.8, 12.6)
            n = max(2, int(round(draw * 20)) + 1)
            for i in range(len(raw)):
                grey, target = matplotlib.colors.to_rgb(GREY), matplotlib.colors.to_rgb(CLUSTER_COLORS[cluster[i]])
                color = tuple((1 - group) * a + group * b for a, b in zip(grey, target, strict=True))
                alpha = (0.55 + 0.35 * (1 - group)) * fade_members * (1 - to_lib)
                pts = starts[i] + scale * paths[i, :n]
                ax.plot(*pts.T, color=color, lw=1.3, alpha=alpha, solid_capstyle="round")
                back = pts[max(0, len(pts) - 3)]
                if group < 0.5 and np.linalg.norm(pts[-1] - back) > 0.5:       # arrowhead: direction of travel
                    ax.annotate("", xy=pts[-1], xytext=back,
                                arrowprops={"arrowstyle": "-|>,head_length=0.45,head_width=0.22", "color": color,
                                            "lw": 1.0, "alpha": alpha * (1 - 2 * group), "shrinkA": 0, "shrinkB": 0})
        for c in range(n_c):
            if mean_in > 0:
                proto_path = col_start[c] + s_col * side(protos[c], 0.0)
                mean_path = col_start[c] + s_col * side(means[c], 0.0)
                cur = (1 - snap) * mean_path + snap * proto_path
                target = in_cell(all_side[int(ks[c])], grid_box[int(ks[c])])
                cur = (1 - to_lib) * cur + to_lib * target
                color = CLUSTER_COLORS[c]
                ls = (0, (3, 2)) if snap < 0.5 else "-"
                ax.plot(*cur.T, color=color, lw=2.8 - 1.0 * to_lib, ls=ls, alpha=mean_in * out, zorder=5,
                        solid_capstyle="round")
                if snap >= 1 and to_lib == 0:
                    ax.text(*(cur[-1] + [6, -4]), "prototype", color=color, fontsize=10.5, va="top",
                            alpha=(1 - phase(t, 13.6, 14.0)) * out, bbox=LABEL_BOX, zorder=7)
                elif snap == 0 and mean_in > 0:
                    ax.text(*(cur[-1] + [6, -4]), "mean", color=color, fontsize=10.5, va="top",
                            alpha=mean_in * (1 - phase(t, 11.8, 12.2)), bbox=LABEL_BOX, zorder=7)
        if to_lib > 0:
            ax.text(gx0, gy0 + 2 * (gcell[1] + ggap) + 2, "Library: K = 256 movement prototypes", fontsize=12,
                    color=GREY, alpha=to_lib * out)
            for j, i in enumerate(grid_ids):
                x0, y0, x1, y1 = grid_box[i]
                ours = j < n_c
                a_cell = to_lib if ours else phase(fill * len(others), j - n_c, j - n_c + 1)
                if a_cell > 0:
                    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=BG, ec=LIGHT, lw=1.0,
                                               alpha=a_cell * out, zorder=2))
                    if not ours:
                        ax.plot(*in_cell(all_side[i], grid_box[i]).T, color=GREY, lw=1.6, alpha=a_cell * out,
                                solid_capstyle="round", zorder=3)
        fig.savefig(tmp / f"{f:04d}.png", facecolor=BG)
        plt.close(fig)
    encode(tmp, "library" + SUFFIX)
    shutil.rmtree(tmp)


# --------------------------------------------------------------------------------------------- panel (c)

# Table 3 of the paper (LIBERO-90, K = 256, beta = 0.5, mean over three seeds, choice-first order):
# success of the reference, of the reference through the library, with the selector's choice, and of the
# selector itself, and the paired terms.
COSTS = [
    ("ACT", "regression heads", (78.8, 80.2, 79.3, 67.1), ("+1.4", "−0.9", "−12.1")),
    ("flow DiT", "flow-matching heads", (81.9, 83.4, 83.0, 83.0), ("+1.5", "−0.4", "+0.0")),
    ("Diffusion Policy", "diffusion heads", (91.7, 92.9, 92.4, 82.0), ("+1.2", "−0.5", "−10.3")),
]


def costs() -> None:
    """Table 3 as three cost stacks: quantization, selection and execution per backbone."""
    w, h = 1040, 340
    lo, hi = 60.0, 96.0
    panel_x, panel_w = [206, 486, 766], 250
    rows = ["reference policy", "through the library", "selector's choice", "selector's heads"]
    row_y = [222, 178, 134, 90]                                              # bar bottoms
    bar_h = 22

    def X(v, j):
        return panel_x[j] + (v - lo) / (hi - lo) * panel_w

    steps = [
        (0.0, "Each reference policy is run as it is,"),
        (2.4, "and through the library: quantization costs nothing."),
        (5.0, "A selector's choice, placed and corrected by the reference, costs at most one point."),
        (7.8, "The selector's own placement and correction cost up to twelve points."),
        (10.6, "The loss lies in executing the chosen motion, not in choosing it."),
    ]
    T_FADE, T_END = 13.6, 14.2
    tmp = Path(tempfile.mkdtemp())
    use_noto()
    grow = [(0.3, 1.5), (2.6, 3.8), (5.2, 6.4), (8.0, 9.2)]
    for f in range(int(T_END * FPS)):
        t = f / FPS
        fig, ax = new_frame(w, h)
        out = 1 - phase(t, T_FADE, T_END - 0.1)
        for j, (t0, text) in enumerate(steps):
            t1 = steps[j + 1][0] if j + 1 < len(steps) else T_FADE
            last = j + 1 == len(steps)
            a = phase(t, t0, t0 + 0.25) * (out if last else 1 - phase(t, t1 - 0.2, t1))
            if a > 0:
                caption(ax, j + 1, text, a, h - 24)
        emphasis = phase(t, 10.8, 11.4)

        for r, label in enumerate(rows):
            g = phase(t, *grow[r])
            if g > 0:
                ax.text(20, row_y[r] + bar_h / 2, label, fontsize=12.5, va="center", color=INK, alpha=g * out)
        for j, (name, heads, (sr_pi, sr_ref, sr_idx, sr_l), (q, sel, exe)) in enumerate(COSTS):
            x0 = panel_x[j]
            ax.text(x0, 258, name, fontsize=13, color=INK, fontweight="bold", alpha=out)
            ax.text(x0 + panel_w, 258, heads, fontsize=10.5, color=GREY, ha="right", alpha=out)
            # axis
            ax.plot([x0, x0 + panel_w], [78, 78], color=GREY, lw=0.8, alpha=out)
            for v in (60, 70, 80, 90):
                ax.plot([X(v, j)] * 2, [78, 74], color=GREY, lw=0.8, alpha=out)
                ax.text(X(v, j), 71, str(v), fontsize=10, color=GREY, ha="center", va="top", alpha=out)
            # the grey bar ends where the cost box begins (as in Fig. 1c); the value sits just left of it
            ends = [sr_pi, sr_pi, sr_idx, sr_l]
            values = [sr_pi, sr_ref, sr_idx, sr_l]
            for r in range(4):
                g = phase(t, *grow[r])
                if g == 0:
                    continue
                y = row_y[r]
                cur = lo + g * (ends[r] - lo)
                ax.add_patch(plt.Rectangle((x0, y), X(cur, j) - x0, bar_h, fc=BAR_GREY, ec="none", alpha=out))
                if g >= 1:
                    ax.text(X(ends[r], j) - 5, y + bar_h / 2, f"{values[r]:.1f}", fontsize=10.5, ha="right",
                            va="center", color=INK, alpha=out)
                box = None
                if r == 1:
                    box = (sr_pi, sr_ref, BLUE, QUANT_FILL, "-", f"quantization {q}", BLUE)
                elif r == 2:
                    box = (sr_idx, sr_ref, SEL_GREY, SEL_FILL, (0, (2, 1.5)), f"selection {sel}", SEL_TEXT)
                elif r == 3:
                    box = (sr_l, sr_idx, RED, EXEC_FILL, (0, (2, 1.5)), f"execution {exe}", RED)
                if box is not None and g >= 0.6:
                    b = phase(g, 0.6, 1.0)
                    a, bnd, ec, fc, ls, text, tc = box
                    if bnd - a > 0.05:
                        lw = 0.9 + (1.2 * emphasis if r == 3 else 0)
                        ax.add_patch(plt.Rectangle((X(a, j), y), X(bnd, j) - X(a, j), bar_h, fc=fc, ec=ec, lw=lw,
                                                   ls=ls, alpha=b * out, zorder=3))
                    weight = "bold" if (r == 3 and emphasis > 0.5) else "normal"
                    ax.text(x0 + panel_w, y + bar_h + 4, text, fontsize=10.5, ha="right", color=tc,
                            alpha=b * out * (1 - 0.6 * emphasis if r in (1, 2) else 1), fontweight=weight)
            if phase(t, *grow[1]) > 0.6:
                xr = X(sr_ref, j)
                ax.plot([xr, xr], [row_y[3] - 4, row_y[1] + bar_h], color=BLUE, lw=0.8, ls=(0, (2, 2)), alpha=out,
                        zorder=4)
        ax.text(panel_x[1] + panel_w / 2, 52, "success rate (%)", fontsize=11, color=GREY, ha="center", va="top",
                alpha=out)
        ax.text(20, 12, "LIBERO-90, K = 256 movement prototypes, mean over three seeds (Table 3 of the paper). "
                "Dotted line: the reference policy through the library.", fontsize=10, color=GREY, alpha=out)
        fig.savefig(tmp / f"{f:04d}.png", facecolor=BG)
        plt.close(fig)
    encode(tmp, "costs" + SUFFIX)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    panels = {"library": library, "execution": execution, "costs": costs}
    p.add_argument("panel", choices=[*panels, "all"])
    p.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    args = p.parse_args()
    for theme in ["light", "dark"] if args.theme == "both" else [args.theme]:
        use_theme(theme)
        for name in panels if args.panel == "all" else [args.panel]:
            panels[name]()
