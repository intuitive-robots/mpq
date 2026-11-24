"""MPQ: quantized execution of action chunks over a library of demonstration segments.

    from mpq import MPQ, build_library, cluster_tolerances
    lib = build_library(windows, k=256)          # windows: [n, 20, 7] demo action segments
    tol = cluster_tolerances(windows, lib)       # how far the demos spread around each prototype
    layer = MPQ(lib, tolerance=tol["tolerance"][:, 1])   # the 0.75 quantile; or MPQ("libraries/libero90_yaw_k256_g5.pt")
    executed = layer.quantize(policy_chunk)      # [T, 7] -> [T, 7], a library segment + bounded residual

Depends on torch and numpy only. Modules: `spline` (B-spline chunk representation),
`canonical` (yaw/scale frame and the Gram metric), `library` (k-means + medoid
projection), `quantizer` (the execution layer), `absolute` (the layer on a robot
stack that commands absolute end-effector poses).
"""

from mpq.absolute import (
    LIBERO90_POS_STD,
    LIBERO90_ROT_STD,
    AbsoluteConfig,
    AbsoluteMPQ,
    absolute_config_path,
    demo_to_deltas,
    from_deltas,
    libero_scales,
    to_deltas,
)
from mpq.canonical import (
    CANONICALIZATIONS,
    GEOM_DIMS,
    GRIPPER_DIM,
    GroupTransform,
    canonicalize,
    invert,
    norm_M,
    place,
    unwhiten,
    whiten,
)
from mpq.library import (
    assign,
    build_library,
    cluster_tolerances,
    kmeans,
    load_library,
    load_tolerances,
    project_medoids,
    save_library,
    save_tolerances,
    segment_windows,
    tolerances_path,
)
from mpq.quantizer import (
    MPQ,
    bound_residual,
    clip_residual,
    residual_cap,
)
from mpq.spline import UniformBSpline

__all__ = [
    "CANONICALIZATIONS",
    "GEOM_DIMS",
    "GRIPPER_DIM",
    "LIBERO90_POS_STD",
    "LIBERO90_ROT_STD",
    "MPQ",
    "AbsoluteConfig",
    "AbsoluteMPQ",
    "GroupTransform",
    "UniformBSpline",
    "absolute_config_path",
    "assign",
    "bound_residual",
    "build_library",
    "canonicalize",
    "clip_residual",
    "cluster_tolerances",
    "demo_to_deltas",
    "from_deltas",
    "invert",
    "kmeans",
    "libero_scales",
    "load_library",
    "load_tolerances",
    "norm_M",
    "place",
    "project_medoids",
    "residual_cap",
    "save_library",
    "save_tolerances",
    "segment_windows",
    "to_deltas",
    "tolerances_path",
    "unwhiten",
    "whiten",
]
