"""Uniform clamped B-spline basis: the chunk representation of MPQ.

A chunk of `T` actions is represented by `B` control points (BEAST's published
tokenizer hyperparameters: `B = 10`, degree 4, no boundary conditions). Decoding
is the linear map `A = N C` with the collocation matrix `N` (`[T, B]`), so every
geometric operation of the layer acts on control points and commutes with
decoding.

The basis is de Boor's recursion on a clamped uniform knot vector, following
BEAST's implementation operation for operation. Depends on torch only.
"""

from __future__ import annotations

import torch

NUM_BASIS = 10
DEGREE = 4
RIDGE = 1e-4


class UniformBSpline:
    """Clamped uniform B-spline basis with `num_basis` free control points.

    All operators are memoized per grid length: a chunk length `T` maps to one
    collocation matrix, one Gram Cholesky factor and one ridge solver, so a
    quantization step costs a handful of small matmuls.
    """

    def __init__(self, num_basis: int = NUM_BASIS, degree: int = DEGREE,
                 dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
        self.num_basis = int(num_basis)
        self.degree = int(degree)
        self.dtype = dtype
        self.device = torch.device(device)
        num_knots = self.degree + 1 + self.num_basis
        inner = torch.linspace(0, 1, num_knots - 2 * self.degree, dtype=dtype, device=self.device)
        self.knots = torch.cat([torch.zeros(self.degree, dtype=dtype, device=self.device), inner,
                                torch.ones(self.degree, dtype=dtype, device=self.device)])
        self._cache: dict[tuple, torch.Tensor] = {}

    # -- basis -----------------------------------------------------------------

    def _basis_function(self, i: int, k: int, u: torch.Tensor) -> torch.Tensor:
        knots = self.knots
        if k == 0:
            if i == self.num_basis - 1:            # closed right endpoint
                return torch.where((u >= knots[i]) & (u <= knots[i + 1]), 1.0, 0.0).to(dtype=self.dtype, device=self.device)
            return torch.where((u >= knots[i]) & (u < knots[i + 1]), 1.0, 0.0).to(dtype=self.dtype, device=self.device)
        denom1 = knots[i + k] - knots[i]
        term1 = 0.0 if denom1 == 0 else (u - knots[i]) / denom1 * self._basis_function(i, k - 1, u)
        denom2 = knots[i + k + 1] - knots[i + 1]
        term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - u) / denom2 * self._basis_function(i + 1, k - 1, u)
        return term1 + term2

    def basis(self, phase: torch.Tensor) -> torch.Tensor:
        """Basis values at phases in [0, 1]: `[len(phase), num_basis]`."""
        phase = phase.to(dtype=self.dtype, device=self.device)
        return torch.stack([self._basis_function(i, self.degree, phase) for i in range(self.num_basis)], dim=-1)

    # -- memoized operators per chunk length --------------------------------------

    def phases(self, horizon: int) -> torch.Tensor:
        """The uniform phase grid of a `horizon`-step chunk."""
        return torch.linspace(0, 1, int(horizon), dtype=self.dtype, device=self.device)

    def _memo(self, name: str, horizon: int, build):
        key = (name, int(horizon))
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def collocation(self, horizon: int) -> torch.Tensor:
        """`N` with `A = N C`: `[horizon, num_basis]`."""
        return self._memo("N", horizon, lambda: self.basis(self.phases(horizon)))

    def gram(self, horizon: int) -> torch.Tensor:
        """`M = N^T N`: `||N (C - P)||_F^2 = tr((C - P)^T M (C - P))`."""
        return self._memo("M", horizon, lambda: self.collocation(horizon).T @ self.collocation(horizon))

    def gram_cholesky(self, horizon: int, jitter: float = 1e-8) -> torch.Tensor:
        """Lower-triangular `L` with `M = L L^T`, so `||X||_M = ||L^T X||_F`."""
        def build():
            M = self.gram(horizon)
            return torch.linalg.cholesky(M + jitter * torch.eye(M.shape[-1], dtype=M.dtype, device=M.device))
        return self._memo("L", horizon, build)

    def fit_solver(self, horizon: int, reg: float = RIDGE) -> torch.Tensor:
        """Ridge least-squares solver `(N^T N + reg I)^-1 N^T`: `[num_basis, horizon]`."""
        def build():
            N = self.collocation(horizon)
            eye = torch.eye(N.shape[-1], dtype=N.dtype, device=N.device)
            return torch.linalg.solve(N.T @ N + reg * eye, N.T)
        return self._memo(f"solver{reg:g}", horizon, build)

    # -- fit / decode --------------------------------------------------------------

    def fit(self, windows: torch.Tensor, reg: float = RIDGE) -> torch.Tensor:
        """Control points of action windows `[n, T, d]` -> `[n, num_basis, d]`.

        `T >= num_basis`: closed-form ridge fit (BEAST's `learn_mp_params_from_trajs`
        without the block-diagonal multi-DOF basis). `T < num_basis` (VQ-BeT's
        5-step and OpenVLA-OFT's 8-step chunks): least squares is underdetermined,
        so the minimum-norm exact interpolation is used.
        """
        windows = windows.to(dtype=self.dtype, device=self.device)
        T = windows.shape[1]
        N = self.collocation(T)
        if T < self.num_basis:
            sol = torch.linalg.solve(N @ N.T, windows)
            return torch.einsum("tb,ntd->nbd", N, sol)
        return torch.einsum("bt,ntd->nbd", self.fit_solver(T, reg), windows)

    def decode(self, coeffs: torch.Tensor, horizon: int) -> torch.Tensor:
        """Actions `[n, horizon, d]` of control points `[n, num_basis, d]`."""
        return torch.einsum("tb,nbd->ntd", self.collocation(horizon), coeffs.to(dtype=self.dtype, device=self.device))
