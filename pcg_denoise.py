"""
Reference-Guided Wearable Heart-Sound Denoising
================================================
Implements the smooth composite objective from the EE 596 project proposal:

    F(x) = 0.5 * ||x - y||^2
         + lambda_g * sum_g sqrt( ||(D x)[g : g+K-1]||^2 + eps^2 )
         + lambda_r * sum_i w_i * huber_delta(x_i - z_i)

where
    - D is the first-difference operator,  (D x)[i] = x[i+1] - x[i]
    - the second term is the eps-smoothed overlapping-group sparsity (OGS) penalty
      on the derivative of x, with sliding window of length K
    - the third term is a robust (Huber) reference-attraction to a phase-aligned
      quiet baseline z, weighted by per-sample weights w_i
    - lambda_g, lambda_r >= 0 are regularization weights, eps > 0 is the OGS
      smoothing constant, delta > 0 is the Huber knee.

Provided:
    - objective(x, ...)                 : evaluate F
    - gradient(x, ...)                  : evaluate grad F (analytical)
    - solve_gd(...)                     : monotone gradient descent w/ backtracking
    - solve_mm(...)                     : majorization-minimization with O(n) tridiag solve
    - check_gradient(...)               : finite-difference gradient verification

References (proposal):
    Selesnick & Chen 2013 (OGS-TV / MM)
    Deng & Han 2018 (adaptive OGS for heart sound)
    Huber 1964 (robust loss)
    Beck & Teboulle 2009 (FISTA)
    O'Donoghue & Candes 2015 (adaptive restart)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded


# ---------------------------------------------------------------------------
# First-difference operator D and its transpose
# ---------------------------------------------------------------------------
def D_apply(x: np.ndarray) -> np.ndarray:
    """Forward difference: returns (x[1:] - x[:-1]),  length n-1."""
    return np.diff(x)


def DT_apply(v: np.ndarray) -> np.ndarray:
    """Transpose of the first-difference operator.

    For v of length n-1, returns r of length n where
        r[0]   = -v[0]
        r[i]   =  v[i-1] - v[i],   1 <= i <= n-2
        r[n-1] =  v[n-2]
    """
    n = len(v) + 1
    r = np.zeros(n, dtype=v.dtype)
    r[:-1] -= v
    r[1:] += v
    return r


# ---------------------------------------------------------------------------
# Huber loss (true, piecewise) -- gradient is continuous (saturates at +/- delta)
# ---------------------------------------------------------------------------
def huber(t: np.ndarray, delta: float) -> np.ndarray:
    """Huber loss elementwise: 0.5 t^2 for |t| <= delta, delta(|t|-delta/2) otherwise."""
    a = np.abs(t)
    quad = 0.5 * t * t
    lin = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)


def huber_grad(t: np.ndarray, delta: float) -> np.ndarray:
    """Derivative of Huber:  clip(t, -delta, +delta)."""
    return np.clip(t, -delta, delta)


# ---------------------------------------------------------------------------
# OGS smoothed group penalty: shared internals
# ---------------------------------------------------------------------------
def _ogs_internals(u: np.ndarray, K: int, eps: float):
    """Compute, for derivative vector u = D x of length m:

        v_g  = sum_{j in [g, g+K-1]} u[j]^2,           g = 0, ..., m-K
        beta_g = 1 / sqrt(v_g + eps^2)
        s_j   = sum over groups containing j of beta_g

    Returns (group_norms_full = sqrt(v + eps^2), beta, s).
    Sliding sums are computed in O(m) via prefix sums.
    """
    m = len(u)
    if m < K:
        raise ValueError(f"signal too short: m={m} < K={K}")

    # Rolling sum of u^2 with window K  ->  v of length m - K + 1
    u2 = u * u
    csum_u2 = np.concatenate(([0.0], np.cumsum(u2)))
    v = csum_u2[K:m + 1] - csum_u2[: m - K + 1]                     # m-K+1
    group_norms_full = np.sqrt(v + eps * eps)                       # >= eps
    beta = 1.0 / group_norms_full                                   # m-K+1

    # s_j = sum_{g = max(0, j-K+1)}^{min(m-K, j)} beta[g]
    n_groups = len(beta)
    csum_beta = np.concatenate(([0.0], np.cumsum(beta)))
    j = np.arange(m)
    lo = np.maximum(0, j - K + 1)
    hi = np.minimum(n_groups - 1, j)
    s = csum_beta[hi + 1] - csum_beta[lo]
    return group_norms_full, beta, s


# ---------------------------------------------------------------------------
# Objective & gradient
# ---------------------------------------------------------------------------
def objective(x, y, z, w, lambda_g, lambda_r, K, eps, delta):
    """Evaluate F(x) (the three-term smoothed objective)."""
    u = D_apply(x)
    group_norms, _, _ = _ogs_internals(u, K, eps)
    f_data = 0.5 * np.sum((x - y) ** 2)
    f_ogs = lambda_g * np.sum(group_norms)
    f_ref = lambda_r * np.sum(w * huber(x - z, delta))
    return f_data + f_ogs + f_ref


def gradient(x, y, z, w, lambda_g, lambda_r, K, eps, delta, return_aux=False):
    """Evaluate grad F(x).

        grad F(x) = (x - y)
                  + lambda_g * D^T ( s . D x )
                  + lambda_r * w . huber'(x - z)
    """
    u = D_apply(x)
    _, _, s = _ogs_internals(u, K, eps)
    g_data = x - y
    g_ogs = lambda_g * DT_apply(s * u)
    g_ref = lambda_r * w * huber_grad(x - z, delta)
    g = g_data + g_ogs + g_ref
    if return_aux:
        return g, s, u
    return g


# ---------------------------------------------------------------------------
# Gradient sanity check (finite differences)
# ---------------------------------------------------------------------------
def check_gradient(n=64, K=5, eps=1e-3, delta=0.1, lambda_g=0.5, lambda_r=0.7,
                   seed=0, h=1e-6):
    """Compare analytic gradient to a central-difference gradient.
    Returns the max relative error.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, size=n)

    g_ana = gradient(x, y, z, w, lambda_g, lambda_r, K, eps, delta)
    g_num = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = h
        f_plus = objective(x + e, y, z, w, lambda_g, lambda_r, K, eps, delta)
        f_minus = objective(x - e, y, z, w, lambda_g, lambda_r, K, eps, delta)
        g_num[i] = (f_plus - f_minus) / (2 * h)
    err = np.max(np.abs(g_ana - g_num)) / max(np.max(np.abs(g_num)), 1e-12)
    return err, g_ana, g_num


# ---------------------------------------------------------------------------
# Solver 1 : Monotone gradient descent with backtracking line search
# ---------------------------------------------------------------------------
def solve_gd(y, z=None, w=None,
             lambda_g=0.05, lambda_r=0.0,
             K=25, eps=1e-3, delta=0.1,
             max_iter=500, tol=1e-7,
             init=None, verbose=False):
    """Monotone gradient descent with Armijo backtracking line search.

    Works on the smoothed objective F. Step size is chosen to guarantee
    F(x_{t+1}) <= F(x_t) - 0.5 * eta * ||grad F(x_t)||^2  (sufficient decrease).
    """
    n = len(y)
    if z is None:
        z = np.zeros(n)
        if lambda_r != 0.0:
            raise ValueError("lambda_r != 0 but no reference z given")
    if w is None:
        w = np.ones(n)
    x = y.copy() if init is None else init.copy()

    history = {"obj": [], "step": []}
    eta = 1.0

    for it in range(max_iter):
        f_x = objective(x, y, z, w, lambda_g, lambda_r, K, eps, delta)
        g = gradient(x, y, z, w, lambda_g, lambda_r, K, eps, delta)
        gnorm2 = float(np.dot(g, g))
        history["obj"].append(f_x)

        if gnorm2 < tol:
            if verbose:
                print(f"[gd] converged at iter {it}, |grad|^2={gnorm2:.3e}")
            break

        eta *= 2.0   # try a larger step each iter; backtrack if needed
        for _ in range(40):
            x_new = x - eta * g
            f_new = objective(x_new, y, z, w, lambda_g, lambda_r, K, eps, delta)
            if f_new <= f_x - 0.5 * eta * gnorm2:
                break
            eta *= 0.5
        else:
            if verbose:
                print(f"[gd] line search exhausted at iter {it}")
            break

        history["step"].append(eta)
        if abs(f_x - f_new) < tol * max(1.0, abs(f_x)):
            x = x_new
            history["obj"].append(f_new)
            if verbose:
                print(f"[gd] objective plateau at iter {it}")
            break
        x = x_new

    return x, history


# ---------------------------------------------------------------------------
# Solver 2 : Majorization-Minimization with O(n) tridiagonal sub-problem
# ---------------------------------------------------------------------------
# Majorizers used:
#   For the smoothed group term  phi(s) = sqrt(s + eps^2),  concave in s = ||u||^2,
#   so the linear majorizer at u_t gives
#       sqrt(||u||^2 + eps^2)  <=  beta_t * ||u||^2 / 2 + const.
#   Summed over groups, the OGS penalty is upper-bounded by
#       (1/2) * (D x)^T diag(s_t) (D x) + const,
#   exactly the same s_t = sliding sum of beta we computed above.
#
#   For the Huber term, the half-quadratic majorizer at t_t = x_t - z is
#       huber_delta(t)  <=  0.5 * a_t * t^2 + const,
#       with a_t = min(1, delta / |t_t|).
#
# The sub-problem becomes a quadratic in x:
#       min_x  0.5 ||x - y||^2
#            + (lambda_g / 2) (D x)^T diag(s_t) (D x)
#            + (lambda_r / 2) sum_i w_i a_t_i (x_i - z_i)^2
# whose normal equations
#       ( I + lambda_g D^T diag(s_t) D + lambda_r diag(w * a_t) ) x = y + lambda_r (w * a_t) . z
# yield a *symmetric tridiagonal* system in x (D^T diag(s) D is tridiagonal).
# We solve it in O(n) via scipy.linalg.solveh_banded.
# ---------------------------------------------------------------------------
def _build_tridiag_and_solve(s, a, y, z, w, lambda_g, lambda_r):
    """Solve  ( I + lambda_g D^T diag(s) D + lambda_r diag(w*a) ) x = y + lambda_r (w*a).z .

    s : (n-1,)   sliding-sum OGS weights
    a : (n,)     Huber HQ weights in [0,1]
    """
    n = len(y)
    wa = lambda_r * w * a               # (n,)

    # main diagonal of (D^T diag(s) D):
    #   diag[0]   = s[0]
    #   diag[i]   = s[i-1] + s[i],  1 <= i <= n-2
    #   diag[n-1] = s[n-2]
    main_DtSD = np.empty(n)
    main_DtSD[0] = s[0]
    main_DtSD[-1] = s[-1]
    main_DtSD[1:-1] = s[:-1] + s[1:]

    main = 1.0 + lambda_g * main_DtSD + wa
    off = -lambda_g * s                  # super- (and sub-) diagonal, length n-1

    # solveh_banded with upper banded matrix:
    #   ab[0, j] = super-diagonal at column j (for j>=1),  ab[1, j] = main diagonal
    ab = np.zeros((2, n))
    ab[1, :] = main
    ab[0, 1:] = off

    rhs = y + wa * z
    return solveh_banded(ab, rhs, lower=False)


def solve_mm(y, z=None, w=None,
             lambda_g=0.05, lambda_r=0.0,
             K=25, eps=1e-3, delta=0.1,
             max_iter=200, tol=1e-7,
             init=None, verbose=False):
    """Majorization-minimization solver with O(n) tridiagonal sub-problem.

    Each outer iteration:
      1. compute u = D x_t  and the sliding-sum OGS weights s_t in O(n)
      2. compute Huber HQ weights a_t = min(1, delta / |x_t - z|) in O(n)
      3. solve the quadratic majorizer (symmetric tridiag) for x_{t+1} in O(n)

    The objective is monotone non-increasing under MM by construction.
    """
    n = len(y)
    if z is None:
        z = np.zeros(n)
        if lambda_r != 0.0:
            raise ValueError("lambda_r != 0 but no reference z given")
    if w is None:
        w = np.ones(n)
    x = y.copy() if init is None else init.copy()

    history = {"obj": []}

    f_prev = objective(x, y, z, w, lambda_g, lambda_r, K, eps, delta)
    history["obj"].append(f_prev)

    for it in range(max_iter):
        # OGS weights from current iterate
        u = D_apply(x)
        _, _, s = _ogs_internals(u, K, eps)

        # Huber HQ weights
        if lambda_r > 0.0:
            r = x - z
            a = np.minimum(1.0, delta / np.maximum(np.abs(r), 1e-12))
        else:
            a = np.zeros(n)

        x_new = _build_tridiag_and_solve(s, a, y, z, w, lambda_g, lambda_r)
        f_new = objective(x_new, y, z, w, lambda_g, lambda_r, K, eps, delta)
        history["obj"].append(f_new)

        rel = (f_prev - f_new) / max(1.0, abs(f_prev))
        if verbose and (it % 10 == 0 or rel < tol):
            print(f"[mm] it={it:3d}  F={f_new:.6e}  rel_dec={rel:.2e}")
        x = x_new
        if rel < tol:
            break
        f_prev = f_new

    return x, history


# ---------------------------------------------------------------------------
# Reference alignment: phase-align z (longer signal) to y via cross-correlation,
# then fit an optimal scale alpha = <y, z_shifted> / ||z_shifted||^2.
# ---------------------------------------------------------------------------
def align_reference(y: np.ndarray, z_full: np.ndarray, fit_scale: bool = True):
    """Return z_aligned of the same length as y.

    Picks the shift tau maximizing |corr(y, z_full[tau:tau+n])| via FFT
    cross-correlation, then optionally fits a scalar gain alpha by least
    squares. If z_full is shorter than y, it is right-padded with zeros.
    """
    from scipy.signal import correlate

    n = len(y)
    if len(z_full) < n:
        z_full = np.concatenate([z_full, np.zeros(n - len(z_full))])

    # Demean for shift search to avoid DC bias dominating cross-correlation.
    y_dm = y - y.mean()
    z_dm = z_full - z_full.mean()

    corr = correlate(z_dm, y_dm, mode="valid")          # length L - n + 1
    tau = int(np.argmax(np.abs(corr)))
    z_shifted = z_full[tau : tau + n].astype(float)

    if fit_scale:
        denom = float(np.dot(z_shifted, z_shifted))
        alpha = float(np.dot(y, z_shifted) / denom) if denom > 1e-12 else 1.0
    else:
        alpha = 1.0
    return alpha * z_shifted, tau, alpha


# ---------------------------------------------------------------------------
# Quick self-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== gradient finite-difference check ===")
    err, ga, gn = check_gradient(n=64, K=5, eps=1e-3, delta=0.1,
                                 lambda_g=0.5, lambda_r=0.7)
    print(f"max relative error vs central differences: {err:.3e}")

    print("\n=== monotonicity of MM on a synthetic problem ===")
    rng = np.random.default_rng(0)
    n = 256
    t = np.linspace(0, 1, n)
    clean = np.sin(8 * np.pi * t) * (t > 0.2) * (t < 0.8)
    y = clean + 0.2 * rng.standard_normal(n)
    z = clean
    x_mm, hist_mm = solve_mm(y, z=z, lambda_g=0.05, lambda_r=0.5,
                             K=8, eps=1e-3, delta=0.05,
                             max_iter=50, verbose=True)
    objs = np.array(hist_mm["obj"])
    print("monotone non-increasing:", bool(np.all(np.diff(objs) <= 1e-10)))

    print("\n=== sanity: GD also reaches a similar objective ===")
    x_gd, hist_gd = solve_gd(y, z=z, lambda_g=0.05, lambda_r=0.5,
                             K=8, eps=1e-3, delta=0.05,
                             max_iter=2000, verbose=False)
    print(f"  MM final F = {hist_mm['obj'][-1]:.6e}")
    print(f"  GD final F = {hist_gd['obj'][-1]:.6e}")
