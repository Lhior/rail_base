"""
A summarizer that takes the input of a photo-z point estimate qp ensemble, and the cluster redshift likelihood
and run logistic Gaussian process to estimate the posteroir of the redshift distribution n(z) for a sample.

Author: Markus Michael Rau, Tianqing Zhang
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import qp
from ceci.config import StageParameter as Param
from numpy.linalg import LinAlgError
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import eval_chebyt

from rail.core.common_params import SharedParams
from rail.core.data import ModelHandle, ModelLike, QPHandle
from rail.estimation.summarizer import PZSummarizer


PDF_FLOOR = 1.0e-12
MIN_RETAINED_SAMPLES = 200
AMP_ACCEPT_LO = 0.15
AMP_ACCEPT_HI = 0.5
AMP_ADAPT_TARGET = 0.25


def compute_bin_widths(mids: np.ndarray) -> np.ndarray:
    mids = np.asarray(mids, dtype=float)
    if mids.size <= 1:
        return np.ones_like(mids)
    edges = np.empty(mids.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (mids[:-1] + mids[1:])
    edges[0] = mids[0] - 0.5 * (mids[1] - mids[0])
    edges[-1] = mids[-1] + 0.5 * (mids[-1] - mids[-2])
    return np.diff(edges)


def choose_reference_bin(pdfs: np.ndarray, widths: np.ndarray) -> int:
    """Return the bin index with the largest median mass across ensemble members."""
    pdfs = np.atleast_2d(np.asarray(pdfs, dtype=float))
    widths = np.asarray(widths, dtype=float)
    mass = np.clip(pdfs, PDF_FLOOR, None) * widths
    median_mass = np.median(mass, axis=0)
    return int(np.argmax(median_mass))


def pdf_to_logit(
    pdf: np.ndarray,
    widths: Optional[np.ndarray] = None,
    reference_bin: int = -1,
) -> np.ndarray:
    """Project a density defined on bins to simplex coordinates (Eq. 54/55 in Rau et al. 2021).

    ``reference_bin`` selects the additive-log-ratio baseline; ``-1`` means the last bin.
    """
    pdf = np.asarray(pdf, dtype=float)
    if widths is None:
        widths = np.ones_like(pdf)
    widths = np.asarray(widths, dtype=float)
    mass = np.clip(pdf, PDF_FLOOR, None) * widths
    total = np.sum(mass)
    if not np.isfinite(total) or total <= 0.0:
        total = PDF_FLOOR * mass.size
    simplex = mass / total
    n = simplex.size
    ref = reference_bin if reference_bin >= 0 else n - 1
    if ref < 0 or ref >= n:
        raise ValueError(f"reference_bin={ref} out of range for {n} bins")
    baseline = np.clip(simplex[ref], PDF_FLOOR, None)
    keep = np.ones(n, dtype=bool)
    keep[ref] = False
    return np.log(np.clip(simplex[keep], PDF_FLOOR, None)) - np.log(baseline)


def logit_to_pdf(
    logit: np.ndarray,
    widths: Optional[np.ndarray] = None,
    reference_bin: int = -1,
) -> np.ndarray:
    """Inverse logistic transform returning a density whose bin-integrals follow the simplex."""
    logit = np.asarray(logit, dtype=float)
    n = logit.size + 1
    if widths is None:
        widths = np.ones(n, dtype=float)
    else:
        widths = np.asarray(widths, dtype=float)
    ref = reference_bin if reference_bin >= 0 else n - 1
    if ref < 0 or ref >= n:
        raise ValueError(f"reference_bin={ref} out of range for {n} bins")

    # Numerically stable softmax: reference bin corresponds to logit 0, so after
    # shifting by max(logit) it contributes exp(-max) rather than 1.
    m = np.max(logit) if logit.size else 0.0
    exp_terms = np.exp(logit - m)
    denom = np.exp(-m) + np.sum(exp_terms)
    simplex = np.empty(n, dtype=float)
    keep = np.ones(n, dtype=bool)
    keep[ref] = False
    simplex[keep] = exp_terms / denom
    simplex[ref] = np.exp(-m) / denom
    clipped_widths = np.clip(widths, PDF_FLOOR, None)
    density = simplex / clipped_widths
    norm = np.sum(density * clipped_widths)
    if not np.isfinite(norm) or norm <= 0.0:
        norm = PDF_FLOOR * density.size
    density /= norm
    return density


def _shrink_cov(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink a covariance towards its diagonal (Ledoit-Wolf style target)."""
    return (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))


def _ledoit_wolf_diag(samples: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate a covariance shrunk towards its diagonal with Ledoit-Wolf intensity."""
    num_samples = samples.shape[0]
    deviation = samples - samples.mean(axis=0, keepdims=True)

    cov = (deviation.T @ deviation) / num_samples
    target = np.diag(np.diag(cov))

    deviation_sq = deviation**2
    pi_mat = (deviation_sq.T @ deviation_sq) / num_samples - cov**2
    pi_hat = pi_mat.sum()
    rho_hat = np.trace(pi_mat)
    gamma = np.sum((cov - target) ** 2)

    if gamma <= 0.0:
        intensity = 0.0
    else:
        intensity = float(np.clip((pi_hat - rho_hat) / (gamma * num_samples), 0.0, 1.0))

    shrunk = _shrink_cov(cov, intensity)
    if num_samples > 1:
        shrunk *= num_samples / (num_samples - 1)
    return shrunk, intensity


def _ledoit_wolf_intensity_from_cov(cov: np.ndarray) -> float:
    """Approximate Ledoit-Wolf intensity from a precomputed covariance matrix.

    Without the underlying jackknife samples we cannot evaluate the full
    Ledoit-Wolf estimator.  Use the fraction of off-diagonal Frobenius power as
    a conservative intensity in ``[0, 1]``.
    """
    cov = np.asarray(cov, dtype=float)
    target = np.diag(np.diag(cov))
    off = cov - target
    gamma = float(np.sum(off**2))
    total = float(np.sum(cov**2))
    if total <= 0.0 or gamma <= 0.0:
        return 0.0
    return float(np.clip(gamma / total, 0.0, 1.0))


def hartlap_factor(n_patches: int, n_bins: int) -> float:
    """Hartlap et al. (2007) debiasing factor for an inverse covariance.

    Returns 1 when the correction is undefined (``n_patches <= n_bins + 2``).
    """
    if n_patches <= 0 or n_bins <= 0:
        return 1.0
    if n_patches <= n_bins + 2:
        return 1.0
    return float((n_patches - n_bins - 2) / (n_patches - 1))


def stabilize_covariance(
    cov: np.ndarray,
    *,
    n_patches: int = 0,
    shrinkage: float = -1.0,
    jitter: float = 1.0e-12,
) -> Tuple[np.ndarray, dict[str, float]]:
    """Apply Ledoit-Wolf shrinkage and Hartlap scaling to a clustering covariance.

    The returned matrix is the *precision-ready* covariance: shrinkage is applied
    to the covariance, then Hartlap multiplies it so that ``C^{-1}`` is
    effectively scaled by the Hartlap factor (``C_out = C_shrunk / hartlap``).
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if shrinkage < 0.0:
        intensity = _ledoit_wolf_intensity_from_cov(cov)
    else:
        intensity = float(np.clip(shrinkage, 0.0, 1.0))
    shrunk = _shrink_cov(cov, intensity)
    h = hartlap_factor(int(n_patches), n)
    if h > 0.0 and h != 1.0:
        # Scaling cov by 1/h scales the precision by h.
        stabilized = shrunk / h
    else:
        stabilized = shrunk
    stabilized = stabilized + np.eye(n) * jitter
    meta = {
        "shrinkage_intensity": float(intensity),
        "hartlap_factor": float(h),
        "n_patches": float(n_patches),
        "n_bins": float(n),
    }
    return stabilized, meta


def midpoints_to_edges(mids: np.ndarray) -> np.ndarray:
    """Convert bin midpoints to edges (same construction as ``compute_bin_widths``)."""
    mids = np.asarray(mids, dtype=float)
    if mids.size == 0:
        return np.zeros(0, dtype=float)
    if mids.size == 1:
        return np.array([mids[0] - 0.5, mids[0] + 0.5], dtype=float)
    edges = np.empty(mids.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (mids[:-1] + mids[1:])
    edges[0] = mids[0] - 0.5 * (mids[1] - mids[0])
    edges[-1] = mids[-1] + 0.5 * (mids[-1] - mids[-2])
    return edges


def bin_average_profile(
    zmid: np.ndarray,
    profile: np.ndarray,
    target_mids: np.ndarray,
) -> np.ndarray:
    """Average ``profile`` defined on ``zmid`` over bins centred on ``target_mids``.

    On a matched grid this is the identity (the papers' bin-averaged histogram
    heights).  Otherwise a linear interpolant is integrated over each target bin
    and divided by the bin width.  The photometric support is the edge span of
    ``zmid``; outside that support the prediction is zero.  Within the support
    but beyond the outermost midpoints the boundary profile value is held.
    """
    zmid = np.asarray(zmid, dtype=float)
    profile = np.asarray(profile, dtype=float)
    target_mids = np.asarray(target_mids, dtype=float)
    if target_mids.size == 0:
        return np.zeros(0, dtype=float)
    if zmid.size == 0:
        return np.zeros_like(target_mids)
    if target_mids.size == zmid.size and np.allclose(target_mids, zmid):
        return profile.copy()

    # Evaluate with boundary-value hold so edge overhangs stay filled; zeros
    # outside the photometric *edge* span are enforced by the clip below.
    spline = InterpolatedUnivariateSpline(zmid, profile, k=1, ext=0)

    def _eval(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        left = z < zmid[0]
        right = z > zmid[-1]
        mid = ~left & ~right
        out[left] = profile[0]
        out[right] = profile[-1]
        if np.any(mid):
            out[mid] = spline(z[mid])
        return out

    edges = midpoints_to_edges(target_mids)
    support = midpoints_to_edges(zmid)
    z_lo = float(support[0])
    z_hi = float(support[-1])
    out = np.zeros(target_mids.size, dtype=float)
    for i in range(target_mids.size):
        a = float(edges[i])
        b = float(edges[i + 1])
        width = b - a
        if width <= 0.0:
            continue
        lo = max(a, z_lo)
        hi = min(b, z_hi)
        if hi <= lo:
            continue
        # Trapezoidal average on a few samples across the overlapping interval.
        zs = np.linspace(lo, hi, 8)
        vals = _eval(zs)
        integral = float(np.trapezoid(vals, zs))
        out[i] = integral / width
    return out


def chebyshev_bias_factor(
    z: np.ndarray,
    coeffs: np.ndarray,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Evaluate B(z) = exp(sum_{k=1}^{K} c_k T_k(u(z))) with u mapped onto [-1, 1].

    The k=0 (constant) mode is absorbed into the free amplitude, so ``coeffs``
    starts at Chebyshev order 1.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.size == 0:
        return np.ones_like(z, dtype=float)
    z = np.asarray(z, dtype=float)
    span = max(z_max - z_min, PDF_FLOOR)
    u = 2.0 * (z - z_min) / span - 1.0
    u = np.clip(u, -1.0, 1.0)
    log_b = np.zeros_like(z, dtype=float)
    for k, c in enumerate(coeffs, start=1):
        log_b += c * eval_chebyt(k, u)
    return np.exp(log_b)


def split_half_rhat(chain: np.ndarray) -> float:
    """Gelman-Rubin R-hat from splitting one chain into two halves."""
    chain = np.asarray(chain, dtype=float).ravel()
    n = chain.size
    if n < 4:
        return np.nan
    half = n // 2
    c1 = chain[:half]
    c2 = chain[half : 2 * half]
    m = 2
    n_eff = half
    means = np.array([c1.mean(), c2.mean()])
    vars_ = np.array([c1.var(ddof=1), c2.var(ddof=1)])
    b = n_eff * np.var(means, ddof=1)
    w = vars_.mean()
    if w <= 0.0:
        return np.nan if b <= 0.0 else np.inf
    var_plus = ((n_eff - 1) / n_eff) * w + b / n_eff
    return float(np.sqrt(var_plus / w))


def effective_sample_size(chain: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Autocorrelation-based effective sample size for a 1-d chain."""
    chain = np.asarray(chain, dtype=float).ravel()
    n = chain.size
    if n < 2:
        return float(n)
    x = chain - chain.mean()
    var = np.dot(x, x) / n
    if var <= 0.0:
        return float(n)
    if max_lag is None:
        max_lag = n // 2
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = np.dot(x[:-lag], x[lag:]) / (n * var)
        if lag > 1 and rho < 0.0:
            break
        acf_sum += rho
    ess = n / (1.0 + 2.0 * acf_sum)
    return float(max(ess, 1.0))


class EllipticalSliceSampler:
    def __init__(self, prior_mean: np.ndarray, prior_cov: np.ndarray):
        """Elliptical slice sampler with reusable prior (Murray et al. 2010)."""
        self.prior_mean = np.asarray(prior_mean, dtype=float)
        self.prior_cov = np.asarray(prior_cov, dtype=float)
        self._n = self.prior_mean.size

        try:
            self._chol = np.linalg.cholesky(self.prior_cov)
        except LinAlgError:
            jitter = np.eye(self._n) * 1.0e-8
            self._chol = np.linalg.cholesky(self.prior_cov + jitter)

    def step(self, current_state: np.ndarray, loglik: Callable[[np.ndarray], float]) -> np.ndarray:
        """Draw a single sample given the current state and log-likelihood."""
        current_state = np.asarray(current_state, dtype=float)
        nu = self._chol @ np.random.randn(self._n)
        log_y = loglik(current_state) + np.log(np.random.uniform())
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        theta_min, theta_max = theta - 2.0 * np.pi, theta
        mu = self.prior_mean

        while True:
            proposal = (current_state - mu) * np.cos(theta) + nu * np.sin(theta) + mu
            if loglik(proposal) > log_y:
                return proposal
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = np.random.uniform(theta_min, theta_max)

    def sample(
        self,
        loglik: Callable[[np.ndarray], float],
        initial_state: Optional[np.ndarray] = None,
        n_samples: int = 1,
        n_burn: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples, returning both the chain segment and the final state."""
        if initial_state is None:
            initial_state = self.prior_mean.copy()
        state = np.asarray(initial_state, dtype=float)
        collected = []
        for i in range(n_samples):
            state = self.step(state, loglik)
            if i >= n_burn:
                collected.append(state.copy())
        return np.array(collected), state


class LogLike(object):

    def __init__(
        self,
        zmid_wx,
        mean_wx,
        cov_wx,
        reference_bin: int = -1,
        *,
        n_patches: int = 0,
        cov_shrinkage: float = -1.0,
    ):
        """
        Initialize the LogLike object.

        Parameters:
        zmid_wx (array-like): Midpoints of redshift bins where cluster redshift is defined.
        mean_wx (array-like): Mean vector for the cluster redshift distribution.
        cov_wx (array-like): Covariance matrix for the cluster redshift distribution.
        reference_bin (int): Additive-log-ratio reference bin for logit transforms.
        n_patches (int): Jackknife patch count for Hartlap correction (<=0 disables).
        cov_shrinkage (float): Ledoit-Wolf intensity; <0 estimates from cov, 0 disables.
        """
        self.zmid_wx = np.asarray(zmid_wx, dtype=float)
        self.mean_wx = np.asarray(mean_wx, dtype=float)
        self.reference_bin = int(reference_bin)

        if len(self.zmid_wx) > 1:
            self.bin_widths = compute_bin_widths(self.zmid_wx)
            self.bin_edges = midpoints_to_edges(self.zmid_wx)
        else:
            self.bin_widths = np.ones_like(self.mean_wx)
            self.bin_edges = midpoints_to_edges(self.zmid_wx)

        self.cov_wx_raw = np.asarray(cov_wx, dtype=float)
        self.cov_wx, self.cov_meta = stabilize_covariance(
            self.cov_wx_raw,
            n_patches=int(n_patches),
            shrinkage=float(cov_shrinkage),
        )

        # Factorize once; every likelihood evaluation then uses triangular solves.
        try:
            self._chol = np.linalg.cholesky(self.cov_wx)
        except LinAlgError:
            jitter = np.eye(self.cov_wx.shape[0]) * 1.0e-8
            self._chol = np.linalg.cholesky(self.cov_wx + jitter)
        self._logdet = 2.0 * np.sum(np.log(np.diag(self._chol)))
        self._n_wx = self.mean_wx.size
        self._log_norm = -0.5 * (self._n_wx * np.log(2.0 * np.pi) + self._logdet)
        self.z_min_wx = float(np.min(self.zmid_wx)) if self.zmid_wx.size else 0.0
        self.z_max_wx = float(np.max(self.zmid_wx)) if self.zmid_wx.size else 1.0

    def _gaussian_logpdf(self, expected: np.ndarray) -> float:
        residual = np.asarray(expected, dtype=float) - self.mean_wx
        solved = np.linalg.solve(self._chol, residual)
        return float(self._log_norm - 0.5 * np.dot(solved, solved))

    def _project_profile(self, zmid: np.ndarray, profile: np.ndarray) -> np.ndarray:
        """Bin-average the model profile onto the clustering redshift grid."""
        return bin_average_profile(zmid, profile, self.zmid_wx)

    def chi2(self, expected: np.ndarray) -> float:
        """Gaussian chi-squared of ``expected`` against the data vector."""
        residual = np.asarray(expected, dtype=float) - self.mean_wx
        solved = np.linalg.solve(self._chol, residual)
        return float(np.dot(solved, solved))

    def draw_replicas(self, expected: np.ndarray, n_replicas: int, rng=None) -> np.ndarray:
        """Draw replicated data vectors ~ N(expected, cov_wx)."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.multivariate_normal(
            mean=np.asarray(expected, dtype=float),
            cov=self.cov_wx,
            size=int(n_replicas),
        )

    def _expected_counts(
        self,
        zmid: np.ndarray,
        logit_vec: np.ndarray,
        amp: float,
        bias_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        widths = compute_bin_widths(zmid)
        probs = logit_to_pdf(logit_vec, widths, reference_bin=self.reference_bin)
        prof_wx = self._project_profile(zmid, probs)
        expected = prof_wx * self.bin_widths * amp
        if bias_coeffs is not None and np.size(bias_coeffs) > 0:
            expected = expected * chebyshev_bias_factor(
                self.zmid_wx, bias_coeffs, self.z_min_wx, self.z_max_wx
            )
        return expected

    def loglike_logit_given_amp(
        self,
        zmid: np.ndarray,
        amp: float,
        bias_coeffs: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], float]:
        def loss(logit_vec: np.ndarray) -> float:
            expected_counts = self._expected_counts(zmid, logit_vec, amp, bias_coeffs)
            return self._gaussian_logpdf(expected_counts)

        return loss

    def loglike_amp_given_logit(
        self,
        zmid: np.ndarray,
        logit_vec: np.ndarray,
        bias_coeffs: Optional[np.ndarray] = None,
    ) -> Callable[[float], float]:
        widths = compute_bin_widths(zmid)
        probs = logit_to_pdf(logit_vec, widths, reference_bin=self.reference_bin)
        prof_wx = self._project_profile(zmid, probs)
        prof_counts = prof_wx * self.bin_widths
        if bias_coeffs is not None and np.size(bias_coeffs) > 0:
            prof_counts = prof_counts * chebyshev_bias_factor(
                self.zmid_wx, bias_coeffs, self.z_min_wx, self.z_max_wx
            )

        def loss(amp: float) -> float:
            if amp <= 0.0:
                return -np.inf
            return self._gaussian_logpdf(prof_counts * amp)

        return loss

    def loglike_bias_given_amp_logit(
        self,
        zmid: np.ndarray,
        amp: float,
        logit_vec: np.ndarray,
        bias_prior_scale: float,
    ) -> Callable[[np.ndarray], float]:
        widths = compute_bin_widths(zmid)
        probs = logit_to_pdf(logit_vec, widths, reference_bin=self.reference_bin)
        prof_wx = self._project_profile(zmid, probs)
        base_counts = prof_wx * self.bin_widths * amp

        def loss(coeffs: np.ndarray) -> float:
            coeffs = np.asarray(coeffs, dtype=float)
            bias = chebyshev_bias_factor(
                self.zmid_wx, coeffs, self.z_min_wx, self.z_max_wx
            )
            loglik = self._gaussian_logpdf(base_counts * bias)
            # Weakly informative N(0, bias_prior_scale^2) prior on each coefficient.
            log_prior = -0.5 * np.sum((coeffs / bias_prior_scale) ** 2)
            log_prior -= coeffs.size * np.log(bias_prior_scale * np.sqrt(2.0 * np.pi))
            return float(loglik + log_prior)

        return loss


def convert_mids_to_breaks(mids):
    """
    Convert midpoints of a grid to the breaks of the grid
    Assume the grid points have equal distances
    The output will have size of len(mids)+1
    """
    breaks = np.copy(mids)
    delta_z = breaks[1] - breaks[0]
    breaks = breaks - delta_z / 2.0
    breaks = breaks.tolist()
    breaks.append(breaks[-1] + delta_z)
    breaks = np.array(breaks)
    return breaks


def convert_breaks_to_mids(breaks):
    """
    Convert midpoints of a grid to the breaks of the grid
    Assume the grid points have equal distances
    The output will have size of len(mids)+1
    """
    mids = breaks[:-1]
    delta = 0.5 * (breaks[1] - breaks[0])
    mids = mids + delta
    return mids


class LogisticGPSummarizer(PZSummarizer):
    """Logistic Gaussian Process summarizer combining photo-z and clustering data.

    The free amplitude is sampled with a random walk in ``log(amp)`` under an
    improper flat prior in log-amplitude (Jeffreys prior for a scale parameter);
    no Hastings correction is required. ``min_amp`` is only a numerical floor.
    """

    name = "LogisticGPSummarizer"
    entrypoint_function = "summarize"
    interactive_function = "logistic_gp_summarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=SharedParams.copy_param("zmin"),
        zmax=SharedParams.copy_param("zmax"),
        nzbins=SharedParams.copy_param("nzbins"),
        n_steps=Param(int, 5000, msg="N-steps for MCMC sampling"),
        afterburner=Param(int, 2000, msg="Remove the samples before chain converge"),
        amp_step=Param(
            float,
            0.5,
            msg="RW proposal scale for log(amp); scale-free under flat-in-log prior",
        ),
        min_amp=Param(
            float,
            1.0e-3,
            msg="Numerical lower bound for amplitude (not a prior boundary)",
        ),
        initial_amp=Param(
            float,
            -1.0,
            msg="Optional user-specified amplitude start (<=0 uses data-driven start)",
        ),
        prior_jitter=Param(
            float,
            1.0e-6,
            msg="Diagonal floor added after Ledoit-Wolf shrinkage of the prior covariance",
        ),
        progress_interval=Param(int, 500, msg="Print progress every N iterations (<=0 disables)"),
        ess_n_samples=Param(int, 20, msg="Number of ESS proposals per joint iteration"),
        ess_n_burn=Param(int, 10, msg="Number of burn-in ESS steps per joint iteration"),
        adapt_amp_step=Param(
            bool,
            True,
            msg="Robbins-Monro adaptation of amp_step during afterburner only",
        ),
        reference_bin=Param(
            int,
            -2,
            msg="ALR reference bin (-2=largest median mass, -1=last bin, >=0=explicit index)",
        ),
        bias_poly_order=Param(
            int,
            0,
            msg="Chebyshev order for unknown-sample bias evolution B(z); 0 disables",
        ),
        bias_prior_scale=Param(
            float,
            1.0,
            msg="Gaussian prior stddev on each Chebyshev bias coefficient",
        ),
        bias_step=Param(
            float,
            0.1,
            msg="RW proposal scale for each Chebyshev bias coefficient",
        ),
        n_patches=Param(
            int,
            64,
            msg="Jackknife patch count for Hartlap correction of cov_wx (<=0 disables)",
        ),
        cov_shrinkage=Param(
            float,
            -1.0,
            msg="Ledoit-Wolf intensity for cov_wx; <0 estimates intensity, 0 disables",
        ),
        run_ppc=Param(bool, True, msg="Compute posterior predictive checks"),
        ppc_n_replicas=Param(int, 200, msg="Number of PPC replicated data vectors"),
    )
    inputs = [("model", ModelHandle), ("input", QPHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.diagnostics: dict[str, Any] = {}
        self.trace_amp0: Optional[np.ndarray] = None
        self.trace_logit0: Optional[np.ndarray] = None
        self.trace_bias0: Optional[np.ndarray] = None
        self.reference_bin_used: int = -1
        self.ppc: dict[str, Any] = {}
        self._loglike_model: Optional[LogLike] = None

    def summarize(
        self, input_data: qp.Ensemble, model: ModelLike, **kwargs
    ) -> QPHandle:
        """Summarize photo-z ensemble data using a clustering redshift likelihood model.

        Parameters
        ----------
        input_data : qp.Ensemble
            Per-galaxy p(z), and any ancillary data associated with it
        model : ModelLike
            Model containing cluster redshift information with keys
            ``zmid_wx``, ``signal_wx``, and ``cov_wx``. Optional keys
            ``prior_mean`` / ``prior_cov`` bypass sample-based prior estimation.

        Returns
        -------
        QPHandle
            Ensemble with n(z), and any ancillary data
        """
        self.set_data("model", model)
        model = self.get_data("model")

        self.zmid_wx = model["zmid_wx"]
        self.signal_wx = model["signal_wx"]
        self.cov_wx = model["cov_wx"]
        self._model_prior_mean = model.get("prior_mean")
        self._model_prior_cov = model.get("prior_cov")
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def _resolve_reference_bin(self, pdf_normed: np.ndarray) -> int:
        cfg_ref = int(self.config.reference_bin)
        n_bins = pdf_normed.shape[1]
        if cfg_ref == -2:
            return choose_reference_bin(pdf_normed, self.zgrid_widths)
        if cfg_ref == -1:
            return n_bins - 1
        if cfg_ref < 0 or cfg_ref >= n_bins:
            raise ValueError(
                f"reference_bin={cfg_ref} out of range for {n_bins} redshift bins"
            )
        return cfg_ref

    def _build_prior(
        self, pdf_normed: np.ndarray, reference_bin: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._model_prior_mean is not None and self._model_prior_cov is not None:
            prior_mean = np.asarray(self._model_prior_mean, dtype=float)
            prior_cov = np.asarray(self._model_prior_cov, dtype=float)
            if prior_mean.ndim != 1:
                raise ValueError("model prior_mean must be 1-d")
            if prior_cov.shape != (prior_mean.size, prior_mean.size):
                raise ValueError(
                    f"model prior_cov shape {prior_cov.shape} incompatible with "
                    f"prior_mean length {prior_mean.size}"
                )
            prior_cov = prior_cov + np.eye(prior_mean.size) * self.config.prior_jitter
            return prior_mean, prior_cov

        logit_samples = np.array(
            [
                pdf_to_logit(p, self.zgrid_widths, reference_bin=reference_bin)
                for p in pdf_normed
            ]
        )
        prior_mean = logit_samples.mean(axis=0)
        n_members = logit_samples.shape[0]
        logit_dim = prior_mean.size
        if n_members <= logit_dim:
            raise ValueError(
                f"LogisticGP prior covariance is rank-deficient: "
                f"{n_members} ensemble members for logit dimension {logit_dim}. "
                f"Increase sample_count above nzbins-1 (currently nzbins="
                f"{self.config.nzbins}) or reduce nzbins so the model grid matches "
                f"the clustering redshift binning."
            )
        if n_members < 2:
            prior_cov = np.eye(logit_dim)
        else:
            prior_cov, _intensity = _ledoit_wolf_diag(logit_samples)
        prior_cov = prior_cov + np.eye(logit_dim) * self.config.prior_jitter
        return prior_mean, prior_cov

    def sample_joint(self):
        """
        Perform joint sampling of amplitude, logit vector, and optional bias coeffs.

        Amplitude is proposed in log-space under a flat prior in log(amp)
        (Jeffreys prior for a scale parameter). Returns traces and populates
        ``self.diagnostics``.
        """
        pdf_values = np.atleast_2d(self.qp_output.pdf(self.zgrid_mid))
        pdf_values = np.clip(pdf_values, PDF_FLOOR, None)

        normalisation = np.array([np.sum(p * self.zgrid_widths) for p in pdf_values])
        pdf_normed = pdf_values / normalisation[:, None]

        reference_bin = self._resolve_reference_bin(pdf_normed)
        self.reference_bin_used = reference_bin

        loglike_model = LogLike(
            self.zmid_wx,
            self.signal_wx,
            self.cov_wx,
            reference_bin=reference_bin,
            n_patches=int(self.config.n_patches),
            cov_shrinkage=float(self.config.cov_shrinkage),
        )
        self._loglike_model = loglike_model
        prior_mean, prior_cov = self._build_prior(pdf_normed, reference_bin)
        sampler = EllipticalSliceSampler(prior_mean, prior_cov)

        if self.config.initial_amp > 0:
            initial_amp = self.config.initial_amp
        else:
            initial_amp = np.sum(self.signal_wx)
        initial_amp = max(float(initial_amp), float(self.config.min_amp))

        bias_order = int(self.config.bias_poly_order)
        current_bias = np.zeros(bias_order, dtype=float)

        n_steps = int(self.config.n_steps)
        afterburner = int(self.config.afterburner)
        retained = n_steps - afterburner
        if retained < MIN_RETAINED_SAMPLES:
            print(
                f"WARNING: LogisticGP retained chain length {retained} "
                f"(n_steps={n_steps}, afterburner={afterburner}) is below "
                f"{MIN_RETAINED_SAMPLES}; posterior diagnostics will be unreliable."
            )

        trace_amp = [initial_amp]
        trace_logit = [prior_mean.copy()]
        trace_bias = [current_bias.copy()]

        current_amp = initial_amp
        current_logit = prior_mean.copy()
        log_amp = np.log(current_amp)
        amp_step = float(self.config.amp_step)
        amp_accepts = 0
        amp_proposals = 0
        bias_accepts = 0
        bias_proposals = 0

        for step in range(n_steps):
            if self.config.progress_interval > 0 and step % self.config.progress_interval == 0:
                print(f"LogisticGP sampler step {step}")

            # --- Amplitude block: random walk in log(amp) ---
            loss_amp = loglike_model.loglike_amp_given_logit(
                self.zgrid_mid, current_logit, bias_coeffs=current_bias
            )
            proposed_log_amp = np.random.normal(log_amp, amp_step)
            proposed_amp = float(np.exp(proposed_log_amp))
            amp_proposals += 1
            accepted_amp = False
            if proposed_amp > self.config.min_amp:
                # Flat prior in log(amp): no Hastings correction for the RW proposal.
                log_accept_ratio = loss_amp(proposed_amp) - loss_amp(current_amp)
                if log_accept_ratio > np.log(np.random.uniform(low=0.0, high=1.0)):
                    current_amp = proposed_amp
                    log_amp = proposed_log_amp
                    accepted_amp = True
                    amp_accepts += 1
            trace_amp.append(current_amp)

            if (
                self.config.adapt_amp_step
                and step < afterburner
                and amp_proposals > 0
            ):
                # Robbins-Monro adaptation during burn-in only; freeze afterwards.
                accept_so_far = amp_accepts / amp_proposals
                gain = 1.0 / (10.0 + step)
                amp_step = float(
                    amp_step * np.exp(gain * (accept_so_far - AMP_ADAPT_TARGET))
                )
                amp_step = float(np.clip(amp_step, 1.0e-4, 10.0))

            # --- Bias Chebyshev block (optional) ---
            if bias_order > 0:
                loss_bias = loglike_model.loglike_bias_given_amp_logit(
                    self.zgrid_mid,
                    current_amp,
                    current_logit,
                    float(self.config.bias_prior_scale),
                )
                proposed_bias = current_bias + np.random.normal(
                    0.0, float(self.config.bias_step), size=bias_order
                )
                bias_proposals += 1
                log_accept_ratio = loss_bias(proposed_bias) - loss_bias(current_bias)
                if log_accept_ratio > np.log(np.random.uniform(low=0.0, high=1.0)):
                    current_bias = proposed_bias
                    bias_accepts += 1
            trace_bias.append(current_bias.copy())

            # --- Logit ESS block ---
            loss_logit = loglike_model.loglike_logit_given_amp(
                self.zgrid_mid, current_amp, bias_coeffs=current_bias
            )
            ess_samples, current_logit = sampler.sample(
                loss_logit,
                initial_state=current_logit,
                n_samples=self.config.ess_n_samples,
                n_burn=self.config.ess_n_burn,
            )
            if ess_samples.size == 0:
                trace_logit.append(current_logit.copy())
            else:
                current_logit = ess_samples[-1]
                trace_logit.append(current_logit.copy())

        amp_accept_rate = amp_accepts / max(amp_proposals, 1)
        if amp_accept_rate < AMP_ACCEPT_LO or amp_accept_rate > AMP_ACCEPT_HI:
            print(
                f"WARNING: LogisticGP amplitude acceptance rate "
                f"{amp_accept_rate:.3f} outside [{AMP_ACCEPT_LO}, {AMP_ACCEPT_HI}]; "
                f"final amp_step={amp_step:.4f}."
            )

        self.diagnostics = {
            "amp_accept_rate": float(amp_accept_rate),
            "amp_step_final": float(amp_step),
            "amp_proposals": int(amp_proposals),
            "amp_accepts": int(amp_accepts),
            "bias_accept_rate": float(bias_accepts / max(bias_proposals, 1))
            if bias_order > 0
            else np.nan,
            "bias_poly_order": bias_order,
            "reference_bin": int(reference_bin),
            "n_steps": n_steps,
            "afterburner": afterburner,
            **{f"cov_{k}": v for k, v in loglike_model.cov_meta.items()},
        }
        return (
            np.array(trace_amp),
            np.array(trace_logit),
            np.array(trace_bias),
        )

    def _compute_convergence_diagnostics(
        self,
        trace_amp: np.ndarray,
        trace_nz: np.ndarray,
        burn: int,
    ) -> None:
        retained_amp = np.asarray(trace_amp[burn:], dtype=float)
        retained_nz = np.asarray(trace_nz[burn:], dtype=float)
        mean_z = np.sum(retained_nz * self.zgrid_mid[None, :], axis=1) / np.sum(
            retained_nz, axis=1
        )
        self.diagnostics.update(
            {
                "amp_rhat": split_half_rhat(retained_amp),
                "amp_ess": effective_sample_size(retained_amp),
                "mean_z_rhat": split_half_rhat(mean_z),
                "mean_z_ess": effective_sample_size(mean_z),
                "n_retained": int(retained_amp.size),
                "amp_mean": float(np.mean(retained_amp)),
                "amp_std": float(np.std(retained_amp)),
                "amp_var": float(np.var(retained_amp)),
            }
        )

    def _compute_ppc(
        self,
        trace_amp: np.ndarray,
        trace_logit: np.ndarray,
        trace_bias: np.ndarray,
        burn: int,
    ) -> None:
        """Posterior predictive check against the clustering data vector.

        For each retained posterior draw, compute the expected ``n_cc`` and a
        chi-squared discrepancy; draw replicated data vectors from the Gaussian
        likelihood and compute their chi-squared.  The PPC p-value is the
        fraction of replicas with chi-squared at least as large as the data.
        """
        if self._loglike_model is None:
            self.ppc = {}
            return

        loglike = self._loglike_model
        retained_amp = np.asarray(trace_amp[burn:], dtype=float)
        retained_logit = np.asarray(trace_logit[burn:], dtype=float)
        retained_bias = np.asarray(trace_bias[burn:], dtype=float)
        n_retained = retained_amp.size
        if n_retained < 1:
            self.ppc = {}
            return

        n_replicas = max(1, int(self.config.ppc_n_replicas))
        # Subsample posterior draws if the chain is much longer than n_replicas.
        if n_retained > n_replicas:
            idx = np.linspace(0, n_retained - 1, n_replicas).astype(int)
        else:
            idx = np.arange(n_retained)

        data_chi2 = np.empty(idx.size, dtype=float)
        expected_stack = np.empty((idx.size, loglike.mean_wx.size), dtype=float)
        for i, j in enumerate(idx):
            bias = retained_bias[j] if retained_bias.ndim == 2 else None
            if bias is not None and bias.size == 0:
                bias = None
            expected = loglike._expected_counts(
                self.zgrid_mid,
                retained_logit[j],
                float(retained_amp[j]),
                bias_coeffs=bias,
            )
            expected_stack[i] = expected
            data_chi2[i] = loglike.chi2(expected)

        # Use the posterior-mean prediction for the replicated bands.
        mean_expected = expected_stack.mean(axis=0)
        replicas = loglike.draw_replicas(mean_expected, n_replicas)
        replica_chi2 = np.array([loglike.chi2(r) for r in replicas], dtype=float)
        # Compare each retained data chi2 against the replica distribution via
        # the mean data chi2 (standard scalar PPC summary).
        data_chi2_mean = float(np.mean(data_chi2))
        p_value = float(np.mean(replica_chi2 >= data_chi2_mean))

        bands = np.percentile(replicas, [2.5, 16, 50, 84, 97.5], axis=0)
        self.ppc = {
            "p_value": p_value,
            "data_chi2_mean": data_chi2_mean,
            "replica_chi2_mean": float(np.mean(replica_chi2)),
            "n_replicas": int(n_replicas),
            "expected_mean": mean_expected,
            "replica_p2p5": bands[0],
            "replica_p16": bands[1],
            "replica_p50": bands[2],
            "replica_p84": bands[3],
            "replica_p97p5": bands[4],
            "zmid_wx": np.asarray(loglike.zmid_wx, dtype=float),
            "signal_wx": np.asarray(loglike.mean_wx, dtype=float),
            "error_wx": np.sqrt(np.clip(np.diag(loglike.cov_wx_raw), 0.0, None)),
        }
        self.diagnostics.update(
            {
                "ppc_p_value": p_value,
                "ppc_data_chi2_mean": data_chi2_mean,
                "ppc_replica_chi2_mean": float(np.mean(replica_chi2)),
            }
        )

    def run(self) -> None:
        """Execute the summarization process."""
        input_data = self.get_data("input")
        self.qp_output = input_data

        self.zgrid_breaks = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins
        )
        self.zgrid_mid = convert_breaks_to_mids(self.zgrid_breaks)
        self.zgrid_widths = compute_bin_widths(self.zgrid_mid)

        self.trace_amp0, self.trace_logit0, self.trace_bias0 = self.sample_joint()

        self.trace_nz0 = np.array(
            [
                logit_to_pdf(vec, self.zgrid_widths, reference_bin=self.reference_bin_used)
                for vec in self.trace_logit0
            ]
        )

        burn = min(self.config.afterburner, len(self.trace_nz0) - 1)
        posterior_samples = self.trace_nz0[burn:]
        self._compute_convergence_diagnostics(self.trace_amp0, self.trace_nz0, burn)
        if bool(self.config.run_ppc):
            self._compute_ppc(
                self.trace_amp0, self.trace_logit0, self.trace_bias0, burn
            )
        else:
            self.ppc = {}

        nzs = qp.Ensemble(
            qp.interp, data=dict(xvals=self.zgrid_mid, yvals=posterior_samples)
        )
        self.add_data("output", nzs)
