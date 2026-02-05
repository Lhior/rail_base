"""
A summarizer that takes the input of a photo-z point estimate qp ensemble, and the cluster redshift likelihood
and run logistic Gaussian process to estimate the posteroir of the redshift distribution n(z) for a sample. 

Author: Markus Michael Rau, Tianqing Zhang
"""


import numpy as np
import qp
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.core.data import QPHandle, ModelHandle

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import multivariate_normal
from typing import Callable, Optional, Tuple
from numpy.linalg import LinAlgError


PDF_FLOOR = 1.0e-12


def compute_bin_widths(mids: np.ndarray) -> np.ndarray:
    mids = np.asarray(mids, dtype=float)
    if mids.size <= 1:
        return np.ones_like(mids)
    edges = np.empty(mids.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (mids[:-1] + mids[1:])
    edges[0] = mids[0] - 0.5 * (mids[1] - mids[0])
    edges[-1] = mids[-1] + 0.5 * (mids[-1] - mids[-2])
    return np.diff(edges)


def pdf_to_logit(pdf: np.ndarray, widths: Optional[np.ndarray] = None) -> np.ndarray:
    """Project a density defined on bins to simplex coordinates (Eq. 65/69 in Rau et al. 2021)."""
    pdf = np.asarray(pdf, dtype=float)
    if widths is None:
        widths = np.ones_like(pdf)
    widths = np.asarray(widths, dtype=float)
    mass = np.clip(pdf, PDF_FLOOR, None) * widths
    total = np.sum(mass)
    if not np.isfinite(total) or total <= 0.0:
        total = PDF_FLOOR * mass.size
    simplex = mass / total
    baseline = simplex[-1]
    return np.log(simplex[:-1]) - np.log(baseline)


def logit_to_pdf(logit: np.ndarray, widths: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse logistic transform returning a density whose bin-integrals follow the simplex."""
    logit = np.asarray(logit, dtype=float)
    if widths is None:
        widths = np.ones(logit.size + 1, dtype=float)
    else:
        widths = np.asarray(widths, dtype=float)
    shift = logit - np.max(logit)
    exp_terms = np.exp(shift)
    denom = 1.0 + np.sum(exp_terms)
    simplex = np.empty(logit.size + 1, dtype=float)
    simplex[:-1] = exp_terms / denom
    simplex[-1] = 1.0 / denom
    clipped_widths = np.clip(widths, PDF_FLOOR, None)
    density = simplex / clipped_widths
    norm = np.sum(density * clipped_widths)
    if not np.isfinite(norm) or norm <= 0.0:
        norm = PDF_FLOOR * density.size
    density /= norm
    return density




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
    
    def __init__(self, zmid_wx, mean_wx, cov_wx): 
        """
        Initialize the LogLike object.

        Parameters:
        zmid_wx (array-like): Midpoints of redshift bins where cluster redshift is defined.
        mean_wx (array-like): Mean vector for the cluster redshift distribution.
        cov_wx (array-like): Covariance matrix for the cluster redshift distribution.
        """
        self.zmid_wx = np.asarray(zmid_wx, dtype=float)
        self.mean_wx = np.asarray(mean_wx, dtype=float)
        self.cov_wx = np.asarray(cov_wx, dtype=float)

        if len(self.zmid_wx) > 1:
            midpoints = self.zmid_wx
            edges = np.empty(midpoints.size + 1, dtype=float)
            edges[1:-1] = 0.5 * (midpoints[:-1] + midpoints[1:])
            delta_first = midpoints[1] - midpoints[0]
            delta_last = midpoints[-1] - midpoints[-2]
            edges[0] = midpoints[0] - 0.5 * delta_first
            edges[-1] = midpoints[-1] + 0.5 * delta_last
            self.bin_widths = np.diff(edges)
        else:
            self.bin_widths = np.ones_like(self.mean_wx)
        
    def _project_profile(self, zmid: np.ndarray, profile: np.ndarray) -> np.ndarray:
        spline = InterpolatedUnivariateSpline(zmid, profile, k=1, ext=1)
        return spline(self.zmid_wx)

    def loglike_logit_given_amp(self, zmid: np.ndarray, amp: float) -> Callable[[np.ndarray], float]:
        widths = compute_bin_widths(zmid)
        def loss(logit_vec: np.ndarray) -> float:
            probs = logit_to_pdf(logit_vec, widths)
            prof_wx = self._project_profile(zmid, probs)
            expected_counts = prof_wx * self.bin_widths * amp
            return multivariate_normal.logpdf(expected_counts, self.mean_wx, self.cov_wx)
        return loss

    def loglike_amp_given_logit(self, zmid: np.ndarray, logit_vec: np.ndarray) -> Callable[[float], float]:
        widths = compute_bin_widths(zmid)
        prof_wx = self._project_profile(zmid, logit_to_pdf(logit_vec, widths))
        prof_counts = prof_wx * self.bin_widths

        def loss(amp: float) -> float:
            if amp <= 0.0:
                return -np.inf
            return multivariate_normal.logpdf(prof_counts * amp, self.mean_wx, self.cov_wx)

        return loss

def convert_mids_to_breaks(mids): 
    """
    Convert midpoints of a grid to the breaks of the grid
    Assume the grid points have equal distances
    The output will have size of len(mids)+1
    """
    breaks = np.copy(mids)
    delta_z = breaks[1]-breaks[0]
    breaks = breaks - delta_z/2.
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
    delta = 0.5*(breaks[1] - breaks[0])
    mids = mids + delta
    return mids


class LogisticGPSummarizer(PZSummarizer):
    """
    Logistic Gaussian Summarizer
    
    Implements the composite likelihood methodology of Rau et al. (2021, 2022) by
    combining photometric redshift ensemble information (approximated as a
    logit-normal prior on the sample redshift distribution) with clustering
    cross-correlation data modelled as Gaussian-distributed counts. Inference is
    performed via joint sampling of the amplitude parameter and the latent
    logit field using elliptical slice sampling.
    """

    name = "LogisticGPSummarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        n_steps=Param(int, 5000, msg="N-steps for MCMC sampling"),
        afterburner=Param(int, 2000, msg='Remove the samples before chain converge'),
        amp_step=Param(float, 0.5, msg="RW proposal scale for the amplitude parameter"),
        min_amp=Param(float, 1.0e-3, msg="Lower bound/initialisation for the amplitude parameter"),
        initial_amp=Param(float, -1.0, msg="Optional user-specified amplitude start (<=0 uses data-driven start)"),
        prior_jitter=Param(float, 1.0e-6, msg="Diagonal jitter added to the photometric prior covariance"),
        progress_interval=Param(int, 500, msg="Print progress every N iterations (<=0 disables)"),
        ess_n_samples=Param(int, 20, msg="Number of ESS proposals per joint iteration"),
        ess_n_burn=Param(int, 10, msg="Number of burn-in ESS steps per joint iteration"),
        )
    inputs = [("input", QPHandle), ("model", ModelHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    
    def summarize(self, input_data, model):
        """
        Summarize the input data using the model.

        Parameters:
        input_data: Input pz distributions from photo-z methods
        model: Model containing cluster redshift information.

        Returns:
        QPHandle: Handle to the output data.
        """
        # read the model
        self.set_data("model", model)
        model = self.get_data('model')
        
        self.zmid_wx = model["zmid_wx"]
        self.signal_wx = model["signal_wx"]
        self.cov_wx = model["cov_wx"]
        # set the photometric data
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")
    
    def sample_joint(self): 
        """
        Perform joint sampling of amplitude and s_vec using MCMC.

        Returns:
        tuple: Arrays of sampled amplitudes and s_vecs.
        """
        loglike_model = LogLike(self.zmid_wx, self.signal_wx, self.cov_wx)

        pdf_values = np.atleast_2d(self.qp_output.pdf(self.zgrid_mid))
        pdf_values = np.clip(pdf_values, PDF_FLOOR, None)

        normalisation = np.array([np.sum(p * self.zgrid_widths) for p in pdf_values])
        pdf_normed = pdf_values / normalisation[:, None]

        logit_samples = np.array([pdf_to_logit(p, self.zgrid_widths) for p in pdf_normed])
        prior_mean = logit_samples.mean(axis=0)
        if logit_samples.shape[0] < 2:
            prior_cov = np.eye(prior_mean.size)
        else:
            prior_cov = np.cov(logit_samples, rowvar=False)
        prior_cov += np.eye(prior_mean.size) * self.config.prior_jitter

        sampler = EllipticalSliceSampler(prior_mean, prior_cov)

        if self.config.initial_amp > 0:
            initial_amp = self.config.initial_amp
        else:
            initial_amp = np.sum(self.signal_wx)
        initial_amp = max(initial_amp, self.config.min_amp)
        trace_amp = [initial_amp]
        trace_logit = [prior_mean.copy()]

        current_amp = initial_amp
        current_logit = prior_mean.copy()

        for step in range(self.config.n_steps): 
            if self.config.progress_interval > 0 and step % self.config.progress_interval == 0:
                print(f"LogisticGP sampler step {step}")

            loss_amp = loglike_model.loglike_amp_given_logit(self.zgrid_mid, current_logit)
            proposed_amp = np.random.normal(current_amp, self.config.amp_step)
            if proposed_amp > self.config.min_amp:
                log_accept_ratio = loss_amp(proposed_amp) - loss_amp(current_amp)
                if log_accept_ratio > np.log(np.random.uniform(low=0.0, high=1.0)):
                    current_amp = proposed_amp
            trace_amp.append(current_amp)

            loss_logit = loglike_model.loglike_logit_given_amp(self.zgrid_mid, current_amp)
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

        return np.array(trace_amp), np.array(trace_logit)

    def run(self):
        """
        Execute the summarization process.
        """
        input_data = self.get_data('input')
        self.qp_output = input_data
        
        self.zgrid_breaks = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        self.zgrid_mid = convert_breaks_to_mids(self.zgrid_breaks)
        self.zgrid_widths = compute_bin_widths(self.zgrid_mid)
        
        self.trace_amp0, self.trace_logit0 = self.sample_joint()

        self.trace_nz0 = np.array([logit_to_pdf(vec, self.zgrid_widths) for vec in self.trace_logit0])

        burn = min(self.config.afterburner, len(self.trace_nz0) - 1)
        posterior_samples = self.trace_nz0[burn:]

        nzs = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid_mid, yvals=posterior_samples))

        self.add_data('output', nzs)
        