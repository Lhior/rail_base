from typing import Any

def logistic_gp_summarizer(**kwargs) -> Any:
    """
    Logistic Gaussian Process summarizer combining photo-z and clustering data.

    ---

    Summarize photo-z ensemble data using a clustering redshift likelihood model.

    ---

    This function was generated from the function
    rail.estimation.algos.log_gp.LogisticGPSummarizer.summarize

    Parameters
    ----------
    input_data : qp.Ensemble, required
        Per-galaxy p(z), and any ancillary data associated with it
    model : ModelLike, required
        Model containing cluster redshift information with keys
        ``zmid_wx``, ``signal_wx``, and ``cov_wx``. Optional keys
        ``prior_mean`` / ``prior_cov`` bypass sample-based prior estimation.
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    n_steps : int, optional
        N-steps for MCMC sampling
        Default: 5000
    afterburner : int, optional
        Remove the samples before chain converge
        Default: 2000
    amp_step : float, optional
        RW proposal scale for log(amp); scale-free under flat-in-log prior
        Default: 0.5
    min_amp : float, optional
        Numerical lower bound for amplitude (not a prior boundary)
        Default: 0.001
    initial_amp : float, optional
        Optional user-specified amplitude start (<=0 uses data-driven start)
        Default: -1.0
    prior_jitter : float, optional
        Diagonal floor added after Ledoit-Wolf shrinkage of the prior covariance
        Default: 1e-06
    progress_interval : int, optional
        Print progress every N iterations (<=0 disables)
        Default: 500
    ess_n_samples : int, optional
        Number of ESS proposals per joint iteration
        Default: 20
    ess_n_burn : int, optional
        Number of burn-in ESS steps per joint iteration
        Default: 10
    adapt_amp_step : bool, optional
        Robbins-Monro adaptation of amp_step during afterburner only
        Default: True
    reference_bin : int, optional
        ALR reference bin (-2=largest median mass, -1=last bin, >=0=explicit index)
        Default: -2
    bias_poly_order : int, optional
        Chebyshev order for unknown-sample bias evolution B(z); 0 disables
        Default: 0
    bias_prior_scale : float, optional
        Gaussian prior stddev on each Chebyshev bias coefficient
        Default: 1.0
    bias_step : float, optional
        RW proposal scale for each Chebyshev bias coefficient
        Default: 0.1
    n_patches : int, optional
        Jackknife patch count for Hartlap correction of cov_wx (<=0 disables)
        Default: 64
    cov_shrinkage : float, optional
        Ledoit-Wolf intensity for cov_wx; <0 estimates intensity, 0 disables
        Default: -1.0
    run_ppc : bool, optional
        Compute posterior predictive checks
        Default: True
    ppc_n_replicas : int, optional
        Number of PPC replicated data vectors
        Default: 200

    Returns
    -------
    qp.core.ensemble.Ensemble
        Ensemble with n(z), and any ancillary data
    """
