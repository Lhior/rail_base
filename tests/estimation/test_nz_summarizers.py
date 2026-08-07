import os
from typing import Any

import numpy as np
import pytest
import qp

from rail.core.data import QPHandle, TableHandle
from rail.estimation.algos.log_gp import (
    LogisticGPSummarizer,
    bin_average_profile,
    chebyshev_bias_factor,
    choose_reference_bin,
    compute_bin_widths,
    hartlap_factor,
    logit_to_pdf,
    pdf_to_logit,
    stabilize_covariance,
)
from rail.estimation.algos.nz_prior import (
    CosmicVarianceStackInformer,
    CosmicVarianceStackSummarizer,
)
from rail.utils.path_utils import RAILDIR


def _write_tmp_varn_file(path: str) -> None:
    """Write a smooth var(N)/N vs z curve for tests."""
    z = np.linspace(0.0, 3.0, 31)
    varn_over_n = 1.0 + 0.1 * (z / 3.0)
    data = np.vstack([z, varn_over_n])
    np.savetxt(path, data)


def _load_qp_test_ensemble() -> QPHandle:
    testdata = os.path.join(RAILDIR, "rail/examples_data/testdata/output_BPZ_lite.hdf5")
    return QPHandle("test_data", path=testdata)


def _load_training_table() -> TableHandle:
    traindata = os.path.join(RAILDIR, "rail/examples_data/testdata/training_100gal.hdf5")
    return TableHandle("training_data", path=traindata)


def _safe_remove(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def _safe_remove_tag(stage, tag: str) -> None:
    """Remove a stage output by tag if it exists for this stage."""
    try:
        aliased = stage.get_aliased_tag(tag)
        path = stage.get_output(aliased, final_name=True)
    except KeyError:
        return
    _safe_remove(path)


def _gaussian_nz(z: np.ndarray, mu: float = 1.0, sigma: float = 0.35) -> np.ndarray:
    pdf = np.exp(-0.5 * ((z - mu) / sigma) ** 2)
    pdf = np.clip(pdf, 1e-12, None)
    pdf /= np.trapezoid(pdf, z)
    return pdf


def _synthetic_prior_ensemble(
    z: np.ndarray,
    n_members: int,
    *,
    mu: float = 1.0,
    sigma: float = 0.35,
    coeff: float = 0.15,
    seed: int = 0,
) -> qp.Ensemble:
    """Lognormal CV-like draws around a Gaussian n(z)."""
    rng = np.random.default_rng(seed)
    mean = _gaussian_nz(z, mu=mu, sigma=sigma)
    sigma_abs = np.maximum(coeff * mean, 1e-12)
    # Match CvSampleRealizations: lognormal with E[draw] ≈ mean.
    sigma2 = np.log1p((sigma_abs / mean) ** 2)
    loc = np.log(mean) - 0.5 * sigma2
    draws = rng.normal(loc=loc, scale=np.sqrt(sigma2), size=(n_members, z.size))
    pdfs = np.exp(draws)
    for i in range(n_members):
        pdfs[i] /= np.trapezoid(pdfs[i], z)
    return qp.Ensemble(qp.interp, data=dict(xvals=z, yvals=pdfs))


def _wx_model_from_truth(
    z: np.ndarray,
    truth: np.ndarray,
    amp: float,
    *,
    err_frac: float = 0.05,
    bias_coeffs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    widths = compute_bin_widths(z)
    signal = truth * widths * amp
    if bias_coeffs is not None and np.size(bias_coeffs) > 0:
        signal = signal * chebyshev_bias_factor(
            z, bias_coeffs, float(z.min()), float(z.max())
        )
    err = np.maximum(err_frac * np.abs(signal), 1e-3)
    return {
        "zmid_wx": z.copy(),
        "signal_wx": signal,
        "cov_wx": np.diag(err**2),
    }


def test_cosmic_variance_stack_summarizer(tmp_path: Any) -> None:
    """End-to-end test for CosmicVarianceStack summarizer with cleanup like other tests."""
    training = _load_training_table()
    varn_path = tmp_path / "varN_N_data.txt"
    _write_tmp_varn_file(str(varn_path))

    inform_cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=101,
        hdf5_groupname="photometry",
        varN_N_filename=str(varn_path),
        redshift_col="redshift",
    )
    informer = CosmicVarianceStackInformer.make_stage(**inform_cfg)
    model_handle = informer.inform(training)

    test_data = _load_qp_test_ensemble()
    summ_cfg: dict[str, Any] = dict(zmin=0.0, zmax=3.0, nzbins=101, ancil_type="zmode")
    summarizer = CosmicVarianceStackSummarizer.make_stage(
        name="CosmicVarianceNZ",
        model=model_handle,
        **summ_cfg,
    )
    out = summarizer.summarize(test_data, model_handle)

    ens = out.data
    assert isinstance(ens, qp.Ensemble)
    assert ens.npdf > 0

    _safe_remove_tag(summarizer, "output")
    _safe_remove_tag(summarizer, "single_NZ")
    _safe_remove(informer.get_output(informer.get_aliased_tag("model"), final_name=True))


def test_logistic_gp_summarizer_fast() -> None:
    """Fast smoke test for the Logistic GP summarizer mirroring cleanup style."""
    z = np.linspace(0.05, 2.95, 30)
    prior = _synthetic_prior_ensemble(z, n_members=80, seed=1)
    truth = _gaussian_nz(z)
    model = _wx_model_from_truth(z, truth, amp=5.0)

    # nzbins = number of edges = n_mids + 1
    cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=31,
        n_steps=300,
        afterburner=50,
        amp_step=0.5,
        adapt_amp_step=True,
        progress_interval=0,
    )
    summarizer = LogisticGPSummarizer.make_stage(name="LogisticGP", **cfg)
    out = summarizer.summarize(prior, model)

    ens = out.data
    assert isinstance(ens, qp.Ensemble)
    assert ens.npdf > 0
    assert summarizer.diagnostics["amp_accept_rate"] >= 0.0

    _safe_remove_tag(summarizer, "output")
    _safe_remove_tag(summarizer, "single_NZ")


def test_pdf_logit_roundtrip_with_reference_bin() -> None:
    z = np.linspace(0.0, 3.0, 21)
    widths = compute_bin_widths(z)
    pdf = _gaussian_nz(z)
    # Normalize consistently with the bin-mass convention used by the transforms.
    pdf = pdf / np.sum(pdf * widths)
    for ref in (0, 5, 10, -1):
        logit = pdf_to_logit(pdf, widths, reference_bin=ref if ref >= 0 else -1)
        recovered = logit_to_pdf(logit, widths, reference_bin=ref if ref >= 0 else -1)
        assert np.allclose(recovered, pdf, rtol=1e-10, atol=1e-12)


def test_choose_reference_bin_picks_peak() -> None:
    z = np.linspace(0.0, 3.0, 31)
    widths = compute_bin_widths(z)
    pdfs = np.vstack([_gaussian_nz(z, mu=1.2) for _ in range(20)])
    ref = choose_reference_bin(pdfs, widths)
    peak = int(np.argmax(_gaussian_nz(z, mu=1.2) * widths))
    assert abs(ref - peak) <= 1


def test_logistic_gp_rank_guard() -> None:
    """Rank-deficiency guard fires when members <= logit dimension."""
    z = np.linspace(0.05, 2.95, 30)
    # logit dim = 29; only 10 members → must raise
    prior = _synthetic_prior_ensemble(z, n_members=10, seed=2)
    truth = _gaussian_nz(z)
    model = _wx_model_from_truth(z, truth, amp=3.0)

    cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=31,
        n_steps=20,
        afterburner=5,
        progress_interval=0,
    )
    summarizer = LogisticGPSummarizer.make_stage(name="LogisticGPRank", **cfg)
    with pytest.raises(ValueError, match="rank-deficient"):
        summarizer.summarize(prior, model)
    _safe_remove_tag(summarizer, "output")


def test_logistic_gp_amplitude_marginalization() -> None:
    """Amplitude chain must move (regression for frozen absolute amp_step)."""
    z = np.linspace(0.05, 2.95, 25)
    prior = _synthetic_prior_ensemble(z, n_members=80, seed=3)
    truth = _gaussian_nz(z)
    true_amp = 8.0
    model = _wx_model_from_truth(z, truth, amp=true_amp, err_frac=0.08)

    cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=26,
        n_steps=800,
        afterburner=200,
        amp_step=0.5,
        adapt_amp_step=True,
        progress_interval=0,
        bias_poly_order=0,
    )
    summarizer = LogisticGPSummarizer.make_stage(name="LogisticGPAmp", **cfg)
    out = summarizer.summarize(prior, model)
    assert out.data.npdf > 0

    diag = summarizer.diagnostics
    assert diag["amp_var"] > 0.0
    assert diag["amp_accept_rate"] > 0.05
    # Recovered amplitude should be in the ballpark of the injected value.
    assert 0.2 * true_amp < diag["amp_mean"] < 5.0 * true_amp

    _safe_remove_tag(summarizer, "output")


def test_logistic_gp_coverage_synthetic() -> None:
    """Posterior credible interval should cover the injected truth n(z)."""
    z = np.linspace(0.05, 2.95, 25)
    prior = _synthetic_prior_ensemble(z, n_members=100, coeff=0.2, seed=4)
    truth = _gaussian_nz(z)
    true_amp = 6.0
    model = _wx_model_from_truth(z, truth, amp=true_amp, err_frac=0.1)

    cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=26,
        n_steps=1000,
        afterburner=250,
        amp_step=0.5,
        adapt_amp_step=True,
        progress_interval=0,
    )
    summarizer = LogisticGPSummarizer.make_stage(name="LogisticGPCov", **cfg)
    out = summarizer.summarize(prior, model)
    samples = np.asarray(out.data.pdf(z), dtype=float)
    lo = np.percentile(samples, 16, axis=0)
    hi = np.percentile(samples, 84, axis=0)
    # Fraction of z-bins where truth lies in the 68% interval.
    covered = np.mean((truth >= lo) & (truth <= hi))
    assert covered > 0.4

    _safe_remove_tag(summarizer, "output")


def test_logistic_gp_bias_poly_order_zero_reproducible() -> None:
    """bias_poly_order=0 with fixed seed is reproducible."""
    z = np.linspace(0.05, 2.95, 20)
    prior = _synthetic_prior_ensemble(z, n_members=60, seed=5)
    truth = _gaussian_nz(z)
    model = _wx_model_from_truth(z, truth, amp=4.0)

    cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=21,
        n_steps=120,
        afterburner=20,
        amp_step=0.5,
        adapt_amp_step=False,
        progress_interval=0,
        bias_poly_order=0,
    )

    np.random.seed(12345)
    s1 = LogisticGPSummarizer.make_stage(name="LogisticGPRepA", **cfg)
    out1 = s1.summarize(prior, model)
    samples1 = np.asarray(out1.data.pdf(z), dtype=float)

    np.random.seed(12345)
    s2 = LogisticGPSummarizer.make_stage(name="LogisticGPRepB", **cfg)
    out2 = s2.summarize(prior, model)
    samples2 = np.asarray(out2.data.pdf(z), dtype=float)

    assert np.allclose(samples1, samples2)
    assert s1.trace_bias0 is not None
    assert s1.trace_bias0.shape[1] == 0

    _safe_remove_tag(s1, "output")
    _safe_remove_tag(s2, "output")


def test_logistic_gp_bias_recovery() -> None:
    """With bias_poly_order=1, a linear tilt in signal_wx is absorbed by B(z)."""
    z = np.linspace(0.05, 2.95, 25)
    prior = _synthetic_prior_ensemble(z, n_members=100, coeff=0.1, seed=6)
    truth = _gaussian_nz(z)
    true_amp = 5.0
    true_c1 = 0.4
    model_tilted = _wx_model_from_truth(
        z, truth, amp=true_amp, err_frac=0.06, bias_coeffs=np.array([true_c1])
    )

    cfg_free: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=26,
        n_steps=1200,
        afterburner=300,
        amp_step=0.5,
        adapt_amp_step=True,
        progress_interval=0,
        bias_poly_order=1,
        bias_step=0.15,
        bias_prior_scale=1.0,
    )
    s_free = LogisticGPSummarizer.make_stage(name="LogisticGPBiasFree", **cfg_free)
    out_free = s_free.summarize(prior, model_tilted)
    samples_free = np.asarray(out_free.data.pdf(z), dtype=float)
    mean_free = samples_free.mean(axis=0)
    burn = min(cfg_free["afterburner"], len(s_free.trace_bias0) - 1)
    c1_mean = float(np.mean(s_free.trace_bias0[burn:, 0]))

    cfg_fixed: dict[str, Any] = dict(cfg_free)
    cfg_fixed["bias_poly_order"] = 0
    s_fixed = LogisticGPSummarizer.make_stage(name="LogisticGPBiasFixed", **cfg_fixed)
    out_fixed = s_fixed.summarize(prior, model_tilted)
    samples_fixed = np.asarray(out_fixed.data.pdf(z), dtype=float)
    mean_fixed = samples_fixed.mean(axis=0)

    # Free bias should recover a non-zero tilt in the same direction.
    assert c1_mean * true_c1 > 0.0
    assert abs(c1_mean - true_c1) < 0.5

    # n(z) bias (L2 vs truth) should be smaller when B(z) is marginalized.
    err_free = np.sqrt(np.mean((mean_free - truth) ** 2))
    err_fixed = np.sqrt(np.mean((mean_fixed - truth) ** 2))
    assert err_free < err_fixed

    _safe_remove_tag(s_free, "output")
    _safe_remove_tag(s_fixed, "output")


def test_logistic_gp_prior_passthrough() -> None:
    """model prior_mean / prior_cov bypass sample-based estimation."""
    z = np.linspace(0.05, 2.95, 20)
    # Intentionally too few members for sample-based cov; passthrough should work.
    prior = _synthetic_prior_ensemble(z, n_members=5, seed=7)
    truth = _gaussian_nz(z)
    widths = compute_bin_widths(z)
    logit = pdf_to_logit(truth, widths, reference_bin=-1)
    model = _wx_model_from_truth(z, truth, amp=3.0)
    model["prior_mean"] = logit
    model["prior_cov"] = np.eye(logit.size) * 0.5

    cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=21,
        n_steps=100,
        afterburner=20,
        amp_step=0.5,
        adapt_amp_step=False,
        progress_interval=0,
        reference_bin=-1,
    )
    summarizer = LogisticGPSummarizer.make_stage(name="LogisticGPPriorPass", **cfg)
    out = summarizer.summarize(prior, model)
    assert out.data.npdf > 0
    _safe_remove_tag(summarizer, "output")


def test_bin_average_profile_constant_and_linear():
    """Constant profiles are exact; linear profiles match the midpoint value."""
    z = np.linspace(0.1, 1.0, 10)
    const = np.full_like(z, 2.5)
    assert np.allclose(bin_average_profile(z, const, z), const)

    linear = 1.0 + 2.0 * z
    avg = bin_average_profile(z, linear, z)
    # For a linear interpolant the integral average over a bin equals the value
    # at the bin midpoint.
    assert np.allclose(avg, linear, rtol=1e-10, atol=1e-12)


def test_bin_average_profile_out_of_range_is_zero():
    z = np.linspace(0.5, 1.0, 6)
    profile = np.ones_like(z)
    target = np.array([0.1, 0.75, 1.5])
    avg = bin_average_profile(z, profile, target)
    assert avg[0] == 0.0
    assert avg[2] == 0.0
    assert avg[1] > 0.0


def test_stabilize_covariance_hartlap_and_shrinkage():
    n = 5
    cov = np.eye(n) * 0.04
    cov[0, 1] = cov[1, 0] = 0.01
    stabilized, meta = stabilize_covariance(cov, n_patches=64, shrinkage=0.5)
    assert meta["hartlap_factor"] == pytest.approx(hartlap_factor(64, n))
    assert meta["shrinkage_intensity"] == pytest.approx(0.5)
    # Hartlap scales cov by 1/h → diagonal grows when h < 1.
    assert stabilized[0, 0] > cov[0, 0]


def test_logistic_gp_ppc_self_consistent() -> None:
    """Noisy self-consistent synthetic data yields a well-defined PPC p-value."""
    z = np.linspace(0.05, 2.95, 20)
    prior = _synthetic_prior_ensemble(z, n_members=80, seed=8)
    truth = _gaussian_nz(z)
    model = _wx_model_from_truth(z, truth, amp=5.0, err_frac=0.1)
    # Realise a noisy data vector so the PPC is not trivially p=1.
    rng = np.random.default_rng(42)
    model["signal_wx"] = rng.multivariate_normal(
        mean=model["signal_wx"], cov=model["cov_wx"]
    )

    cfg: dict[str, Any] = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=21,
        n_steps=400,
        afterburner=100,
        amp_step=0.5,
        adapt_amp_step=True,
        progress_interval=0,
        run_ppc=True,
        ppc_n_replicas=100,
        n_patches=64,
    )
    summarizer = LogisticGPSummarizer.make_stage(name="LogisticGPPPC", **cfg)
    out = summarizer.summarize(prior, model)
    assert out.data.npdf > 0
    assert "p_value" in summarizer.ppc
    p = float(summarizer.ppc["p_value"])
    assert 0.0 <= p <= 1.0
    assert "replica_p16" in summarizer.ppc
    assert summarizer.ppc["replica_p16"].shape == model["signal_wx"].shape
    assert "ppc_p_value" in summarizer.diagnostics
    _safe_remove_tag(summarizer, "output")
