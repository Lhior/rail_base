import os
from typing import Any

import numpy as np
import qp

from rail.core.data import QPHandle, TableHandle
from rail.estimation.algos.log_gp import LogisticGPSummarizer
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
    test_data = _load_qp_test_ensemble()

    zmid_wx = np.linspace(0.0, 3.0, 51)
    mean_wx = np.exp(-0.5 * ((zmid_wx - 1.0) / 0.5) ** 2)
    mean_wx /= np.trapezoid(mean_wx, zmid_wx)
    cov_wx = np.diag(0.05 * np.ones_like(zmid_wx))
    model = dict(zmid_wx=zmid_wx, signal_wx=mean_wx, cov_wx=cov_wx)

    cfg: dict[str, Any] = dict(zmin=0.0, zmax=3.0, nzbins=101, n_steps=200, afterburner=50)
    summarizer = LogisticGPSummarizer.make_stage(name="LogisticGP", **cfg)
    out = summarizer.summarize(test_data, model)

    ens = out.data
    assert isinstance(ens, qp.Ensemble)
    assert ens.npdf > 0

    _safe_remove_tag(summarizer, "output")
    _safe_remove_tag(summarizer, "single_NZ")
