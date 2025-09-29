import os
import numpy as np
import qp

import pytest

from rail.core.stage import RailStage
from rail.core.data import QPHandle, TableHandle
from rail.utils.path_utils import RAILDIR

from rail.estimation.algos.nz_prior import (
    CosmicVarianceStackInformer,
    CosmicVarianceStackSummarizer,
)
from rail.estimation.algos.log_gp import LogisticGPSummarizer


DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def _write_tmp_varn_file(path):
    # Minimal smooth curve for var(N)/N vs z to satisfy spline
    z = np.linspace(0.0, 3.0, 31)
    # Poisson (1.0) plus small clustering term rising with z
    varn_over_n = 1.0 + 0.1 * (z / 3.0) ** 1.0
    data = np.vstack([z, varn_over_n])
    np.savetxt(path, data)


def _load_qp_test_ensemble():
    # Use the same input ensemble used by other summarizer tests
    testdata = os.path.join(RAILDIR, "rail/examples_data/testdata/output_BPZ_lite.hdf5")
    return DS.read_file("test_data", QPHandle, testdata)


def _load_training_table():
    traindata = os.path.join(RAILDIR, "rail/examples_data/testdata/training_100gal.hdf5")
    return DS.read_file("training_data", TableHandle, traindata)


def test_cosmic_variance_stack_summarizer(tmp_path):
    DS.clear()
    # Prepare informer inputs
    training = _load_training_table()
    varn_path = tmp_path / "varN_N_data.txt"
    _write_tmp_varn_file(str(varn_path))

    inform_cfg = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=101,
        hdf5_groupname="photometry",
        varN_N_filename=str(varn_path),
        redshift_col="redshift",
    )
    informer = CosmicVarianceStackInformer.make_stage(**inform_cfg)
    informer.inform(training)

    # Prepare summarizer inputs: existing QP ensemble must include point estimates
    qp_input = _load_qp_test_ensemble()
    summ_cfg = dict(zmin=0.0, zmax=3.0, nzbins=101, ancil_type="mean")
    summarizer = CosmicVarianceStackSummarizer.make_stage(
        model=informer.get_handle("model"), **summ_cfg
    )
    out = summarizer.summarize(qp_input, informer.get_handle("model"))

    # Basic sanity checks
    ens = out.data
    assert isinstance(ens, qp.Ensemble)
    # Expect multiple realizations; ensure xvals length equals nzbins
    assert ens.n_pdfs > 0
    assert ens.xvals().shape[-1] == 101


def test_logistic_gp_summarizer_fast():
    DS.clear()
    # Input QP ensemble (per-galaxy PDFs)
    qp_input = _load_qp_test_ensemble()

    # Construct a simple Gaussian model for cluster redshift likelihood
    # Define grid for model consistent with summarizer z-grid
    zmid_wx = np.linspace(0.0, 3.0, 51)
    # Mean signal resembling a broad n(z) shape
    mean_wx = np.exp(-0.5 * ((zmid_wx - 1.0) / 0.5) ** 2)
    mean_wx /= np.trapz(mean_wx, zmid_wx)
    # Small diagonal covariance to make the distribution well-conditioned
    cov_wx = np.diag(0.05 * np.ones_like(zmid_wx))

    model = dict(zmid_wx=zmid_wx, signal_wx=mean_wx, cov_wx=cov_wx)

    # Keep runtime short
    cfg = dict(zmin=0.0, zmax=3.0, nzbins=101, n_steps=200, afterburner=50)
    summarizer = LogisticGPSummarizer.make_stage(**cfg)
    out = summarizer.summarize(qp_input, model)

    ens = out.data
    assert isinstance(ens, qp.Ensemble)
    assert ens.n_pdfs > 0
    # Output grid should match requested nzbins
    assert ens.xvals().shape[-1] == 101


