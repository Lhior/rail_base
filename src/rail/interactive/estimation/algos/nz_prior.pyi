from typing import Any

def cosmic_variance_stack_informer(**kwargs) -> Any:
    """
    Informer that parameterizes cosmic variance from a spectroscopic training set.

    ---

    The main interface method for Informers

    This will attach the input_data to this `Informer`
    (for introspection and provenance tracking).

    Then it will call the run(), validate() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the model that it creates to this Estimator
    by using `self.add_data('model', model)`.

    Finally, this will return a ModelHandle providing access to the trained model.

    ---

    This function was generated from the function
    rail.estimation.algos.nz_prior.CosmicVarianceStackInformer.inform

    Parameters
    ----------
    training_data : TableLike, required
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    varN_N_filename : str, optional
        var N / N for the training set
        Default: varN_N_data.txt

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """

def cosmic_variance_stack_summarizer(**kwargs) -> Any:
    """
    Summarizer that applies cosmic variance to a photo-z point-estimate stack.

    ---

    Summarize photo-z data using a cosmic-variance model from the informer.

    ---

    This function was generated from the function
    rail.estimation.algos.nz_prior.CosmicVarianceStackSummarizer.summarize

    Parameters
    ----------
    input_data : qp.Ensemble, required
        Per-galaxy p(z), and any ancillary data associated with it
    model : ModelLike, required
        Model from `CosmicVarianceStackInformer` with cosmic variance parameters
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
    ancil_type : str, optional
        Type of point estimate used for histogram
        Default: zmean

    Returns
    -------
    qp.core.ensemble.Ensemble
        Ensemble with n(z), and any ancillary data
    """
