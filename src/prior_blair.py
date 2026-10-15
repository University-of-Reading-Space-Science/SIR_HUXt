"""
prior_blair.py

Correlated Gaussian prior for the CME parameters with the covariance estimated from Blair's CME list
(a CSV file with columns including "V" (speed), "Ang_rad" (half-width), "lon" and "lat").

Rows / columns follow the cme_par_array order (t_init, v, width, lon, lat, thick). The list has no
launch-time or thickness information, so those get zero covariance apart from the variances
sd_t_init^2 and sd_thick^2.
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import Generator

from cme_par_dict_structure import cme_par_get_indices, required_dict_keys, cme_par_key_to_index
from prior_cov_plots import plot_sample_cov


def make_cov_blair(
        blair_file_path: str,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        use_log_v: bool = False,
        plot_cme_cov: bool=False,
)-> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
        Generate the covariance matrix for the initial parameters using Blair's CME list
        :param blair_file_path: File path to CME list to estimate covariance from
        :param vars_req: Parameters to perturb in the ensemble (others get zero (co)variance)
            Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
        :param sd_t_init: Standard deviation of CME launch time
        :param sd_v: Standard deviation of CME velocity
        :param sd_width: Standard deviation of CME width
        :param sd_lon: Standard deviation of CME longitude
        :param sd_lat: Standard deviation of CME latitude
        :param sd_thick: Standard deviation of CME thickness
        :param use_log_v: Boolean to determine whether to use log CME speeds
        :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
        :return: cov_out: Covariance matrix of CME parameters
        :return: corr_out: Correlation matrix of CME parameters
    """
    assert all(v in required_dict_keys() for v in vars_req)

    ######################################################
    # Open and read in CME list
    ######################################################
    dfCMEpar = pd.read_csv(blair_file_path)

    # Rename column headers to be consistent with SIR-HUXt and then delete columns that aren't in vars_req
    dfCMEpar.rename({"V": "v", "Ang_rad": "width"}, axis=1, inplace=True)
    dfCMEpar.drop(
        columns=[col for col in dfCMEpar if col not in required_dict_keys()], inplace=True
    )
    cme_par_keys_ordered = cme_par_key_to_index().keys()
    list_req_order = [
        r for r in cme_par_keys_ordered if r in dfCMEpar.columns
    ]
    dfCMEpar = dfCMEpar[list_req_order]

    #####################################################
    # Calculate covariance matrix from dfCMEpar
    #####################################################
    blair_cov = dfCMEpar.cov(numeric_only=True)

    # Add in t_init and thick rows
    blair_cov = blair_cov.reindex(cme_par_keys_ordered, fill_value=0.0)
    blair_cov.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    blair_cov.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
    blair_cov.loc["t_init", "t_init"] = sd_t_init * sd_t_init
    blair_cov.loc["thick", "thick"] = sd_thick * sd_thick

    ####################################################
    # Calculate correlation matrix from dfCMEpar
    ####################################################
    blair_corr = dfCMEpar.corr(numeric_only=True)

    # Add in t_init and thick rows
    blair_corr = blair_corr.reindex(cme_par_keys_ordered, fill_value=0.0)
    blair_corr.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    blair_corr.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
    blair_corr.loc["t_init", "t_init"] = 1.0
    blair_corr.loc["thick", "thick"] = 1.0

    ####################################################
    # Transform to an array for output
    ####################################################
    cov_out = blair_cov.values
    corr_out = blair_corr.values
    for m, par_key in enumerate(cme_par_keys_ordered):
        # Remove any covariance from variables that do not need to be perturbed
        if par_key not in vars_req:
            cov_out[m, :] = 0
            cov_out[:, m] = 0

            corr_out[m, :] = 0
            corr_out[:, m] = 0

    ################################################
    # Plot CME covariance and correlation matrix
    ################################################
    if plot_cme_cov:
        plot_sample_cov(
            cov_out,
            corr_out,
            "Blair's list",
            use_log_v=use_log_v,
        )

    print(f"blair_cov={cov_out}")
    print(f"blair_corr={corr_out}")

    return cov_out, corr_out


def make_blair_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        blair_file_path: str,
        rng: Generator,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        use_log_v: bool = False,
        plot_cme_cov: bool=False,
) -> npt.NDArray[float]:
    """
    Draw Gaussian samples around mean_cme_pars with the covariance from Blair's CME list
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters to perturb around
    :param blair_file_path: Path to Blair's CME list file
    :param rng: Random number generator
    :param vars_req: Parameters to perturb in the ensemble
        Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Boolean to determine whether to use log CME speeds
    :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
    :return: blair_samples: (6, n_ens) array of ensemble members
        (NB: use_log_v only affects the plot labels here; speeds are sampled linearly)
    """
    blair_samples: npt.NDArray[float] = np.zeros((6, n_ens))

    blair_tuple: tuple[npt.NDArray[float], npt.NDArray[float]] = make_cov_blair(
        blair_file_path=blair_file_path,
        vars_req=vars_req,
        sd_t_init=sd_t_init,
        sd_v=sd_v,
        sd_width=sd_width,
        sd_lon=sd_lon,
        sd_lat=sd_lat,
        sd_thick=sd_thick,
        use_log_v = use_log_v,
        plot_cme_cov=plot_cme_cov,
    )
    blair_cov: npt.NDArray[float] = blair_tuple[0]
    blair_corr: npt.NDArray[float] = blair_tuple[1]

    for m in range(n_ens):
        blair_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=blair_cov
        )

    print(blair_samples)
    return blair_samples
