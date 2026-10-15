"""
prior_uncorrelated.py

Uncorrelated priors for the CME parameters:
    uncorrelated_cov_mat         : diagonal 6x6 covariance matrix from per-parameter standard deviations
    make_uncorrelated_samples    : Gaussian samples with that covariance
    make_uncorr_uniform_samples  : independent *uniform* samples (used for cme_cov_type="uncorr")

All samplers return a (6, n_ens) array whose rows follow the cme_par_array column order
(t_init, v, width, lon, lat, thick); see cme_par_dict_structure.py.
"""
import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from cme_par_dict_structure import cme_par_get_indices, cme_par_key_to_index
from prior_cov_plots import plot_sample_cov


def uncorrelated_cov_mat(
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        plot_cme_cov: bool=False,
) -> npt.NDArray[float]:
    """
    Generate an uncorrelated (diagonal) covariance matrix, diag(sd^2), in cme_par_array order
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param plot_cme_cov: Plot the covariance matrix
    :return: uncorr_cov_mat: Uncorrelated covariance matrix
    """

    # Generate the covariance matrix, checking that the
    # required standard deviations have been provided
    uncorr_cov_mat = np.zeros((6, 6))
    sd_vals = [sd_t_init, sd_v, sd_width, sd_lon, sd_lat, sd_thick]

    cme_par_keys_ordered = cme_par_key_to_index().keys()

    for iv, var_name in enumerate(cme_par_keys_ordered):
        ind_req = cme_par_get_indices(var_name)
        uncorr_cov_mat[ind_req, ind_req] = sd_vals[ind_req] * sd_vals[ind_req]

    # (previously always plotted, which blocks on plt.show() during DA runs)
    if plot_cme_cov:
        plot_sample_cov(
            uncorr_cov_mat,
            np.identity(6),
            "Uncorrelated covariance matrix",
        )
    return uncorr_cov_mat


def make_uncorrelated_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng: Generator,
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        use_log_v: bool = False,
) -> npt.NDArray[float]:
    """
    Draw Gaussian samples around mean_cme_pars with an uncorrelated (diagonal) covariance matrix
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters (cme_par_array order)
    :param rng: Random number generator
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity (of log(v) if use_log_v)
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Sample log(speed) instead of speed (samples are returned as speeds)
    :return: uncorr_samples: (6, n_ens) array of ensemble members
    """
    uncorr_samples: npt.NDArray[float] = np.zeros((6, n_ens))

    uncorr_cov: npt.NDArray[float] = uncorrelated_cov_mat(
        sd_t_init=sd_t_init,
        sd_v=sd_v,
        sd_width=sd_width,
        sd_lon=sd_lon,
        sd_lat=sd_lat,
        sd_thick=sd_thick
    )
    print(f"uncorr_cov = {uncorr_cov}")

    # Copy so the caller's list isn't modified in place
    mean_cme_pars = list(mean_cme_pars)
    if use_log_v:
        mean_cme_pars[1] = np.log(mean_cme_pars[1])

    for m in range(n_ens):
        uncorr_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=uncorr_cov
        )
    if use_log_v:
        uncorr_samples[1, :] = np.exp(uncorr_samples[1, :])
    return uncorr_samples


def make_uncorr_uniform_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng: Generator,
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0
) -> npt.NDArray[float]:
    """
    Draw independent uniform samples for each CME parameter:
        x ~ U(mean - sd, mean + sd)               for t_init, width, lon, lat, thick
        v ~ U((1 - sd_v) mean_v, (1 + sd_v) mean_v)  (NB: sd_v is a *fractional* half-width here)
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters (cme_par_array order)
    :param rng: Random number generator
    :param sd_*: Half-widths of the uniform distributions (fractional for sd_v)
    :return: samples: (6, n_ens) array of ensemble members
    """

    samples = np.zeros((6, n_ens))

    low_cme_t_init = mean_cme_pars[0] - sd_t_init
    high_cme_t_init = mean_cme_pars[0] + sd_t_init
    samples[0, :] = [
        rng.uniform(low=low_cme_t_init, high=high_cme_t_init)
        for _ in range(n_ens)
    ]

    low_cme_speed = (1 - sd_v) * mean_cme_pars[1]
    high_cme_speed = (1 + sd_v) * mean_cme_pars[1]
    samples[1, :] = [
        rng.uniform(low=low_cme_speed, high=high_cme_speed) for _ in range(n_ens)
    ]


    low_cme_width = mean_cme_pars[2] - sd_width
    high_cme_width = mean_cme_pars[2] + sd_width
    samples[2, :] = [
        rng.uniform(low=low_cme_width, high=high_cme_width) for _ in range(n_ens)
    ]

    low_cme_lon = mean_cme_pars[3] - sd_lon
    high_cme_lon = mean_cme_pars[3] + sd_lon
    samples[3, :] = [
        rng.uniform(low=low_cme_lon, high=high_cme_lon) for _ in range(n_ens)
    ]

    low_cme_lat = mean_cme_pars[4] - sd_lat
    high_cme_lat = mean_cme_pars[4] + sd_lat
    samples[4, :] = [
        rng.uniform(low=low_cme_lat, high=high_cme_lat) for _ in range(n_ens)
    ]

    low_cme_thick = mean_cme_pars[5] - sd_thick
    high_cme_thick = mean_cme_pars[5] + sd_thick
    samples[5, :] = [
        rng.uniform(low=low_cme_thick, high=high_cme_thick) for _ in range(n_ens)
    ]

    return samples
