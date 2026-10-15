"""
prior_donki.py

Correlated Gaussian prior for the CME parameters with the covariance estimated from the NASA DONKI
cone-CME catalogue (retrieved with surf_inputs / huxt_inputs ``get_DONKI_coneCMEs``).

Optionally (``scale_corr=True``) the DONKI *correlation* matrix is rescaled by the user's standard
deviations, C = D R D with D = diag(sd). Longitude is always decorrelated from the other parameters.
Rows / columns follow the cme_par_array order (t_init, v, width, lon, lat, thick).
"""
import datetime
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import Generator

from huxt import huxt_inputs
from surf import surf_inputs

from cme_par_dict_structure import cme_par_get_indices, cme_par_key_to_index
from prior_cov_plots import plot_sample_cov, plot_pair_plot


def make_donki_cov(
        start_time: datetime.datetime,
        end_time:datetime.datetime,
        use_model: str,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        scale_corr: bool = False,
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        use_log_v: bool=False,
        plot_cme_cov: bool=False,
        most_acc_only: str="true",
        catalog: str="ALL",
        feature: str="LE",
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Make covariance matrix using DONKI CME catalogue
    :param start_time: Start time of DONKI CME catalogue
    :param end_time: End time of DONKI CME catalogue
    :param use_model: Model in use ("surf", "surf3d", "compress_surf", "compress_surf3d" or "huxt");
        selects which package's DONKI reader is used
    :param vars_req: Parameters to perturb in the ensemble (others get zero (co)variance)
        Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param scale_corr: Boolean to determine whether to scale the correlation
        matrix to make the covariance matrix
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Boolean to determine whether to use log CME speeds
    :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
    :param most_acc_only: DONKI query option: only the most accurate analysis per CME ("true"/"false")
    :param catalog: DONKI catalogue to query (e.g. "ALL")
    :param feature: DONKI feature to query (e.g. "LE" = leading edge)
    :return: cov_out: Covariance matrix made with the DONKI CME catalogue
    :return: corr_out: Correlation matrix made with the DONKI CME catalogue
    """

    ########################################################################
    # Use routine in surf.surf_inputs to retrieve DONKI ConeCME parameters
    ########################################################################
    if use_model in ["surf", "surf3d", "compress_surf", "compress_surf3d"]:
        donki_cone_cme_dict = surf_inputs.get_DONKI_coneCMEs(
            startdate=start_time,
            enddate=end_time,
            mostAccOnly=most_acc_only,
            catalog=catalog,
            feature=feature
        )
    elif use_model in ["huxt"]:
        donki_cone_cme_dict = huxt_inputs.get_DONKI_coneCMEs(
            startdate=start_time,
            enddate=end_time,
            mostAccOnly=most_acc_only,
            catalog=catalog,
            feature=feature
        )
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")
    print(donki_cone_cme_dict)
    ########################################################
    # Put CME parameters into a dataframe
    ########################################################
    len_cme_dict = len(donki_cone_cme_dict)
    dfCMEpar = pd.DataFrame(
        columns=["v", "width", "lon", "lat"], index=np.arange(len_cme_dict)
    )
    # (DONKI gives the cone half-angle "rmajor", so width = 2 * rmajor; NaN/slow speeds are reported)
    for i in range(len_cme_dict):
        if np.isnan(donki_cone_cme_dict[i]['vcld']):
            print(f"v[{i}] = {donki_cone_cme_dict[i]['vcld']}")

        if (donki_cone_cme_dict[i]['vcld'] < 200):
            print(f"v[{i}] = {donki_cone_cme_dict[i]['vcld']}")

        if use_log_v:
            dfCMEpar.loc[i, "v"] = np.log(donki_cone_cme_dict[i]["vcld"])
        else:
            dfCMEpar.loc[i, "v"] = donki_cone_cme_dict[i]["vcld"]

        dfCMEpar.loc[i, "width"] = 2 * donki_cone_cme_dict[i]["rmajor"]
        dfCMEpar.loc[i, "lon"] = donki_cone_cme_dict[i]["lon"]
        dfCMEpar.loc[i, "lat"] = donki_cone_cme_dict[i]["lat"]

    cme_par_keys_ordered = cme_par_key_to_index().keys()
    list_req_order = [
        r for r in cme_par_keys_ordered if r in dfCMEpar.columns
    ]
    dfCMEpar = dfCMEpar[list_req_order]

    ####################################################
    # Calculate covariance matrix from dfCMEpar
    ####################################################
    donki_cov = dfCMEpar.cov(numeric_only=False)

    # Add in t_init and thick rows
    donki_cov = donki_cov.reindex(cme_par_keys_ordered, fill_value=0.0)
    donki_cov.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    donki_cov.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
    donki_cov.loc["t_init", "t_init"] = sd_t_init * sd_t_init
    donki_cov.loc["thick", "thick"] = sd_thick * sd_thick

    # Assume zero covariance if covariance value is NaN
    donki_cov = donki_cov.fillna(0)
    print(f"donki_cov={donki_cov}")

    #####################################################
    # Calculate correlation matrix from dfCMEpar
    #####################################################
    donki_corr = dfCMEpar.corr(numeric_only=False)


    # Add in t_init and thick rows
    donki_corr = donki_corr.reindex(cme_par_keys_ordered, fill_value=0.0)
    donki_corr.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    donki_corr.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)

    # Force diagonal's to be 1
    donki_corr.loc["t_init", "t_init"] = 1.0
    donki_corr.loc["v", "v"] = 1.0
    donki_corr.loc["width", "width"] = 1.0
    donki_corr.loc["lon", "lon"] = 1.0
    donki_corr.loc["lat", "lat"] = 1.0
    donki_corr.loc["thick", "thick"] = 1.0

    donki_corr = donki_corr.fillna(0)
    print(f"donki_corr={donki_corr}")

    # Update covariance matrix if we require it to be a scaled correlation matrix
    if scale_corr:
        sd_diag_elements = [sd_t_init, sd_v, sd_width, sd_lon, sd_lat, sd_thick]
        assert all(sd_diag_elements[i] >= 0 for i in range(len(sd_diag_elements))), "Ensure all sd_diag_elements >= 0"
        assert (np.sum(sd_diag_elements) > 0), "At least one standard deviation must be greater than 0"

        # Make matrix with diagonal equal to the standard deviation of the parameters as required
        sd_diag = np.diag(sd_diag_elements)
        print(sd_diag)
        donki_cov = sd_diag.dot(donki_corr).dot(sd_diag)

    #################################################
    # Transform to an array for output
    #################################################
    if isinstance(donki_cov, np.ndarray):
        cov_out = donki_cov
    else:
        cov_out = donki_cov.values

    corr_out = donki_corr.values


    for m, par_key in enumerate(cme_par_keys_ordered):
        if par_key == "lon":
            # Remove all non-diagonal entries from longitudinal covariances
            lon_var = cov_out[m, m]

            cov_out[m, :] = 0
            cov_out[:, m] = 0
            cov_out[m, m] = lon_var

            corr_out[m, :] = 0
            corr_out[:, m] = 0
            corr_out[m, m] = 1

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
            "DONKI",
            use_log_v=use_log_v,
        )

        plot_pair_plot(
            dfCMEpar, "DONKI", use_log_v=use_log_v
        )

    return cov_out, corr_out


def make_donki_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng: Generator,
        start_time: datetime.datetime,
        end_time:datetime.datetime,
        use_model: str="surf",
        scale_corr: bool=False,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        use_log_v: bool = False,
        plot_cme_cov: bool=False,
        most_acc_only: str="true",
        catalog: str="ALL",
        feature: str="LE"
) -> npt.NDArray[float]:
    """
    Draw Gaussian samples around mean_cme_pars with the covariance from the DONKI catalogue
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters to perturb around (cme_par_array order)
    :param rng: Random number generator
    :param use_model: Model in use (selects the DONKI reader, see make_donki_cov)
    :param start_time: Start time
    :param end_time: End time
    :param scale_corr: Boolean to determine whether to scale the correlation
        matrix to make the covariance matrix
    :param vars_req: Parameters to perturb in the ensemble
        Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Sample log(speed) instead of speed (samples are returned as speeds)
    :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
    :param most_acc_only, catalog, feature: DONKI query options (see make_donki_cov)
    :return: donki_samples: (6, n_ens) array of ensemble members
    """
    donki_samples: npt.NDArray[float] = np.zeros((6, n_ens))

    # (use_model is a required argument of make_donki_cov and was previously not passed)
    donki_tuple: tuple[npt.NDArray[float], npt.NDArray[float]] = make_donki_cov(
        start_time=start_time,
        end_time=end_time,
        use_model=use_model,
        scale_corr=scale_corr,
        vars_req=vars_req,
        sd_t_init=sd_t_init,
        sd_v=sd_v,
        sd_width=sd_width,
        sd_lon=sd_lon,
        sd_lat=sd_lat,
        sd_thick=sd_thick,
        use_log_v=use_log_v,
        plot_cme_cov=plot_cme_cov,
        most_acc_only=most_acc_only,
        catalog=catalog,
        feature=feature
    )

    donki_cov: npt.NDArray[float] = donki_tuple[0]
    donki_corr: npt.NDArray[float] = donki_tuple[1]

    # Work in log-speed if required (exponentiated back after sampling)
    v_index = cme_par_get_indices("v")

    mean_cme_pars = list(mean_cme_pars)
    if use_log_v:
        mean_cme_pars[v_index]: list[float] = np.log(mean_cme_pars[v_index])

    for m in range(n_ens):
        donki_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=donki_cov
        )

    if use_log_v:
        donki_samples[v_index, :]: npt.NDArray[float] = np.exp(donki_samples[v_index, :])

    return donki_samples
