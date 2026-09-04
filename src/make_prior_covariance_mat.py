import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import glob

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from huxt import huxt_inputs
from surf import surf_inputs

from cme_par_dict_structure import cme_par_get_indices, required_dict_keys, cme_par_key_to_index, cme_par_get_keys

def plot_sample_cov(
        df_cov,
        df_corr,
        cov_name,
        use_log_v=False,
):
    fig, ax = plt.subplots(1, 1)
    tick_labels = [
        cme_par_get_keys(i) for i in range(len(required_dict_keys()))
    ]
    if use_log_v:
        tick_labels[cme_par_get_indices("v")] = "log(v)"

    sns.heatmap(
        df_cov, annot=True, fmt=".3g", cmap=plt.cm.RdBu_r,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    ax.set_title(f"Covariance matrix for {cov_name}")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        df_corr, annot=True, fmt=".4g", cmap=plt.cm.RdBu_r,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    ax.set_title(f"Correlation matrix for {cov_name}")
    plt.show()

    # Basic correlogram
    # sns.pairplot(df)
    # plt.show()
    # sns.heatmap(df.cov(), annot=True, cmap=plt.cm.RdBu_r, vmin=-1600, vmax=1600)
    # plt.show()
    #cov_var_array = np.cov(var_array, rowvar=False)

    #print(cov_var_array)

    return None


def plot_pair_plot(
        df_cme_par,
        cov_name,
        use_log_v=False,
):

    im = sns.pairplot(df_cme_par)
    im.map_lower(sns.kdeplot, levels=10, color="k")
    plt.show()

    return None

def uncorrelated_cov_mat(
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0
) -> npt.NDArray[float]:
    """
    Generate uncorrelated covariance matrix
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
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

    plot_sample_cov(
        uncorr_cov_mat,
        np.identity(6),
        "Uncorrelated covariance matrix",
    )
    return uncorr_cov_mat


def make_uncorrelated_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng,#: Generator,
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        use_log_v: bool = False,
) -> npt.NDArray[float]:
    """
    Generate uncorrelated covariance matrix
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :return: uncorr_samples: Ensemble members made with uncorrelated covariance matrix
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

    if use_log_v:
        mean_cme_pars[1] = np.log(mean_cme_pars[1])

    for m in range(n_ens):
        uncorr_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=uncorr_cov
        )
    if use_log_v:
        uncorr_samples[1, :] = np.exp(uncorr_samples[1, :])
    #print(uncorr_samples)
    return uncorr_samples


def make_uncorr_uniform_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng,
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0
) -> npt.NDArray[float]:

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
        provided in file_path
        :param file_path: File path to CME list to estimate covariance from
        :param vars_req: List of variables to estimate covariance of
            Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
            :param vars_req: List of variables that need to be perturbed in ensemble
        :param sd_t_init: Standard deviation of CME launch time
        :param sd_v: Standard deviation of CME velocity
        :param sd_width: Standard deviation of CME width
        :param sd_lon: Standard deviation of CME longitude
        :param sd_lat: Standard deviation of CME latitude
        :param sd_thick: Standard deviation of CME thickness
        :param use_log_v: Boolean to determine whether to use log CME speeds
        :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
        :return: cov_out: Covariance matrix of CME parameters
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
    #print(f"blair_cov={blair_cov}")

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
        rng,
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
    Generate uncorrelated covariance matrix
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters to perturb around
    :param blair_file_path: Path to Blair's CME list file
    :param rng: Random number generator
    :param vars_req: List of variables to estimate covariance of
        Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
        :param vars_req: List of variables that need to be perturbed in ensemble
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Boolean to determine whether to use log CME speeds
    :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
    :return: blair_samples: Ensemble members made with uncorrelated covariance matrix
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
    # print(f"blair_cov = {blair_cov}")

    for m in range(n_ens):
        blair_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=blair_cov
        )

    print(blair_samples)
    return blair_samples


def make_donki_cov(
        start_time: datetime.datetime,
        end_time:datetime.datetime,
        use_model: str,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        scale_corr=False,
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
):
    """
    Make covariance matrix using DONKI CME catalogue
    :param start_time: Start time of DONKI CME catalogue
    :param end_time: End time of DONKI CME catalogue
    :param use_model: String to determine which model to use, must be ["surf", "compress_surf", "huxt"]
    :param vars_req: List of variables to estimate covariance of
        Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param vars_req: List of variables that need to be perturbed in ensemble
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
    :param most_acc_only: Whether to use most accurate value
    :param catalog: Catalog to use
    :param feature: Feature to use
    :return: cov_out: Covariance matrix made with the DONKI CME catalogue
    :return: corr_out: Correlation matrix made with the DONKI CME catalogue
    """

    ########################################################################
    # Use routine in surf.surf_inputs to retrieve DONKI ConeCME parameters
    ########################################################################
    if use_model in ["surf", "compress_surf"]:
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
    for i in range(len_cme_dict):
        #dfCMEpar.loc[i, "t_init"] = donki_cone_cme_dict[i]["vcld"]
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
        rng,
        start_time: datetime.datetime,
        end_time:datetime.datetime,
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
    Generate uncorrelated covariance matrix
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters to perturb around
    :param blair_file_path: Path to Blair's CME list file
    :param rng: Random number generator
    :param start_time: Start time
    :param end_time: End time
    :param scale_corr: Boolean to determine whether to scale the correlation
        matrix to make the covariance matrix
    :param vars_req: List of variables to estimate covariance of
    Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param vars_req: List of variables that need to be perturbed in ensemble
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Boolean to determine whether to use log CME speeds
    :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
    :param most_acc_only: Whether to use only the most accurate value
    :param catalog: Catalog to use
    :param feature: Feature to use
    :return: blair_samples: Ensemble members made with Blair's CME covariance matrix
    """
    donki_samples: npt.NDArray[float] = np.zeros((6, n_ens))

    donki_tuple: tuple[npt.NDArray[float], npt.NDArray[float]] = make_donki_cov(
        start_time=start_time,
        end_time=end_time,
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

    # Take exponential of speed values
    v_index = cme_par_get_indices("v")

    if use_log_v:
        mean_cme_pars[v_index]: list[float] = np.log(mean_cme_pars[v_index])

    for m in range(n_ens):
        donki_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=donki_cov
        )

    if use_log_v:
        donki_samples[v_index, :]: npt.NDArray[float] = np.exp(donki_samples[v_index, :])

    return donki_samples


def filter_first_mo_cone_cme(
        file_dir: str,
        start_time: datetime.datetime,
        path_list: list[str],
) -> list[str]:
    """
    Filter out the initial ConeCME files that are before the date range
    :param file_dir: Directory containing CME files
    :param start_time: Datetime object of start time
    :param path_list: List of paths to CME files to append to
    :return: path_list: List of paths to CME files with new files appended to
    """
    # Get start_year and start_month from start_time
    start_year = start_time.year
    start_month = start_time.month

    # Get required file directory for first month of files and sort them
    file_dir_month = os.path.join(file_dir, f"{start_year:04d}", f"{start_month:02d}")
    glob_lists = sorted(glob.glob(os.path.join(file_dir_month, "cone_cme_*.in")))

    # Filter out any cone_cme file names prior to start_date
    glob_file_names = [
        gl.split(os.path.sep)[-1] for gl in glob_lists
    ]
    glob_dates = np.array([
        datetime.datetime.strptime(gl, "cone_cme_%Y%m%d%H.in") for gl in glob_file_names
    ])
    glob_file_paths = [
        gl for i, gl in enumerate(glob_lists) if glob_dates[i] >= start_time
    ]
    path_list.extend(glob_file_paths)

    return path_list


def filter_final_mo_cone_cme(
        file_dir: str,
        end_time: datetime.datetime,
        path_list: list[str],
) -> list[str]:
    """
    Filter out the initial ConeCME files that are before the date range
    :param file_dir: Directory containing CME files
    :param end_time: Datetime object of end time
    :param path_list: List of paths to CME files to append to
    :return: path_list: List of paths to CME files with new files appended to
    """
    # Get start_year and start_month from start_time
    end_year = end_time.year
    end_month = end_time.month

    # Get required file directory for first month of files and sort them
    file_dir_month = os.path.join(file_dir, f"{end_year:04d}", f"{end_month:02d}")
    glob_lists = sorted(glob.glob(os.path.join(file_dir_month, "cone_cme_*.in")))

    # Filter out any cone_cme file names prior to start_date
    glob_file_names = [
        gl.split(os.path.sep)[-1] for gl in glob_lists
    ]
    glob_dates = np.array([
        datetime.datetime.strptime(gl, "cone_cme_%Y%m%d%H.in") for gl in glob_file_names
    ])
    glob_file_paths = [
        gl for i, gl in enumerate(glob_lists) if glob_dates[i] <= end_time
    ]
    path_list.extend(glob_file_paths)

    return path_list


def filter_mo_cone_cme_1month(
        file_dir: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime
) -> list[str]:
    """
    Filter out the initial ConeCME files that are before the date range
    :param file_dir: Directory containing CME files
    :param start_time: Datetime object of start time
    :param path_list: List of paths to CME files to append to
    :return: path_list: List of paths to CME files with new files appended to
    """
    # Get start_year and start_month from start_time
    start_year = start_time.year
    start_month = start_time.month

    end_year = end_time.year
    end_month = end_time.month

    # Assert that this is being called only over a single month
    assert ((start_year == end_year) and (start_month == end_month))

    # Get required file directory for first month of files and sort them
    file_dir_month = os.path.join(file_dir, f"{start_year:04d}", f"{start_month:02d}")
    glob_lists = sorted(glob.glob(os.path.join(file_dir_month, "cone_cme_*.in")))

    # Filter out any cone_cme file names prior to start_date
    glob_file_names = [
        gl.split(os.path.sep)[-1] for gl in glob_lists
    ]
    glob_dates = np.array([
        datetime.datetime.strptime(gl, "cone_cme_%Y%m%d%H.in") for gl in glob_file_names
    ])
    path_list = [
        gl for i, gl in enumerate(glob_lists)
        if ((glob_dates[i] >= start_time) and (glob_dates[i] <= end_time))
    ]

    return path_list


def make_mo_cone_file_list(
        file_dir: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
) -> list[str]:
    """
    Make a covariance matrix based upon the Met Office's Cone CME files
    :param file_dir: Directory containing Met Office Cone CME files
    :param start_time: Get start time of first file to be read
    :param end_time: Get end time of last file to be read
    :return: mo_cone_file_list: List of MO ConeCME files to be read
    """

    # Get years of start_time and end_time
    start_year: int = start_time.year
    start_month: int = start_time.month

    end_year: int = end_time.year
    end_month: int = end_time.month

    # Get all years required
    years_req = np.array(range(start_year, end_year + 1))
    n_years = len(years_req)

    path_list: list[str] = []

    if n_years == 1:
        assert (start_year == end_year)
        months_req = np.array(range(start_month, end_month + 1))
        n_months = len(months_req)
        if n_months == 1:
            assert (start_month == end_month)
            path_list = filter_mo_cone_cme_1month(
                file_dir, start_time, end_time
            )
        else:
            assert (start_month < end_month)
            for imonth, month in enumerate(months_req):
                if imonth == 0:
                    path_list = filter_first_mo_cone_cme(
                        file_dir, start_time, path_list
                    )
                elif month == end_month:
                    path_list = filter_final_mo_cone_cme(
                        file_dir, end_time, path_list
                    )
                else:
                    file_dir_month = os.path.join(file_dir, f"{start_year:04d}", f"{month:02d}")
                    glob_lists = sorted(glob.glob(
                        os.path.join(file_dir_month, "cone_cme_*.in")
                    ))
                    path_list.extend(glob_lists)
    elif n_years > 1:
        for iyear, year in enumerate(years_req):
            file_dir_year: str = os.path.join(file_dir, f"{year:04d}")
            if iyear == 0:
                for imonth, month in enumerate(range(start_month, 13)):
                    if imonth == 0:
                        path_list = filter_first_mo_cone_cme(
                            file_dir, start_time, path_list
                        )
                    else:
                        file_dir_month = os.path.join(file_dir_year, f"{month:02d}")
                        glob_lists = sorted(glob.glob(os.path.join(file_dir_month, "cone_cme_*.in")))
                        path_list.extend(glob_lists)
            elif year == end_year:
                for imonth, month in enumerate(range(1, end_month + 1)):
                    if month == end_month:
                        path_list = filter_final_mo_cone_cme(
                            file_dir, end_time, path_list
                        )
                    else:
                        file_dir_month = os.path.join(file_dir_year, f"{month:02d}")
                        glob_lists = sorted(glob.glob(
                            os.path.join(file_dir_month, "cone_cme_*.in")
                        ))
                        path_list.extend(glob_lists)
            else:
                for month in range(1, 13):
                    file_dir_month = os.path.join(file_dir_year, f"{month:02d}")
                    glob_lists = sorted(glob.glob(
                        os.path.join(file_dir_month, "cone_cme_*.in")
                    ))
                    path_list.extend(glob_lists)

    return path_list


def make_mo_cone_cme_cov(
        file_dir: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        overwrite_cov: bool=False,
        mo_cone_cov_dir: str=None,
        scale_corr:bool=False,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0,
        sd_v: float=0,
        sd_width: float=0,
        sd_lon: float=0,
        sd_lat: float=0,
        sd_thick: float=0,
        use_log_v: bool=False,
        plot_cme_cov: bool=False,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Make a covariance matrix based upon the Met Office's Cone CME files
    :param file_dir: Directory containing Met Office Cone CME files
    :param start_time: Get start time of first file to be read
    :param end_time: Get end time of last file to be read
    :param overwrite_cov: Boolean to determine whether to overwrite existing covariance/correlation matrices
    :param mo_cone_cov_dir: Directory containing precalculated cme_pars and covariance/correlation matrices
        Defaults to current working directory
    :param scale_corr: Boolean to determine whether to scale the correlation
        matrix to make the covariance matrix
    :param vars_req: List of variables that need to be perturbed in ensemble
        Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Boolean to determine whether to use log CME speeds
    :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
    :return: mo_cone_cov: Covariance matrix of Met Office Cone CME files
    :return: mo_cone_corr: Correlation matrix of Met Office Cone CME files
    """

    # Define output file names
    start_date_str = start_time.strftime("%Y%m%d")
    end_date_str = end_time.strftime("%Y%m%d")

    #If mo_cone_cov_dir is not specified, use current directory
    if mo_cone_cov_dir is None:
        mo_cone_cov_dir = os.path.dirname(os.path.abspath(__file__))

    mo_cone_par_file_name = os.path.join(
        mo_cone_cov_dir, f"mo_cone_par_{start_date_str}-{end_date_str}.nc"
    )
    cov_out_file_name = os.path.join(
        mo_cone_cov_dir, f"mo_cone_cme_cov_{start_date_str}-{end_date_str}.nc"
    )
    corr_out_file_name = os.path.join(
        mo_cone_cov_dir, f"mo_cone_corr_{start_date_str}-{end_date_str}.nc"
    )

    # If one of the files doesn't exist, default to writing all files
    par_file_bool = os.path.exists(mo_cone_par_file_name)
    cov_file_bool = os.path.exists(cov_out_file_name)
    corr_file_bool = os.path.exists(corr_out_file_name)

    make_cov_corr_mat_from_nc_pars = False

    cme_par_keys_ordered = cme_par_key_to_index().keys()

    if not par_file_bool:
        overwrite_cov = True
    else:
        assert par_file_bool
        if not (cov_file_bool and corr_file_bool):
            make_cov_corr_mat_from_nc_pars = True

    if overwrite_cov:
        assert os.path.exists(file_dir), f"File directory for file_dir:\n {file_dir} does not exist."

        ########################################################################
        # Make the file list of required Met Office ConeCME parameters
        ########################################################################
        file_list = make_mo_cone_file_list(
            file_dir, start_time, end_time
        )
        df_all_list = []

        ##############################################################################
        # Use routine in surf.surf_inputs to retrieve Met Office ConeCME parameters
        ##############################################################################
        for file_name in file_list:

            mo_cone_dict = surf_inputs.import_cone2bc_parameters(file_name)
            len_mo_dict = len(mo_cone_dict)

            file_name_end = file_name.split("_")[-1]
            file_name_end5 = file_name_end[-7:]

            if file_name_end5 == "0100.in":
                print(f"filename={file_name}")

            dfCMEpar = pd.DataFrame(
                columns=["v", "width", "lon", "lat"], index=range(len_mo_dict)
            )
            for i in range(len_mo_dict):
                if use_log_v:
                    dfCMEpar.loc[i, "v"] = np.log(mo_cone_dict[i + 1]["vcld"])
                else:
                    dfCMEpar.loc[i, "v"] = mo_cone_dict[i + 1]["vcld"]

                dfCMEpar.loc[i, "width"] = 2 * mo_cone_dict[i + 1]["rmajor"]
                dfCMEpar.loc[i, "lon"] = mo_cone_dict[i + 1]["lon"]
                dfCMEpar.loc[i, "lat"] = mo_cone_dict[i + 1]["lat"]

                # Reorder the parameters so they're in the correct order
                cme_par_keys_ordered = cme_par_key_to_index().keys()
                list_req_order = [
                    r for r in cme_par_keys_ordered if r in dfCMEpar.columns
                ]
                dfCMEpar = dfCMEpar[list_req_order]

            # Append dataframe to a list
            df_all_list.append(dfCMEpar)

        # Concatenate all dataframes in the list to a single dataframe
        mo_cone_cme_df = pd.concat(df_all_list)
        mo_cone_cme_df.drop_duplicates(keep='first', inplace=True, ignore_index=True)

        #cme_par_keys_ordered = cme_par_key_to_index().keys()
        list_req_order = [
            r for r in cme_par_keys_ordered if r in mo_cone_cme_df.columns
        ]
        mo_cone_cme_df = mo_cone_cme_df[list_req_order]

        ###############################################################
        # Calculate covariance matrix from mo_cone_cme_df
        ###############################################################
        mo_cone_cme_cov = mo_cone_cme_df.cov(numeric_only=False)

        # Add in t_init and thick rows
        mo_cone_cme_cov = mo_cone_cme_cov.reindex(cme_par_keys_ordered, fill_value=0.0)
        mo_cone_cme_cov.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
        mo_cone_cme_cov.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
        mo_cone_cme_cov.loc["t_init", "t_init"] = sd_t_init * sd_t_init
        mo_cone_cme_cov.loc["thick", "thick"] = sd_thick * sd_thick

        ################################################################
        # Calculate correlation matrix from mo_cone_df
        ################################################################
        mo_cone_cme_corr = mo_cone_cme_df.corr(numeric_only=False)

        # Add in t_init and thick rows
        mo_cone_cme_corr = mo_cone_cme_corr.reindex(cme_par_keys_ordered, fill_value=0.0)
        mo_cone_cme_corr.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
        mo_cone_cme_corr.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
        mo_cone_cme_corr.loc["t_init", "t_init"] = 1.0
        mo_cone_cme_corr.loc["thick", "thick"] = 1.0

        # Write the CME parameters and the covariance/correlation matrices to predefined files
        if not os.path.exists(mo_cone_cov_dir):
            os.makedirs(mo_cone_cov_dir)

        mo_cone_cme_ds = mo_cone_cme_df.to_xarray()
        mo_cone_cme_ds.to_netcdf(mo_cone_par_file_name)

        mo_cone_cme_cov_ds = mo_cone_cme_cov.to_xarray()
        mo_cone_cme_cov_ds.to_netcdf(cov_out_file_name)

        mo_cone_cme_corr_ds = mo_cone_cme_corr.to_xarray()
        mo_cone_cme_corr_ds.to_netcdf(corr_out_file_name)

    elif make_cov_corr_mat_from_nc_pars:
        # Open premade nc files containing all cme parameters
        mo_cone_cme_ds = xr.open_dataset(mo_cone_par_file_name)
        mo_cone_cme_df = mo_cone_cme_ds.to_pandas()

        ###############################################################
        # Calculate covariance matrix from mo_cone_cme_df
        ###############################################################
        mo_cone_cme_cov = mo_cone_cme_df.cov(numeric_only=False)

        # Add in t_init and thick rows
        mo_cone_cme_cov = mo_cone_cme_cov.reindex(cme_par_keys_ordered, fill_value=0.0)
        mo_cone_cme_cov.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
        mo_cone_cme_cov.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
        mo_cone_cme_cov.loc["t_init", "t_init"] = sd_t_init * sd_t_init
        mo_cone_cme_cov.loc["thick", "thick"] = sd_thick * sd_thick

        ################################################################
        # Calculate correlation matrix from mo_cone_df
        ################################################################
        mo_cone_cme_corr = mo_cone_cme_df.corr(numeric_only=False)

        # Add in t_init and thick rows
        mo_cone_cme_corr = mo_cone_cme_corr.reindex(cme_par_keys_ordered, fill_value=0.0)
        mo_cone_cme_corr.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
        mo_cone_cme_corr.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
        mo_cone_cme_corr.loc["t_init", "t_init"] = 1.0
        mo_cone_cme_corr.loc["thick", "thick"] = 1.0

    else:
        # Open premade nc files containing all cme parameters
        mo_cone_cme_ds = xr.open_dataset(mo_cone_par_file_name)
        mo_cone_cme_df = mo_cone_cme_ds.to_pandas()

        # Open premade nc files containing cme parameter covariance and correlation matrices
        mo_cone_cme_cov_ds = xr.open_dataset(cov_out_file_name)
        mo_cone_cme_corr_ds = xr.open_dataset(corr_out_file_name)

        mo_cone_cme_cov = mo_cone_cme_cov_ds.to_pandas()
        mo_cone_cme_corr = mo_cone_cme_corr_ds.to_pandas()

    # Update covariance matrix if we require it to be a scaled correlation matrix
    if scale_corr:
        sd_diag_elements = [sd_t_init, sd_v, sd_width, sd_lon, sd_lat, sd_thick]
        assert all(sd_diag_elements[i] >= 0 for i in range(len(sd_diag_elements))), "Ensure all sd_diag_elements >= 0"
        assert (np.sum(sd_diag_elements) > 0), "At least one standard deviation must be greater than 0"

        # Make matrix with diagonal equal to the standard deviation of the parameters as required
        sd_diag = np.diag(sd_diag_elements)
        mo_cone_cme_cov = sd_diag.dot(mo_cone_cme_corr).dot(sd_diag)

    #########################################################################
    # Transform covariance and correlation matrices to an array for output
    #########################################################################
    if isinstance(mo_cone_cme_cov, np.ndarray):
        cov_out = mo_cone_cme_cov.copy()
    else:
        cov_out = mo_cone_cme_cov.values
    len_cov_out = len(cov_out[0, :])

    if isinstance(mo_cone_cme_corr, np.ndarray):
        corr_out = mo_cone_cme_corr.copy()
    else:
        corr_out = mo_cone_cme_corr.values.copy()
    len_corr_out = len(corr_out[0, :])

    for m, par_key in enumerate(cme_par_keys_ordered):
        if par_key == "lon":
            print(type(cov_out))
            # Remove all non-diagonal entries from longitudinal covariances
            lon_var = cov_out[m, m]

            cov_out[m, :] = 0
            cov_out[:, m] = 0
            cov_out[m, m] = lon_var

        # Remove any covariance from variables that do not need to be perturbed
        if par_key not in vars_req:
            cov_out[m, :] = 0
            cov_out[:, m] = 0

    #################################################################
    # Plot CME covariance and correlation matrices, if necessary
    #################################################################
    if plot_cme_cov:
        plot_sample_cov(
            cov_out,
            corr_out,
            "MO Cone CME",
            use_log_v=use_log_v,
        )

        plot_pair_plot(
            mo_cone_cme_df, "MO Cone CME", use_log_v=use_log_v
        )

    print(f"cov_out={cov_out}")
    print(f"corr_out={corr_out}")

    return cov_out, corr_out


def make_mo_cone_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng,
        mo_cone_file_dir: str,
        start_time: datetime.datetime,
        end_time:datetime.datetime,
        overwrite_cov: bool=False,
        mo_cone_cov_dir: str=None,
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
) -> npt.NDArray[float]:
    """
    Generate uncorrelated covariance matrix
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters to perturb around
    :param rng: Random number generator
    :param overwrite_cov: Boolean to determine whether to overwrite existing covariance matrix
    :param mo_cone_cov_dir: Directory containing precalculated cme_pars and covariance/correlation matrices
        Defaults to current working directory
    :param scale_corr: Boolean to determine whether to scale the correlation
        matrix to make the covariance matrix
    :param mo_cone_file_dir: Path to the Met Office's Cone CME files
    :param start_time: Start time
    :param end_time: End time
    :param vars_req: List of variables to estimate covariance of
    Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param vars_req: List of variables that need to be perturbed in ensemble
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_v: Standard deviation of CME velocity
    :param sd_width: Standard deviation of CME width
    :param sd_lon: Standard deviation of CME longitude
    :param sd_lat: Standard deviation of CME latitude
    :param sd_thick: Standard deviation of CME thickness
    :param use_log_v: Boolean to determine whether to use log CME speeds
    :param plot_cme_cov: Boolean to determine whether to plot CME covariance matrix
    :return: mo_cone_samples: Ensemble members made with Blair's CME covariance matrix
    """
    mo_cone_samples: npt.NDArray[float] = np.zeros((6, n_ens))

    mo_cone_tuple: tuple[npt.NDArray[float], npt.NDArray[float]] = make_mo_cone_cme_cov(
        file_dir=mo_cone_file_dir,
        start_time=start_time,
        end_time=end_time,
        overwrite_cov = overwrite_cov,
        mo_cone_cov_dir = mo_cone_cov_dir,
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
    )

    mo_cone_cov: npt.NDArray[float] = mo_cone_tuple[0]
    mo_cone_corr: npt.NDArray[float] = mo_cone_tuple[1]

    mean_cme_pars[cme_par_get_indices("v")]: list[float] = np.log(mean_cme_pars[cme_par_get_indices("v")])
    for m in range(n_ens):
        mo_cone_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=mo_cone_cov
        )

    # Take exponential of speed values
    v_index = cme_par_get_indices("v")
    if use_log_v:
        mo_cone_samples[v_index, :]: npt.NDArray[float] = np.exp(mo_cone_samples[v_index, :])

    print(mo_cone_samples)

    return mo_cone_samples


def main(
        uncorr_samp: bool=True,
        blair_samp: bool=True,
        donki_samp: bool=True,
        mo_cone_samp: bool=True,
):
    n_ens: int = 3
    mean_cme_pars: list[float] = [0.1, 1000.2, 2000.3, 3000.4, 4000.5, 5000.6]
    vars_req = np.array(["t_init", "v", "width", "lon", "lat", "thick"])

    sd_t_init = 0
    sd_v = 50
    sd_width = 10.6
    sd_lon = 4
    sd_lat = 2.5
    sd_thick = 0

    use_log_v = True
    plot_cme_cov = True

    if use_log_v:
        sd_v = np.log(sd_v)


    if uncorr_samp:
        rng = np.random.default_rng(42)
        uncorr_samp = make_uncorrelated_samples(
            n_ens, mean_cme_pars, rng,
            sd_t_init=sd_t_init, sd_v=sd_v, sd_width=sd_width,
            sd_lon=sd_lon, sd_lat=sd_lat, sd_thick=sd_thick
        )


    if blair_samp:
        rng = np.random.default_rng(42)
        blair_file_path = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "blairCMElistSingle.csv"
        )
        #blair_cov = make_cov_blair(blair_file_path = blair_file_path)
        blair_samples = make_blair_samples(
            n_ens=n_ens,
            mean_cme_pars=mean_cme_pars,
            blair_file_path=blair_file_path,
            rng=rng,
            vars_req=vars_req,
            sd_t_init=sd_t_init,
            sd_v=sd_v,
            sd_width=sd_width,
            sd_lon=sd_lon,
            sd_lat=sd_lat,
            sd_thick=sd_thick,
            use_log_v=use_log_v,
            plot_cme_cov=plot_cme_cov,
        )

    if donki_samp:
        start_time = datetime.datetime(2017, 1, 1, 0, 0, 0)
        end_time = datetime.datetime(2026, 2, 1, 0, 0, 0)
        rng = np.random.default_rng(42)
        most_acc_only = "true"
        catalog = "ALL"
        feature = "LE"

        donki_samples = make_donki_samples(
            n_ens=n_ens,
            mean_cme_pars=mean_cme_pars,
            rng=rng,
            start_time=start_time,
            end_time=end_time,
            scale_corr=True,
            vars_req=vars_req,
            sd_t_init=sd_t_init,
            sd_v=sd_v,
            sd_width=sd_width,
            sd_lon=sd_lon,
            sd_lat=sd_lat,
            sd_thick=sd_thick,
            use_log_v=use_log_v,
            plot_cme_cov=plot_cme_cov,
            most_acc_only = most_acc_only,
            catalog = catalog,
            feature = feature
        )

    if mo_cone_samp:
        mo_start_time = datetime.datetime(2017, 1, 1, 5, 0, 0)
        mo_end_time = datetime.datetime(2026, 2, 1, 0, 0, 0)
        rng = np.random.default_rng(42)
        mo_cone_cov_dir = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "moConeCMECov"
        )
        if not os.path.exists(mo_cone_cov_dir):
            os.makedirs(mo_cone_cov_dir)

        mo_cme_cone_file_dir = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "moswoc_cone"
        )
        make_mo_cone_samples(
            n_ens=n_ens,
            mean_cme_pars=mean_cme_pars,
            rng=rng,
            mo_cone_file_dir=mo_cme_cone_file_dir,
            start_time=mo_start_time,
            end_time=mo_end_time,
            overwrite_cov=False,
            mo_cone_cov_dir=mo_cone_cov_dir,
            scale_corr=True,
            vars_req=vars_req,
            sd_t_init=sd_t_init,
            sd_v=sd_v,
            sd_width=sd_width,
            sd_lon=sd_lon,
            sd_lat=sd_lat,
            sd_thick=sd_thick,
            use_log_v=use_log_v,
            plot_cme_cov=plot_cme_cov,
        )
    return None

if __name__ == "__main__":
    main(
        uncorr_samp=False,
        blair_samp=False,
        donki_samp=True,
        mo_cone_samp=True,
    )