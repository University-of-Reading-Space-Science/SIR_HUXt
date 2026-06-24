import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import numpy.typing as npt
import pandas as pd
from huxt import huxt_inputs
from sklearn.decomposition import dict_learning_online

from cme_par_dict_structure import cme_par_get_indices, required_dict_keys, cme_par_key_to_index

def plot_sample_cov(
        df_cov,
        df_corr,
        cov_name,
        vars_req=["t_init", "v", "width", "lon", "lat", "thick"],
):
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(df_cov, annot=True, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Covariance matrix for {cov_name}")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(df_corr, annot=True, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Correlation matrix for {cov_name}")
    plt.show()

    # sns.heatmap(df.cov(), annot=True, cmap=plt.cm.RdBu_r, vmin=-1600, vmax=1600)
    # plt.show()
    #cov_var_array = np.cov(var_array, rowvar=False)

    #print(cov_var_array)

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

    for iv, var_name in enumerate(required_dict_keys()):
        ind_req = cme_par_get_indices(var_name)
        uncorr_cov_mat[ind_req, ind_req] = sd_vals[iv] * sd_vals[iv]

    plot_sample_cov(
        uncorr_cov_mat,
        np.identity(6),
        "Uncorrelated covariance matrix",
        vars_req=["t_init", "v", "width", "lon", "lat", "thick"],
    )
    return uncorr_cov_mat


def make_uncorrelated_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng,#: Generator,
        sd_t_init: float=0, sd_v: float=0, sd_width: float=0,
        sd_lon: float=0, sd_lat: float=0, sd_thick: float=0
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

    for m in range(n_ens):
        uncorr_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=uncorr_cov
        )

    #print(uncorr_samples)
    return uncorr_samples


def make_cov_blair(
        blair_file_path: str,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0,
        sd_thick: float=0
)-> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
        Generate the covariance matrix for the initial parameters using Blair's CME list
        provided in file_path
        :param file_path: File path to CME list to estimate covariance from
        :param vars_req: List of variables to estimate covariance of
            Accepted inputs = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
        :param sd_t_init: Standard deviation of CME launch time
        :param sd_thick: Standard deviation of CME thickness
        :return: cov_out: Covariance matrix of CME parameters
    """
    assert all(v in required_dict_keys() for v in vars_req)

    # Open and read in CME list
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
    print(list_req_order)
    dfCMEpar = dfCMEpar[list_req_order]
    print(f"dfCMEpar={dfCMEpar}")
    # Calculate covariance matrix from dfCMEpar
    blair_cov = dfCMEpar.cov(numeric_only=True)

    # Add in t_init and thick rows
    blair_cov = blair_cov.reindex(cme_par_keys_ordered, fill_value=0.0)
    blair_cov.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    blair_cov.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
    blair_cov.loc["t_init", "t_init"] = sd_t_init * sd_t_init
    blair_cov.loc["thick", "thick"] = sd_thick * sd_thick

    print(f"blair_cov={blair_cov}")
    blair_corr = dfCMEpar.corr(numeric_only=True)

    # Add in t_init and thick rows
    blair_corr = blair_corr.reindex(cme_par_keys_ordered, fill_value=0.0)
    blair_corr.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    blair_corr.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
    blair_corr.loc["t_init", "t_init"] = 1.0
    blair_corr.loc["thick", "thick"] = 1.0
    print(f"blair_corr={blair_corr}")

    # Transform to an array for output
    cov_out = blair_cov.values
    corr_out = blair_corr.values
    for m, par_key in enumerate(cme_par_keys_ordered):
        # Remove any covariance from variables that do not need to be perturbed
        if par_key not in vars_req:
            cov_out[m, :] = 0
            cov_out[:, m] = 0

            corr_out[m, :] = 0
            corr_out[:, m] = 0

    plot_sample_cov(
        cov_out,
        corr_out,
        "Blair's list",
        vars_req=["t_init", "v", "width", "lon", "lat", "thick"],
    )
    return cov_out, corr_out


def make_blair_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        blair_file_path: str,
        rng,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0, sd_thick: float=0
) -> npt.NDArray[float]:
    """
    Generate uncorrelated covariance matrix
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters to perturb around
    :param blair_file_path: Path to Blair's CME list file
    :param rng: Random number generator
    :param vars_req: List of variables that need to be perturbed in ensemble
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_thick: Standard deviation of CME thickness
    :return: blair_samples: Ensemble members made with uncorrelated covariance matrix
    """
    blair_samples: npt.NDArray[float] = np.zeros((6, n_ens))

    blair_tuple: tuple[npt.NDArray[float], npt.NDArray[float]] = make_cov_blair(
        blair_file_path=blair_file_path,
        vars_req=vars_req,
        sd_t_init=sd_t_init,
        sd_thick=sd_thick
    )
    blair_cov: npt.NDArray[float] = blair_tuple[0]
    blair_corr: npt.NDArray[float] = blair_tuple[1]
    print(f"blair_cov = {blair_cov}")

    for m in range(n_ens):
        blair_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=blair_cov
        )

    print(blair_samples)
    return blair_samples


def make_donki_cov(
        start_time: datetime.datetime,
        end_time:datetime.datetime,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0,
        sd_thick: float=0,
        most_acc_only: str="true",
        catalog: str="ALL",
        feature: str="LE"
):
    # Use routine in huxt.huxt_inputs to retrieve DONKI ConeCME parameters
    donki_cone_cme_dict = huxt_inputs.get_DONKI_coneCMEs(
        startdate=start_time,
        enddate=end_time,
        mostAccOnly=most_acc_only,
        catalog=catalog,
        feature=feature
    )

    # Put CME parameters into a dataframe
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

        dfCMEpar.loc[i, "v"] = np.log(donki_cone_cme_dict[i]["vcld"])
        dfCMEpar.loc[i, "width"] = 2 * donki_cone_cme_dict[i]["rmajor"]
        dfCMEpar.loc[i, "lon"] = donki_cone_cme_dict[i]["lon"]
        dfCMEpar.loc[i, "lat"] = donki_cone_cme_dict[i]["lat"]
        #dfCMEpar.loc[i, "thick"] = donki_cone_cme_dict[i]["vcld"]


    cme_par_keys_ordered = cme_par_key_to_index().keys()
    list_req_order = [
        r for r in cme_par_keys_ordered if r in dfCMEpar.columns
    ]
    print(list_req_order)
    dfCMEpar = dfCMEpar[list_req_order]
    print(f"dfCMEpar={dfCMEpar}")

    # Calculate covariance matrix from dfCMEpar
    donki_cov = dfCMEpar.cov(numeric_only=False)
    print(f"donki_cov={donki_cov}")

    # Add in t_init and thick rows
    donki_cov = donki_cov.reindex(cme_par_keys_ordered, fill_value=0.0)
    donki_cov.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    donki_cov.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
    donki_cov.loc["t_init", "t_init"] = sd_t_init * sd_t_init
    donki_cov.loc["thick", "thick"] = sd_thick * sd_thick

    print(f"donki_cov={donki_cov}")
    donki_corr = dfCMEpar.corr(numeric_only=False)

    # Add in t_init and thick rows
    donki_corr = donki_corr.reindex(cme_par_keys_ordered, fill_value=0.0)
    donki_corr.insert(loc=cme_par_get_indices("t_init"), column="t_init", value=0.0)
    donki_corr.insert(loc=cme_par_get_indices("thick"), column="thick", value=0.0)
    donki_corr.loc["t_init", "t_init"] = 1.0
    donki_corr.loc["thick", "thick"] = 1.0
    print(f"donki_corr={donki_corr}")

    # Transform to an array for output
    cov_out = donki_cov.values
    corr_out = donki_corr.values
    for m, par_key in enumerate(cme_par_keys_ordered):
        # Remove any covariance from variables that do not need to be perturbed
        if par_key not in vars_req:
            cov_out[m, :] = 0
            cov_out[:, m] = 0

            corr_out[m, :] = 0
            corr_out[:, m] = 0

    print(f"cov_out={cov_out}")
    print(f"corr_out={corr_out}")

    plot_sample_cov(
        cov_out,
        corr_out,
        "DONKI",
        vars_req=["t_init", "v", "width", "lon", "lat", "thick"],
    )
    return cov_out, corr_out


def make_donki_samples(
        n_ens: int,
        mean_cme_pars: list[float],
        rng,
        start_time: datetime.datetime,
        end_time:datetime.datetime,
        vars_req: npt.NDArray[str]=np.array(["t_init", "v", "width", "lon", "lat", "thick"]),
        sd_t_init: float=0,
        sd_thick: float=0,
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
    :param vars_req: List of variables that need to be perturbed in ensemble
    :param sd_t_init: Standard deviation of CME launch time
    :param sd_thick: Standard deviation of CME thickness
    :return: blair_samples: Ensemble members made with uncorrelated covariance matrix
    """
    donki_samples: npt.NDArray[float] = np.zeros((6, n_ens))

    donki_tuple: tuple[npt.NDArray[float], npt.NDArray[float]] = make_donki_cov(
        start_time,
        end_time,
        vars_req=vars_req,
        sd_t_init=sd_t_init,
        sd_thick=sd_thick,
        most_acc_only=most_acc_only,
        catalog=catalog,
        feature=feature
    )
    """make_cov_blair(
        blair_file_path=blair_file_path,
        vars_req=vars_req,
        sd_t_init=sd_t_init,
        sd_thick=sd_thick
    ))"""
    donki_cov: npt.NDArray[float] = donki_tuple[0]
    donki_corr: npt.NDArray[float] = donki_tuple[1]
    print(f"donki_cov = {donki_cov}")

    mean_cme_pars[cme_par_get_indices("v")]: list[float] = np.log(mean_cme_pars[cme_par_get_indices("v")])
    for m in range(n_ens):
        donki_samples[:, m] = rng.multivariate_normal(
            mean=mean_cme_pars, cov=donki_cov
        )

    # Take exponential of speed values
    v_index = cme_par_get_indices("v")
    donki_samples[v_index, :]: npt.NDArray[float] = np.exp(donki_samples[v_index, :])
    print(donki_samples)

    return donki_samples


def main(
        uncorr_samp: bool=True,
        blair_samp: bool=True,
        donki_samp: bool=True
):
    n_ens: int = 3
    mean_cme_pars: list[float] = [0.1, 1000.2, 2000.3, 3000.4, 4000.5, 5000.6]
    vars_req = np.array(["t_init", "v", "width", "lon", "lat", "thick"])

    sd_t_init = 1
    sd_v = 400
    sd_width = 5
    sd_lon = 5
    sd_lat = 5
    sd_thick = 1

    if uncorr_samp:
        rng = np.random.default_rng(42)
        uncorr_samp = make_uncorrelated_samples(
            n_ens, mean_cme_pars, rng,
            sd_t_init=sd_t_init, sd_v=sd_v, sd_width=sd_width,
            sd_lon=sd_lon, sd_lat=sd_lat, sd_thick=sd_thick
        )
        print(f"uncorr_samp = {uncorr_samp}")


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
            sd_t_init=1,
            sd_thick=1
        )

        print(f"blair_samples = {blair_samples}")

    if donki_samp:
        start_time = datetime.datetime(2011, 1, 1, 0, 0, 0)
        end_time = datetime.datetime(2025, 12, 31, 23, 59, 59)
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
            vars_req=vars_req,
            sd_t_init=sd_t_init,
            sd_thick=sd_thick,
            most_acc_only = most_acc_only,
            catalog = catalog,
            feature = feature
        )
        print(f"donki_samples = {donki_samples}")
    return None

if __name__ == "__main__":
    main(
        uncorr_samp=False,
        blair_samp=True,
        donki_samp=True
    )