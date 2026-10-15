"""
prior_mo_cone.py

Correlated Gaussian prior for the CME parameters with the covariance estimated from the Met Office
cone-CME files (``<file_dir>/YYYY/MM/cone_cme_YYYYMMDDHH.in``, read with surf_inputs).

Because reading many years of files is slow, the CME parameters and the covariance / correlation
matrices are cached as netCDF files in ``mo_cone_cov_dir``:
    mo_cone_par_<start>-<end>.nc, mo_cone_cme_cov_<start>-<end>.nc, mo_cone_corr_<start>-<end>.nc
(NB: the cache file names do not record use_log_v, so delete them / set overwrite_cov=True after
changing it.)

As for the DONKI prior, ``scale_corr=True`` rescales the correlation matrix by the user's standard
deviations, and longitude is always decorrelated from the other parameters.
"""
import datetime
import glob
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from numpy.random import Generator

from surf import surf_inputs

from cme_par_dict_structure import cme_par_get_indices, cme_par_key_to_index
from prior_cov_plots import plot_sample_cov, plot_pair_plot


def filter_first_mo_cone_cme(
        file_dir: str,
        start_time: datetime.datetime,
        path_list: list[str],
) -> list[str]:
    """
    Append the ConeCME files of start_time's month that are at or after start_time
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
    Append the ConeCME files of end_time's month that are at or before end_time
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
    List the ConeCME files between start_time and end_time when both are in the same month
    :param file_dir: Directory containing CME files
    :param start_time: Datetime object of start time
    :param end_time: Datetime object of end time (same year and month as start_time)
    :return: path_list: List of paths to CME files in the range
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
    List all Met Office Cone CME files between start_time and end_time (inclusive), walking the
    YYYY/MM directory tree: partial first and last months are filtered by file date, intermediate
    months are taken whole.
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
        mo_cone_cov_dir: str | None = None,
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
        Defaults to the directory containing this module
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

    # If mo_cone_cov_dir is not specified, use the directory containing this module
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

    # Three cases:
    #   overwrite_cov (or no cached parameter file): read all cone files, compute and cache everything
    #   parameters cached but not the matrices:      recompute the matrices from the cached parameters
    #   everything cached:                           read the cached matrices
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
        #  (context manager so the file handle is released once the data are in memory)
        with xr.open_dataset(mo_cone_par_file_name) as mo_cone_cme_ds:
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
        #  (context managers so the file handles are released once the data are in memory;
        #   previously three handles were left open on every call, i.e. every DA run)
        with xr.open_dataset(mo_cone_par_file_name) as mo_cone_cme_ds:
            mo_cone_cme_df = mo_cone_cme_ds.to_pandas()

        # Open premade nc files containing cme parameter covariance and correlation matrices
        with xr.open_dataset(cov_out_file_name) as mo_cone_cme_cov_ds:
            mo_cone_cme_cov = mo_cone_cme_cov_ds.to_pandas()
        with xr.open_dataset(corr_out_file_name) as mo_cone_cme_corr_ds:
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
        rng: Generator,
        mo_cone_file_dir: str,
        start_time: datetime.datetime,
        end_time:datetime.datetime,
        overwrite_cov: bool=False,
        mo_cone_cov_dir: str | None = None,
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
    Draw Gaussian samples around mean_cme_pars with the covariance from the Met Office cone-CME files
    :param n_ens: Number of ensemble members required
    :param mean_cme_pars: Mean CME parameters to perturb around
    :param rng: Random number generator
    :param overwrite_cov: Boolean to determine whether to overwrite existing covariance matrix
    :param mo_cone_cov_dir: Directory containing precalculated cme_pars and covariance/correlation matrices
        Defaults to the directory containing this module
    :param scale_corr: Boolean to determine whether to scale the correlation
        matrix to make the covariance matrix
    :param mo_cone_file_dir: Path to the Met Office's Cone CME files
    :param start_time: Start time
    :param end_time: End time
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
    :return: mo_cone_samples: (6, n_ens) array of ensemble members
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

    # Only work in log-speed if use_log_v (previously the mean speed was always logged,
    #  but only exponentiated back when use_log_v=True -> CME speeds of ~log(v) ~ 7 km/s otherwise)
    mean_cme_pars = list(mean_cme_pars)
    if use_log_v:
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
