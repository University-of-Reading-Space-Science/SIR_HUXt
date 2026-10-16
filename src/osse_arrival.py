"""
osse_arrival.py

CME arrival-time / arrival-speed analysis of data-assimilation output: run the forward model for the
prior and posterior ensembles (and, for OSSEs, the truth CME), find when each CME reaches the target
body, and summarise / plot / save the statistics (``cme_arrival_stats.json`` and
``histogram_arrival_time_error.png`` in each run directory).
"""
import datetime
import json
import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import huxt.huxt as H
import surf.surf as S

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.time import Time

from osse_stats import weighted_median
from timing import timed


def _ds_scalar(ds: xr.Dataset, name: str) -> float:
    """A scalar stored in ds (its first element if the dataset has a run dimension)."""
    return float(np.ravel(ds[name].values)[0])


def launch_delay_to_r_min(ds: xr.Dataset, speed: u.Quantity) -> u.Quantity:
    """
    Time for a CME moving at constant ``speed`` to travel from cme_init_rad (where t_init is defined) to
    the model inner boundary r_min. The data assimilation launches CMEs at t_init + this delay
    (ObservationOperator.extract_cme_launch_time_and_speed), so the arrival analysis must too.
    :param ds: DA output dataset (gives r_min and cme_init_rad, in solar radii)
    :param speed: CME speed(s)
    :return: Delay in seconds (same shape as speed)
    """
    dist = ((_ds_scalar(ds, "r_min") - _ds_scalar(ds, "cme_init_rad")) * u.solRad).to(u.km)
    return (dist / speed.to(u.km / u.s)).to(u.s)


def solved_cmes(model: S.SURF | S.SURF3d | H.HUXt) -> list:
    """
    The solved CMEs of a model. SURF3d has no ``cmes`` attribute (each latitude slice holds its own),
    so for SURF3d the slice closest to Earth's latitude is used.
    """
    if hasattr(model, "SURFlat"):
        earth_lat = np.mean(model.get_observer("EARTH").lat.to(u.rad).value)
        slice_lats = np.array([m.latitude.to(u.rad).value for m in model.SURFlat])
        return model.SURFlat[int(np.argmin(np.abs(slice_lats - earth_lat)))].cmes
    return model.cmes


def get_prior_post_cme_pars_from_ds(ds: xr.Dataset, runNo: int = 0) -> tuple[dict, dict]:
    """
    Function to retrieve prior and posterior cme parameters: [t_init, v, width, lon, lat, thick]
      from dataset (and the posterior logarithm of the weights)
    :param ds: Dataset containing all CME parameters from data assimilation run
    :param runNo: Run index to extract if ds contains multiple runs (previously undefined here)
    :return: prior_dict, Dictionary containing the prior CME parameters extracted
    :return: posterior_dict, Dictionary containing the posterior CME parameters extracted plus the log_weights
    """
    # Get the number of ensemble members from ds
    n_members = len(ds["ens_no"].values)

    # Extract the prior and posterior variables, accounting for scenarios
    #  where multiple runs are being performed
    try:
        prior_t_init_in = ds["t_init"][0, :].values
        prior_v_in = ds["v"][0, :].values
        prior_width_in = ds["width"][0, :].values
        prior_lon_in = ds["lon"][0, :].values
        prior_lat_in = ds["lat"][0, :].values
        prior_thick_in = ds["thick"][0, :].values

        post_t_init_in = ds["t_init"][-1, :].values
        post_v_in = ds["v"][-1, :].values
        post_width_in = ds["width"][-1, :].values
        post_lon_in = ds["lon"][-1, :].values
        post_lat_in = ds["lat"][-1, :].values
        post_thick_in = ds["thick"][-1, :].values

        log_weights_in = ds["log_weight"][-1, :].values
    except:
        prior_t_init_in = ds["t_init"][runNo, 0, :].values
        prior_v_in = ds["v"][runNo, 0, :].values
        prior_width_in = ds["width"][runNo, 0, :].values
        prior_lon_in = ds["lon"][runNo, 0, :].values
        prior_lat_in = ds["lat"][runNo, 0, :].values
        prior_thick_in = ds["thick"][runNo, 0, :].values

        # Posterior is the final observation index (previously index 0, i.e. the prior)
        post_t_init_in = ds["t_init"][runNo, -1, :].values
        post_v_in = ds["v"][runNo, -1, :].values
        post_width_in = ds["width"][runNo, -1, :].values
        post_lon_in = ds["lon"][runNo, -1, :].values
        post_lat_in = ds["lat"][runNo, -1, :].values
        post_thick_in = ds["thick"][runNo, -1, :].values

        log_weights_in = ds["log_weight"][runNo, -1, :].values
    else:
        prior_t_init_in = ds["t_init"][0, :].values
        prior_v_in = ds["v"][0, :].values
        prior_width_in = ds["width"][0, :].values
        prior_lon_in = ds["lon"][0, :].values
        prior_lat_in = ds["lat"][0, :].values
        prior_thick_in = ds["thick"][0, :].values

        post_t_init_in = ds["t_init"][-1, :].values
        post_v_in = ds["v"][-1, :].values
        post_width_in = ds["width"][-1, :].values
        post_lon_in = ds["lon"][-1, :].values
        post_lat_in = ds["lat"][-1, :].values
        post_thick_in = ds["thick"][-1, :].values

        log_weights_in = ds["log_weight"][-1, :].values

    sec_to_cme_prior = [
        float(prior_t_init_in[i]) * u.s for i in range(n_members)
    ]
    v_prior = [prior_v_in[i] * u.km / u.s for i in range(n_members)]
    width_prior = [prior_width_in[i] * u.deg for i in range(n_members)]
    lon_prior = [prior_lon_in[i] * u.deg for i in range(n_members)]
    lat_prior = [prior_lat_in[i] * u.deg for i in range(n_members)]
    thick_prior = [prior_thick_in[i] * u.solRad for i in range(n_members)]

    sec_to_cme_post = [
        float(post_t_init_in[i]) * u.s for i in range(n_members)
    ]
    v_post = [post_v_in[i] * u.km / u.s for i in range(n_members)]
    width_post = [post_width_in[i] * u.deg for i in range(n_members)]
    lon_post = [post_lon_in[i] * u.deg for i in range(n_members)]
    lat_post = [post_lat_in[i] * u.deg for i in range(n_members)]
    thick_post = [post_thick_in[i] * u.solRad for i in range(n_members)]
    log_weights_post = [log_weights_in[i] for i in range(n_members)]
    weights_post = np.array([
        np.exp(log_weights_post[i]) for i in range(n_members)
    ])

    # Make dictionaries for output
    prior_dict_out = {
        "t_init": sec_to_cme_prior,
        "v": v_prior,
        "width": width_prior,
        "lon": lon_prior,
        "lat": lat_prior,
        "thick": thick_prior,
    }

    post_dict_out = {
        "t_init": sec_to_cme_post,
        "v": v_post,
        "width": width_post,
        "lon": lon_post,
        "lat": lat_post,
        "thick": thick_post,
        "log_weights": log_weights_post,
        "weights": weights_post,
    }

    return prior_dict_out, post_dict_out


def get_true_cme_arrival_time(
        ds: xr.Dataset,
        use_model: str,
        true_cme_t_init: datetime.datetime,
        true_cme_speed: float,
        true_cme_width: float,
        true_cme_lat: float,
        true_cme_lon: float,
        true_cme_thick: float,
        cme_hit_object: str = "EARTH"
) -> tuple[datetime.datetime | float, float]:
    """
    Arrival time and speed of the truth CME (OSSEs), from a 5-day run of the model with the ambient
    wind / inner boundary stored in ds.
    :param ds: DA output dataset (gives model_init_time, r_min and ambient_vr)
    :param use_model: "surf", "compress_surf" or "huxt"
    :param true_cme_*: Truth CME parameters (t_init as a datetime; speed km/s; angles deg; thick rS)
    :param cme_hit_object: Body to compute the arrival at
    :return: cme_arrival_time (datetime, or NaN if it misses), cme_arrival_speed (km/s, or NaN)
    """
    print(ds["model_init_time"].values)
    try:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s
    except:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values[0]), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values[0] * u.solRad
        v_bound_val = ds["ambient_vr"].values[0] * u.km / u.s
    else:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s

    print(start_time)
    start_time_astro = Time(start_time, format='datetime')
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))

    if use_model in ["surf", "compress_surf"]:
        ert = S.Observer("EARTH", start_time_astro)
    elif use_model in ["huxt"]:
        ert = H.Observer("EARTH", start_time_astro)
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    lon_start = 300 * u.deg
    lon_end = 420 * u.deg

    # Initialise list
    cme_arrival_prior = []
    cme_arr_prior_speed = []
    cme_arrival_post = []
    cme_arr_post_speed = []

    n_members = len(ds["ens_no"].values)

    if use_model in ["surf"]:
        model: S.SURF = S.SURF(
            v_boundary=v_bound_val,#ds["ambient_vr"].values * u.km / u.s,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_value,#ds["r_min"].values * u.solRad,
            solver='huxt',
            # accel_limit=accel_limit
        )
    elif use_model in ["compress_surf"]:
        model: S.SURF = S.SURF(
            v_boundary=v_bound_val,#ds["ambient_vr"].values * u.km / u.s,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_value,#ds["r_min"].values * u.solRad,
            solver='hydro',
            # accel_limit=accel_limit
        )
    elif use_model in ["huxt"]:
        model: H.HUXt = H.HUXt(
            v_boundary=v_bound_val,#ds["ambient_vr"].values * u.km / u.s,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_value,#ds["r_min"].values * u.solRad,
            # accel_limit=accel_limit
        )
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    # Generate CME object
    sec_to_cme_true = float(
        (true_cme_t_init - start_time).total_seconds()
    ) * u.s
    v_true = float(true_cme_speed) * u.km / u.s
    # t_init is defined at cme_init_rad: launch from r_min as the data assimilation does
    sec_to_cme_true = sec_to_cme_true + launch_delay_to_r_min(ds, v_true)
    width_true = float(true_cme_width) * u.deg
    lon_true =  float(true_cme_lon) * u.deg
    lat_true =  float(true_cme_lat) * u.deg
    thick_true = float(true_cme_thick) * u.solRad

    if use_model in ["surf", "compress_surf"]:
        cme_object: S.ConeCME = S.ConeCME(
            t_launch=sec_to_cme_true,
            v=v_true,
            width=width_true,
            longitude=lon_true,
            latitude=lat_true,
            thickness=thick_true,
            cme_fixed_duration=True,
            fixed_duration=12 * 3600 * u.s
        )

    elif use_model in ["huxt"]:
        cme_object: H.ConeCME = H.ConeCME(
            t_launch=sec_to_cme_true,
            v=v_true,
            width=width_true,
            longitude=lon_true,
            latitude=lat_true,
            thickness=thick_true,
            cme_fixed_duration=True,
            fixed_duration=12 * 3600 * u.s
        )
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    # Make true CME arrival time
    model.solve([cme_object])
    cme = solved_cmes(model)[0]
    true_stats = cme.compute_arrival_at_body(cme_hit_object)

    if true_stats['hit']:
        cme_arrival_time = true_stats['t_arrive'].datetime
        cme_arrival_speed = (true_stats["v"].to(u.km / u.s)).value
    else:
        cme_arrival_time = np.nan
        cme_arrival_speed = np.nan

    print(f"True CME arrival time at {cme_hit_object}: {cme_arrival_time}")
    print(f"True CME arrival speed at {cme_hit_object}: {cme_arrival_speed}")

    return cme_arrival_time, cme_arrival_speed


def get_cme_stats_prior_post(
    ens_no: int,
    n_members: int,
    model: S.SURF | H.HUXt,
    prior_cme: S.ConeCME | H.ConeCME,
    post_cme: S.ConeCME | H.ConeCME,
    cme_hit_object: str
) -> tuple:
    """
    Function to output CME stats + arrival time and speed, primarily for use with parallel application
    :param ens_no: Current ensemble member
    :param n_members: Total number of members
    :param model: SURF/HUXt model to solve for
    :param prior_cme: prior_cme_object corresponding to ensemble member, ens_no
    :param post_cme: posterior_cme_object corresponding to ensemble member, ens_no
    :param cme_hit_object: Object that CME is projected to hit
    :return: (prior arrival time, prior arrival speed, prior hit (0/1), prior stats dict,
              posterior arrival time, posterior arrival speed, posterior hit (0/1), posterior stats dict);
              times/speeds are NaN for a miss
    """
    if np.mod(ens_no, 1) == 0:
        print(f"{ens_no}/{n_members}")

    # Make prior CME arrival times
    print(f"prior_cme: {prior_cme}")
    with timed("arrival.solve_prior_member"):
        model.solve([prior_cme])
    cme = solved_cmes(model)[0]
    prior_stats = cme.compute_arrival_at_body(cme_hit_object)

    if prior_stats['hit']:
        cme_arrival_prior = prior_stats['t_arrive'].datetime
        cme_arr_prior_speed = (prior_stats["v"].to(u.km / u.s)).value
        prior_count_hit = 1
    else:
        cme_arrival_prior = np.nan
        cme_arr_prior_speed = np.nan
        prior_count_hit = 0

    # Make posterior CME arrival times
    with timed("arrival.solve_posterior_member"):
        model.solve([post_cme])
    cme = solved_cmes(model)[0]
    post_stats = cme.compute_arrival_at_body(cme_hit_object)

    if post_stats['hit']:
        cme_arrival_post = post_stats['t_arrive'].datetime
        cme_arr_post_speed = (post_stats["v"].to(u.km / u.s)).value
        post_count_hit = 1
    else:
        cme_arrival_post = np.nan
        cme_arr_post_speed = np.nan
        post_count_hit = 0

    print(
        f"cme_arrival_prior={cme_arrival_prior},\ncme_arr_prior_speed={cme_arr_prior_speed},\n"
        f" prior_count_hit={prior_count_hit},\n prior_stats={prior_stats},\n,"
        f"cme_arrival_post={cme_arrival_post},\n cme_arr_post_speed={cme_arr_post_speed},\n"
        f" post_count_hit={post_count_hit},\n post_stats={post_stats}"
    )
    return (
        cme_arrival_prior, cme_arr_prior_speed, prior_count_hit, prior_stats,
        cme_arrival_post, cme_arr_post_speed, post_count_hit, post_stats
    )


@timed("arrival.get_cme_arrival_time")
def get_cme_arrival_time(
        ds: xr.Dataset,
        use_model: str,
        runNo: int = 0,
        real_arrival_time: datetime.datetime | None = None,
        cme_hit_object: str = "EARTH",
        base_file_path: str | None = None,
        parallel_ens: bool = False,
) -> None:
    """
    Arrival-time / speed statistics of the prior and posterior ensembles of one run.

    Steps: build the model from the ambient wind stored in ds; make prior (obs index 0) and posterior
    (last obs index) cone CMEs for every member; solve each and record its arrival at cme_hit_object
    (in parallel over members if parallel_ens); compute hit rates and the (weighted) mean / median /
    standard deviation of arrival time and speed. If real_arrival_time is given, also compute arrival
    errors, plot their histograms and save everything to <base_file_path>/run_NNN/cme_arrival_stats.json.

    :param ds: DA output dataset
    :param use_model: "surf", "compress_surf", "compress_surf3d" or "huxt"
    :param runNo: Run to analyse (indexes ds if it has a run dimension; used in the output path)
    :param real_arrival_time: Observed (or truth) arrival time
    :param cme_hit_object: Body to compute the arrival at
    :param base_file_path: If None show the histogram, otherwise save outputs under this directory
    :param parallel_ens: Solve the ensemble members in parallel
    """
    print(ds["model_init_time"].values)
    try:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s
    except:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values[0]), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values[0] * u.solRad
        v_bound_val = ds["ambient_vr"].values[0] * u.km / u.s
    else:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s

    print(start_time)
    start_time_astro = Time(start_time, format='datetime')
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))

    if use_model in ["surf", "compress_surf", "compress_surf3d"]:
        ert = S.Observer("EARTH", start_time_astro)
    elif use_model in ["huxt"]:
        ert = H.Observer("EARTH", start_time_astro)
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    lon_start = 300 * u.deg
    lon_end = 420 * u.deg

    n_members = len(ds["ens_no"].values)


    # Initialise list
    cme_arrival_prior = [None for _ in range(n_members)]
    cme_arr_prior_speed = [None for _ in range(n_members)]
    cme_arrival_post = [None for _ in range(n_members)]
    cme_arr_post_speed = [None for _ in range(n_members)]

    if use_model in ["huxt", "surf", "compress_surf"]:
        try:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad
        except:
            v_bound_in = ds["ambient_vr"].values * u.km / u.s
            r_min_in = ds["r_min"].values * u.solRad
        else:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad
    elif use_model in ["compress_surf3d"]:
        try:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            vr_longs_in = ds["vr_longs"].values[runNo, :, :] * u.rad
            vr_lats_in = ds["vr_lats"].values[runNo, :, :] * u.rad
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad
        except:
            v_bound_in = ds["ambient_vr"].values * u.km / u.s
            # (stored without units; the WSA map coordinates are in radians)
            vr_longs_in = ds["vr_longs"].values * u.rad
            vr_lats_in = ds["vr_lats"].values * u.rad
            r_min_in = ds["r_min"].values * u.solRad
        else:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            vr_longs_in = ds["vr_longs"].values[runNo, :, :] * u.rad
            vr_lats_in = ds["vr_lats"].values[runNo, :, :] * u.rad
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad

    if use_model in ["surf"]:
        model: S.SURF = S.SURF(
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            solver='huxt',
            # accel_limit=accel_limit
        )
    elif use_model in ["compress_surf"]:
        model: S.SURF = S.SURF(
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=3 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            solver='hydro',
            # accel_limit=accel_limit
        )
    elif use_model in ["compress_surf3d"]:
        # TODO: Need to remove hard-coded numbers here
        model: S.SURF3d = S.SURF3d(
            v_map=v_bound_in,
            v_map_long=vr_longs_in,
            v_map_lat=vr_lats_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude_min=-15 * u.deg,
            latitude_max=15 * u.deg,
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=3 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            solver='hydro',
        )
    elif use_model in ["huxt"]:
        model: H.HUXt = H.HUXt(
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=3 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            # accel_limit=accel_limit
        )
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf', 'compress_surf3d' or 'huxt'")

    # Extract the CME parameters from the xr.Dataset
    prior_dict, post_dict = get_prior_post_cme_pars_from_ds(ds, runNo=runNo)
    # (u.Quantity keeps the units; np.array of a list of Quantities would not)
    sec_to_cme_prior = u.Quantity(prior_dict["t_init"]).to(u.s)
    v_prior = u.Quantity(prior_dict["v"])
    width_prior = u.Quantity(prior_dict["width"])
    lon_prior = u.Quantity(prior_dict["lon"])
    lat_prior = u.Quantity(prior_dict["lat"])
    thick_prior = u.Quantity(prior_dict["thick"])

    v_post = u.Quantity(post_dict["v"])
    # t_init is defined at cme_init_rad: launch from r_min as the data assimilation does
    sec_to_cme_prior = sec_to_cme_prior + launch_delay_to_r_min(ds, v_prior)
    sec_to_cme_post = u.Quantity(post_dict["t_init"]).to(u.s) + launch_delay_to_r_min(ds, v_post)
    width_post = post_dict["width"]
    lon_post = post_dict["lon"]
    lat_post = post_dict["lat"]
    thick_post = post_dict["thick"]
    log_weights_post = post_dict["log_weights"]
    weights_post = post_dict["weights"]

    print(f"t_init: {np.shape(sec_to_cme_prior.value)}, {sec_to_cme_prior}")
    print(f"v_prior: {np.shape(v_prior.value)}, {v_prior}")
    print(f"width_prior: {np.shape(width_prior.value)}, {width_prior}")
    print(f"lon_prior: {np.shape(lon_prior)}, {lon_prior}")
    print(f"lat_prior: {np.shape(lat_prior)}, {lat_prior}")
    print(f"thick_prior: {np.shape(thick_prior)}, {thick_prior}")

    print(f"v_post: {v_post}")
    print(f"width_post: {width_post}")

    if use_model in ["surf", "compress_surf", "compress_surf3d"]:
        prior_cme_objects: list[S.ConeCME] = [
            S.ConeCME(
                t_launch=sec_to_cme_prior[i],
                v=v_prior[i],
                width=width_prior[i],
                longitude=lon_prior[i],
                latitude=lat_prior[i],
                thickness=thick_prior[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]

        post_cme_objects: list[S.ConeCME] = [
            S.ConeCME(
                t_launch=sec_to_cme_post[i],
                v=v_post[i],
                width=width_post[i],
                longitude=lon_post[i],
                latitude=lat_post[i],
                thickness=thick_post[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]
    elif use_model in ["huxt"]:
        prior_cme_objects: list[H.ConeCME] = [
            H.ConeCME(
                t_launch=sec_to_cme_prior[i],
                v=v_prior[i],
                width=width_prior[i],
                longitude=lon_prior[i],
                latitude=lat_prior[i],
                thickness=thick_prior[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]

        post_cme_objects: list[H.ConeCME] = [
            H.ConeCME(
                t_launch=sec_to_cme_post[i],
                v=v_post[i],
                width=width_post[i],
                longitude=lon_post[i],
                latitude=lat_post[i],
                thickness=thick_post[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    prior_count_hit_list = np.zeros(n_members)
    post_count_hit_list = np.zeros(n_members)
    prior_stat_dict = {}
    post_stat_dict = {}

    if parallel_ens:
        def log_result(result):
            # This is called whenever get_cme_stats_prior_post
            #  returns a result in parallel computation.
            # result_list is modified only by the main process, not the pool workers.
            result_list.append(result)

        print("Use parallel execution on ensemble.")

        # Initialise variables required for parallel computations
        result_list = []
        obj_result = [None] * n_members

        n_proc = int(os.cpu_count() - 1)
        if n_proc < 1:
            n_proc = 1
        pool = Pool(processes=n_proc)
        try:

            for i in range(n_members):
                obj_result[i] = pool.apply_async(
                    get_cme_stats_prior_post,
                    kwds={
                        'ens_no': i,
                        'n_members': n_members,
                        'model': model,
                        'prior_cme': prior_cme_objects[i],
                        'post_cme': post_cme_objects[i],
                        'cme_hit_object': cme_hit_object,
                    },
                    callback=log_result,
                )

            pool.close()
            pool.join()
        except BaseException:
            # Release the worker processes if anything fails (or on Ctrl+C)
            pool.terminate()
            pool.join()
            raise

        for i in range(n_members):
            obs_tuple = obj_result[i].get()

            cme_arrival_prior[i] = obs_tuple[0]
            cme_arr_prior_speed[i] = obs_tuple[1]
            prior_count_hit_list[i] = obs_tuple[2]
            prior_stat_dict[f"{i}"] = obs_tuple[3]

            cme_arrival_post[i] = obs_tuple[4]
            cme_arr_post_speed[i] = obs_tuple[5]
            post_count_hit_list[i] = obs_tuple[6]
            post_stat_dict[f"{i}"] = obs_tuple[7]

        prior_count_hit = np.sum(prior_count_hit_list)
        post_count_hit = np.sum(post_count_hit_list)

    else:
        print("Use serial execution on ensemble.")


        for i in range(n_members):
            obs_tuple = get_cme_stats_prior_post(
                ens_no=i,
                n_members=n_members,
                model=model,
                prior_cme=prior_cme_objects[i],
                post_cme=post_cme_objects[i],
                cme_hit_object=cme_hit_object,
            )

            cme_arrival_prior[i] = obs_tuple[0]
            cme_arr_prior_speed[i] = obs_tuple[1]
            prior_count_hit_list[i] = obs_tuple[2]
            prior_stat_dict[f"{i}"] = obs_tuple[3]

            cme_arrival_post[i] = obs_tuple[4]
            cme_arr_post_speed[i] = obs_tuple[5]
            post_count_hit_list[i] = obs_tuple[6]
            post_stat_dict[f"{i}"] = obs_tuple[7]

        prior_count_hit = np.sum(prior_count_hit_list)
        post_count_hit = np.sum(post_count_hit_list)

    prior_hit_rate = prior_count_hit / n_members
    post_hit_rate = post_count_hit / n_members

    print(f"Cme_arrival_prior = {cme_arrival_prior}")
    print(f"Cme_arr_prior_speed = {cme_arr_prior_speed}")
    print(f"cme_arrival_post = {cme_arrival_post}")
    print(f"cme_arr_post_speed = {cme_arr_post_speed}")

    # Arrival times as seconds after 2000-01-01 (NaN for a miss), so they can be averaged
    base_time = datetime.datetime(2000, 1, 1, 0, 0, 0)
    seconds_arrival_prior = np.array([
        (cme_arrival_prior[i] - base_time).total_seconds()
        if isinstance(cme_arrival_prior[i], datetime.datetime) else np.nan
        for i in range(n_members)
    ])
    seconds_arrival_post = np.array([
        (cme_arrival_post[i] - base_time).total_seconds()
        if isinstance(cme_arrival_post[i], datetime.datetime) else np.nan
        for i in range(n_members)
    ])
    cme_arr_prior_speed = np.array(cme_arr_prior_speed)
    cme_arr_post_speed = np.array(cme_arr_post_speed)

    print(f"prior_CME_hit_rate = {prior_hit_rate * 100}%")
    if prior_hit_rate > 1e-7:
        mean_prior_arr_sec = np.nanmean(seconds_arrival_prior)
        mean_prior_arr_speed = np.nanmean(cme_arr_prior_speed)

        median_prior_arr_sec = np.nanmedian(seconds_arrival_prior)
        median_prior_arr_speed = np.nanmedian(cme_arr_prior_speed)

        sd_prior_arr_sec = np.nanstd(seconds_arrival_prior)
        sd_prior_arr_hr = sd_prior_arr_sec / 3600.0

        mean_prior_arrival = base_time + datetime.timedelta(
            seconds=mean_prior_arr_sec
        )
        median_prior_arrival = base_time + datetime.timedelta(
            seconds=median_prior_arr_sec
        )
        sd_prior_arr_speed = np.nanstd(cme_arr_prior_speed)

        print(f"\nMean_prior_arrival = {mean_prior_arrival} +/- {sd_prior_arr_hr} hours")
        print(f"Mean_prior_speed = {mean_prior_arr_speed} +/- {sd_prior_arr_speed} km/s")
        print(f"\nMedian_prior_arrival = {median_prior_arrival}")
        print(f"Median_prior_speed = {median_prior_arr_speed} km/s")
    else:
        print(f"Prior CME not detected at {cme_hit_object}")
        mean_prior_arr_sec = np.nan
        sd_prior_arr_sec = np.nan
        mean_prior_arr_speed = np.nan
        sd_prior_arr_speed = np.nan
        median_prior_arr_sec = np.nan
        median_prior_arr_speed = np.nan

    print(f"\nPost CME_hit_rate = {post_hit_rate * 100}%")
    if post_hit_rate > 1e-7:
        ind_req = ~np.isnan(seconds_arrival_post)
        renorm_weights_post = weights_post[ind_req] / np.sum(weights_post[ind_req])

        ################################################################
        # Calculate unweighted statistics
        mean_post_arr_sec = np.nanmean(seconds_arrival_post)
        median_post_arr_sec = np.nanmedian(seconds_arrival_post)
        sd_post_arr_sec = np.nanstd(seconds_arrival_post)
        sd_post_arr_hr = sd_post_arr_sec / 3600.0

        mean_post_arrival = base_time + datetime.timedelta(
            seconds=mean_post_arr_sec
        )
        median_post_arrival = base_time + datetime.timedelta(
            seconds=median_post_arr_sec
        )
        mean_post_arr_speed = np.nanmean(cme_arr_post_speed)
        median_post_arr_speed = np.nanmedian(cme_arr_post_speed)
        sd_post_arr_speed = np.nanstd(cme_arr_post_speed)
        #################################################################

        print(f"\nMean_post_arrival = {mean_post_arrival} +/- {sd_post_arr_hr} hours")
        print(f"Mean_post_speed = {mean_post_arr_speed} +/- {sd_post_arr_speed} km/s")
        print(f"\nMedian_post_arrival_time = {median_post_arrival}")
        print(f"Median_post_speed = {median_post_arr_speed}")

        if np.sum(renorm_weights_post) != 0:
            #################################################################
            # Calculate weighted statistics
            weighted_mean_post_arr_sec = np.average(seconds_arrival_post[ind_req], weights=renorm_weights_post)
            weighted_mean_post_arrival = base_time + datetime.timedelta(
                seconds=weighted_mean_post_arr_sec
            )

            weighted_median_post_arr_sec = weighted_median(
                seconds_arrival_post[ind_req], weights=renorm_weights_post
            )

            weighted_median_post_arrival = base_time + datetime.timedelta(
                seconds=weighted_median_post_arr_sec
            )

            weighted_mean_post_arr_speed = np.average(cme_arr_post_speed[ind_req], weights=renorm_weights_post)
            weighted_median_post_arr_speed = weighted_median(
                cme_arr_post_speed[ind_req], weights=renorm_weights_post
            )
            # Calculate (biased) weighted posterior variance
            weighted_var_post_arr_sec_biased = np.average(
                (seconds_arrival_post[ind_req] - weighted_mean_post_arr_sec) ** 2,
                weights=renorm_weights_post
            )
            weighted_var_post_arr_speed_biased = np.average(
                (cme_arr_post_speed[ind_req] - weighted_mean_post_arr_speed) ** 2,
                weights=renorm_weights_post
            )

            # Transform to unbiased variance estimator
            n_non_nan_samples = np.sum(ind_req)
            b_factor = n_non_nan_samples / (n_non_nan_samples - 1)
            weighted_var_post_arr_sec_unbiased = b_factor * weighted_var_post_arr_sec_biased
            weighted_var_post_arr_speed_unbiased = b_factor * weighted_var_post_arr_speed_biased

            # Calculate weighted standard deviation
            weighted_sd_post_arr_sec = np.sqrt(weighted_var_post_arr_sec_unbiased)
            weighted_sd_post_arr_hr = weighted_sd_post_arr_sec / 3600.0
            weighted_sd_post_arr_speed = np.sqrt(weighted_var_post_arr_speed_unbiased)
            ##################################################################

            print(f"\nWeighted_mean_post_arrival = {weighted_mean_post_arrival} +/- {weighted_sd_post_arr_hr} hours")
            print(f"Weighted_mean_post_speed = {weighted_mean_post_arr_speed} +/- {weighted_sd_post_arr_speed} km/s")
            print(f"\nWeighted_median_arrival_time = {weighted_median_post_arrival}")
            print(f"Weighted_median_post_speed = {weighted_median_post_arr_speed}")
        else:
            weighted_mean_post_arr_sec = np.nan
            weighted_median_post_arr_sec = np.nan
            weighted_sd_post_arr_sec = np.nan
            weighted_mean_post_arr_speed = np.nan
            weighted_median_post_arr_speed = np.nan
            weighted_sd_post_arr_speed = np.nan

    else:
        print(f"Posterior CME not detected at {cme_hit_object}")
        weighted_mean_post_arr_sec = np.nan
        weighted_median_post_arr_sec = np.nan
        weighted_sd_post_arr_sec = np.nan
        weighted_mean_post_arr_speed = np.nan
        weighted_median_post_arr_speed = np.nan
        weighted_sd_post_arr_speed = np.nan

        mean_post_arr_sec = np.nan
        median_post_arr_sec = np.nan
        sd_post_arr_sec = np.nan
        mean_post_arr_speed = np.nan
        median_post_arr_speed = np.nan
        sd_post_arr_speed = np.nan

        renorm_weights_post = np.nan


    if real_arrival_time is not None:
        prior_arrival_error_hours = [
            ((pri_arr - real_arrival_time).total_seconds()) / 3600.0
            for pri_arr in cme_arrival_prior if isinstance(pri_arr, datetime.datetime)
        ]
        post_arrival_error_hours = [
            ((post_arr - real_arrival_time).total_seconds()) / 3600.0
            for post_arr in cme_arrival_post if isinstance(post_arr, datetime.datetime)
        ]

        if len(prior_arrival_error_hours) > 0:
            mean_prior_arrival_error_hours = np.nanmean(prior_arrival_error_hours)
            median_prior_arrival_error_hours = np.nanmedian(prior_arrival_error_hours)
        else:
            mean_prior_arrival_error_hours = np.nan
            median_prior_arrival_error_hours = np.nan

        if len(post_arrival_error_hours) > 0:
            mean_post_arrival_error_hours = np.nanmean(post_arrival_error_hours)
            median_post_arrival_error_hours = np.nanmedian(post_arrival_error_hours)
        else:
            mean_post_arrival_error_hours = np.nan
            median_post_arrival_error_hours = np.nan

        weights_post_plot = [
            w
            for i, w in enumerate(weights_post)
            if isinstance(cme_arrival_post[i], datetime.datetime)
        ]

        if len(weights_post_plot) > 0:
            weights_post_plot = weights_post_plot / np.sum(weights_post_plot)
            print(np.max(weights_post_plot))
            print(f"weights_post_plot - renorm_weights_post = {weights_post_plot-renorm_weights_post}")

            weighted_mean_post_arrival_error_hours = np.average(
                post_arrival_error_hours, weights=weights_post_plot
            )
            weighted_median_post_arrival_error_hours = weighted_median(
                post_arrival_error_hours, weights=weights_post_plot
            )
        else:
            weighted_mean_post_arrival_error_hours = np.nan
            weighted_median_post_arrival_error_hours = np.nan

        print(f"\n Observed CME arrival time at {cme_hit_object}: {real_arrival_time}")
        print(f"\nPrior mean_arrival_time_error = {mean_prior_arrival_error_hours} hours")
        print(f"Post mean_arrival_time_error = {mean_post_arrival_error_hours} hours")
        print(f"\nPrior Median_arrival_time_error = {median_prior_arrival_error_hours} hours")
        print(f"Post Median_arrival_time_error = {median_post_arrival_error_hours} hours")
        print(f"\nWeighted Mean_arrival_time_error = {weighted_mean_post_arrival_error_hours} hours")
        print(f"Weighted Median_arrival_time_error = {weighted_median_post_arrival_error_hours} hours")

        max_y_axis = 10 * np.ceil(n_members / 40)
        fig, ax = plt.subplots(figsize=(16, 12), nrows=2, ncols=1, sharex=True, layout="constrained")
        ax[0].hist(prior_arrival_error_hours, bins=np.arange(-10, 20.5, 1))
        ax[0].set_title(f"Prior CME Arrival Error at {cme_hit_object}", fontsize=24)
        ax[0].set_ylim([0, max_y_axis])
        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].set_ylabel("No. of particles", fontsize=20)

        ax[1].hist(post_arrival_error_hours, bins=np.arange(-10, 20.5, 1))
        ax[1].set_title(f"Posterior CME Arrival Error at {cme_hit_object}", fontsize=24)
        ax[1].set_ylim([0, max_y_axis])
        ax[1].set_ylabel("No. of particles", fontsize=20)
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[-1].set_xlabel("Arrival time error (hours)", fontsize=20)

        if base_file_path is None:
            plt.show()
            plt.close(fig)
        else:
            out_dir_path = os.path.join(base_file_path, f"run_{runNo:03d}")
            filePath = os.path.join(
                out_dir_path, "histogram_arrival_time_error.png"
            )
            plt.savefig(filePath)
            plt.close(fig)

            out_dict = {
                "prior_cme_arrival_s_from_20000101": list(seconds_arrival_prior),
                "post_cme_arrival_s_from_20000101": list(seconds_arrival_post),
                "prior_arrival_speed": list(cme_arr_prior_speed),
                "post_arrival_speed": list(cme_arr_post_speed),
                "prior_cme_hit_rate": prior_hit_rate,
                "post_cme_hit_rate": post_hit_rate,
                "mean_prior_arrival": float(mean_prior_arr_sec),
                "mean_post_arrival": float(mean_post_arr_sec),
                "mean_prior_speed": float(mean_prior_arr_speed),
                "mean_post_speed": float(mean_post_arr_speed),
                "median_prior_arrival": float(median_prior_arr_sec),
                "median_post_arrival": float(median_post_arr_sec),
                "median_prior_speed": float(median_prior_arr_speed),
                "median_post_speed": float(median_post_arr_speed),
                "weighted_mean_post_arrival": float(weighted_mean_post_arr_sec),
                "weighted_mean_post_speed": float(weighted_mean_post_arr_speed),
                "weighted_median_post_arrival": float(weighted_median_post_arr_sec),
                "weighted_median_post_speed": float(weighted_median_post_arr_speed),
                "sd_prior_arrival": float(sd_prior_arr_sec),
                "sd_post_arrival": float(sd_post_arr_sec),
                "sd_prior_speed": float(sd_prior_arr_speed),
                "sd_post_speed": float(sd_post_arr_speed),
                "weighted_sd_post_arrival": float(weighted_sd_post_arr_sec),
                "weighted_sd_post_speed": float(weighted_sd_post_arr_speed),
            }
            json_file_path = os.path.join(out_dir_path, f"cme_arrival_stats.json")
            with open(json_file_path, "w") as f:
                json.dump(out_dict, f)

    return None
