# @package init_sir
# This module will take inputs from SURF and perform an SIR to
#  generate an updated set of weights and particles
import numpy as np
import numpy.typing as npt
import datetime

import huxt.huxt as H
import surf.surf as S

import surf.surf as S

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from cme_par_ens import CmeParEns


def setup_huxt(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
#        accel_limit: bool = False
) -> H.HUXt:
    """
    Initialise HUXt with some predetermined boundary/initial conditions
    Here a uniform 400km/s wind is used, and HUXt time is set to 2008-01-01T00:00:00.
    :param start_datetime: Initial datetime of HUXt simulation
    :param vr_in: Initial radial solar wind speed in km/s
    :param lon_start: Initial longitude of HUXt simulation in degrees
    :param lon_stop: Final longitude of HUXt simulation in degrees
    :param sim_time: HUXt simulation time in seconds
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii

    :return model: HUXt model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: Observer = H.Observer("EARTH", start_time)

    # Set up HUXt for a sim_time-day simulation, outputting every dt_scale
    model: HUXt = H.HUXt(
        v_boundary=vr_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=ert.lat.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
#        accel_limit=accel_limit
    )

    return model


def setup_surf(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
        solver: str = "huxt",
#        accel_limit: bool = False
) -> S.SURF:
    """
    Initialise SURF with some predetermined boundary/initial conditions
    Here a uniform 400km/s wind is used, and SURF time is set to 2008-01-01T00:00:00.
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: Initial radial solar wind speed in km/s
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param sim_time: SURF simulation time in seconds
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use, string, must be in ["huxt", "hydro"]

    :return model: SURF model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["huxt", "hydro"], "Solver must be in ['huxt', 'hydro']"

    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: Observer = S.Observer("EARTH", start_time)

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    model: SURF = S.SURF(
        v_boundary=vr_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=0 * u.deg,#ert.lat.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
#        accel_limit=accel_limit
    )

    # model1d = S.SURF(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
    #                  lon_out=0 * u.deg, simtime=5 * u.day, dt_scale=4)
    # print(type(model))

    return model


def setup_surf_compress(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s,
        br_in=np.nan * u.nT,
        n_in=np.nan * (u.cm ** -3),
        T_in=np.nan * u.K,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
        solver: str = "huxt",
#        accel_limit: bool = False
) -> S.SURF:
    """
    Initialise SURF with some predetermined boundary/initial conditions
    Here a uniform 400km/s wind is used, and SURF time is set to 2008-01-01T00:00:00.
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: Initial radial solar wind speed in km/s
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param sim_time: SURF simulation time in seconds
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use, string, must be in ["huxt", "hydro"]

    :return model: SURF model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["hydro"], "Solver must be in ['hydro']"

    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: Observer = S.Observer("EARTH", start_time)

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    model: SURF = S.SURF(
        v_boundary=vr_in,
        b_boundary=br_in,
        rho_boundary=n_in,
        temp_boundary=T_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=0 * u.deg,#ert.lat.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
#        accel_limit=accel_limit
    )

    # model1d = S.SURF(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
    #                  lon_out=0 * u.deg, simtime=5 * u.day, dt_scale=4)
    # print(type(model))

    return model

def setup_surf3d(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones((45, 128)) * 400 * u.km / u.s,
        n_in: npt.NDArray[Quantity[u.cm ** -3]]=np.nan * (u.cm ** -3),
        T_in: npt.NDArray[Quantity[u.K]]=np.nan * u.K,
        br_in: npt.NDArray[Quantity[u.nT]] | Quantity[u.nT]=np.nan * u.nT,
        br_lons: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
        br_lats: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
        vr_lons: npt.NDArray[Quantity[u.deg]] = np.arange(0, 361, 10) * u.deg,
        vr_lats: npt.NDArray[Quantity[u.deg]] = np.arange(-90, 90, 10) * u.deg,
        lat_min: Quantity[u.deg] = -30 * u.deg,
        lat_max: Quantity[u.deg] = 30 * u.deg,
        n_lat_surf: int=45,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        n_lon_surf: int=128,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
        solver: str = "huxt",
) -> S.SURF3d:
    """
    Initialise SURF with some predetermined boundary/initial conditions
    Here a uniform 400km/s wind is used, and SURF time is set to 2008-01-01T00:00:00.
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_3d_in: Initial radial solar wind speed in km/s
    :param vr_lons: Initial longitude of SURF simulation in degrees
    :param vr_lats: Initial latitude of SURF simulation in degrees
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param sim_time: SURF simulation time in seconds
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use, string, must be in ["huxt", "hydro"]

    :return model3d: SURF3d model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["huxt", "hydro"], "Solver must be in ['huxt', 'hydro']"

    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: Observer = S.Observer("EARTH", start_time)

    # Verify that v_map has the correct shape
    if isinstance(vr_lats.value, float):
        len_lat = 1
    else:
        len_lat = len(vr_lats.value)

    if isinstance(vr_lons.value, float):
        len_lon = 1
    else:
        len_lon = len(vr_lons.value)

    #len_lon = len(vr_lons)
    shape_vr = np.shape(vr_in)
    #print(f"shape_vr={shape_vr}, len_lat={len_lat}, len_lon={len_lon}")
    assert len(shape_vr) == 2
    assert shape_vr == (len_lat, len_lon)

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    model3d = S.SURF3d(
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        v_map=vr_in,
        v_map_lat=vr_lats.to(u.rad),
        v_map_long=vr_lons.to(u.rad),
        latitude_min=lat_min.to(u.deg),
        latitude_max=lat_max.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        nlon=n_lon_surf,
        nlat=n_lat_surf,
        dt_scale=dt_scale,
        simtime=sim_time,
        r_min=r_min,
        solver=solver,
    )

    # model1d = S.SURF(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
    #                  lon_out=0 * u.deg, simtime=5 * u.day, dt_scale=4)
    # print(type(model))

    return model3d


def initialise_cme_parameter_ensemble_dict(
        n_ensemble: int,
        surf_init_time: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
) -> CmeParEns:
    """
    Function to initialise empty arrays for storing the CME parameters for each ensemble member at each analysis step
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :param surf_init_time: Initial datetime of SURF simulation
    :return parameter_arrays: A dictionary of parameter keys and an empty array for
        storing each ensemble member value conforming to class CmeParEns
    """

    parameter_dict: CmeParEns = {
        "surf_init_time": surf_init_time,
        "t_init": [
            datetime.datetime(2008, 1, 1, 0, 0, 0)
            for _ in range(n_ensemble)
        ],
        "v": [0 for _ in range(n_ensemble)] * u.km / u.s,
        "width": [0 for _ in range(n_ensemble)] * u.deg,
        "lon": [0 for _ in range(n_ensemble)] * u.deg,
        "lat": [0 for _ in range(n_ensemble)] * u.deg,
        "thick": [0 for _ in range(n_ensemble)] * u.solRad,
        "bias": [0.0 for _ in range(n_ensemble)],  # observation-bias coefficient (deg), see obs_bias.py
        "v_transit": [0 for _ in range(n_ensemble)],
        "v_hit": [0 for _ in range(n_ensemble)],
        "likelihood": [0 for _ in range(n_ensemble)],
        "weight": [(1.0 / n_ensemble) for _ in range(n_ensemble)],
        "log_weight": [-np.log(n_ensemble) for _ in range(n_ensemble)],
        "n_members": n_ensemble
    }

    return parameter_dict