"""
init_sir.py

Factory functions for the forward models used by the data assimilation, plus the initialiser for the
CME parameter ensemble dictionary.

    setup_huxt           -> HUXt (incompressible, 2D in longitude at one latitude)
    setup_surf           -> SURF (2D; solver "huxt" = incompressible, "hydro" = compressible)
    setup_surf_compress  -> SURF with the compressible "hydro" solver and a magnetic-field boundary
    setup_surf3d         -> SURF3d (stack of SURF latitudes, built from a 2D lat-lon speed map)

    initialise_cme_parameter_ensemble_dict -> empty ``CmeParEns`` dictionary for n ensemble members

All models are initialised for the Carrington rotation containing ``start_datetime``, with the
Carrington longitude of Earth at that time as the initial longitude.
"""
import datetime

import numpy as np
import numpy.typing as npt

import huxt.huxt as H
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
) -> H.HUXt:
    """
    Initialise HUXt for the Carrington rotation containing start_datetime, at Earth's latitude.
    (By default a uniform 400 km/s wind, starting at 2008-01-01T00:00:00.)
    :param start_datetime: Initial datetime of HUXt simulation
    :param vr_in: Radial solar wind speed at the inner boundary, in km/s (one value per longitude)
    :param lon_start: Initial longitude of HUXt simulation in degrees
    :param lon_stop: Final longitude of HUXt simulation in degrees
    :param sim_time: HUXt simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii

    :return model: HUXt model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: H.Observer = H.Observer("EARTH", start_time)

    # Set up HUXt for a sim_time-day simulation, outputting every dt_scale
    # (NB: HUXt's accel_limit option is not used)
    model: H.HUXt = H.HUXt(
        v_boundary=vr_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=ert.lat.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
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
) -> S.SURF:
    """
    Initialise 2D SURF for the Carrington rotation containing start_datetime, in the equatorial plane
    (latitude 0 deg). (By default a uniform 400 km/s wind, starting at 2008-01-01T00:00:00.)
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: Radial solar wind speed at the inner boundary, in km/s (one value per longitude)
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param sim_time: SURF simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use, must be in ["huxt", "hydro"]
        ("huxt" = incompressible, "hydro" = compressible)

    :return model: SURF model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["huxt", "hydro"], "Solver must be in ['huxt', 'hydro']"

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: S.Observer = S.Observer("EARTH", start_time)

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    # (NB: latitude fixed at 0 deg rather than Earth's latitude, ert.lat)
    model: S.SURF = S.SURF(
        v_boundary=vr_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=0 * u.deg,
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
    )

    return model


def setup_surf_compress(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s,
        br_in: npt.NDArray[Quantity[u.nT]] | Quantity[u.nT] = np.nan * u.nT,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
        solver: str = "hydro",
) -> S.SURF:
    """
    Initialise 2D compressible SURF (with a radial magnetic field boundary condition) for the Carrington
    rotation containing start_datetime, in the equatorial plane (latitude 0 deg).
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: Radial solar wind speed at the inner boundary, in km/s (one value per longitude)
    :param br_in: Radial magnetic field at the inner boundary, in nT
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param sim_time: SURF simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use; must be "hydro" (compressible)

    :return model: SURF model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["hydro"], "Solver must be in ['hydro']"

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: S.Observer = S.Observer("EARTH", start_time)

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    model: S.SURF = S.SURF(
        v_boundary=vr_in,
        b_boundary=br_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=0 * u.deg,
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
    )

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
    Initialise 3D SURF (SURF3d) from a 2D (latitude x longitude) map of inner-boundary wind speeds.
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: (n_vr_lats, n_vr_lons) map of radial solar wind speed at the inner boundary, km/s
    :param n_in, T_in: Inner-boundary density and temperature (currently unused)
    :param br_in, br_lons, br_lats: Inner-boundary magnetic field map and its coordinates (currently unused)
    :param vr_lons: Longitudes of the columns of vr_in, in degrees
    :param vr_lats: Latitudes of the rows of vr_in, in degrees
    :param lat_min, lat_max: Latitude range of the SURF3d grid
    :param n_lat_surf: Number of SURF latitudes between lat_min and lat_max
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param n_lon_surf: Number of SURF longitudes
    :param sim_time: SURF simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use, must be in ["huxt", "hydro"]

    :return model3d: SURF3d model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["huxt", "hydro"], "Solver must be in ['huxt', 'hydro']"

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: S.Observer = S.Observer("EARTH", start_time)

    # Verify that the speed map (vr_in) has shape (n_vr_lats, n_vr_lons)
    if isinstance(vr_lats.value, float):
        len_lat = 1
    else:
        len_lat = len(vr_lats.value)

    if isinstance(vr_lons.value, float):
        len_lon = 1
    else:
        len_lon = len(vr_lons.value)

    shape_vr = np.shape(vr_in)
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

    return model3d


def initialise_cme_parameter_ensemble_dict(
        n_ensemble: int,
        surf_init_time: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
) -> CmeParEns:
    """
    Initialise a CME parameter ensemble dictionary with placeholder values (zeros, uniform weights)
    for each ensemble member. The CME parameters are filled in afterwards (e.g. from prior samples).
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :param surf_init_time: Initial datetime of SURF simulation
    :return parameter_dict: Dictionary conforming to ``CmeParEns``
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
        "v_transit": [0 for _ in range(n_ensemble)],
        "v_hit": [0 for _ in range(n_ensemble)],
        "likelihood": [0 for _ in range(n_ensemble)],
        # Uniform prior weights: 1/N (linear) and -log(N) (log)
        "weight": [(1.0 / n_ensemble) for _ in range(n_ensemble)],
        "log_weight": [-np.log(n_ensemble) for _ in range(n_ensemble)],
        "n_members": n_ensemble
    }

    return parameter_dict
