# @package init_sir
# This module will take inputs from HUXt and perform an SIR to
#  generate an updated set of weights and particles
import huxt.huxt
import numpy as np
import numpy.typing as npt
import datetime

import huxt.huxt as H
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
        r_min: Quantity[u.solRad] = 30 * u.solRad,
#        accel_limit: bool = False
) -> huxt.huxt.HUXt:
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
    cr_num: int = np.fix(sn.carrington_rotation_number(start_time))
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

    # model1d = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
    #                  lon_out=0 * u.deg, simtime=5 * u.day, dt_scale=4)
    # print(type(model))

    return model


def initialise_cme_parameter_ensemble_dict(
        n_ensemble: int,
        huxt_init_time: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
) -> CmeParEns:
    """
    Function to initialise empty arrays for storing the CME parameters for each ensemble member at each analysis step
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :param huxt_init_time: Initial datetime of HUXt simulation
    :return parameter_arrays: A dictionary of parameter keys and an empty array for
        storing each ensemble member value conforming to class CmeParEns
    """

    parameter_dict: CmeParEns = {
        "huxt_init_time": huxt_init_time,
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
        "weight": [(1.0 / n_ensemble) for _ in range(n_ensemble)],
        "log_weight": [-np.log(n_ensemble) for _ in range(n_ensemble)],
        "n_members": n_ensemble
    }

    return parameter_dict