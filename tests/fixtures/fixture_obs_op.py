import pytest
import numpy as np
import astropy.units as u
import datetime
import pandas as pd
import sys

import numpy.typing as npt
from astropy.units import Quantity
from astropy.time import Time

# sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
from sir_observation_operator import ObservationOperator
from to_state_vector import ToStateVector
from from_state_vector import FromStateVector

# sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code\\tests')
from tests.cme_par_array import CmeParArray
from tests.cme_par_dict import CmeParDict

#################################################################################################
# Initialise variables for inputting into SURF and initialising the observation operator class
#################################################################################################
@pytest.fixture
def get_surf_init_time_obs_op() -> datetime.datetime:
    surf_init_time = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0)
    return surf_init_time

@pytest.fixture
def get_surf_vr_in() -> npt.NDArray[float]:
    vr_in = np.ones(128) * 450
    return vr_in

@pytest.fixture
def get_surf_lon_start() -> Quantity[u.deg]:
    lon_start = 316 * u.deg
    return lon_start

@pytest.fixture
def get_surf_lon_stop() -> Quantity[u.deg]:
    lon_stop = 382 * u.deg
    return lon_stop

@pytest.fixture
def get_surf_simtime() -> Quantity[u.day]:
    sim_time = 2 * u.day
    return sim_time

@pytest.fixture
def get_surf_dt_scale() -> int | float:
    dt_scale=10
    return dt_scale

@pytest.fixture
def get_surf_r_min() -> Quantity[u.solRad]:
    r_min = 30 * u.solRad
    return r_min

@pytest.fixture
def get_cme_init_rad() -> Quantity[u.solRad]:
    cme_init_rad = 12 * u.solRad
    return cme_init_rad

@pytest.fixture
def get_cme_fixed_duration() -> bool:
    cme_fixed_duration = True
    return cme_fixed_duration

@pytest.fixture
def get_fixed_duration() -> Quantity[u.s]:
    fixed_duration = 12 * 3600 * u.s
    return fixed_duration

@pytest.fixture
def get_obs_lon():
    obs_lon = -60.0 * u.deg
    return obs_lon


@pytest.fixture
def get_obs_times(obs_op_tests) -> list[datetime.datetime]:
    # if hasattr(request, 'param'):
    #     test_no = request.param
    # else:
    #     test_no = -1

    obs_times = obs_op_tests[0]

    return obs_times


@pytest.fixture
def get_use_model() -> str:
    use_model = "huxt"

    return use_model


@pytest.fixture
def get_obs_cov() -> float:
    obs_cov = 0.25
    return obs_cov


@pytest.fixture
def init_obs_op(
        par_dict: CmeParDict,
        get_use_model: str,
        get_obs_lon: Quantity[u.deg],
        get_obs_times: list[datetime.datetime],
        get_obs_cov: float,
        get_surf_init_time_obs_op: datetime.datetime,
        get_surf_vr_in: Quantity[u.solRad],
        get_surf_lon_start: Quantity[u.deg],
        get_surf_lon_stop: Quantity[u.deg],
        get_surf_simtime: Quantity[u.day],
        get_surf_dt_scale: int | float,
        get_surf_r_min: Quantity[u.solRad],
        get_cme_init_rad: Quantity[u.solRad],
        get_cme_fixed_duration: bool,
        get_fixed_duration: Quantity[u.s],
        request=None
) -> ObservationOperator:
    if hasattr(request, 'param'):
        test_no = request.param
    else:
        test_no = -1

    obs_op_obj = ObservationOperator(
        use_model=get_use_model,
        cme_par_dict=par_dict,
        obs_lon = get_obs_lon,
        obs_time_in_datetime = get_obs_times,
        obs_cov = get_obs_cov,
        surf_init_time = get_surf_init_time_obs_op,
        vr_in = get_surf_vr_in,
        lon_start = get_surf_lon_start,
        lon_stop = get_surf_lon_stop,
        sim_time = get_surf_simtime,
        dt_scale = get_surf_dt_scale,
        r_min = get_surf_r_min,
        cme_init_rad = get_cme_init_rad,
        cme_fixed_duration = get_cme_fixed_duration,
        fixed_duration = get_fixed_duration,
    )

    return obs_op_obj


@pytest.fixture
def init_test_cme_flank_single_ens(get_surf_init_time_obs_op: datetime.datetime, request) -> pd.DataFrame:
    if hasattr(request, 'param'):
        ens_member = request.param
    else:
        ens_member = None

    assert (ens_member >= 0) and (ens_member < 5)

    # For test CME flanks, set timestep to 10 minutes
    #  (for ease of checking the rounding of the times)
    deltaT: Quantity[u.s] = 6000 * u.s

    # Define times over one day
    n_timesteps_day: int = 144
    times: Time = Time([
        get_surf_init_time_obs_op + datetime.timedelta(seconds=i * deltaT.value)
        for i in range(n_timesteps_day)
    ])

    # Initialise flank dataframe
    flank: pd.DataFrame = pd.DataFrame(
        index=np.arange(times.size),
        columns=['time', 'el', 'r', 'lon']
    )
    flank['time'] = times.jd

    # Create data for flank
    init_flank_el = [4, 3, 4.5, 3.9, 4.2]
    init_flank_r = [0.05, 0.07, 0.06, 0.08, 0.075]
    init_flank_lon = [-5.0, -5.5, -4.5, -4.0, -6.0]
    # Elongation profile increases by 0.2 every 10 minutes (range=[0.4, 28.4])
    flank.loc[:, 'el']: npt.NDArray[float] = np.array(
        [init_flank_el[ens_member] + (i / 5.0) for i in range(n_timesteps_day)]
    )

    # Flank radius increases linear by 0.025 AU every 10 minutes (range=[0.05, 0.43])
    flank.loc[:, 'r']: npt.NDArray[float] = np.array(
        [init_flank_r[ens_member] + (i / 400.0) for i in range(n_timesteps_day)]
    )

    # Flank longitude increases linearly by 1.0/14.4 degrees every 10 minutes (range=[-10, 10] degrees)
    # and converted to radians
    flank.loc[:, 'lon']: npt.NDArray[float] = np.array(
        [(init_flank_lon[ens_member] + (i / 14.4)) * np.pi / 180.0 for i in range(n_timesteps_day)]
    )

    return flank


@pytest.fixture
def init_test_cme_flanks(get_surf_init_time_obs_op: datetime.datetime, request) -> list[pd.DataFrame]:
    if hasattr(request, 'param'):
        n_ensemble = request.param
    else:
        n_ensemble = None

    # For test CME flanks, set timestep to 10 minutes
    #  (for ease of checking the rounding of the times)
    deltaT: Quantity[u.s] = 600 * u.s

    # Define times over one day
    n_timesteps_day: int = 144
    times: Time = Time([
        get_surf_init_time_obs_op + datetime.timedelta(seconds=i * deltaT.value)
        for i in range(n_timesteps_day)
    ])
    print(f"n_ensemble = {n_ensemble}")
    # Create data for flank
    init_flank_el = [4, 3, 4.5, 3.9, 4.2]
    init_flank_r = [0.05, 0.07, 0.06, 0.08, 0.075]
    init_flank_lon = [-5.0, -5.5, -4.5, -4.0, -6.0]

    flank_list: list[pd.DataFrame] = [pd.DataFrame() for _ in range(n_ensemble)]
    # Elongation profile increases by 0.2 every 10 minutes (range=[0.4, 28.4])
    for m in range(n_ensemble):
        # Initialise flank dataframe
        flank: pd.DataFrame = pd.DataFrame(
            index=np.arange(times.size),
            columns=['time', 'el', 'r', 'lon']
        )
        flank['time'] = times.jd

        flank.loc[:, 'el']: npt.NDArray[float] = np.array(
            [init_flank_el[m] + (i / 5.0) for i in range(n_timesteps_day)]
        )

        # Flank radius increases linear by 0.025 AU every 10 minutes (range=[0.05, 0.43])
        flank.loc[:, 'r']: npt.NDArray[float] = np.array(
            [init_flank_r[m] + (i / 400.0) for i in range(n_timesteps_day)]
        )

        # Flank longitude increases linearly by 1.0/14.4 degrees every 10 minutes (range=[-10, 10] degrees)
        # and converted to radians
        flank.loc[:, 'lon']: npt.NDArray[float] = np.array(
            [(init_flank_lon[m] + (i / 14.4)) * np.pi / 180.0 for i in range(n_timesteps_day)]
        )

        flank_list[m] = flank.copy()

    # print(f"flank_list = {flank_list}")
    # print(f"flank_list.shape = {np.shape(flank_list)}")

    return flank_list


@pytest.fixture
def obs_op_tests(request) -> tuple[list[datetime.datetime], list[int]]:
    if hasattr(request, 'param'):
        test_no = request.param
    else:
        test_no = -1

    if test_no == 0:
        obs_times = [
                datetime.datetime(2021, 1, 1, 0, 0 ,0)
                + datetime.timedelta(hours=6 * i) for i in range(4)
        ]
        index_output = [0, 36, 72, 108]
    elif test_no == 1:
        obs_times = [
                datetime.datetime(2021, 1, 1, 0, 0, 0)
                + datetime.timedelta(hours=3 * i) for i in range(8)
        ]
        index_output = [0, 18, 36, 54, 72, 90, 108, 126]
    elif test_no == 2:
        obs_times = [
                datetime.datetime(2021, 1, 1, 0, 0, 0)
                + datetime.timedelta(minutes=270 * i) for i in range(6)
        ]
        index_output = [0, 27, 54, 81, 108, 135]
    elif test_no == 3:
        obs_times = [
                datetime.datetime(2021, 1, 1, 0, 0, 0)
                + datetime.timedelta(minutes=95 * i) for i in range(16)
        ]
        index_output = [0, 9, 19, 28, 38, 47, 57, 66, 76, 85, 95, 104, 114, 123, 133, 142]
    elif test_no == 4:
        obs_times = [
                datetime.datetime(2021, 1, 1, 2, 0, 0)
                + datetime.timedelta(minutes=127 * i) for i in range(11)
        ]
        index_output = [12, 25, 37, 50, 63, 75, 88, 101, 114, 126, 139]
    elif test_no == 5:
        obs_times = [
                datetime.datetime(2021, 1, 1, 0, 0, 0)
                + datetime.timedelta(minutes=60 * i) for i in range(24)
        ]
        index_output = [
            0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60,
            66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
            126, 132, 138
        ]
    elif test_no == 6:
        obs_times = [
            datetime.datetime(2021, 1, 1, 3, 12, 0),
            datetime.datetime(2021, 1, 1, 4, 38, 0),
            datetime.datetime(2021, 1, 1, 6, 25, 0),
            datetime.datetime(2021, 1, 1, 7, 38, 0),
            datetime.datetime(2021, 1, 1, 8, 21, 0),
            datetime.datetime(2021, 1, 1, 11, 11, 0),
            datetime.datetime(2021, 1, 1, 12, 51, 0),
            datetime.datetime(2021, 1, 1, 13, 24, 0),
            datetime.datetime(2021, 1, 1, 16, 44, 0),
            datetime.datetime(2021, 1, 1, 17, 39, 0),
            datetime.datetime(2021, 1, 1, 19, 33, 0),
            datetime.datetime(2021, 1, 1, 20, 57, 0)
        ]
        index_output = [
            19, 28, 38, 46, 50, 67, 77, 80, 100, 106, 117, 126
        ]
    elif test_no==7:
        obs_times = datetime.datetime(2021, 1, 1, 5, 0, 0)
        index_output = 30
    else:
        obs_times = [
                datetime.datetime(2021, 1, 1, 0, 0, 0)
                + datetime.timedelta(hours=4 * i) for i in range(6)
        ]
        index_output = [0, 24, 48, 72, 96, 120]

    return obs_times, index_output