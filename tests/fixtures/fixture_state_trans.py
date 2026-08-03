import pytest
import numpy as np
import astropy.units as u
import datetime
import sys

import numpy.typing as npt
from astropy.units import Quantity

sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\src')
from cme_par_ens import CmeParEns
from to_state_vector import ToStateVector
from from_state_vector import FromStateVector

#sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\tests')
from cme_par_array import CmeParArray
from cme_par_dict import CmeParDict

@pytest.fixture
def get_n_members() -> int:
    n_members = 5
    return n_members


@pytest.fixture
def get_surf_init_time_state_trans() -> datetime.datetime:
    surf_init_time = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0)
    return surf_init_time

#########################################################################################
######################## GET THE VALUES FOR THE DICTIONARY ##############################
@pytest.fixture
def get_t_init_dict(get_n_members) -> list[datetime.datetime]:
    t_init = [
        datetime.datetime(year=2021, month=1, day=1, hour=12, minute=0, second=0),
        datetime.datetime(year=2021, month=1, day=1, hour=13, minute=0, second=0),
        datetime.datetime(year=2021, month=1, day=1, hour=14, minute=0, second=0),
        datetime.datetime(year=2021, month=1, day=1, hour=18, minute=0, second=0),
        datetime.datetime(year=2021, month=1, day=1, hour=15, minute=30, second=0)
    ]
    assert (len(t_init) == get_n_members)

    return t_init


@pytest.fixture
def get_v_dict(get_n_members) -> list[Quantity]:
    v = [
        500 * 1000 * u.m / u.s,
        550 * u.km / u.s,
        750 * u.km / u.s,
        740 * 3600 * u.km / u.hour,
        1200 * 3600 * 1000 * u.m / u.hour
    ]

    assert (len(v) == get_n_members)

    return v


@pytest.fixture
def get_width_dict(get_n_members) -> list[Quantity]:
    width = [
        50 * u.deg,
        30 * np.pi / 180 * u.rad,
        48 * np.pi / 180 * u.rad,
        56 * u.deg,
        12 * u.deg
    ]

    assert (len(width) == get_n_members)

    return width


@pytest.fixture
def get_lon_dict(get_n_members) -> list[Quantity]:
    lon = [
        -34 * u.deg,
        240 * np.pi / 180 * u.rad,
        49 * np.pi / 180 * u.rad,
        -120 * np.pi / 180.0 * u.rad,
        185 * u.deg
    ]

    assert (len(lon) == get_n_members)

    return lon


@pytest.fixture
def get_lat_dict(get_n_members) -> list[Quantity]:
    lat = [
        2 * u.deg,
        4 * np.pi / 180 * u.rad,
        -49 * u.deg,
        -12 * np.pi / 180.0 * u.rad,
        8 * u.deg
    ]

    assert (len(lat) == get_n_members)

    return lat


@pytest.fixture
def get_thick_dict(get_n_members) -> list[Quantity]:
    thick = [
        3 * u.solRad,
        12 * 695700 * u.km,
        5 * u.solRad,
        7 * 695700 * 1000 * u.m,
        8 * 0.00465047 * u.AU
    ]

    assert (len(thick) == get_n_members)

    return thick


@pytest.fixture
def par_dict(
    get_n_members,
    get_surf_init_time_state_trans,
    get_t_init_dict,
    get_v_dict,
    get_width_dict,
    get_lon_dict,
    get_lat_dict,
    get_thick_dict
) -> CmeParEns:

    par_class = CmeParDict(
        n_members=get_n_members,
        surf_init_time=get_surf_init_time_state_trans,
        t_init=get_t_init_dict,
        v=get_v_dict,
        width=get_width_dict,
        lon=get_lon_dict,
        lat=get_lat_dict,
        thick=get_thick_dict
    )

    return par_class.make_par_dict()

# Initialise ToStateVector object
@pytest.fixture
def init_to_state_vector(par_dict, request) -> ToStateVector:
    if hasattr(request, 'param'):
        pars_in_state_vector = request.param
    else:
        pars_in_state_vector = None
    print(f"pars_in_state_vector={pars_in_state_vector}")
    print(f"par_dict = {par_dict}")
    return ToStateVector(par_dict, pars_in_state_vector=pars_in_state_vector)
############################################################################################


#########################################################################################
######################## GET THE VALUES FOR THE STATE VECTOR ############################
@pytest.fixture
def get_t_init_state(get_n_members) -> list[datetime.datetime]:
    t_init = [
        datetime.datetime(year=2021, month=1, day=1, hour=13, minute=40, second=0),
        datetime.datetime(year=2021, month=1, day=1, hour=14, minute=6, second=40),
        datetime.datetime(year=2021, month=1, day=1, hour=14, minute=50, second=0),
        datetime.datetime(year=2021, month=1, day=1, hour=13, minute=36, second=40),
        datetime.datetime(year=2021, month=1, day=1, hour=18, minute=20, second=0)
    ]
    assert (len(t_init) == get_n_members)

    return t_init


@pytest.fixture
def get_v_state(get_n_members) -> list[Quantity]:
    v = [
        550 * u.km / u.s,
        1000 * u.km / u.s,
        940 * u.km / u.s,
        800 * u.km / u.s,
        650 * u.km / u.s
    ]

    assert (len(v) == get_n_members)

    return v


@pytest.fixture
def get_width_state(get_n_members) -> list[Quantity]:
    width = [
        40 * u.deg,
        38 * u.deg,
        34 * u.deg,
        53 * u.deg,
        27 * u.deg
    ]

    assert (len(width) == get_n_members)

    return width


@pytest.fixture
def get_lon_state(get_n_members) -> list[Quantity]:
    lon = [
        -30 * u.deg,
        -100 * u.deg,
        84 * u.deg,
        -90 * u.deg,
        -74 * u.deg
    ]

    assert (len(lon) == get_n_members)

    return lon


@pytest.fixture
def get_lat_state(get_n_members) -> list[Quantity]:
    lat = [
        3 * u.deg,
        5 * u.deg,
        -45 * u.deg,
        -26 * u.deg,
        11 * u.deg
    ]

    assert (len(lat) == get_n_members)

    return lat


@pytest.fixture
def get_thick_state(get_n_members) -> list[Quantity]:
    thick = [
        7 * u.solRad,
        11 * u.solRad,
        6 * u.solRad,
        19 * u.solRad,
        2 * u.solRad
    ]

    assert (len(thick) == get_n_members)

    return thick


@pytest.fixture
def par_state(
        get_n_members,
        get_surf_init_time_state_trans,
        get_t_init_state,
        get_v_state,
        get_width_state,
        get_lon_state,
        get_lat_state,
        get_thick_state
) -> CmeParEns:

    par_class = CmeParDict(
        n_members=get_n_members,
        surf_init_time=get_surf_init_time_state_trans,
        t_init=get_t_init_state,
        v=get_v_state,
        width=get_width_state,
        lon=get_lon_state,
        lat=get_lat_state,
        thick=get_thick_state
    )

    return par_class.make_par_dict()


#########################################################################################
########################## GET THE VALUES FOR THE ARRAY #################################
@pytest.fixture
def get_t_init_array(get_n_members) -> npt.NDArray[datetime.datetime]:
    t_init = np.array([
        43200, 46800, 50400, 64800, 55800
    ])
    assert (len(t_init) == get_n_members)

    return t_init


@pytest.fixture
def get_v_array(get_n_members) -> npt.NDArray[float]:
    v = np.array([500, 550, 750, 740, 1200])

    assert (len(v) == get_n_members)

    return v


@pytest.fixture
def get_width_array(get_n_members) -> npt.NDArray[float]:
    width = np.array([50, 30, 48, 56, 12])

    assert (len(width) == get_n_members)

    return width


@pytest.fixture
def get_lon_array(get_n_members) -> npt.NDArray[float]:
    lon = np.array([-34, -120, 49, -120, -175])

    assert (len(lon) == get_n_members)

    return lon


@pytest.fixture
def get_lat_array(get_n_members) -> npt.NDArray[float]:
    lat = np.array([ 2, 4, -49, -12, 8 ])

    assert (len(lat) == get_n_members)

    return lat


@pytest.fixture
def get_thick_array(get_n_members) -> npt.NDArray[float]:
    thick = np.array([3, 12, 5, 7, 8])

    assert (len(thick) == get_n_members)

    return thick


@pytest.fixture
def par_array(
        get_n_members,
        get_surf_init_time_state_trans,
        get_t_init_array,
        get_v_array,
        get_width_array,
        get_lon_array,
        get_lat_array,
        get_thick_array
) -> npt.NDArray[float]:

    par_class = CmeParArray(
        n_members=get_n_members,
        surf_init_time=get_surf_init_time_state_trans,
        t_init=get_t_init_array,
        v=get_v_array,
        width=get_width_array,
        lon=get_lon_array,
        lat=get_lat_array,
        thick=get_thick_array
    )

    return par_class.make_par_array()

"""
@pytest.fixture
def init_state_vector(par_dict, request) -> npt.NDArray[float]:
    if hasattr(request, 'param'):
        pars_in_state_vector = request.param
    else:
        pars_in_state_vector = None

    state_obj = ToStateVector(par_dict, pars_in_state_vector=pars_in_state_vector)

    return state_obj.cme_par_dict_to_state_vector()"""


# Initialise FromStateVector object
@pytest.fixture
def init_from_state_vector(par_array, par_dict, par_state, request) -> FromStateVector:
    if hasattr(request, 'param'):
        pars_in_state_vector = request.param
    else:
        pars_in_state_vector = None
    print(f"pars_in_state_vector={pars_in_state_vector}")

    state_obj = ToStateVector(par_state, pars_in_state_vector=pars_in_state_vector)

    init_state_vector = state_obj.cme_par_dict_to_state_vector()

    from_state_obj = FromStateVector(
        par_array,
        par_dict,
        state_ens=init_state_vector,
        pars_in_state_vector=pars_in_state_vector
    )

    return from_state_obj
##################################################################################################
