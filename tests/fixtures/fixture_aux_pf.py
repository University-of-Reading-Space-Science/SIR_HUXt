import numpy as np
import numpy.typing as npt
import datetime

import sir_huxt_mono_obs as shmo
import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose

from astropy.time import Time
from init_sir import initialise_cme_parameter_ensemble_dict
from sir_observation_operator import ObservationOperator
from tests.expected_obs_op import ExpectedObservationOperator, ExpectedCMEFlankSingle
import matplotlib.pyplot as plt
from cme_par_ens import CmeParEns
import seaborn as sns
import colorcet as cc
import pytest
from cme_par_dict_structure import required_dict_keys
from aux_pf import AuxPF

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def get_n_ens():
    return 5


@pytest.fixture
def get_surf_init_time_aux_pf():
    return datetime.datetime(2008, 1, 1, 0, 0, 0)


@pytest.fixture
def get_cme_par_dict(get_n_ens, get_surf_init_time_aux_pf) -> CmeParEns:
    n_ens = get_n_ens

    w: npt.NDArray[float] = np.ones(n_ens) / n_ens
    cme_par_dictionary: CmeParEns = initialise_cme_parameter_ensemble_dict(n_ens)

    cme_par_dictionary["surf_init_time"] = get_surf_init_time_aux_pf
    cme_par_dictionary["t_init"] = [datetime.datetime(2008, 1, 1)] * n_ens
    cme_par_dictionary["v"] = np.linspace(400, 600, n_ens) * u.km / u.s
    #print(f"cme_par_dictionary[v]: {cme_par_dictionary['v']}")
    cme_par_dictionary["width"] = np.ones(n_ens) * 30 * u.deg
    cme_par_dictionary["lon"] = np.linspace(-20, 20, n_ens) * u.deg
    cme_par_dictionary["lat"] = np.zeros(n_ens) * u.deg
    cme_par_dictionary["thick"] = np.ones(n_ens) * 0 * u.solRad
    cme_par_dictionary["weight"] = w.tolist()
    cme_par_dictionary["log_weight"] = np.log(w).tolist()
    cme_par_dictionary["likelihood"] = np.zeros(n_ens).tolist()
    cme_par_dictionary["n_members"] = n_ens

    return cme_par_dictionary


@pytest.fixture
def get_true_cme_par_dict(get_surf_init_time_aux_pf) -> CmeParEns:

    true_cme_par_dict = initialise_cme_parameter_ensemble_dict(1, get_surf_init_time_aux_pf)

    # Initialise true CME parameters
    true_cme_t_init: datetime.datetime = datetime.datetime(2008, 1, 1, 1, 0, 0)
    true_cme_speed: float = 495
    true_cme_width: float = 37.4
    true_cme_lon: float = 0
    true_cme_lat: float = 0
    true_cme_thick: float = 0

    true_cme_par_dict["t_init"]: datetime.datetime = true_cme_t_init

    true_cme_par_dict["v"]: Quantity[u.km / u.s] = true_cme_speed * u.km / u.s
    true_cme_par_dict["width"]: Quantity[u.deg] = true_cme_width * u.deg
    true_cme_par_dict["lon"]: Quantity[u.deg] = true_cme_lon * u.deg
    true_cme_par_dict["lat"]: Quantity[u.deg] = true_cme_lat * u.deg
    true_cme_par_dict["thick"]: Quantity[u.solRad] = true_cme_thick * u.solRad

    return true_cme_par_dict


@pytest.fixture
def get_aux_pf(get_cme_par_dict, get_true_cme_par_dict) -> AuxPF:
    aux_pf_class = AuxPF(
        use_model="surf",
        cme_par_dict=get_cme_par_dict,
        obs=[10.0, 12.0],
        obs_cov=0.1,
        obs_lon=0 * u.deg,
        obs_time=[
            datetime.datetime(2008, 1, 1, 9, 0, 0),
            datetime.datetime(2008, 1, 1, 10, 0, 0)
        ],
        true_cme_par_dict=get_true_cme_par_dict,
        pars_in_state = ["v", "width"],
        rng=np.random.default_rng(42)
    )

    return aux_pf_class
