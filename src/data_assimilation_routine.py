import numpy as np
import numpy.typing as npt
import datetime
import os
import sys

import xarray as xr

import surf.surf as S
import surf.surf_inputs as Sin

import huxt.huxt_inputs as Hin

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from dotenv import load_dotenv
from pathlib import Path

import make_prior_covariance_mat as mp_cov
from wsa_reader import ReadWSAFiles

from init_sir import initialise_cme_parameter_ensemble_dict
from sir_observations import Observations
from aux_pf import AuxPF
from cme_par_ens import CmeParEns

from multiprocessing import Pool

env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path, override=True)


def print_environment_variables():
    print(f"obs_dir = {os.getenv('OBS_DIR')}")
    print(f"mo_cone_cov_dir = {os.getenv('MO_CONE_COV_DIR')}")
    print(f"donki_cov_dir = {os.getenv('DONKI_COV_DIR')}")
    print(f"mo_cme_cone_file_dir = {os.getenv('MO_CME_CONE_FILE_DIR')}")
    print(f"out_base_dir = {os.getenv('OUT_BASE_DIR')}")
    print(f"wsa_dir = {os.getenv('WSA_DIR')}")

    return None


def allowed_cov_types():
    return {"uncorr", "donki", "mo_cone"}


def get_mas_wsa_const():
    mas_wsa_const = "const"
    
    return mas_wsa_const


def get_use_synthetic_obs():
    use_synthetic_obs = True

    return use_synthetic_obs


def get_ssw_event_craft():
    ssw_event = "ssw_012"
    craft = "stb"
    img = "diff"

    return ssw_event, craft, img


def get_surf_init_time():

    ssw_event, _, _ = get_ssw_event_craft()
    if get_use_synthetic_obs():
        surf_init_time: datetime.datetime = datetime.datetime(2012, 1, 1, 0, 0)
    else:
        if ssw_event == "ssw_007":
            surf_init_time: datetime.datetime = datetime.datetime(2012, 8, 31, 0, 0, 0)

        elif ssw_event == "ssw_008":
            surf_init_time: datetime.datetime = datetime.datetime(2012, 9, 27, 15, 0, 0)

        elif ssw_event == "ssw_009":
            surf_init_time: datetime.datetime = datetime.datetime(2012, 10, 4, 20, 0, 0)

        elif ssw_event == "ssw_012":
            surf_init_time: datetime.datetime = datetime.datetime(2012, 11, 20, 0, 0, 0)

        else:
            sys.exit("Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'")

    return surf_init_time


def get_use_model():
    """
    Function to get model we require to use
    :return: use_model: String containing either "surf", "surf_compress", "huxt"
    """

    poss_models = ["surf", "compress_surf", "huxt"]
    use_model = "compress_surf"

    assert use_model.lower() in poss_models

    return use_model.lower()


def initialise_surf_parameters():
    """
    Function to initialise the SURF parameters for simulation
    :return:
    """
    mas_or_wsa = get_mas_wsa_const()

    surf_init_time = get_surf_init_time()
    use_model = get_use_model().lower()

    surf_init_time_astro = Time(surf_init_time, format="datetime")
    ert = S.Observer("EARTH", surf_init_time_astro)
    earth_lat = ert.lat.to(u.deg)
    earth_lon = ert.lat.to(u.deg)

    lon_start: Quantity[u.deg] = -70 * u.deg
    lon_stop: Quantity[u.deg] = 70 * u.deg
    sim_time: Quantity[u.day] = 3 * u.day
    dt_scale: int = 1
    cme_init_rad: Quantity[u.solRad] = 12 * u.solRad
    r_min: Quantity[u.solRad] = 21.5 * u.solRad
    cme_fixed_duration: bool = True
    fixed_duration: Quantity[u.s] = 12 * 3600 * u.s


    if mas_or_wsa.upper() == "MAS":
        cr_num: int = np.trunc(sn.carrington_rotation_number(surf_init_time))

        if use_model in ["surf", "compress_surf"]:
            vr_in = Sin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
        elif use_model in ["huxt"]:
            vr_in = Hin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
        else:
            sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    elif mas_or_wsa.upper() == "WSA":
        wsa_dir = os.getenv('WSA_DIR')
        readWSAClass = ReadWSAFiles(surf_init_time, output_dir=wsa_dir)
        output_path = readWSAClass.download_file()

        # Get input speeds for all longitudes for input into surf
        # Decelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
        # Load in map and Earth lon cut, decaccelerate both from 215 to 21.5.
        surf_init_time_astro = Time(surf_init_time, format="datetime")
        ert = S.Observer("EARTH", surf_init_time_astro)
        earth_lat = ert.lat.to(u.deg)

        # Can't use hin.map_vmap_inwards as that shifts the longitudes and interpolates, which we do not want for
        # WSA
        if use_model in ["surf", "compress_surf"]:
            vr_in = Sin.get_WSA_long_profile(output_path, lat=earth_lat)
            n_lon = len(vr_in)
            v_lon_grid_step = 2 * np.pi / n_lon
            v_lons = np.array([
                i * v_lon_grid_step for i in range(n_lon)
            ]) * u.rad

            vr_in, lon_temp = Sin.map_v_inwards(vr_in, 215 * u.solRad, v_lons, r_min)
        elif use_model in ["huxt"]:
            vr_in = Hin.get_WSA_long_profile(output_path, lat=earth_lat)
            n_lon = len(vr_in)
            v_lon_grid_step = 2 * np.pi / n_lon
            v_lons = np.array([
                i * v_lon_grid_step for i in range(n_lon)
            ]) * u.rad

            vr_in, lon_temp = Hin.map_v_inwards(vr_in, 215 * u.solRad, v_lons, r_min)
        else:
            sys.exit("Unknown model name, expected either 'surf', 'compress_surf' or 'huxt'")


    elif mas_or_wsa.upper() == "CONST":
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.zeros(128) + 400 * u.km / u.s
    else:
        print("Defaulting to constant ambient solar wind of 400 km/s")
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.zeros(128) + 400 * u.km / u.s

    out_surf_par = {
        "use_model": use_model,
        "surf_init_time": surf_init_time,
        "vr_in": vr_in,
        "lon_start": lon_start,
        "lon_stop": lon_stop,
        "sim_time": sim_time,
        "dt_scale": dt_scale,
        "r_min": r_min,
        "cme_init_rad": cme_init_rad,
        "cme_fixed_duration": cme_fixed_duration,
        "fixed_duration": fixed_duration,
    }

    return out_surf_par


def initialise_true_cme_par_dict():
    surf_init_time = get_surf_init_time()
    true_cme_par_dict = initialise_cme_parameter_ensemble_dict(
        1, surf_init_time
    )

    # Initialise true CME parameters
    true_cme_t_init: datetime.datetime = surf_init_time + datetime.timedelta(hours=1)
    true_cme_speed: float = 495
    true_cme_width: float = 37.4
    true_cme_lon: float = 0
    true_cme_lat: float = 0
    true_cme_thick: float = 0

    true_cme_par_dict["t_init"]: datetime.datetime = true_cme_t_init
    true_t_init_str: str = true_cme_par_dict["t_init"].strftime("%Y%m%d-%H%M")

    true_cme_par_dict["v"]: Quantity[u.km / u.s] = true_cme_speed * u.km / u.s
    true_cme_par_dict["width"]: Quantity[u.deg] = true_cme_width * u.deg
    true_cme_par_dict["lon"]: Quantity[u.deg] = true_cme_lon * u.deg
    true_cme_par_dict["lat"]: Quantity[u.deg] = true_cme_lat * u.deg
    true_cme_par_dict["thick"]: Quantity[u.solRad] = true_cme_thick * u.solRad

    return true_cme_par_dict


def initialise_fg_cme_parameters():
    # Initialise CME parameters
    ssw_event, _, _ = get_ssw_event_craft()
    donki_cme = True

    if get_use_synthetic_obs():
        surf_init_time = get_surf_init_time()
        cme_at_12rs: datetime.datetime = (
                surf_init_time + datetime.timedelta(hours=1)
        )
        surf_init_time_astro = Time(surf_init_time, format="datetime")
        ert = S.Observer("EARTH", surf_init_time_astro)
        earth_lat = ert.lat.to(u.deg)
        earth_lon = ert.lon.to(u.deg)

        fg_cme_speed: float = 495
        fg_cme_width: float = 37.4
        fg_cme_lon: float = earth_lon.value
        fg_cme_lat: float = earth_lat.value
        fg_cme_thick: float = 0
    else:
        if donki_cme:
            if ssw_event == "ssw_007":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 8, 31, 22, 43, 0)
                print(f"CR = {sn.carrington_rotation_number(cme_at_21rs)}")
                fg_cme_speed: float = 1498

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                    cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 150
                fg_cme_lon: float = -63
                fg_cme_lat: float = -15
                fg_cme_thick: float = 0

            elif ssw_event == "ssw_008":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 9, 28, 1, 56, 0)

                fg_cme_speed: float = 1160

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                        cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 170
                fg_cme_lon: float = 30
                fg_cme_lat: float = 5
                fg_cme_thick: float = 0

            elif ssw_event == "ssw_009":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 10, 5, 9, 15, 0)

                fg_cme_speed: float = 650

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                        cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 94
                fg_cme_lon: float = 10
                fg_cme_lat: float = -28
                fg_cme_thick: float = 0

            elif ssw_event == "ssw_012":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 11, 20, 17, 32, 0)

                fg_cme_speed: float = 725

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                        cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 60
                fg_cme_lon: float = 90
                fg_cme_lat: float = 5
                fg_cme_thick: float = 0

            else:
                sys.exit("Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'")
        else:
            if ssw_event == "ssw_007":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 8, 31, 22, 46, 0)
                print(f"CR = {sn.carrington_rotation_number(cme_at_21rs)}")
                fg_cme_speed: float = 1010

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                        cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 66
                fg_cme_lon: float = -30
                fg_cme_lat: float = 0
                fg_cme_thick: float = 0

            elif ssw_event == "ssw_008":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 9, 28, 3, 49, 0)

                fg_cme_speed: float = 872

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                        cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 110
                fg_cme_lon: float = 20
                fg_cme_lat: float = 4
                fg_cme_thick: float = 0

            elif ssw_event == "ssw_009":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 10, 5, 8, 47, 0)

                fg_cme_speed: float = 698

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                        cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 84
                fg_cme_lon: float = 9
                fg_cme_lat: float = -24
                fg_cme_thick: float = 0

            elif ssw_event == "ssw_012":
                cme_at_21rs: datetime.datetime = datetime.datetime(2012, 11, 20, 17, 40, 0)

                fg_cme_speed: float = 664

                dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                cme_at_12rs: datetime.datetime = (
                        cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                )

                fg_cme_width: float = 94
                fg_cme_lon: float = 22
                fg_cme_lat: float = 20
                fg_cme_thick: float = 0

            else:
                sys.exit("Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'")

    # Time taken to get from cme_init_rad to r_min
    surf_dict = initialise_surf_parameters()
    surf_cme_ir: Quantity[u.solRad] = surf_dict["cme_init_rad"]
    surf_r_min: Quantity[u.solRad] = surf_dict["r_min"]

    dist_in_km: Quantity[u.km] = (surf_r_min - surf_cme_ir).to(u.km)
    seconds_to_r_min: float = dist_in_km.to(u.km).value / fg_cme_speed

    fg_cme_t_init: datetime.datetime = (
        cme_at_12rs + datetime.timedelta(seconds=seconds_to_r_min)
    )

    if get_use_synthetic_obs():
        sd_cme_t_init: float = 0  # In seconds
        sd_cme_speed: float = 50  # In km/s
        sd_cme_width: float = 5  # In deg
        sd_cme_lon: float = 5  # In deg
        sd_cme_lat: float = 0 # In deg
        sd_cme_thick: float = 0  # In solRad
    else:
        sd_cme_t_init: float = 3600  # In seconds
        sd_cme_speed: float = 50  # In km/s
        sd_cme_width: float = 5  # In deg
        sd_cme_lon: float = 5  # In deg
        sd_cme_lat: float = 5  # In deg
        sd_cme_thick: float = 0  # In solRad

    fg_cme_par_dict = {
        "fg_cme_t_init": fg_cme_t_init,
        "fg_cme_speed": fg_cme_speed,
        "fg_cme_width": fg_cme_width,
        "fg_cme_lon": fg_cme_lon,
        "fg_cme_lat": fg_cme_lat,
        "fg_cme_thick": fg_cme_thick,
        "sd_cme_t_init": sd_cme_t_init,
        "sd_cme_speed": sd_cme_speed,
        "sd_cme_width": sd_cme_width,
        "sd_cme_lon": sd_cme_lon,
        "sd_cme_lat": sd_cme_lat,
        "sd_cme_thick": sd_cme_thick
    }

    return fg_cme_par_dict


def initialise_prior_cme_cov():
    if get_use_synthetic_obs():
        cme_cov_type = "uncorr"
    else:
        cme_cov_type = "mo_cone"

    """# Initialise how to build the prior CME covariance matrix
    if cme_cov_type is None:
        cme_cov_type = "uncorr"
    
    try:
        assert cme_cov_type.lower() in allowed_cov_types()
    except AssertionError:
        print("Defaulting to uncorrelated covariance matrix")
        cme_cov_type = "uncorr"
    else:
        cme_cov_type = cme_cov_type.lower()"""

    # Scale correlation matrix to make covariance matrix
    scale_corr = True
    use_log_v = True

    if get_use_synthetic_obs():
        # Initialise CME parameters for uniform distribution
        sd_t_init: float = 0  # In seconds

        if use_log_v:
            sd_speed: float = 0.1  # Multiplicative variance
        else:
            sd_speed: float = 50  # In km/s
        sd_width: float = 5  # In deg
        sd_lon: float = 5  # In deg
        sd_lat: float = 0  # In deg
        sd_thick: float = 0  # In solRad
    else:
        # Initialise CME parameters for uniform distribution
        sd_t_init: float = 3600  # In seconds

        if use_log_v:
            sd_speed: float = 0.1 # Multiplicative variance
        else:
            sd_speed: float = 50 # In km/s
        sd_width: float = 5  # In deg
        sd_lon: float = 5  # In deg
        sd_lat: float = 5  # In deg
        sd_thick: float = 0  # In solRad

    prior_cme_cov_dict = {
        "cme_cov_type": cme_cov_type,
        "scale_corr": scale_corr,
        "use_log_v": use_log_v,
        "sd_t_init": sd_t_init,
        "sd_speed": sd_speed,
        "sd_width": sd_width,
        "sd_lon": sd_lon,
        "sd_lat": sd_lat,
        "sd_thick": sd_thick
    }

    return prior_cme_cov_dict
    
    
def initialise_observation_parameters():
    n_obs: int = 3
    obs_cadence_hr = 8 # Observations cadence in hours
    true_cme_dict = initialise_true_cme_par_dict()
    true_cme_t_init: datetime.datetime = true_cme_dict["t_init"]

    surf_init_time = get_surf_init_time()
    obs_radius: Quantity[u.AU] = 1.0 * u.AU
    obs_lon: Quantity[u.deg] = 300 * u.deg
    obs_lat: Quantity[u.deg] = 0 * u.deg

    obs_cov: list[float] = [0.5]  # * np.eye(len(obs))

    obs_times: list[datetime.datetime] = [
        true_cme_t_init + datetime.timedelta(hours=10 + (obs_cadence_hr * i))
        for i in range(1, 1 + n_obs)
    ]

    use_synthetic_obs: bool = get_use_synthetic_obs()
    obs_rng_seed: int = 4096
    obs_filenames: list[str] = [
        os.path.join(
            str(os.getenv("OBS_DIR")), "SSW_cme_classifications.hdf5"
        )
    ]

    ssw_event, craft, img = get_ssw_event_craft()
    if use_synthetic_obs:
        bias_term_bool = False #True
    else:
        bias_term_bool = True

    bias_term_5 = 2.5
    bias_term_21 = 2.5

    bias_term_5 = 2.5
    bias_term_21 = 2.5

    obs_par_dict = {
        "n_obs": n_obs,
        "obs_radius": obs_radius,
        "obs_lon": obs_lon,
        "obs_lat": obs_lat,
        "obs_cov": obs_cov,
        "obs_times": obs_times,
        "use_synthetic_obs": use_synthetic_obs,
        "obs_filenames": obs_filenames,
        "obs_rng_seed": obs_rng_seed,
        "ssw_event": ssw_event,
        "craft": craft,
        "img": img,
        "bias_term_bool": bias_term_bool,
        "bias_term_5rs": bias_term_5,
        "bias_term_21rs": bias_term_21,
    }

    return obs_par_dict


def initialise_da_parameters():
    n_members: int = 50
    n_runs: int = 25

    delta_aux_pf: float = 0.98
    if get_use_synthetic_obs():
        pars_in_state_vector: list[str] = ["v", "width", "lon"]
    else:
        pars_in_state_vector: list[str] = ["t_init", "v", "width", "lon", "lat"]
    time_tolerance: Quantity[u.day] = 0.5 * u.day

    da_par_dict = {
        "n_members": n_members,
        "n_runs": n_runs,
        "delta_aux_pf": delta_aux_pf,
        "pars_in_state": pars_in_state_vector,
        "time_tolerance": time_tolerance
    }

    return da_par_dict


class RunDataAssimilationRoutine:
    def __init__(self):

        # Get SURF parameters
        surf_par_dict = initialise_surf_parameters()
        self.use_model: str = surf_par_dict["use_model"]
        self.surf_init_time: datetime.datetime = surf_par_dict["surf_init_time"]
        self.vr_in: npt.NDArray[Quantity[u.km / u.s]] = surf_par_dict["vr_in"]
        self.n_lon: int = len(self.vr_in)
        self.lon_start: Quantity[u.deg] = surf_par_dict["lon_start"]
        self.lon_stop: Quantity[u.deg] = surf_par_dict["lon_stop"]
        self.sim_time: Quantity[u.days] = surf_par_dict["sim_time"]
        self.dt_scale: int | float = surf_par_dict["dt_scale"]
        self.r_min: Quantity[u.solRad] = surf_par_dict["r_min"]
        self.cme_init_rad: Quantity[u.solRad] = surf_par_dict["cme_init_rad"]
        self.cme_fixed_duration: bool = surf_par_dict["cme_fixed_duration"]
        self.fixed_duration: Quantity[u.s] = surf_par_dict["fixed_duration"]

        # Initialise observation variables
        obs_par_dict = initialise_observation_parameters()
        self.n_obs: int = obs_par_dict["n_obs"]
        self.obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = obs_par_dict["obs_radius"]
        self.obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lon"]
        self.obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lat"]
        self.obs_cov: float | npt.NDArray[float] = obs_par_dict["obs_cov"]
        self.obs_times: list[datetime.datetime] = obs_par_dict["obs_times"]
        self.use_synthetic_obs: bool = obs_par_dict["use_synthetic_obs"]
        self.obs_filenames: list[str] = obs_par_dict["obs_filenames"]
        self.obs_rng_seed: int = obs_par_dict["obs_rng_seed"]
        self.ssw_event:str = obs_par_dict["ssw_event"]
        self.craft:str = obs_par_dict["craft"]
        self.img:str = obs_par_dict["img"]
        self.bias_term_bool: bool = obs_par_dict["bias_term_bool"]
        self.bias_term_5rs: float = obs_par_dict["bias_term_5rs"]
        self.bias_term_21rs: float = obs_par_dict["bias_term_21rs"]

        # Ensure all bias terms exist, if not, set bias_term_bool to False
        if (self.bias_term_5rs is None) or (self.bias_term_21rs is None):
            self.bias_term_bool = False
        self.bias_term_5rs: float = obs_par_dict["bias_term_5rs"]
        self.bias_term_21rs: float = obs_par_dict["bias_term_21rs"]

        # Ensure all bias terms exist, if not, set bias_term_bool to False
        if (self.bias_term_5rs is None) or (self.bias_term_21rs is None):
            self.bias_term_bool = False

        # If we're using synthetic observations/ running OSSEs define true_cme_par_dict
        self.true_cme_par_dict: CmeParEns = initialise_true_cme_par_dict()

        if self.use_synthetic_obs:
            # Generate observations
            self.observations: list[float] = self.get_observations()
            if not isinstance(self.obs_radius, list):
                print("Hi")
                self.obs_radius = [self.obs_radius for _ in range(self.n_obs)]

            if not isinstance(self.obs_lon, list):
                self.obs_lon = [self.obs_lon for _ in range(self.n_obs)]

            if not isinstance(self.obs_lat, list):
                self.obs_lat = [self.obs_lat for _ in range(self.n_obs)]
        else:
            obs_tuple: tuple[
                list[datetime.datetime], list[Quantity[u.deg]], list[float], int
            ] = self.get_observations()

            self.obs_times: list[datetime.datetime] = obs_tuple[0]
            self.obs_lon: list[Quantity[u.deg]] = obs_tuple[1]
            self.observations: list[float] = obs_tuple[2]
            self.n_obs: int = obs_tuple[3]


        # Generate prior cme_parameters
        fg_cme_par_dict = initialise_fg_cme_parameters()
        self.fg_mean_cme_t_init: datetime.datetime = fg_cme_par_dict["fg_cme_t_init"]
        self.fg_mean_cme_speed: float = fg_cme_par_dict["fg_cme_speed"]
        self.fg_mean_cme_width: float = fg_cme_par_dict["fg_cme_width"]
        self.fg_mean_cme_lon: float = fg_cme_par_dict["fg_cme_lon"]
        self.fg_mean_cme_lat: float = fg_cme_par_dict["fg_cme_lat"]
        self.fg_mean_cme_thick: float = fg_cme_par_dict["fg_cme_thick"]

        self.fg_sd_cme_t_init: float = fg_cme_par_dict["sd_cme_t_init"]
        self.fg_sd_cme_speed: float = fg_cme_par_dict["sd_cme_speed"]
        self.fg_sd_cme_width: float = fg_cme_par_dict["sd_cme_width"]
        self.fg_sd_cme_lon: float = fg_cme_par_dict["sd_cme_lon"]
        self.fg_sd_cme_lat: float = fg_cme_par_dict["sd_cme_lat"]
        self.fg_sd_cme_thick: float = fg_cme_par_dict["sd_cme_thick"]

        # Get prior CME standard deviations
        prior_cme_par_dict = initialise_prior_cme_cov()
        self.cme_cov_type: str = prior_cme_par_dict["cme_cov_type"]
        self.scale_corr: bool = prior_cme_par_dict["scale_corr"]
        self.use_log_v: bool = prior_cme_par_dict["use_log_v"]
        self.prior_sd_t_init: float = prior_cme_par_dict["sd_t_init"]
        self.prior_sd_speed: float = prior_cme_par_dict["sd_speed"]
        self.prior_sd_width: float = prior_cme_par_dict["sd_width"]
        self.prior_sd_lon: float = prior_cme_par_dict["sd_lon"]
        self.prior_sd_lat: float = prior_cme_par_dict["sd_lat"]
        self.prior_sd_thick: float = prior_cme_par_dict["sd_thick"]

        # Get DA parameters
        da_par_dict = initialise_da_parameters()
        self.n_members: int = da_par_dict["n_members"]
        self.n_runs: int = 1 #da_par_dict["n_runs"]
        self.delta_aux_pf: float = da_par_dict["delta_aux_pf"]
        self.pars_in_state: list[str] = da_par_dict["pars_in_state"]
        self.time_tolerance: Quantity[u.day] = da_par_dict["time_tolerance"]
        #self.rng: Generator = np.random.default_rng(da_par_dict["rng_seed"])


    def generate_random_seed(
            self,
            add_const: int=None,
            mult_const: int=None,
            run_no: int=None
    ):
        """
        Function to generate random seed for each model run
        :return: random_seed: Random seed for each model run
        """
        if add_const is None:
            add_const = 42

        if mult_const is None:
            mult_const = 10000

        if run_no is None:
            run_no = 0

        if run_no >= 0:
            random_seed = add_const + (mult_const * run_no)
        else:
            random_seed = int(np.abs((add_const + (mult_const / 2.0)) + (run_no * mult_const)))

        return random_seed


    def make_output_dir(self, cme_par_dict: CmeParEns, run_no: int):
        """
        Function to make the output directory
        :return: out_dir: Output directory
        """
        base_dir = os.path.join(
            os.getenv("OUT_BASE_DIR"),
            f"ens_{self.n_members}", self.cme_cov_type
        )
        model_dir = f"model_{self.use_model.upper()}"

        model_dir = f"model_{self.use_model.upper()}"

        if self.use_synthetic_obs:
            obs_dir = (
                f"truth_{self.true_cme_par_dict["v"].value}_{self.true_cme_par_dict["width"].value}"
                f"_{self.true_cme_par_dict["lon"].value}_{self.true_cme_par_dict["lat"].value}"
                f"_{self.true_cme_par_dict["thick"].value}"
            )
        else:
            obs_dir = (
                f"obs_{self.craft}_{self.ssw_event}_{self.img}"
            )

        prior_dir = (
            f"prior_{self.fg_mean_cme_speed:.1f}_{self.fg_mean_cme_width:.1f}"
            f"_{self.fg_mean_cme_lon:.1f}_{self.fg_mean_cme_lat:.1f}"
            f"_{self.fg_mean_cme_thick:.1f}"
        )
        run_dir = f"run_{run_no:03d}"

        if self.bias_term_bool:
            bias_dir = os.path.join(
                "bias_correction", f"bias5_{self.bias_term_5rs}_bias21_{self.bias_term_21rs}"
            )
            output_dir = os.path.join(
                base_dir, model_dir, bias_dir, obs_dir,
                prior_dir, f"{self.delta_aux_pf}", run_dir
            )
        else:
            output_dir = os.path.join(
                base_dir, model_dir, obs_dir, prior_dir,
                f"{self.delta_aux_pf}", run_dir
            )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir


    def get_observations(self) -> (
            list[float] | tuple[list[datetime.datetime], list[Quantity[u.deg]], list[float], int]
    ):
        """
        Function to get observations from files or generate synthetic observations
        :return: observations: Observations to be assimilated
        """
        # Initialise observations
        obs_class = Observations(
            use_model=self.use_model,
            obs_lon=self.obs_lon,
            obs_times_in_datetime=self.obs_times,
            obs_filenames=self.obs_filenames,
            use_synthetic_obs=self.use_synthetic_obs,
            true_cme_par_dict=self.true_cme_par_dict,
            obs_cov=self.obs_cov,
            surf_init_time=self.surf_init_time,
            vr_in=self.vr_in,
            lon_start=self.lon_start,
            lon_stop=self.lon_stop,
            sim_time=self.sim_time,
            dt_scale=self.dt_scale,
            r_min=self.r_min,
            cme_init_rad=self.cme_init_rad,
            cme_fixed_duration=self.cme_fixed_duration,
            fixed_duration=self.fixed_duration,
            plot_surf_output=False,
            obs_rng_seed=self.obs_rng_seed,
            ssw_event=self.ssw_event,
            craft=self.craft,
            img=self.img,
        )

        # Get observations and update obs_times, n_obs and obs_lon if necessary
        if self.use_synthetic_obs:
            obs_out = obs_class.observations
            return obs_out
        else:
            obs_out = obs_class.observations
            obs_times_out = obs_class.obs_times_in_datetime
            obs_lon_out = obs_class.obs_lon
            n_obs = len(obs_out)

            assert len(obs_times_out) == n_obs
            assert len(obs_lon_out) == n_obs

            return obs_times_out, obs_lon_out, obs_out, n_obs


    def initialise_mean_cme_pars(self, rng):
        #######################################################################
        if get_use_synthetic_obs():
            # Initialise CME parameters
            mean_cme_t_init = self.fg_mean_cme_t_init
            mean_cme_speed = self.fg_mean_cme_speed + (
                    self.fg_sd_cme_speed * rng.uniform(low=-1, high=1)
            )
            mean_cme_width = self.fg_mean_cme_width + (
                    self.fg_sd_cme_width * rng.uniform(low=-1, high=1)
            )
            mean_cme_lon = self.fg_mean_cme_lon + (self.fg_sd_cme_lon * rng.uniform(low=-1, high=1))
            mean_cme_lat = self.fg_mean_cme_lat
            mean_cme_thick = self.fg_mean_cme_thick
        else:
            # Initialise CME parameters
            mean_cme_t_init = self.fg_mean_cme_t_init + datetime.timedelta(
                seconds=self.fg_sd_cme_t_init * rng.uniform(low=-1, high=1)
            )
            mean_cme_t_init_str = mean_cme_t_init.strftime("%Y%m%d-%H%M")
            mean_cme_speed = self.fg_mean_cme_speed + (
                self.fg_sd_cme_speed * rng.uniform(low=-1, high=1)
            )
            mean_cme_width = self.fg_mean_cme_width + (
                    self.fg_sd_cme_width * rng.uniform(low=-1, high=1)
            )
            mean_cme_lon = self.fg_mean_cme_lon + (self.fg_sd_cme_lon * rng.uniform(low=-1, high=1))
            mean_cme_lat = self.fg_mean_cme_lat + (self.fg_sd_cme_lat * rng.uniform(low=-1, high=1))
            mean_cme_thick = self.fg_mean_cme_thick + (
                    self.fg_sd_cme_thick * rng.uniform(low=-1, high=1)
            )

        mean_cme_par_dict = {
            "mean_cme_t_init": mean_cme_t_init,
            "mean_cme_speed": mean_cme_speed,
            "mean_cme_width": mean_cme_width,
            "mean_cme_lon": mean_cme_lon,
            "mean_cme_lat": mean_cme_lat,
            "mean_cme_thick": mean_cme_thick,
        }

        return mean_cme_par_dict


    def make_cme_par_dict(self, rng):
        # Get mean CME parameters
        mean_cme_par_dict = self.initialise_mean_cme_pars(rng=rng)
        mean_cme_t_init = (
            mean_cme_par_dict["mean_cme_t_init"] - self.surf_init_time
        ).total_seconds()
        mean_cme_speed = mean_cme_par_dict["mean_cme_speed"]
        mean_cme_width = mean_cme_par_dict["mean_cme_width"]
        mean_cme_lon = mean_cme_par_dict["mean_cme_lon"]
        mean_cme_lat = mean_cme_par_dict["mean_cme_lat"]
        mean_cme_thick = mean_cme_par_dict["mean_cme_thick"]

        mean_cme_par_array = [
            mean_cme_t_init,
            mean_cme_speed,
            mean_cme_width,
            mean_cme_lon,
            mean_cme_lat,
            mean_cme_thick
        ]

        # Initialise cme_par_dict
        cme_par_dict = initialise_cme_parameter_ensemble_dict(
            n_ensemble=self.n_members, surf_init_time=self.surf_init_time
        )

        if self.cme_cov_type == "uncorr":
            print(mean_cme_par_array)
            samples = mp_cov.make_uncorr_uniform_samples(
                n_ens=self.n_members,
                mean_cme_pars=mean_cme_par_array,
                rng=rng,
                sd_t_init=self.prior_sd_t_init,
                sd_v=self.prior_sd_speed,
                sd_width=self.prior_sd_width,
                sd_lon=self.prior_sd_lon,
                sd_lat=self.prior_sd_lat,
                sd_thick=self.prior_sd_thick,
            )
            # samples = mp_cov.make_uncorrelated_samples(
            #     n_ens=self.n_members,
            #     mean_cme_pars=mean_cme_par_array,
            #     rng=rng,
            #     sd_t_init=self.prior_sd_t_init,
            #     sd_v=self.prior_sd_speed,
            #     sd_width=self.prior_sd_width,
            #     sd_lon=self.prior_sd_lon,
            #     sd_lat=self.prior_sd_lat,
            #     sd_thick=self.prior_sd_thick,
            # )

        elif self.cme_cov_type == "donki":
            start_time_donki = datetime.datetime(2017, 1, 1, 0, 0, 0)
            end_time_donki = datetime.datetime(2026, 2, 1, 0, 0, 0)

            samples = mp_cov.make_donki_samples(
                n_ens=self.n_members,
                mean_cme_pars=mean_cme_par_array,
                rng=rng,
                start_time=start_time_donki,
                end_time=end_time_donki,
                scale_corr=self.scale_corr,
                vars_req=np.array(self.pars_in_state),
                sd_t_init=self.prior_sd_t_init,
                sd_v=self.prior_sd_speed,
                sd_width=self.prior_sd_width,
                sd_lon=self.prior_sd_lon,
                sd_lat=self.prior_sd_lat,
                sd_thick=self.prior_sd_thick,
                use_log_v=self.use_log_v,
                plot_cme_cov=False,
                most_acc_only="true",
                catalog="ALL",
                feature="LE"
            )

        elif self.cme_cov_type == "mo_cone":
            mo_cone_cov_dir = os.environ.get("MO_CONE_COV_DIR")

            if not os.path.exists(mo_cone_cov_dir):
                os.makedirs(mo_cone_cov_dir)

            mo_cme_cone_file_dir = os.environ.get("MO_CME_CONE_FILE_DIR")

            start_time_mo_cone = datetime.datetime(2017, 1, 1, 0, 0, 0)
            end_time_mo_cone = datetime.datetime(2026, 2, 1, 0, 0, 0)

            samples = mp_cov.make_mo_cone_samples(
                n_ens=self.n_members,
                mean_cme_pars=mean_cme_par_array,
                rng=rng,
                mo_cone_file_dir=mo_cme_cone_file_dir,
                start_time=start_time_mo_cone,
                end_time=end_time_mo_cone,
                overwrite_cov=False,
                mo_cone_cov_dir=mo_cone_cov_dir,
                scale_corr=self.scale_corr,
                vars_req=np.array(self.pars_in_state),
                sd_t_init=self.prior_sd_t_init,
                sd_v=self.prior_sd_speed,
                sd_width=self.prior_sd_width,
                sd_lon=self.prior_sd_lon,
                sd_lat=self.prior_sd_lat,
                sd_thick=self.prior_sd_thick,
                use_log_v=self.use_log_v,
                plot_cme_cov=False,
            )
        else:
            print(f"cme_cov_type provided was: {self.cme_cov_type}")
            print(f"cme_cov_type must be in {allowed_cme_cov_types}.")
            sys.exit()

        cme_par_dict["t_init"] = [
            self.surf_init_time + datetime.timedelta(seconds=samples[0, i])
            for i in range(self.n_members)
        ]
        cme_par_dict["v"] = list(samples[1, :]) * u.km / u.s
        cme_par_dict["width"] = list(samples[2, :]) * u.deg
        cme_par_dict["lon"] = list(samples[3, :]) * u.deg
        cme_par_dict["lat"] = list(samples[4, :]) * u.deg
        cme_par_dict["thick"] = list(samples[5, :]) * u.solRad

        return cme_par_dict


    def run_data_assimilation(self, run_start=0):
        """
        Function to run the data assimilation routine
        :return: None
        """
        if run_start < 0:
            self.n_runs = -self.n_runs
            step = -1
        else:
            step = 1

        for run_no in range(run_start, run_start + self.n_runs, step):
            print(f"run_no = {run_no}")
            if run_no < 0:
                self.fg_sd_cme_t_init: float = 0
                self.fg_sd_cme_speed: float = 0
                self.fg_sd_cme_width: float = 0
                self.fg_sd_cme_lon: float = 0
                self.fg_sd_cme_lat: float = 0
                self.fg_sd_cme_thick: float = 0

            # Initialise random generator
            rand_seed = self.generate_random_seed(
                add_const=42,
                mult_const=10000,
                run_no=run_no
            )
            rng: Generator = np.random.default_rng(rand_seed)

            cme_par_dict = self.make_cme_par_dict(rng=rng)
            cme_saved_pars = np.zeros((self.n_obs + 1, self.n_members, 7))

            # Save prior state
            cme_saved_pars[0, :, 0] = [
                (
                        cme_par_dict["t_init"][i] - cme_par_dict["surf_init_time"]
                ).total_seconds()
                for i in range(self.n_members)
            ]
            cme_saved_pars[0, :, 1] = cme_par_dict["v"].to(u.km / u.s).value
            cme_saved_pars[0, :, 2] = cme_par_dict["width"].to(u.deg).value

            cme_saved_pars[0, :, 3] = cme_par_dict["lon"].to(u.deg).value
            lon_cond = cme_saved_pars[0, :, 3] > 180
            cme_saved_pars[0, lon_cond, 3] = (
                    cme_saved_pars[0, lon_cond, 3] - 360
            )

            cme_saved_pars[0, :, 4] = cme_par_dict["lat"].to(u.deg).value
            cme_saved_pars[0, :, 5] = cme_par_dict["thick"].to(u.solRad).value
            cme_saved_pars[0, :, 6] = cme_par_dict["log_weight"]

            # Make output directory
            output_dir = self.make_output_dir(
                cme_par_dict=cme_par_dict, run_no=run_no
            )
            #print(f"obs={self.observations}")
            #print(f"Input bias_term_bool = {self.bias_term_bool}")
            for yi, obs in enumerate(self.observations):
                print(f"run_no = {run_no}/{self.n_runs}, obs = {yi}/{len(self.observations)}")
                #print(f"obs = {obs}, obs_lon = {self.obs_lon[yi]}, obs_time = {self.obs_times[yi]}")
                aux_pf_class = AuxPF(
                    use_model=self.use_model,
                    cme_par_dict=cme_par_dict,
                    obs=obs,
                    obs_cov=self.obs_cov,
                    obs_radius=self.obs_radius[yi],
                    obs_lon=self.obs_lon[yi],
                    obs_lat=self.obs_lat[yi],
                    obs_time=self.obs_times[yi],
                    true_cme_par_dict=self.true_cme_par_dict,
                    infl_fact=1,
                    pars_in_state=self.pars_in_state,
                    vr_in=self.vr_in,
                    lon_start=self.lon_start,
                    lon_stop=self.lon_stop,
                    time_tolerance=self.time_tolerance,
                    dt_scale=self.dt_scale,
                    delta_aux_pf=self.delta_aux_pf,
                    r_min=self.r_min,
                    cme_init_rad=self.cme_init_rad,
                    cme_fixed_duration = self.cme_fixed_duration,
                    fixed_duration=self.fixed_duration,
                    bias_term_bool=self.bias_term_bool,
                    bias_term_5rs=self.bias_term_5rs,
                    bias_term_21rs=self.bias_term_21rs,
                    rng=rng
                )

                cme_par_dict = aux_pf_class.aux_pf()

                # Standardise the units and remove the astropy units
                cme_saved_pars[yi + 1, :, 0] = [
                    (
                        cme_par_dict["t_init"][i] - cme_par_dict["surf_init_time"]
                    ).total_seconds()
                    for i in range(self.n_members)
                ]
                # print(cme_par_dict['width'])
                cme_saved_pars[yi + 1, :, 1] = cme_par_dict["v"].to(u.km / u.s).value
                cme_saved_pars[yi + 1, :, 2] = cme_par_dict["width"].to(u.deg).value

                cme_saved_pars[yi + 1, :, 3] = cme_par_dict["lon"].to(u.deg).value
                lon_cond = cme_saved_pars[yi + 1, :, 3] > 180
                cme_saved_pars[yi + 1, lon_cond, 3] = (
                        cme_saved_pars[yi + 1, lon_cond, 3] - 360
                )

                cme_saved_pars[yi + 1, :, 4] = cme_par_dict["lat"].to(u.deg).value
                cme_saved_pars[yi + 1, :, 5] = cme_par_dict["thick"].to(u.solRad).value
                cme_saved_pars[yi + 1, :, 6] = cme_par_dict["log_weight"]

                # for par_ind in [1, 2, 3]:
                #     print(f"cme_saved_pars[{yi + 1}] = {cme_saved_pars[yi + 1, :, par_ind]}")

            for par_ind in range(7):
                if par_ind == 6:
                    print(f"weights_post = {np.exp(cme_saved_pars[-1, :, par_ind])}")
                print(f"mean_cme_saved_pars[{par_ind}] = {np.mean(cme_saved_pars[:, :, par_ind], axis=1)}")

            # Save cme_parameters into a .nc file
            # Make an xarray object
            cme_par_ds = xr.Dataset(
                data_vars=dict(
                    model_init_time=self.surf_init_time,
                    cme_init_rad=self.cme_init_rad,
                    r_min=self.r_min,
                    ambient_vr=(["n_lon"], self.vr_in),
                    t_init=(["n_obs", "n_ens"], cme_saved_pars[:, :, 0]),
                    v=(["n_obs", "n_ens"], cme_saved_pars[:, :, 1]),
                    width=(["n_obs", "n_ens"], cme_saved_pars[:, :, 2]),
                    lon=(["n_obs", "n_ens"], cme_saved_pars[:, :, 3]),
                    lat=(["n_obs", "n_ens"], cme_saved_pars[:, :, 4]),
                    thick=(["n_obs", "n_ens"], cme_saved_pars[:, :, 5]),
                    log_weight=(["n_obs", "n_ens"], cme_saved_pars[:, :, 6]),
                ),
                coords=dict(
                    obs_no=("n_obs", range(self.n_obs + 1)),
                    ens_no=("n_ens", range(self.n_members)),
                    surf_lon=("n_lon", (2 * np.pi / self.n_lon) * np.arange(self.n_lon)),
                ),
            )
            print(cme_par_ds)
            outParFile = os.path.join(output_dir, "cme_pars.nc")
            cme_par_ds.to_netcdf(outParFile, mode='w')

        return None


def main():
    print_environment_variables()
    run_da_class = RunDataAssimilationRoutine()

    run_start = 0
    da_par_dict = initialise_da_parameters()
    n_runs = da_par_dict["n_runs"]
    pool_size = os.cpu_count() - 1


    if get_use_synthetic_obs():
        with Pool(pool_size) as pool:
            pool.map(
                run_da_class.run_data_assimilation,
                range(run_start, run_start + n_runs)
            )
    else:
        run_da_class.run_data_assimilation(run_start=-1)

    return None

if __name__ == "__main__":
    main()