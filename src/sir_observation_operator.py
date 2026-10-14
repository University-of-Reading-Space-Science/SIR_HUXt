# @package sir_observation_operator
# This module will take a HUXt/SURF and ConeCME object and calculate its elongation profile
#  for use as an observation operator in the data assimilation algorithm
import numpy as np
import numpy.typing as npt
import pandas as pd
import sys
import os

import surf.surf as S
import surf.surf_inputs as Sin
import surf.surf_analysis as SA
import surf.surf_imaging as Sim

import huxt.huxt as H
import huxt.huxt_analysis as HA

import datetime
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
import matplotlib.pyplot as plt

from init_sir import setup_huxt, setup_surf, setup_surf3d, initialise_cme_parameter_ensemble_dict
from cme_par_ens import CmeParEns
from cme_par_dict_structure import required_dict_keys

import asyncio
from functools import partial
import time
from multiprocessing import Pool


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(
            None,
            partial(f, *args, **kwargs)
        )

    return wrapped


class SynthObsEphem:
    """
    Class to hold observation time, radius, latitude and longitude for use in initialising
      Sim.SyntheticImager class
    """
    def __init__(self, surf_init_time, obs_times, obs_rads, obs_lats, obs_lons):
        if isinstance(obs_times.value, float):
            self.datetime = np.array([surf_init_time + datetime.timedelta(seconds=obs_times.value)])
            self.time = np.array([obs_times.value]) * u.s
            self.r = np.array([obs_rads.to(u.m).value]) * u.m
            self.lat = np.array([obs_lats.to(u.rad).value]) * u.rad
            self.lon = np.array([obs_lons.to(u.rad).value]) * u.rad
        else:
            self.datetime = [
                surf_init_time + datetime.timedelta(seconds=i) for i in obs_times.value
            ]
            self.time = obs_times
            self.r = obs_rads
            self.lat = obs_lats
            self.lon = obs_lons


class ObserverTracer:
    """
    The function of this observer class is to provide pseudo-observations of a ConeCME's flank elongation from a SURF simulation
    for an observer at a specified longitude relative to Earth. The observers distance defaults to 1 AU, and has the same latitude as Earth
    during the SURF run.
    Author: Luke Barnard
    """
    @u.quantity_input(longitude=u.deg)
    def __init__(
            self,
            use_model: str,
            model: S.SURF | H.HUXt,
            cme: S.ConeCME | H.ConeCME,
            longitude: Quantity[u.deg],
            el_min: float=4.0,
            el_max: float=30.0
    ):
        self.use_model: str = use_model
        if self.use_model in ["surf", "compress_surf"]:
            ert_ephem: S.Observer = model.get_observer('EARTH')
        elif self.use_model in ["huxt"]:
            ert_ephem: H.Observer = model.get_observer('EARTH')
        else:
            sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")


        self.time: npt.NDArray[Time] = ert_ephem.time
        self.r: npt.NDArray[Quantity[u.AU]] = ert_ephem.r * 0 + 1 * u.AU
        self.lon: npt.NDArray[Quantity[u.deg]] = ert_ephem.lon + longitude
        self.lat: npt.NDArray[Quantity[u.deg]] = ert_ephem.lat
        self.el_min: float = el_min
        self.el_max: float = el_max

        # Force longitude into 0-360 domain
        id_over: list[bool] | bool = self.lon > 360 * u.deg
        id_under: list[bool] | bool = self.lon < 0 * u.deg
        if np.any(id_over):
            self.lon[id_over] = self.lon[id_over] - 360 * u.deg

        if np.any(id_under):
            self.lon[id_under] = self.lon[id_under] + 360 * u.deg

        self.model_flank: pd.DataFrame = self.compute_flank_profile(cme)


    def compute_flank_profile(
            self,
            cme: H.ConeCME | S.ConeCME
    ) -> pd.DataFrame:
        """
        Compute the time elongation profile of the flank of a ConeCME in SURF. The observer longtiude is specified
        relative to Earth but otherwise matches Earth's coords.

        Parameters
        ----------
        cme: A ConeCME object from a completed SURF run (i.e the ConeCME.coords dictionary has been populated).
        Returns
        -------
        obs_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STA's perspective, including the
                    time, elongation, position angle, and HEEQ radius and longitude.
        """
        times: Time = Time([coord['time'] for i, coord in cme.coords.items()])

        # Compute observers location using earth ephem, adding on observers longitude offset from Earth
        # and correct for runover 2*pi
        flank: pd.DataFrame = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon'])
        flank['time'] = times.jd

        for i, coord in cme.coords.items():

            if len(coord['r']) == 0:
                flank.loc[i, ['lon', 'r', 'el']] = np.nan
                continue

            r_obs: npt.NDArray[Quantity[u.AU]] = self.r[i]
            x_obs: npt.NDArray[Quantity[u.AU]] = self.r[i] * np.cos(self.lat[i]) * np.cos(self.lon[i])
            y_obs: npt.NDArray[Quantity[u.AU]] = self.r[i] * np.cos(self.lat[i]) * np.sin(self.lon[i])
            z_obs: npt.NDArray[Quantity[u.AU]] = self.r[i] * np.sin(self.lat[i])

            lon_cme: npt.NDArray[Quantity[u.deg]] = coord['lon']
            lat_cme: npt.NDArray[Quantity[u.deg]] = coord['lat']
            r_cme: npt.NDArray[Quantity[u.AU]] = coord['r']

            x_cme: npt.NDArray[Quantity[u.AU]] = r_cme * np.cos(lat_cme) * np.cos(lon_cme)
            y_cme: npt.NDArray[Quantity[u.AU]] = r_cme * np.cos(lat_cme) * np.sin(lon_cme)
            z_cme: npt.NDArray[Quantity[u.AU]] = r_cme * np.sin(lat_cme)

            #########################################################
            # Compute the observer CME distance, S, and elongation
            x_cme_s: npt.NDArray[Quantity[u.AU]] = x_cme - x_obs
            y_cme_s: npt.NDArray[Quantity[u.AU]] = y_cme - y_obs
            z_cme_s: npt.NDArray[Quantity[u.AU]] = z_cme - z_obs
            s: npt.NDArray[Quantity[u.AU]] = np.sqrt(x_cme_s ** 2 + y_cme_s ** 2 + z_cme_s ** 2)

            numer: npt.NDArray[float] = (r_obs ** 2 + s ** 2 - r_cme ** 2).value
            denom: npt.NDArray[float] = (2.0 * r_obs * s).value
            e_obs: npt.NDArray[float] = np.arccos(numer / denom)

            # Restrict those CME points to those in FOV
            # For those ahead of Earth, this is negative y_cme_s
            # For those behind Earth, this is positive y_cme_s
            if self.lon[i] < np.pi * u.rad:
                id_sub: list[bool] = y_cme_s.value < 0
                e_obs = e_obs[id_sub]
                lon_cme = lon_cme[id_sub]
                r_cme = r_cme[id_sub]
            elif self.lon[i] > np.pi * u.rad:
                id_sub = y_cme_s.value > 0
                e_obs = e_obs[id_sub]
                lon_cme = lon_cme[id_sub]
                r_cme = r_cme[id_sub]

            # Find the flank coordinate and update output
            id_obs_flank: int = int(np.argmax(e_obs))
            flank.loc[i, 'lon'] = lon_cme[id_obs_flank].value
            flank.loc[i, 'r'] = r_cme[id_obs_flank].value
            flank.loc[i, 'el'] = np.rad2deg(e_obs[id_obs_flank])

        # Force values to be floats.
        keys: list[str] = ['lon', 'r', 'el']
        flank[keys] = flank[keys].astype(np.float64)

        return flank


class ObserverSynthHI:
    """
    The function of this observer class is to provide pseudo-observations of a ConeCME's flank elongation from a SURF simulation
    for an observer at a specified longitude relative to Earth. The observers distance defaults to 1 AU, and has the same latitude as Earth
    during the SURF run.
    Author: Matthew Lang, based off code written by Luke Barnard
    """
    @u.quantity_input(longitude=u.deg)
    def __init__(
            self,
            use_model: str,
            model: S.SURF,
            obs_time: npt.NDArray[datetime.datetime],
            craft: str=None,
            psi_ert_avg: float=90.0,
            obs_radius: npt.NDArray[Quantity[u.AU]]=1.0 * u.AU,
            obs_longitude: npt.NDArray[Quantity[u.deg]]=300.0 * u.deg,
            obs_latitude: npt.NDArray[Quantity[u.deg]]=0.0 * u.deg,
            el_min: float=4.0,
            el_max: float=30.0,
            hi_resolution: Quantity[u.s] | Quantity[u.hour] | Quantity[u.day]=1.0 * u.hour,
    ):
        """
        Class to generate synthetic HI profiles and extract the elongation profiles
        :param use_model: The model type being used in this simulation (must be 'compress_surf')
        :param model: The model after it has been solved with the CMEs required
        :param cme: CME object
        :param obs_time: Observation time
        :param obs_radius: Observation radius
        :param obs_longitude: Observation longitude
        :param obs_latitude: Observation latitude
        :param el_min: Minimum elongation
        :param el_max: Maximum elongation
        :param hi_resolution: Temporal resolution to model the HI data in
        """

        self.use_model: str = use_model
        if self.use_model in ["compress_surf", "compress_surf3d"]:
            ert_ephem: S.Observer = model.get_observer('EARTH')
        else:
            sys.exit("Synthetic HI can only be generated with use_model = 'compress_surf' and 'compress_surf3d'")

        self.obs_time: npt.NDArray[datetime.datetime] = obs_time

        if self.use_model == "compress_surf":
            surf_model_init_time: datetime.datetime = model.time_init.datetime
            surf_model_sim_time: float = float(model.simtime.to(u.s).value)
        else:
            surf_model_init_time: datetime.datetime = model.SURFlat[0].time_init.datetime
            surf_model_sim_time: float = float(model.SURFlat[0].simtime.to(u.s).value)

        try:
            obs_time_from_start_sec = [
                (ot - surf_model_init_time).total_seconds() for ot in self.obs_time
            ]
        except:
            obs_time_from_start_sec = [
                (self.obs_time - surf_model_init_time).total_seconds()
            ]
            self.obs_time = np.array([self.obs_time])
        else:
            obs_time_from_start_sec = [
                (ot - surf_model_init_time).total_seconds() for ot in self.obs_time
            ]
        self.obs_time_sec: npt.NDArray[Quantity[u.s]] = np.array(obs_time_from_start_sec) * u.s #ert_ephem.time
        self.obs_time_day: npt.NDArray[Quantity[u.day]] = self.obs_time_sec.to(u.day)
        self.n_obs = len(self.obs_time)
        self.el_min: float = el_min
        self.el_max: float = el_max

        # Get Heliospheric Imager resolution required
        hi_resolution_hr = hi_resolution.to(u.hour).value
        hi_resolution_sec = hi_resolution.to(u.s).value
        hi_resolution_day = hi_resolution.to(u.day).value

        if use_model == "compress_surf":
            self.model_times = model.time

        elif use_model == "compress_surf3d":
            self.model_times = model.SURFlat[0].time

        self.synth_obs_times = np.arange(
            0,
            surf_model_sim_time + (hi_resolution_sec / 2.0),
            hi_resolution_sec
        ) * u.s

        self.n_synth_obs_times = len(self.synth_obs_times)
        self.ast_times = Time([
            surf_model_init_time + datetime.timedelta(seconds=s)
            for s in self.model_times.value
        ])

        if craft is not None:
            self.craft_observer = model.get_observer(craft)
        else:
            self.obs_times: npt.NDArray[Time] = Time([
            surf_model_init_time + datetime.timedelta(seconds=s)
            for s in self.synth_obs_times.value
            ])

            self.obs_r: npt.NDArray[Quantity[u.AU]] = obs_radius * np.ones(self.n_synth_obs_times)
            self.obs_lon: npt.NDArray[Quantity[u.deg]] = obs_longitude * np.ones(self.n_synth_obs_times)
            self.obs_lat: npt.NDArray[Quantity[u.deg]] = obs_latitude * np.ones(self.n_synth_obs_times)

            # Make dictionary for input into the Observer class for a synthetic observer
            synth_heeq_pos = {
                "time": self.obs_times,
                "r": self.obs_r,
                "lon": self.obs_lon,
                "lat": self.obs_lat,
            }
            self.craft_observer = S.Observer(
                body="SYNTHETIC",
                times=self.ast_times,
                synth_heeq_pos=synth_heeq_pos,
            )

        synth_obs_class = SynthObsEphem(
            surf_init_time=surf_model_init_time,
            obs_times=self.synth_obs_times,
            obs_rads=self.obs_r,
            obs_lats=self.obs_lat,
            obs_lons=self.obs_lon,
        )

        # Calculate the j-maps and delta j-maps and track the CME
        if use_model == "compress_surf":
            synth_imager_class = Sim.SyntheticImager2D(synth_obs_class)
            jmap, djmap = synth_imager_class.compute_jmap(model)
            cme_profile = synth_imager_class.track_cmes(model, djmap)[0]
            #fig, ax = synth_imager_class.plot_jmap(model, jmap, djmap)
            #plt.show()

            #print(cme_profile)
            # Extract time and elongation profiles of synthetic HI-data
            try:
                synth_imager_time = cme_profile["feature_00"]["t"][:]
                synth_imager_elon = cme_profile["feature_00"]["e"][:]
            except:
                synth_imager_time = np.nan
                synth_imager_elon = np.nan
            else:
                synth_imager_time = cme_profile["feature_00"]["t"][:]
                synth_imager_elon = cme_profile["feature_00"]["e"][:]

        elif use_model == "compress_surf3d":
            synth_imager_class = Sim.SyntheticImager3DOrig(
                observer=self.craft_observer,
                pa=psi_ert_avg,
                elon_min=el_min,
                elon_max=el_max
            )
            jmap, djmap = synth_imager_class.compute_jmap(model)
            cme_profile = synth_imager_class.track_cmes(model, djmap)[0]
            #fig, ax = synth_imager_class.plot_jmap(model, jmap, djmap)
            #plt.show()

            # Extract time and elongation profiles of synthetic HI-data
            try:
                synth_imager_time = cme_profile["feature_00"]["t"][:]
                synth_imager_elon = cme_profile["feature_00"]["e"][:]
            except:
                synth_imager_time = np.nan
                synth_imager_elon = np.nan
            else:
                synth_imager_time = cme_profile["feature_00"]["t"][:]
                synth_imager_elon = cme_profile["feature_00"]["e"][:]

        # Interpolate the synthetic elongations to observation times
        cond1 = np.isnan(synth_imager_time)
        cond2 = np.isnan(synth_imager_elon)
        all_nan_cond = cond1.all() or cond2.all()
        if not all_nan_cond:
            synth_imager_elon_interp = np.interp(
                self.obs_time_day.value, synth_imager_time, synth_imager_elon
            )
        else:
            synth_imager_elon_interp = np.nan * np.ones(self.n_obs)

        flank_data = {
            "time": obs_time,
            "el": synth_imager_elon_interp,
            "r": np.nan * np.ones(self.n_obs),
            "lon": np.nan * np.ones(self.n_obs)
        }

        self.model_flank: pd.DataFrame = pd.DataFrame(flank_data)


class ObservationOperator:
    def __init__(
            self,
            use_model: str,
            cme_par_dict: CmeParEns,
            obs_time_in_datetime: list[datetime.datetime] | datetime.datetime,
            obs_cov: npt.NDArray[float] | float,
            craft: str | None = None,
            obs_radius: list[Quantity[u.AU]] | Quantity[u.AU]=1 * u.AU,
            obs_lon: list[Quantity[u.deg]] | Quantity[u.deg]=0 * u.deg,
            obs_lat: list[Quantity[u.deg]] | Quantity[u.deg]=0 * u.deg,
            surf_init_time: datetime.datetime=datetime.datetime(2008, 1, 1, 0, 0, 0),
            vr_in: npt.NDArray[Quantity[u.km / u.s]]=np.nan * u.km / u.s,
            vr_longs: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
            vr_lats: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
            br_in: npt.NDArray[Quantity[u.nT]] | Quantity[u.nT]=np.nan * u.nT,
            br_longs: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
            br_lats: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
            lon_start: Quantity[u.deg]=290 * u.deg,
            lon_stop: Quantity[u.deg]=430 * u.deg,
            n_lon_surf: int=128,
            lat_min: Quantity[u.deg]=np.nan * u.deg,
            lat_max: Quantity[u.deg]=np.nan * u.deg,
            n_lat_surf: int=45,
            sim_time: Quantity[u.day]=5.0 * u.day,
            dt_scale: int | float=1.0,
            r_min: Quantity[u.solRad]=21.5 * u.solRad,
            cme_init_rad: Quantity[u.solRad]=12 * u.solRad,
            cme_fixed_duration:bool=True,
            fixed_duration: Quantity[u.s] = 12 * 60 * 60 * u.s,
            plot_surf_output: bool = False,
            bias_term_bool: bool = False,
            bias_term_5rs: float = None,
            bias_term_21rs: float = None,
            use_parallel: bool = True,
    ) -> None:
        """
        Class to get observations operator for DA from
         (what model thinks observations are)
        :param use_model: String to determine whether to use SURF, Compressible SURF or HUXt
        :param obs_time_in_datetime: Times observations are taken
        :param obs_cov: To be used with synthetic observations, an observation covariance matrix to be used to
            add random noise to  truth CME elongations
        :param craft: Spacecraft that HI observations are being taken from.
            Possible inputs are Valid bodies are Earth, Venus, Mercury, STA, STB and twin.
            (Twin for specifying synthetic observations)
        :param obs_radius: Observation radius in AU
        :param obs_lon: Longitude of observation source in degrees
        :param obs_lat: Latitude of observation source in degrees
        :param cme_par_dict: To be used to create synthetic observations, a dictionary containing true parameters
            to create a truth CME simulation
        :param surf_init_time: Initial datetime for SURF model
        :param vr_in: Velocities at the inner boundary of the SURF model domain
        :param lon_start: Longitude where we start the SURF simulation
        :param lon_stop: Longitude where we stop the SURF simulation
        :param sim_time: How long the SURF simulation will run for
        :param dt_scale: Frequency with which the model will output
        :param r_min: Minimum radius of SURF simulation
        :param cme_init_rad: Initial radius the CME is observed at
        :param cme_fixed_duration: Boolean to determine whether CMEs will be input with fixed duration
        :param fixed_duration: If CME is fixed duration, this variable defines that fixed duration in seconds
        :param plot_surf_output: Boolean to determine whether to plot SURF output at observation time
        :param bias_term_bool: Boolean to determine whether to use bias term or not
        :param bias_term_5rs: Float to determine bias term at 5Rs
        :param bias_term_21rs: Float to determine bias term at 21Rs
        :param bias_term_5rs: Float to determine bias term at 5Rs
        :param bias_term_21rs: Float to determine bias term at 21Rs
        :param use_parallel: Boolean to determine whether to use multiprocessing
        :TODO: CHANGED SUCH THAT MULTIPLE LONGITUDES AND RADII CAN BE
               INPUT AND TIMES OF OBSERVATIONS ARE TAKEN FROM INPUT FILE
               IN CASE OF REAL OBS
        :TODO: CHANGE cme_init_rad SUCH THAT IT CAN BE VARIED BETWEEN ENSEMBLE MEMBERS
        """

        # Get model to use and whether to use compressible SURF, incompressible SURF or HUXt;
        #  and define the solver accordingly.
        self.use_model = use_model.lower()
        assert (self.use_model in ["surf", "surf3d", "compress_surf", "compress_surf3d", "huxt"])

        if (use_model == "surf") or (use_model == "surf3d"):
            self.solver = "huxt"
        elif (use_model == "compress_surf") or (use_model == "compress_surf3d"):
            self.solver = "hydro"
        else:
            self.solver = "huxt"

        assert (self.solver in ["huxt", "hydro"])

        assert obs_time_in_datetime is not None
        self.obs_time_in_datetime: datetime.datetime | list[datetime.datetime] = obs_time_in_datetime
        self.obs_time_in_jd = Time(self.obs_time_in_datetime, format='datetime').jd

        if isinstance(self.obs_time_in_datetime, datetime.datetime):
            self.obs_time_in_sec = (self.obs_time_in_datetime - surf_init_time).total_seconds() * u.s
            self.n_obs_in_time = 1
        elif isinstance(self.obs_time_in_datetime, list):
            # Store as a single Quantity array (SynthObsEphem needs .value on it)
            self.obs_time_in_sec = np.array([
                (t - surf_init_time).total_seconds() for t in self.obs_time_in_datetime
            ]) * u.s
            self.n_obs_in_time = len(self.obs_time_in_datetime)
        else:
            sys.exit(
                "Expected self.obs_time_in_datetime to be a datetime.datetime"
                " or a list. Exiting..."
            )

        # Define all variables required to initialise SURF are provided
        # (the previous `assert (all([...]) is not None)` was always True, and obs_cov=0 is valid)
        assert cme_par_dict is not None, "cme_par_dict must be provided"
        assert obs_cov is not None, "obs_cov must be provided"
        assert all(p in cme_par_dict.keys() for p in required_dict_keys())

        # Read in CME parameter dictionary and obtain number of ensemble members
        self.cme_par_dict = cme_par_dict
        self.obs_cov = obs_cov
        self.n_members = self.cme_par_dict["n_members"]

        # Initialise ambient solar wind conditions for SURF
        self.vr_in = vr_in
        self.vr_longs = vr_longs
        self.vr_lats = vr_lats
        self.br_in = br_in
        self.br_longs = br_longs
        self.br_lats = br_lats

        # Initialise SURF input parameters
        self.surf_init_time = surf_init_time
        self.lon_start = lon_start
        self.lon_stop = lon_stop
        self.n_lon_surf = n_lon_surf
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.n_lat_surf = n_lat_surf
        self.sim_time = sim_time
        self.dt_scale = dt_scale
        self.r_min = r_min
        self.cme_init_rad = cme_init_rad
        self.cme_fixed_duration = cme_fixed_duration
        self.fixed_duration = fixed_duration

        self.carr_rot, self.carr_rot_lon_init = Sin.datetime2surfinputs(surf_init_time)

        if self.lon_start > self.lon_stop:
            self.lon_stop = self.lon_stop + 360 * u.deg

        if not (np.isnan(self.lat_min) or np.isnan(self.lat_max)):
            assert (self.lat_min < self.lat_max)

        # Position angle of Earth as seen from the observer (only overwritten for
        #  compress_surf3d with a real craft). Default matches SURF's SyntheticImager default.
        self.psi_ert_avg: float = 90.0

        # Initialise observation related quantities.
        #  craft=None is treated like "twin": use the observer position provided.
        if (craft is None) or (str(craft).lower() == "twin"):
            self.obs_radius = obs_radius
            self.obs_lon = obs_lon
            self.obs_lat = obs_lat
        else:
            # Generate dummy model instance to get the craft's and Earth's position
            dum_model = S.SURF(
                v_boundary=np.ones(128) * 400 * u.km / u.s,
                cr_num=self.carr_rot,
                cr_lon_init=self.carr_rot_lon_init,
            )
            craft_observer = dum_model.get_observer(craft)
            craft_observer_datetime: datetime.datetime = [
                surf_init_time + datetime.timedelta(seconds=t_out.to('s').value)
                for t_out in dum_model.time_out
            ]
            # Use the (first) observation time to locate the craft
            if isinstance(obs_time_in_datetime, datetime.datetime):
                obs_ref_time = obs_time_in_datetime
            else:
                obs_ref_time = obs_time_in_datetime[0]
            time_diffs = [
                (cod - obs_ref_time).total_seconds()
                for cod in craft_observer_datetime
            ]

            ind_req: int = int(np.argmin(
                np.abs(time_diffs)
            ))
            self.obs_radius = craft_observer.r[ind_req].to(u.AU)
            self.obs_lon = craft_observer.lon[ind_req].to(u.deg)
            self.obs_lat = craft_observer.lat[ind_req].to(u.deg)

            craft_obs_class = SynthObsEphem(
                surf_init_time=surf_init_time,
                obs_times=self.obs_time_in_sec,
                obs_rads=self.obs_radius,
                obs_lats=self.obs_lat,
                obs_lons=self.obs_lon,
            )

            if use_model in ["compress_surf3d"]:
                # Compute position angle of Earth from observing craft
                self.ert_observer = dum_model.get_observer("EARTH")
                self.psi_ert, self.elon_ert = Sim.compute_target_hpr_coords(craft_observer, self.ert_observer)
                self.psi_ert_avg = np.mean(self.psi_ert.to(u.deg).value)

                # Setup a SyntheticImager3D class instance to get the min and max latitudes
                # TODO: Remove hard-coded values for min and max elongations and buff_lat
                print(f"psi_ert_avg = {self.psi_ert_avg}")

                imgr = Sim.SyntheticImager3DOrig(
                    craft_observer,#_class,
                    pa=self.psi_ert_avg,
                    elon_min=5.0,
                    elon_max=30.0
                )
                buff_lat = 2.0 * u.deg
                imgr._compute_fov_geometry(0)
                self.lat_min = imgr.lat_grid.min().to(u.deg) - buff_lat
                self.lat_max = imgr.lat_grid.max().to(u.deg) + buff_lat
                print(f"lat_min: {self.lat_min}, lat_max: {self.lat_max}")

        # SURF3d needs a finite latitude range; fall back to setup_surf3d's defaults
        if self.use_model in ["compress_surf3d"]:
            if np.isnan(self.lat_min) or np.isnan(self.lat_max):
                print("lat_min/lat_max not set for compress_surf3d, defaulting to -30/+30 deg")
                self.lat_min = -30 * u.deg
                self.lat_max = 30 * u.deg

        self.obs_cov = obs_cov

        # Read in bias terms
        self.plot_surf_output = plot_surf_output
        self.bias_term_bool = bias_term_bool
        self.bias_term_5rs = bias_term_5rs
        self.bias_term_21rs = bias_term_21rs

        if (self.bias_term_5rs is None) or (self.bias_term_21rs is None):
            self.bias_term_bool = False

        # Determine whether to use parallel computation for each ensemble member
        self.use_parallel = use_parallel


    def extract_cme_launch_time_and_speed(self) -> tuple[list[Quantity[u.s]], list[Quantity[u.km / u.s]]]:
        """
        Function to extract the CME launch time and speed at r_min
        :return: cme_launch_time: CME launch time in seconds from start of SURF model run
        :return cme_speed: CME launch speed at r_min
        """
        # Calculate time taken to go from cme's initial radius to surf inner boundary
        # Ensure all variables are lists, if not make them into lists
        # Get CME launch time
        try:
            [
                (t - self.cme_par_dict["surf_init_time"]).total_seconds()
                for t in self.cme_par_dict["t_init"]
            ]
        except TypeError:
            seconds_to_cme: list[float] = [
                (t - self.cme_par_dict["surf_init_time"]).total_seconds()
                for t in [self.cme_par_dict["t_init"]]
            ]
        else:
            seconds_to_cme: list[float] = [
                (t - self.cme_par_dict["surf_init_time"]).total_seconds()
                for t in self.cme_par_dict["t_init"]
            ]

        # Get CME_speeds
        try:
            [v.to(u.km / u.s).value for v in self.cme_par_dict['v']]
        except TypeError:
            cme_speed = [v.to(u.km / u.s).value for v in [self.cme_par_dict['v']]]
        else:
            cme_speed = [v.to(u.km / u.s).value for v in self.cme_par_dict['v']]

        # Calculate the time it will take for CME to travel from cme_init_rad
        #  to SURF/HUXt model's inner radial boundary
        dist_to_inner_rad: float = (
            self.r_min.to(u.solRad) - self.cme_init_rad.to(u.solRad)
        ).to(u.km).value
        cme_time_to_rmin: list[float] = [
            (dist_to_inner_rad / v) for v in cme_speed
        ]

        # Extract the CME speed and launch_times
        cme_speed: list[Quantity[u.km / u.s]] = cme_speed * u.km / u.s
        cme_launch_time: list[Quantity[u.s]] = [
            cme_time_to_rmin[i] + seconds_to_cme[i]
            for i in range(self.n_members)
        ] * u.s

        return cme_launch_time, cme_speed


    def extract_cme_width(self) -> list[Quantity[u.deg]]:
        """
        Function to extract the CME width at r_min
        :return: cme_width: CME width in degrees
        """
        try:
            [w.to(u.deg).value for w in self.cme_par_dict["width"]]
        except TypeError:
            cme_width_values = [w.to(u.deg).value for w in [self.cme_par_dict["width"]]]
        else:
            cme_width_values: list[float] = [w.to(u.deg).value for w in self.cme_par_dict["width"]]

        cme_width: list[Quantity[u.deg]] = cme_width_values * u.deg

        return cme_width


    def extract_cme_lon(self) -> list[Quantity[u.deg]]:
        """
        Function to extract the CME longitude at r_min
        :return: cme_lon: CME longitude in degrees between +/- 180
        """
        try:
            [lon.to(u.deg).value for lon in self.cme_par_dict["lon"]]
        except TypeError:
            cme_lon: list[float] = [
                lon.to(u.deg).value for lon in [self.cme_par_dict["lon"]]
            ]
        else:
            cme_lon: list[float] = [
                lon.to(u.deg).value for lon in self.cme_par_dict["lon"]
            ]
        cme_lon: list[Quantity[u.deg]] = cme_lon * u.deg

        lon_cond: list[bool] = list(cme_lon > 180 * u.deg)
        cme_lon[lon_cond]: list[Quantity[u.deg]] = cme_lon[lon_cond] - (360 * u.deg)

        return cme_lon


    def extract_cme_lat(self) -> list[Quantity[u.deg]]:
        """
        Function to extract the CME latitude at r_min
        :return: cme_lat: CME latitude in degrees
        """
        try:
            [lat.to(u.deg).value for lat in self.cme_par_dict["lat"]]
        except TypeError:
            cme_lat_values: list[float] = [
                lat.to(u.deg).value for lat in [self.cme_par_dict["lat"]]
            ]
        else:
            cme_lat_values: list[float] = [
                lat.to(u.deg).value for lat in self.cme_par_dict["lat"]
            ]

        cme_lat: list[Quantity[u.deg]] = cme_lat_values * u.deg

        return cme_lat


    def extract_cme_thickness(self) -> list[Quantity[u.solRad]]:
        """
        Function to extract the CME thickness at r_min
        :return: cme_thickness: CME thickness in solar radii
        """
        try:
            [thick.to(u.solRad).value for thick in self.cme_par_dict["thick"]]
        except TypeError:
            cme_thick_values: list[float] = [
                thick.to(u.solRad).value for thick in [self.cme_par_dict["thick"]]
            ]
        else:
            cme_thick_values: list[float] = [
                thick.to(u.solRad).value for thick in self.cme_par_dict["thick"]
            ]

        cme_thick: list[Quantity[u.solRad]] = cme_thick_values * u.solRad

        return cme_thick


    def extract_cme_parameters(self) -> tuple[
        list[Quantity[u.s]],
        list[Quantity[u.km / u.s]],
        list[Quantity[u.deg]],
        list[Quantity[u.deg]],
        list[Quantity[u.deg]],
        list[Quantity[u.solRad]]
    ]:
        """
        Function to extract the true CME parameters at r_min
        :return: cme_launch_time: CME launch time in seconds
        :return: cme_speed: CME launch speed at r_min in km/s
        :return: cme_width: CME launch width at r_min in deg
        :return: cme_lon: CME launch longitude at r_min in deg
        :return: cme_lat: CME launch latitude at r_min in deg
        :return: cme_thickness: CME launch thickness at r_min in solar radii
        """
        cme_launch_time, cme_speed = self.extract_cme_launch_time_and_speed()
        cme_width = self.extract_cme_width()
        cme_lon = self.extract_cme_lon()
        cme_lat = self.extract_cme_lat()
        cme_thickness = self.extract_cme_thickness()

        return (
            cme_launch_time, cme_speed, cme_width,
            cme_lon, cme_lat, cme_thickness
        )


    def make_cme_objects(self) -> list[S.ConeCME] | list[H.ConeCME]:
        """
        Function to make a SURF ConeCME model object to be propagated through SURF
        :return: cme_obj: SURF ConeCME model object
        """
        # Extract CME parameters from true_cme_dict
        cme_pars = self.extract_cme_parameters()
        cme_launch_time: list[Quantity[u.s]] = cme_pars[0]
        cme_speed: list[Quantity[u.km / u.s]]= cme_pars[1]
        cme_width: list[Quantity[u.deg]] = cme_pars[2]
        cme_lon: list[Quantity[u.deg]] = cme_pars[3]
        cme_lat: list[Quantity[u.deg]] = cme_pars[4]
        cme_thickness: list[Quantity[u.solRad]] = cme_pars[5]

        # Generate CME object
        if self.use_model in ["surf", "compress_surf", "compress_surf3d"]:
            cme_objects: list[S.ConeCME] = [
                S.ConeCME(
                    t_launch=cme_launch_time[i],
                    v=cme_speed[i],
                    width=cme_width[i],
                    longitude=cme_lon[i],
                    latitude=cme_lat[i],
                    thickness=cme_thickness[i],
                    cme_fixed_duration=self.cme_fixed_duration,
                    fixed_duration=self.fixed_duration
                ) for i in range(self.n_members)
            ]
        elif self.use_model in ["huxt"]:
            cme_objects: list[H.ConeCME] = [
                H.ConeCME(
                    t_launch=cme_launch_time[i],
                    v=cme_speed[i],
                    width=cme_width[i],
                    longitude=cme_lon[i],
                    latitude=cme_lat[i],
                    thickness=cme_thickness[i],
                    cme_fixed_duration=self.cme_fixed_duration,
                    fixed_duration=self.fixed_duration
                ) for i in range(self.n_members)
            ]
        else:
            cme_objects = []
            sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

        return cme_objects


    def plot_surf(self, model) -> None:
        if isinstance(self.obs_time_in_datetime, datetime.datetime):
            obs_times_to_plot = [self.obs_time_in_datetime]
        else:
            obs_times_to_plot = list(self.obs_time_in_datetime)

        for obs_t in obs_times_to_plot:
            # Get nearest timestep to required observation time
            t_interest = (obs_t - self.surf_init_time).total_seconds() * u.s

            # Make the plot using SURF's plotting routine
            if self.use_model in ["surf", "compress_surf"]:
                fig, ax = SA.plot(model, t_interest)
                ax.set_title(f"Synth obs CME at {obs_t}")
                plt.show()
            elif self.use_model in ["huxt"]:
                fig, ax = HA.plot(model, t_interest)
                ax.set_title(f"Synth obs CME at {obs_t}")
                plt.show()
            else:
                sys.exit(
                    f"Unknown use_model name, {self.use_model}, "
                    f"expected either 'surf', 'compress_surf' "
                    f"or 'huxt'")

        return None


    def get_cme_flank_single_ens_member(
            self,
            cme: H.ConeCME | S.ConeCME=None,
            obs_longitude: Quantity[u.deg] | list[Quantity[u.deg]]=None,
            ens_no: int | None=None,
    ) -> pd.DataFrame:
        """
        Function to retrieve the CME's flank for a single ensemble member
        :param cme: CME object to calculate CME flank for
        :param obs_longitude: Longitude of observation source
        :param ens_no: Ensemble number
        :return: cme_flank: Dataframe containing calculated CME flank
        """
        if obs_longitude is None:
            obs_longitude = self.obs_lon

        # Initialise SURF model object for each ensemble member
        if self.use_model in ["surf"]:
            model: SURF = setup_surf(
                start_datetime=self.surf_init_time,
                vr_in=self.vr_in,
                lon_start=self.lon_start,
                lon_stop=self.lon_stop,
                sim_time=self.sim_time,
                dt_scale=self.dt_scale,
                r_min=self.r_min,
                solver=self.solver,
            )

            # Run CME through SURF
            model.solve([cme])
            cme_member: S.ConeCME = model.cmes[0]

        elif self.use_model in ["compress_surf"]:
            model: SURF = setup_surf(
                start_datetime=self.surf_init_time,
                vr_in=self.vr_in,
                lon_start=self.lon_start,
                lon_stop=self.lon_stop,
                sim_time=3 * u.day,
                dt_scale=self.dt_scale,
                r_min=self.r_min,
                solver=self.solver,
            )

            # Run CME through SURF
            model.solve([cme])
            cme_member: S.ConeCME = model.cmes[0]

        elif self.use_model in ["compress_surf3d"]:
            model: SURF3d = setup_surf3d(
                start_datetime=self.surf_init_time,
                vr_in=self.vr_in,
                vr_lons=self.vr_longs,
                vr_lats=self.vr_lats,
                lat_min=self.lat_min,
                lat_max=self.lat_max,
                n_lat_surf=self.n_lat_surf,
                lon_start=self.lon_start,
                lon_stop=self.lon_stop,
                n_lon_surf=self.n_lon_surf,
                sim_time=3 * u.day,
                dt_scale=self.dt_scale,
                r_min=self.r_min,
                solver=self.solver,
            )

            # Run CME through SURF
            model.solve([cme])
            cme_member: S.ConeCME = model.SURFlat[0].cmes[0]

        elif self.use_model in ["huxt"]:
            model: HUXt = setup_huxt(
                start_datetime=self.surf_init_time,
                vr_in=self.vr_in,
                lon_start=self.lon_start,
                lon_stop=self.lon_stop,
                sim_time=self.sim_time,
                dt_scale=self.dt_scale,
                r_min=self.r_min
            )
            # Run CME through SURF
            model.solve([cme])

            cme_member: H.ConeCME = model.cmes[0]
        else:
            sys.exit(
                f"Unknown use_model name, {self.use_model}, expected one of "
                f"'surf', 'compress_surf', 'compress_surf3d' or 'huxt'"
            )

        #print(f"self.use_model = {self.use_model}")
        #print(f"model = {model}")
        if ens_no is not None:
            if np.mod(ens_no, 10) == 0:
                print(f"Model solved cme_flank_ens_no: {ens_no}")

        # Calculate CME flank
        if self.use_model in ["huxt", "surf"]:
            observer_object: ObserverTracer = ObserverTracer(
                use_model=self.use_model,
                model=model,
                cme=cme_member,
                longitude=obs_longitude
            )
            cme_flank: pd.DataFrame = observer_object.model_flank

        elif self.use_model in ["compress_surf", "compress_surf3d"]:
            obs_datetime = np.array(self.obs_time_in_datetime)
            obs_radius = np.array(self.obs_radius.to(u.AU).value) * u.AU
            obs_longitude = np.array(self.obs_lon.to(u.deg).value) * u.deg
            obs_latitude = np.array(self.obs_lat.to(u.deg).value) * u.deg

            # TODO: Remove hard-coded numbers input into el_min and el_max
            obs_synth_HI_class = ObserverSynthHI(
                use_model=self.use_model,
                model=model,
                psi_ert_avg=self.psi_ert_avg,
                obs_time=obs_datetime,
                obs_radius=obs_radius,
                obs_longitude=obs_longitude,
                obs_latitude=obs_latitude,
                el_min=4.0,
                el_max=30.0
            )
            cme_flank: pd.DataFrame = obs_synth_HI_class.model_flank

            if np.isnan(cme_flank['el'].values).any():
                print("=====================================================")
                print(f"Observation operator failed: ens_no = {ens_no}")
                print(f"cme_t_init = {cme_member.t_launch}")
                print(f"cme_speed = {cme_member.v}")
                print(f"cme_width = {cme_member.width}")
                print(f"cme_lon = {cme_member.longitude.to(u.deg)}")
                print(f"cme_lat = {cme_member.latitude.to(u.deg)}")
                print("=====================================================\n")
        else:
            sys.exit(
                f"Unrecognised model type, {self.use_model} input"
                f" into self.use_model"
            )

        if ens_no is not None:
            cme_flank.index = [ens_no for _ in range(len(cme_flank.index))]

        if self.plot_surf_output:
            self.plot_surf(model)

        return cme_flank


    def get_cme_flanks(self) -> list[pd.DataFrame]:
        """
        Function to retrieve the CME flank dataframes for all ensemble members
         and store them in a list
        :return: cme_flanks: List of CME flank dataframes
        """
        start_time = time.time()
        cme_objects: list[S.ConeCME] | list[H.ConeCME] = self.make_cme_objects()

        if self.use_parallel:
            def log_result(result):
                # This is called whenever self.get_cme_flank_single_ens_member(i)
                #  returns a result in parallel computation.
                # result_list is modified only by the main process, not the pool workers.
                result_list.append(result)

            print("Using parallel execution")

            # Initialise variables required for parallel computations
            result_list = []
            obj_res = [None] * self.n_members

            n_proc = int(os.cpu_count() - 1)
            if n_proc < 1:
                n_proc = 1

            cme_flanks: list[DataFrame] | list[None] = [None] * self.n_members

            pool = Pool(processes=n_proc)
            if isinstance(self.obs_lon, np.ndarray):
                if np.ndim(self.obs_lon) == 0:
                    # Get CME flanks for each ensemble member in parallel
                    #  when self.obs_lon is a single value in an array
                    obs_lon_val: Quantity[u.deg] = float(self.obs_lon.value) * u.deg

                    for i, i_cme in enumerate(cme_objects):
                        obj_res[i] = pool.apply_async(
                            self.get_cme_flank_single_ens_member,
                            kwds={
                                'cme': i_cme,
                                'obs_longitude': obs_lon_val,
                                'ens_no': i,
                            },
                            callback=log_result
                        )
                else:
                    # Get CME flanks for each ensemble member in parallel
                    #  when self.obs_lon is an array with multiple values and
                    #  extract relevant value
                    for i, i_cme in enumerate(cme_objects):
                        obj_res[i] = pool.apply_async(
                            self.get_cme_flank_single_ens_member,
                            kwds={
                                'cme': i_cme,
                                'obs_longitude': self.obs_lon[i],
                                'ens_no': i,
                            },
                            callback=log_result
                        )
            else:
                # Get CME flanks for each ensemble member in parallel
                #  when self.obs_lon is not in an array (i.e. a float)
                for i, i_cme in enumerate(cme_objects):
                    obj_res[i] = pool.apply_async(
                        self.get_cme_flank_single_ens_member,
                        kwds={
                            'cme': i_cme,
                            'obs_longitude': self.obs_lon,
                            'ens_no': i,
                        },
                        callback=log_result
                    )

            pool.close()
            pool.join()

            for i, _ in enumerate(cme_objects):
                cme_flanks[i] = obj_res[i].get()
            print(f"Parallel cme_flanks={cme_flanks}")

        else:
            print("Using serial execution")
            if isinstance(self.obs_lon, np.ndarray):
                if np.ndim(self.obs_lon) == 0:
                    # Get CME flanks for each ensemble member in serial
                    #  when self.obs_lon is an array containing a single value
                    obs_lon_val: Quantity[u.deg] = float(self.obs_lon.value) * u.deg

                    cme_flanks: list[pd.DataFrame] = [
                        self.get_cme_flank_single_ens_member(
                            cme=i_cme,
                            obs_longitude=obs_lon_val,
                            ens_no=i
                        )
                        for i, i_cme in enumerate(cme_objects)
                    ]
                else:
                    # Get CME flanks for each ensemble member in serial
                    #  when self.obs_lon is an array containing multiple values
                    #  and extract relevant value
                    cme_flanks: list[pd.DataFrame] = [
                        self.get_cme_flank_single_ens_member(
                            cme=i_cme,
                            obs_longitude=self.obs_lon[i],
                            ens_no=i
                        )
                        for i, i_cme in enumerate(cme_objects)
                    ]
            else:
                # Get CME flanks for each ensemble member in serial
                #  when self.obs_lon is not in an array (i.e. a float)
                cme_flanks: list[pd.DataFrame] = [
                    self.get_cme_flank_single_ens_member(
                        cme=i_cme,
                        obs_longitude=self.obs_lon,
                        ens_no=i
                    )
                    for i, i_cme in enumerate(cme_objects)
                ]
            print(f"Serial cme_flanks = {cme_flanks}")

        end_time = time.time()
        time_taken = end_time - start_time

        time_taken_minutes = int(time_taken / 60.0)
        time_taken_rem_seconds = time_taken - (time_taken_minutes * 60.0)

        print(
            f"Time taken to complete get_cme_flanks:"
            f" {time_taken_minutes} minutes and {time_taken_rem_seconds} seconds"
        )

        return cme_flanks


    def obs_op_bias_correction(
            self,
            elon: float
    ):
        """
        Function to calculate the bias correction due to using tracer particles to measure CME flank
        Initial experiments will use a simple linear relation
        :param elon: Elongation to calculate the bias correction for
        :return: bias_corr: bias correction
        """
        print(f"bias terms = 5rS: {self.bias_term_5rs}, 21rS: {self.bias_term_21rs}")

        # Calculate gradient and constant terms for linear relation
        m: float = (self.bias_term_5rs - self.bias_term_21rs) / 15.0
        c: float = ((21 * self.bias_term_5rs) - (5 * self.bias_term_21rs)) / 15.0

        # Calculate bias required
        bias_corr = (m * elon) + c

        return bias_corr


    def make_obs_op(self) -> npt.NDArray[float]:
        """
        obs_op: The purpose of this definition is to perform the observation operator
                  function that maps from the cme_parameters to observation space (in this
                  case, the CME's flank position)
        :return: obs_op: CME flank estimated by model
        """
        cme_flanks = self.get_cme_flanks()

        # Get type of a list of datetimes
        type_list_datetime = isinstance(self.obs_time_in_datetime, list) and all(
            isinstance(i, datetime.datetime) for i in self.obs_time_in_datetime
        )

        # Initialise an array to contain the observation operator
        if isinstance(self.obs_time_in_datetime, datetime.datetime):
            obs_op: npt.NDArray[float] = np.zeros((self.n_members, 1))
        elif type_list_datetime:
            obs_op: npt.NDArray[float] = np.zeros((self.n_members, len(self.obs_time_in_datetime)))
        else:
            raise Exception(
                "Admissable types for obs_time_in_datetime are: datetime.datetime"
                " or list[datetime.datetime]"
            )

        if self.use_model in ["surf", "huxt"]:
            for i, cme_flank in enumerate(cme_flanks):
                if isinstance(self.obs_time_in_datetime, datetime.datetime):
                    # Get the CME elongation at the nearest timestep to the observation time
                    ind_req: int = np.argmin(
                        abs(cme_flank["time"].values - Time(self.obs_time_in_datetime).jd)
                    )

                    if self.bias_term_bool:
                        obs_op[i, :] = [(
                            cme_flank["el"].values[ind_req]
                            + self.obs_op_bias_correction(elon=cme_flank["el"].values[ind_req])
                        )]
                    else:
                        obs_op[i, :] = [cme_flank["el"].values[ind_req]]

                elif type_list_datetime:
                    obs_times_in_jd: list[float] = Time(
                        self.obs_time_in_datetime, format='datetime'
                    ).jd

                    inds_req: list[int] = [
                        int(np.argmin(np.round(abs(cme_flank["time"].values - obs_time), 9)))
                        for obs_time in obs_times_in_jd
                    ]

                    obs_op[i, :]: list[float] = [
                        cme_flank["el"].values[j] + self.obs_op_bias_correction(
                            elon=cme_flank["el"].values[j]
                        )
                        if self.bias_term_bool else cme_flank["el"].values[j]
                        for j in inds_req
                    ]

        elif self.use_model in ["compress_surf", "compress_surf3d"]:
            # NB: compress_surf3d previously fell through here, so obs_op was left as zeros
            for i, cme_flank in enumerate(cme_flanks):
                if isinstance(self.obs_time_in_datetime, datetime.datetime):
                    # Get the CME elongation at the nearest timestep to the observation time
                    obs_op[i, :]: list[float] = [
                        cme_flank["el"].values[0]
                    ]
                elif type_list_datetime:
                    obs_op[i, :]: list[float] = [
                        x for x in cme_flank["el"].values
                    ]
                else:
                    sys.exit(
                        "Admissable types for obs_time_in_datetime when using "
                        "compressible SURF are: datetime.datetime or list[datetime.datetime]."
                    )
        else:
            sys.exit(f"make_obs_op: unsupported use_model '{self.use_model}'")

        return obs_op


def main():
    surf_init_time = datetime.datetime(2012, 1, 1, 0, 0, 0, 0)
    vr_in = np.ones(128) * 400 * u.km / u.s
    lon_start = 290 * u.deg
    lon_stop = 430 * u.deg
    sim_time = 3 * u.day
    dt_scale = 1
    r_min = 21.5 * u.solRad
    use_model = "compress_surf"
    solver = "hydro"

    model: SURF = setup_surf(
        start_datetime=surf_init_time,
        vr_in=vr_in,
        lon_start=lon_start,
        lon_stop=lon_stop,
        sim_time=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
    )

    cme_launch_datetime = datetime.datetime(2012, 1, 1, 1, 0, 0, 0)
    cme_launch_time = (cme_launch_datetime - surf_init_time).total_seconds() * u.s
    print(cme_launch_time)
    cme_speed = 600 * u.km / u.s
    cme_width = 40 * u.deg
    cme_lon = 0 * u.deg
    cme_lat = 0 * u.deg
    cme_thickness = 0.0 * u.solRad
    cme_fixed_duration = True
    fixed_duration = 12 * 60 * 60 * u.s

    # Initialise CME parameter dictionary
    cme_par_dict = initialise_cme_parameter_ensemble_dict(
        n_ensemble=1,
        surf_init_time=surf_init_time
    )
    cme_par_dict["t_init"] = [cme_launch_datetime]
    cme_par_dict["v"] = [cme_speed]
    cme_par_dict["width"] = [cme_width]
    cme_par_dict["lon"] = [cme_lon]
    cme_par_dict["lat"] = [cme_lat]
    cme_par_dict["thick"] = [cme_thickness]

    cme_object = S.ConeCME(
        t_launch=cme_launch_time,
        v=cme_speed,
        width=cme_width,
        longitude=cme_lon,
        latitude=cme_lat,
        thickness=cme_thickness,
        cme_fixed_duration=cme_fixed_duration,
        fixed_duration=fixed_duration
    )

    n_obs = 1
    obs_from_start_sec = 3600

    obs_cadence_sec = 3600
    obs_times = np.array([
        obs_from_start_sec + (obs_cadence_sec * i)
        for i in range(n_obs)
    ]) * u.s
    obs_datetime = np.array([
        surf_init_time + datetime.timedelta(
            seconds=obs_from_start_sec + (obs_cadence_sec * i)
        ) for i in range(n_obs)
    ])

    obs_rads = np.array([
        1.0 for i in range(n_obs)
    ]) * u.AU
    obs_lats = np.array([
        0.0 for i in range(n_obs)
    ]) * u.deg
    obs_lons = np.array([
       -60
       for i in range(n_obs)
   ]) * u.deg


    print(cme_object.t_launch)
    model.solve([cme_object])

    # obs_synth_hi_class = ObserverSynthHI(
    #     use_model='compress_surf',
    #     model=model,
    #     cme=cme_object,
    #     obs_time=obs_datetime,
    #     obs_radius=obs_rads,
    #     obs_longitude=obs_lons,
    #     obs_latitude=obs_lats,
    #     el_min=4.0,
    #     el_max=30.0
    # )
    obs_op_class = ObservationOperator(
        use_model=use_model,
        cme_par_dict=cme_par_dict,
        obs_radius=obs_rads,
        obs_lon=obs_lons,
        obs_lat=obs_lats,
        obs_time_in_datetime=obs_datetime[0],
        obs_cov=0,
        surf_init_time=surf_init_time,
        vr_in=vr_in,
        lon_start=lon_start,
        lon_stop=lon_stop,
        sim_time=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        cme_init_rad=r_min,
        cme_fixed_duration=cme_fixed_duration,
        fixed_duration=fixed_duration
    )

    obs_op = obs_op_class.make_obs_op()
    print(f"obs_op: {obs_op}")

    return None

if __name__ == "__main__":
    main()