"""
sir_observation_operator.py

The observation operator H(x) for the data assimilation: maps an ensemble of CME parameters to the
elongations a heliospheric imager would observe.

For every ensemble member, ``ObservationOperator.make_obs_op()``:
    1. builds a cone CME from the member's parameters (``make_cme_objects``),
    2. runs the forward model (SURF, compressible SURF, SURF3d or HUXt) with that CME,
    3. extracts the CME's time-elongation profile with an observer class (see sir_observers.py), and
    4. samples it at the observation time(s),
returning an (n_members, n_obs_times) array.

The observer classes used to live in this module and are re-exported from here, so
``from sir_observation_operator import ObserverSynthHI`` (etc.) keeps working.
"""
import datetime
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

import surf.surf as S
import surf.surf_inputs as Sin
import surf.surf_analysis as SA
import surf.surf_imaging as Sim

import huxt.huxt as H
import huxt.huxt_analysis as HA

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
import matplotlib.pyplot as plt

from init_sir import setup_huxt, setup_surf, setup_surf3d, initialise_cme_parameter_ensemble_dict
from cme_par_ens import CmeParEns
from cme_par_dict_structure import required_dict_keys

# Re-exported for backwards compatibility (these classes were previously defined here)
from sir_observers import background, SynthObsEphem, ObserverTracer, ObserverSynthHI  # noqa: F401


class ObservationOperator:
    """
    Observation operator: run the forward model for every ensemble member and return the modelled
    elongation(s) at the observation time(s). See the module docstring.

    Example::

        hx = ObservationOperator(use_model, cme_par_dict, obs_time, obs_cov, craft="sta", ...).make_obs_op()
    """
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
        :param vr_longs, vr_lats: Coordinates of vr_in (needed for compress_surf3d)
        :param br_in, br_longs, br_lats: Inner-boundary magnetic field and coordinates (not currently used)
        :param n_lon_surf, n_lat_surf: Grid sizes for SURF3d
        :param lat_min, lat_max: Latitude range for SURF3d (derived from the imager field of view when a
            real craft is given; otherwise -30/+30 deg if not set)
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

        # Carrington rotation number and initial longitude for surf_init_time
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
            # Real spacecraft: the observer is fixed at the craft's position at the (first) observation time
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
                #print(f"psi_ert_avg = {self.psi_ert_avg}")

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
                #print(f"lat_min: {self.lat_min}, lat_max: {self.lat_max}")

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


    # ------------------------------------------------------------------------------------------
    # CME parameters -> ConeCME objects
    #  Each extract_* method accepts a cme_par_dict holding either one value per member (list)
    #  or a single value (scalar), and returns one value per member.
    # ------------------------------------------------------------------------------------------
    def extract_cme_launch_time_and_speed(self) -> tuple[list[Quantity[u.s]], list[Quantity[u.km / u.s]]]:
        """
        Function to extract the CME launch time and speed at r_min
        :return: cme_launch_time: CME launch time in seconds from start of SURF model run
        :return cme_speed: CME launch speed at r_min
        """
        # t_init is defined at cme_init_rad; the model needs the time the CME crosses r_min, so add the
        #  travel time (r_min - cme_init_rad) / v, assuming constant speed.
        # Get CME launch time (the try/except accepts a list or a single datetime)
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


    # ------------------------------------------------------------------------------------------
    # Running the model for each member and extracting the elongation profile
    # ------------------------------------------------------------------------------------------
    def plot_surf(self, model: S.SURF | H.HUXt) -> None:
        """
        Plot the model solution at each observation time (only used when plot_surf_output=True).
        :param model: Solved SURF/HUXt model
        """
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
                plt.close(fig)
            elif self.use_model in ["huxt"]:
                fig, ax = HA.plot(model, t_interest)
                ax.set_title(f"Synth obs CME at {obs_t}")
                plt.show()
                plt.close(fig)
            else:
                sys.exit(
                    f"Unknown use_model name, {self.use_model}, "
                    f"expected either 'surf', 'compress_surf' "
                    f"or 'huxt'")

        return None


    def get_cme_flank_single_ens_member(
            self,
            cme: H.ConeCME | S.ConeCME | None = None,
            obs_longitude: Quantity[u.deg] | list[Quantity[u.deg]] | None = None,
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

        # Build and solve a fresh model for this member (compressible runs use a fixed 3-day run length)
        if self.use_model in ["surf"]:
            model: S.SURF = setup_surf(
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
            model: S.SURF = setup_surf(
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
            model: S.SURF3d = setup_surf3d(
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
            model: H.HUXt = setup_huxt(
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

        if ens_no is not None:
            if np.mod(ens_no, 10) == 0:
                print(f"Model solved cme_flank_ens_no: {ens_no}")

        # Extract the CME's time-elongation profile:
        #  incompressible models -> tracer-particle flank; compressible models -> synthetic HI j-maps
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
        Retrieve the CME flank dataframes for all ensemble members, in serial or in parallel
        (one multiprocessing task per member). The three branches in each case only differ in how
        the observer longitude is passed (0-d array, one value per member, or a single Quantity).
        :return: cme_flanks: List of CME flank dataframes, in ensemble-member order
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
            try:
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
            except BaseException:
                # Release the worker processes if anything fails (or on Ctrl+C)
                pool.terminate()
                pool.join()
                raise

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


    # ------------------------------------------------------------------------------------------
    # Observation operator
    # ------------------------------------------------------------------------------------------
    def obs_op_bias_correction(
            self,
            elon: float
    ) -> float:
        """
        Function to calculate the bias correction due to using tracer particles to measure CME flank
        Initial experiments will use a simple linear relation in elongation, defined by the bias at
        5 and 21 solar radii (bias_term_5rs, bias_term_21rs)
        :param elon: Elongation to calculate the bias correction for
        :return: bias_corr: bias correction
        """
        print(f"bias terms = 5rS: {self.bias_term_5rs}, 21rS: {self.bias_term_21rs}")

        # Straight line through (5, bias_term_5rs) and (21, bias_term_21rs)
        #  (previously the gradient had the wrong sign and both terms were divided by 15 rather than
        #   21 - 5 = 16, so e.g. equal biases of 2.5 gave a constant correction of 2.67)
        m: float = (self.bias_term_21rs - self.bias_term_5rs) / 16.0
        c: float = ((21 * self.bias_term_5rs) - (5 * self.bias_term_21rs)) / 16.0

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
            # Tracer flank profiles are on the model time grid: take the nearest model time
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
            # Synthetic-HI profiles are already interpolated to the observation times
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


def main() -> None:
    """Example: observation operator for a single 600 km/s CME in compressible SURF."""
    surf_init_time = datetime.datetime(2012, 1, 1, 0, 0, 0, 0)
    vr_in = np.ones(128) * 400 * u.km / u.s
    lon_start = 290 * u.deg
    lon_stop = 430 * u.deg
    sim_time = 3 * u.day
    dt_scale = 1
    r_min = 21.5 * u.solRad
    use_model = "compress_surf"
    solver = "hydro"

    model: S.SURF = setup_surf(
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