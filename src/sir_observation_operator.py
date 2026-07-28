# @package sir_observation_operator
# This module will take a HUXt and ConeCME object and calculate its elongation profile
#  for use as an observation operator in the data assimilation algorithm
import numpy as np
import numpy.typing as npt
import pandas as pd

import huxt.huxt as H
import huxt.huxt_analysis as HA

import datetime
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

#from .to_state_vector import ToStateVector
from init_sir import setup_huxt
from cme_par_ens import CmeParEns
from cme_par_dict_structure import required_dict_keys

class Observer:
    """
    The function of this observer class is to provide pseudo-observations of a ConeCME's flank elongation from a HUXt simulation
    for an observer at a specified longitude relative to Earth. The observers distance defaults to 1 AU, and has the same latitude as Earth
    during the HUXt run.
    Author: Luke Barnard
    """

    @u.quantity_input(longitude=u.deg)
    def __init__(
            self,
            model: H.HUXt,
            cme: H.ConeCME,
            longitude: Quantity[u.deg],
            el_min: float=4.0,
            el_max: float=30.0
    ):

        ert_ephem: H.Observer = model.get_observer('EARTH')

        self.time: npt.NDArray[Time] = ert_ephem.time
        self.r: npt.NDArray[Quantity[u.AU]] = ert_ephem.r * 0 + 1 * u.AU
        self.lon: npt.NDArray[Quantity[u.deg]] = ert_ephem.lon + longitude
        self.lat: npt.NDArray[Quantity[u.deg]] = ert_ephem.lat
        self.el_min: float = el_min
        self.el_max: float = el_max

        # Force longitude into 0-360 domain
        id_over: list[bool] = self.lon > 360 * u.deg
        id_under: list[bool] = self.lon < 0 * u.deg
        if np.any(id_over):
            self.lon[id_over] = self.lon[id_over] - 360 * u.deg

        if np.any(id_under):
            self.lon[id_under] = self.lon[id_under] + 360 * u.deg

        self.model_flank: pd.DataFrame = self.compute_flank_profile(cme)


    def compute_flank_profile(self, cme: H.ConeCME) -> pd.DataFrame:
        """
        Compute the time elongation profile of the flank of a ConeCME in HUXt. The observer longtiude is specified
        relative to Earth but otherwise matches Earth's coords.

        Parameters
        ----------
        cme: A ConeCME object from a completed HUXt run (i.e the ConeCME.coords dictionary has been populated).
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

            #############
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
            id_obs_flank: int = np.argmax(e_obs)
            flank.loc[i, 'lon'] = lon_cme[id_obs_flank].value
            flank.loc[i, 'r'] = r_cme[id_obs_flank].value
            flank.loc[i, 'el'] = np.rad2deg(e_obs[id_obs_flank])

        # Force values to be floats.
        keys: list[str] = ['lon', 'r', 'el']
        flank[keys] = flank[keys].astype(np.float64)

        return flank


class ObservationOperator:
    def __init__(
            self,
            cme_par_dict: CmeParEns,
            obs_lon: Quantity[u.deg],
            obs_time_in_datetime: list[datetime.datetime] | datetime.datetime,
            obs_cov: npt.NDArray[float] | float = None,
            huxt_init_time: datetime.datetime = None,
            vr_in: npt.NDArray[Quantity[u.km / u.s]] = None,
            lon_start: Quantity[u.deg] = None,
            lon_stop: Quantity[u.deg] = None,
            sim_time: Quantity[u.day] = None,
            dt_scale: int | float = None,
            r_min: Quantity[u.solRad] = None,
            cme_init_rad: Quantity[u.solRad] = None,
            cme_fixed_duration:bool = True,
            fixed_duration: Quantity[u.s] = 12 * 60 * 60 * u.s,
            plot_huxt_output: bool = False,
            bias_term_bool: bool = False,
    ) -> None:
        """
        Class to get observations for DA from
        :param obs_lon: Longitude of observation source
        :param obs_time_in_datetime: Times observations are taken
        :param cme_par_dict: To be used to create synthetic observations, a dictionary containing true parameters
            to create a truth CME simulation
        :param obs_cov: To be used with synthetic observations, an observation covariance matrix to be used to
            add random noise to  truth CME elongations
        :param huxt_init_time: Initial datetime for HUXt model
        :param vr_in: Velocities at the inner boundary of the HUXt model domain
        :param lon_start: Longitude where we start the HUXt simulation
        :param lon_stop: Longitude where we stop the HUXt simulation
        :param sim_time: How long the HUXt simulation will run for
        :param dt_scale: Frequency with which the model will output
        :param r_min: Minimum radius of HUXt simulation
        :param cme_init_rad: Initial radius the CME is observed at
        :param cme_fixed_duration: Boolean to determine whether CMEs will be input with fixed duration
        :param fixed_duration: If CME is fixed duration, this variable defines that fixed duration in seconds
        :param plot_huxt_output: Boolean to determine whether to plot HUXt output at observation time
        :param bias_term_bool: Boolean to determine whether to use bias term or not
        :TODO: CHANGED SUCH THAT MULTIPLE LONGITUDES AND RADII CAN BE
            INPUT AND TIMES OF OBSERVATIONS ARE TAKEN FROM INPUT FILE IN CASE OF REAL OBS
        :TODO: CHANGE cme_init_rad SUCH THAT IT CAN BE VARIED BETWEEN ENSEMBLE MEMBERS
        :TODO: Change such that observations are downloaded if need be
        """

        if obs_lon is None:
            self.obs_lon = 0 * u.deg
        else:
            self.obs_lon = obs_lon

        assert obs_time_in_datetime is not None
        self.obs_time_in_datetime = obs_time_in_datetime
        self.obs_time_in_jd = Time(self.obs_time_in_datetime, format='datetime').jd

        # Define all variables required to initialise HUXt are provided
        #print(cme_par_dict)
        assert (all([cme_par_dict, obs_cov]) is not None)
        assert all(p in cme_par_dict.keys() for p in required_dict_keys())

        self.cme_par_dict = cme_par_dict
        self.obs_cov = obs_cov

        self.n_members = self.cme_par_dict["n_members"]
        #print(f"self.n_members = {self.n_members}")
        # Initialise default huxt setup
        if huxt_init_time is None:
            self.huxt_init_time = datetime.datetime(2008, 1, 1, 0, 0, 0)
        else:
            self.huxt_init_time = huxt_init_time

        if vr_in is None:
            self.vr_in = np.ones(128) * 400 * u.km / u.s
        else:
            self.vr_in = vr_in

        if lon_start is None:
            self.lon_start = 290 * u.deg
        else:
            self.lon_start = lon_start

        if lon_stop is None:
            self.lon_stop = 380 * u.deg
        else:
            self.lon_stop = lon_stop

        if sim_time is None:
            self.sim_time = 5 * u.day
        else:
            self.sim_time = sim_time

        if dt_scale is None:
            self.dt_scale = 20
        else:
            self.dt_scale = dt_scale

        if r_min is None:
            self.r_min = 30 * u.solRad
        else:
            self.r_min = r_min

        if cme_init_rad is None:
            self.cme_init_rad = 12 * u.solRad
        else:
            self.cme_init_rad = cme_init_rad


        if cme_fixed_duration is None:
            self.cme_fixed_duration = False
        else:
            self.cme_fixed_duration = cme_fixed_duration

        if fixed_duration is None:
            self.fixed_duration = 12 * 60 * 60 * u.s
        else:
            self.fixed_duration = fixed_duration

        self.plot_huxt_output = plot_huxt_output
        self.bias_term_bool = bias_term_bool


    def extract_cme_launch_time_and_speed(self) -> tuple[list[Quantity[u.s]], list[Quantity[u.km / u.s]]]:
        """
        Function to extract the CME launch time and speed at r_min
        :return: cme_launch_time: CME launch time in seconds from start of HUXt model run
        :return cme_speed: CME launch speed at r_min
        """
        # Calculate time taken to go from cme's initial radius to huxt inner boundary
        # Ensure all variables are lists, if not make them into lists
        # Get CME launch time
        try:
            [
                (t - self.cme_par_dict["huxt_init_time"]).total_seconds()
                for t in self.cme_par_dict["t_init"]
            ]
        except TypeError:
            seconds_to_cme: list[float] = [
                (t - self.cme_par_dict["huxt_init_time"]).total_seconds()
                for t in [self.cme_par_dict["t_init"]]
            ]
        else:
            seconds_to_cme: list[float] = [
                (t - self.cme_par_dict["huxt_init_time"]).total_seconds()
                for t in self.cme_par_dict["t_init"]
            ]
        #print(f"seconds_to_cme = {seconds_to_cme}")

        # Get CME_speeds
        try:
            [v.to(u.km / u.s).value for v in self.cme_par_dict['v']]
        except TypeError:
            cme_speed = [v.to(u.km / u.s).value for v in [self.cme_par_dict['v']]]
        else:
            cme_speed = [v.to(u.km / u.s).value for v in self.cme_par_dict['v']]

        dist_to_inner_rad: float = (
            self.r_min.to(u.solRad) - self.cme_init_rad.to(u.solRad)
        ).to(u.km).value
        cme_time_to_rmin: list[float] = [
            (dist_to_inner_rad / v) for v in cme_speed
        ]
        #print(f"cme_speed = {cme_speed}")
        cme_speed: list[Quantity[u.km / u.s]] = cme_speed * u.km / u.s
        #print(f"len(cme_time_to_rmin) = {len(cme_time_to_rmin)}")
        cme_launch_time: list[Quantity[u.s]] = [
            cme_time_to_rmin[i] + seconds_to_cme[i]
            for i in range(self.n_members)
        ] * u.s
        #print(f"cme_lt = {cme_launch_time}")

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

        return cme_launch_time, cme_speed, cme_width, cme_lon, cme_lat, cme_thickness


    def make_cme_objects(self) -> list[H.ConeCME]:
        """
        Function to make a HUXt ConeCME model object to be propagated through HUXt
        :return: cme_obj: HUXt ConeCME model object
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
        #print(f"cme_launch_time = {cme_launch_time}")
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

        return cme_objects


    def plot_huxt(self, model) -> None:
        for it, obs_t in enumerate(self.obs_time_in_datetime):
            # Get nearest timestep to required observation time
            t_interest = (self.obs_time_in_datetime[it] - self.huxt_init_time).total_seconds() * u.s

            # Make the plot using HUXt's plotting routine
            fig, ax = HA.plot(model, t_interest)
            ax.set_title(f"Synth obs CME at {obs_time_in_datetime[it]}")
            plt.show()

        return None


    def get_cme_flank_single_ens_member(self, cme: H.ConeCME) -> pd.DataFrame:
        """
        Function to retrieve the CME's flank for a single ensemble member
        :return: cme_flank: CME flank dataframe
        """
        # Initialise HUXt model object for each ensemble member
        model: HUXt = setup_huxt(
            start_datetime=self.huxt_init_time,
            vr_in=self.vr_in,
            lon_start=self.lon_start,
            lon_stop=self.lon_stop,
            sim_time=self.sim_time,
            dt_scale=self.dt_scale,
            r_min=self.r_min
        )

        # Run CME through HUXt
        model.solve([cme])

        cme_member: H.ConeCME = model.cmes[0]
        #print(f"cme_member={cme_member}")
        # Calculate CME flank
        observer_object: Observer = Observer(model, cme_member, self.obs_lon)
        cme_flank: pd.DataFrame = observer_object.model_flank #compute_flank_profile(cme_member)

        if self.plot_huxt_output:
            self.plot_huxt(model)

        return cme_flank


    def get_cme_flanks(self) -> list[pd.DataFrame]:
        """
        Function to retrieve the CME flank dataframes for all ensemble members
         and store them in a list
        :return: cme_flanks: List of CME flank dataframes
        """
        cme_objects: list[H.ConeCME] = self.make_cme_objects()

        #print(f"make_cme_objects = {cme_objects}")
        cme_flanks: list[pd.DataFrame] = [
            self.get_cme_flank_single_ens_member(i_cme)
            for i_cme in cme_objects
        ]
        #print(f"cme_flanks = {cme_flanks}")

        return cme_flanks

    def obs_op_bias_correction(
            self,
            corr_at_elon6: float,
            corr_at_elon_21: float,
            elon: float
    ):
        """
        Function to calculate the bias correction due to using tracer particles to measure CME flank
        Initial experiments will use a simple linear relation
        :param corr_at_elon6: Correction required at elongation 5deg
        :param corr_at_elon_21: Correction required at elongation 21deg
        :param elon: Elongation to calculate the bias correction for
        :return: bias_corr: bias correction
        """

        # Calculate gradient and constant terms for linear relation
        m: float = (corr_at_elon6 - corr_at_elon_21) / 15.0
        c: float = ((21 * corr_at_elon6) - (6 * corr_at_elon_21)) / 15.0

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
            obs_op: npt.NDArray[np.float] = np.zeros((self.n_members, 1))
        elif type_list_datetime:
            obs_op: npt.NDArray[float] = np.zeros((self.n_members, len(self.obs_time_in_datetime)))
        else:
            raise Exception(
                "Admissable types for obs_time_in_datetime are: datetime.datetime"
                " or list[datetime.datetime]"
            )

        for i, cme_flank in enumerate(cme_flanks):
            if isinstance(self.obs_time_in_datetime, datetime.datetime):
                # Get the CME elongation at the nearest timestep to the observation time
                # obs_time_in_jd: float = Time(self.obs_time_in_datetime).jd.value

                ind_req: int = np.argmin(
                    abs(cme_flank["time"].values - Time(self.obs_time_in_datetime).jd)
                )
                #print(f"ind_req={ind_req}")
                #print(f"self.bias_term_bool1 = {self.bias_term_bool}")
                if self.bias_term_bool:
                    obs_op[i, :] = [
                        cme_flank["el"].values[ind_req] + self.obs_op_bias_correction(
                            corr_at_elon6=3.5,
                            corr_at_elon_21=1.5,
                            elon=cme_flank["el"].values[ind_req]
                        )
                    ]
                else:
                    obs_op[i, :] = [cme_flank["el"].values[ind_req]]

            elif type_list_datetime:
                obs_times_in_jd: list[float] = Time(
                    self.obs_time_in_datetime, format='datetime'
                ).jd
                # print(cme_flank["time"].values)
                # print(obs_times_in_jd)
                # float(np.round(np.log(x), 9))
                inds_req: list[int] = [
                    int(np.argmin(np.round(abs(cme_flank["time"].values - obs_time), 9)))
                    for obs_time in obs_times_in_jd
                ]
                #print(f"inds_req={inds_req}")
                #print(f"self.bias_term_bool = {self.bias_term_bool}")
                obs_op[i, :] = [
                    cme_flank["el"].values[j] + self.obs_op_bias_correction(
                        corr_at_elon6=1.5,
                        corr_at_elon_21=1.5,
                        elon=cme_flank["el"].values[j]
                    )
                    if self.bias_term_bool else cme_flank["el"].values[j]
                    for j in inds_req
                ]


        return obs_op