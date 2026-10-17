"""
sir_observers.py

Observer classes that turn a solved SURF/HUXt run into a time-elongation profile of a CME, i.e. what a
heliospheric imager (HI) would measure. They are used by ``ObservationOperator`` (sir_observation_operator.py).

    SynthObsEphem   : container for an observer's times and positions, in the form
                      surf_imaging's SyntheticImager expects
    ObserverTracer  : elongation of the CME *flank* from the model's CME tracer particles
                      (incompressible "surf"/"huxt" models)
    ObserverSynthHI : elongation from synthetic HI images (j-maps) of a compressible SURF run
                      ("compress_surf" / "compress_surf3d"), tracked with surf_imaging

Each observer exposes its result as ``self.model_flank``, a DataFrame with columns
["time", "el", "r", "lon"] (elongation in degrees).
"""
import asyncio
import datetime
import sys
from functools import partial
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

import surf.surf as S
import surf.surf_imaging as Sim

import huxt.huxt as H

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from timing import timed



def fixed_observer_position(obs_radius, obs_longitude, obs_latitude) -> tuple[Quantity, Quantity, Quantity]:
    """
    The single position of a fixed synthetic observer: (r in AU, lon in deg, lat in deg) as scalars. Each input may
    be a scalar or an array / list of identical values (e.g. the same position repeated for every observation
    time). Different values (a moving synthetic observer) are not supported and raise a ValueError.
    """
    out = []
    for name, x, unit in (("obs_radius", obs_radius, u.AU), ("obs_longitude", obs_longitude, u.deg),
                          ("obs_latitude", obs_latitude, u.deg)):
        v = np.atleast_1d(u.Quantity(x).to(unit).value).astype(float).ravel()
        if v.size == 0:
            raise ValueError(f"{name} is empty")
        if np.any(v != v[0]):
            raise ValueError(f"{name} varies ({v.min()}..{v.max()} {unit}): only a fixed synthetic observer is "
                             f"supported; give one position")
        out.append(v[0] * unit)
    return out[0], out[1], out[2]


def background(f: Callable) -> Callable:
    """
    Decorator that runs ``f`` in the default asyncio executor and returns a future
    (not currently used by the data assimilation).
    """
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return asyncio.get_event_loop().run_in_executor(
            None,
            partial(f, *args, **kwargs)
        )

    return wrapped


def sample_track(
        track_time: npt.ArrayLike,
        track_elon: npt.ArrayLike,
        times: npt.ArrayLike,
) -> npt.NDArray[float]:
    """
    Linearly interpolate a tracked time-elongation profile to ``times`` (same time units as track_time).
    Unlike the observation operator's np.interp, times outside the tracked range give NaN rather than the
    first/last tracked value.
    :param track_time: Tracked times (may be a scalar NaN if nothing was tracked)
    :param track_elon: Tracked elongations (deg)
    :param times: Times to sample
    :return: Elongations at ``times`` (NaN where unavailable)
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    t = np.atleast_1d(np.asarray(track_time, dtype=float))
    e = np.atleast_1d(np.asarray(track_elon, dtype=float))
    out = np.full(times.shape, np.nan)
    if t.shape != e.shape:
        return out
    ok = np.isfinite(t) & np.isfinite(e)
    if not ok.any():
        return out
    t, e = t[ok], e[ok]
    order = np.argsort(t)
    t, e = t[order], e[order]
    inside = (times >= t[0]) & (times <= t[-1])
    out[inside] = np.interp(times[inside], t, e)
    return out


def n_output_steps(run, sim_time: Quantity) -> int:
    """
    Number of output times a SURF run of length sim_time has (SURF's time_grid: nt = floor(T / dt),
    nt_out = int(nt / dt_scale)); dt depends only on the grid and v_max, not on the run length.
    """
    dt = run.dt.to(u.s).value
    dt_scale = float(np.ravel(u.Quantity(run.dt_scale).value)[0])
    nt = int(np.floor(sim_time.to(u.s).value / dt))
    return int(nt / dt_scale)


def _truncated_run(run, n_keep: int):
    """Shallow copy of one solved SURF run with its output-time axis (and its CMEs' tracer coords) cut."""
    import copy as _copy
    r = _copy.copy(run)
    r.time_out = run.time_out[:n_keep]
    cmes = []
    for cme in getattr(run, "cmes", []) or []:
        c = _copy.copy(cme)
        if isinstance(getattr(cme, "coords", None), dict):
            c.coords = {j: v for j, v in cme.coords.items() if j < n_keep}
        cmes.append(c)
    r.cmes = cmes
    return r


def truncated_model_view(model, end: Quantity):
    """
    A shallow view of a solved SURF / SURF3d model whose output-time axis (time_out, and the CMEs' tracer
    coords) stops where a run of length ``end`` would have stopped. The solution arrays are shared, not
    copied, and indexing by output step is unchanged, so the synthetic imagers (which loop over time_out /
    the observer times, index rho_grid[time_step] and use the CMEs' coords) see exactly what a run of length
    ``end`` would give: SURF's time step depends only on the grid and v_max, and the solution is causal.
    Returns the model itself if it is not longer than ``end``.
    """
    import copy as _copy
    runs = getattr(model, "SURFlat", None)
    reference = runs[0] if runs else model
    try:
        n_keep = n_output_steps(reference, end)
    except AttributeError:
        return model
    if n_keep >= len(reference.time_out):
        return model
    if runs:
        view = _copy.copy(model)
        view.SURFlat = [_truncated_run(run, n_keep) for run in runs]
        return view
    return _truncated_run(model, n_keep)


class SynthObsEphem:
    """
    Class to hold observation time, radius, latitude and longitude for use in initialising
      Sim.SyntheticImager class
    """
    def __init__(
            self,
            surf_init_time: datetime.datetime,
            obs_times: Quantity[u.s],
            obs_rads: Quantity[u.AU],
            obs_lats: Quantity[u.deg],
            obs_lons: Quantity[u.deg],
    ) -> None:
        """
        :param surf_init_time: Start datetime of the model run
        :param obs_times: Observation time(s), in seconds after surf_init_time (scalar or array)
        :param obs_rads, obs_lats, obs_lons: Observer heliocentric distance, latitude and longitude
            (same shape as obs_times)
        """
        # Scalar inputs are wrapped in length-1 arrays (r converted to m, angles to rad)
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
    Pseudo-observations of a ConeCME's flank elongation from the CME tracer particles of a solved
    SURF/HUXt run, for an observer at a specified longitude *relative to Earth*. The observer is at 1 AU
    and at Earth's latitude throughout the run.
    Result: ``self.model_flank`` (one row per model output time).
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
        """
        :param use_model: "surf", "compress_surf" or "huxt"
        :param model: Solved SURF/HUXt model
        :param cme: ConeCME from the solved model (its .coords are populated)
        :param longitude: Observer longitude relative to Earth
        :param el_min, el_max: Imager elongation range (stored, not applied here)
        """
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

        # Observer ephemeris: Earth's, shifted in longitude and placed at 1 AU
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

            # Compute the observer-CME distance, s, and the elongation from the cosine rule:
            #   cos(e) = (r_obs^2 + s^2 - r_cme^2) / (2 r_obs s)
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
    Pseudo-observations of a CME's elongation from synthetic heliospheric-imager data of a solved
    compressible SURF run: j-maps are computed with surf_imaging, the CME front is tracked in the
    differenced j-map, and the tracked elongation is interpolated to the observation time(s).

    Result: ``self.model_flank`` with one row per observation time (``el`` is NaN when no CME was tracked).
    NB: in 2D the imager assumes the observer is in the model plane, so a real craft more than 3 deg in
    latitude from the model's latitude (0 deg) raises a ValueError in surf_imaging.
    NB: np.interp holds the first/last tracked elongation for observation times outside the tracked range.
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
            profile_times: list[datetime.datetime] | None = None,
            imaging_end: Quantity[u.s] | None = None,
    ):
        """
        Generate the synthetic HI profile and extract the elongation at the observation time(s)
        :param use_model: The model type being used in this simulation ('compress_surf' or 'compress_surf3d')
        :param model: The model after it has been solved with the CMEs required
        :param obs_time: Observation time(s) (datetime or array of datetimes)
        :param craft: Spacecraft to observe from. If None, a synthetic observer fixed at
            (obs_radius, obs_longitude, obs_latitude) is used.
        :param psi_ert_avg: Position angle (deg) of the imager slice (3D only)
        :param obs_radius: Observation radius
        :param obs_longitude: Observation longitude
        :param obs_latitude: Observation latitude
            (synthetic observer: a FIXED position, one value each; a moving synthetic observer is not supported)
        :param el_min: Minimum elongation
        :param el_max: Maximum elongation
        :param hi_resolution: Temporal resolution to model the HI data in
        :param profile_times: Optional extra times at which to sample the tracked profile (diagnostics);
            the result is ``self.model_profile`` (NaN outside the tracked time range, no extrapolation)
        :param imaging_end: Only image (and track) the model up to this time after the model start, even if
            the model ran longer (e.g. so that the same run also gives the CME's arrival at Earth). SURF's time
            step does not depend on the run length and the solution is causal, so the j-map, the tracked
            profile and the elongations are then the same as for a run of length ``imaging_end``.
        """

        self.use_model: str = use_model
        if self.use_model in ["compress_surf", "compress_surf3d"]:
            ert_ephem: S.Observer = model.get_observer('EARTH')
        else:
            sys.exit("Synthetic HI can only be generated with use_model = 'compress_surf' and 'compress_surf3d'")

        self.obs_time: npt.NDArray[datetime.datetime] = obs_time

        # Model start time and run length (SURF3d: taken from its first latitude slice)
        if self.use_model == "compress_surf":
            surf_model_init_time: datetime.datetime = model.time_init.datetime
            surf_model_sim_time: float = float(model.simtime.to(u.s).value)
        else:
            surf_model_init_time: datetime.datetime = model.SURFlat[0].time_init.datetime
            surf_model_sim_time: float = float(model.SURFlat[0].simtime.to(u.s).value)

        # Observation time(s) in seconds since the model start (a single datetime is wrapped in an array)
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

        # Model OUTPUT times: the 3D imager looks the observer up by output-time index (as for a real craft,
        #  whose ephemeris model.get_observer gives at time_out); time_out == time only for dt_scale = 1
        if use_model == "compress_surf":
            self.model_times = model.time_out

        elif use_model == "compress_surf3d":
            self.model_times = model.SURFlat[0].time_out

        # Times at which to make synthetic images: 0 .. simtime (or imaging_end) at the HI cadence
        imaging_window_sec = surf_model_sim_time
        if imaging_end is not None:
            imaging_window_sec = min(surf_model_sim_time, float(imaging_end.to(u.s).value))
        self.synth_obs_times = np.arange(
            0,
            imaging_window_sec + (hi_resolution_sec / 2.0),
            hi_resolution_sec
        ) * u.s

        self.n_synth_obs_times = len(self.synth_obs_times)
        with timed("synth_hi.model_time_array"):
            self.ast_times = Time([
                surf_model_init_time + datetime.timedelta(seconds=s)
                for s in self.model_times.value
            ])

        # Observer: a real body, or a synthetic observer at a fixed position (see fixed_observer_position)
        if craft is not None:
            # 3D imager: the craft at the model output times (as for the synthetic observer below)
            self.craft_observer = model.get_observer(craft)

            # 2D imager (SynthObsEphem): positions must be at the synthetic-image times, since
            #  SynthObsEphem pairs position i with self.synth_obs_times[i]. (The model-time ephemeris
            #  above has a different length and spacing.)
            self.obs_times: npt.NDArray[Time] = Time([
                surf_model_init_time + datetime.timedelta(seconds=s)
                for s in self.synth_obs_times.value
            ])
            craft_at_hi_times = S.Observer(craft, self.obs_times)
            self.obs_r: npt.NDArray[Quantity[u.AU]] = craft_at_hi_times.r.to(u.AU)
            self.obs_lon: npt.NDArray[Quantity[u.deg]] = craft_at_hi_times.lon.to(u.deg)
            self.obs_lat: npt.NDArray[Quantity[u.deg]] = craft_at_hi_times.lat.to(u.deg)
        else:
            self.obs_times: npt.NDArray[Time] = Time([
                surf_model_init_time + datetime.timedelta(seconds=s)
                for s in self.synth_obs_times.value
            ])

            # The synthetic observer is fixed: one position
            r0, lon0, lat0 = fixed_observer_position(obs_radius, obs_longitude, obs_latitude)

            # 2D imager (SynthObsEphem): the position at every synthetic-image time
            self.obs_r: npt.NDArray[Quantity[u.AU]] = r0 * np.ones(self.n_synth_obs_times)
            self.obs_lon: npt.NDArray[Quantity[u.deg]] = lon0 * np.ones(self.n_synth_obs_times)
            self.obs_lat: npt.NDArray[Quantity[u.deg]] = lat0 * np.ones(self.n_synth_obs_times)

            # 3D imager (S.Observer): SURF requires the given positions to span every requested (model output)
            #  time, i.e. the whole run, which can be longer than the imaging window (imaging_end, e.g. a 5-day
            #  run imaged over 3 days). The position is fixed, so its value at the start and end of the run is
            #  enough.
            run_end_s = max(float(self.model_times[-1].to(u.s).value), float(self.synth_obs_times[-1].to(u.s).value))
            synth_heeq_pos = {
                "time": Time([surf_model_init_time,
                              surf_model_init_time + datetime.timedelta(seconds=run_end_s)]),
                "r": r0 * np.ones(2),
                "lon": lon0 * np.ones(2),
                "lat": lat0 * np.ones(2),
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
            # Image a view of the model that ends at imaging_end (see truncated_model_view)
            imaging_model = truncated_model_view(model, imaging_window_sec * u.s)
            with timed(f"synth_hi.compute_jmap.{use_model}"):
                jmap, djmap = synth_imager_class.compute_jmap(imaging_model)
            with timed(f"synth_hi.track_cmes.{use_model}"):
                cme_profile = synth_imager_class.track_cmes(imaging_model, djmap)[0]
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
            # The 3D imager images every model output time: give it a view of the model that ends at
            #  imaging_end (only the output-time axis is shortened; see truncated_model_view)
            imaging_model = truncated_model_view(model, imaging_window_sec * u.s)
            with timed(f"synth_hi.compute_jmap.{use_model}"):
                jmap, djmap = synth_imager_class.compute_jmap(imaging_model)
            with timed(f"synth_hi.track_cmes.{use_model}"):
                cme_profile = synth_imager_class.track_cmes(imaging_model, djmap)[0]
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

        # Interpolate the tracked elongations to the observation times (NaN if nothing was tracked)
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

        # Tracked profile sampled at the optional diagnostic times
        self.model_profile: npt.NDArray[float] | None = None
        if profile_times is not None:
            profile_days = np.array([
                (pt - surf_model_init_time).total_seconds() for pt in profile_times
            ]) / 86400.0
            self.model_profile = sample_track(synth_imager_time, synth_imager_elon, profile_days)
