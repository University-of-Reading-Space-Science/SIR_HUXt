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
import skimage as ski

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
from obs_bias import apply_bias

import asyncio
import atexit
from functools import partial
import time
from multiprocessing import Pool


###############################################################################################
# Persistent worker pool
#  get_cme_flanks used to create (and tear down) a new multiprocessing Pool on every call, i.e.
#  twice per observation in the Aux-PF plus once for the synthetic truth. On Windows each new
#  worker is spawned and re-imports astropy/sunpy/SURF and reloads the numba cache, which costs
#  tens of seconds per pool. The pool below is created once, on first use, and reused for the
#  whole run. Call shutdown_worker_pool() when the run is finished (it is also registered with
#  atexit as a fallback).
###############################################################################################
_WORKER_POOL = None
_WORKER_POOL_SIZE = None


def get_worker_pool(n_proc: int = None) -> Pool:
    """
    Return the shared multiprocessing pool, creating it on first use.
    :param n_proc: Number of worker processes (defaults to os.cpu_count() - 1, minimum 1).
        If a pool of a different size already exists it is replaced.
    :return: multiprocessing.Pool
    """
    global _WORKER_POOL, _WORKER_POOL_SIZE

    if n_proc is None:
        n_proc = max(1, int((os.cpu_count() or 2) - 1))

    if (_WORKER_POOL is not None) and (_WORKER_POOL_SIZE != n_proc):
        shutdown_worker_pool()

    if _WORKER_POOL is None:
        print(f"Starting persistent worker pool with {n_proc} processes")
        _WORKER_POOL = Pool(processes=n_proc)
        _WORKER_POOL_SIZE = n_proc

    return _WORKER_POOL


def shutdown_worker_pool() -> None:
    """
    Close the shared worker pool (if one is running) and wait for its workers to exit.
    """
    global _WORKER_POOL, _WORKER_POOL_SIZE

    if _WORKER_POOL is not None:
        _WORKER_POOL.close()
        _WORKER_POOL.join()
        _WORKER_POOL = None
        _WORKER_POOL_SIZE = None

    return None


atexit.register(shutdown_worker_pool)


###############################################################################################
# Synthetic imager with cached field-of-view geometry
###############################################################################################
class CachedSyntheticImager3DOrig(Sim.SyntheticImager3DOrig):
    """
    SURF's SyntheticImager3DOrig, but the line-of-sight (field-of-view) geometry is only recomputed
    when the observer's position changes.

    SyntheticImager3DOrig.compute_jmap calls _compute_fov_geometry at every model output step. That
    builds a ~1230 x 251 SkyCoord, transforms it HelioprojectiveRadial -> HeliographicStonyhurst and
    recomputes the Thomson-scattering factors, and was ~half the cost of each ensemble member.
    The geometry depends only on the observer's HEEQ position (r, lon, lat): the LOS grids, the
    pointing (pa, elongations) and the HPR -> HGS transform for an observer given in HGS do not depend
    on time otherwise. In this pipeline ObserverSynthHI always builds a fixed-position observer, so the
    geometry is now computed once per imager instead of once per output step. The cache key is the
    exact position, so a moving observer (e.g. a real craft ephemeris) still recomputes whenever its
    position changes and gives identical results to the parent class.

    _fov_mask (which depends only on that geometry and the model grid) is cached alongside it.
    """
    def __init__(self, *args, **kwargs):
        # Must exist before the parent __init__ calls _compute_fov_geometry(0)
        self._fov_geometry_key = None
        self._fov_mask_cache = None
        self._fov_mask_model_id = None
        super().__init__(*args, **kwargs)

    def _observer_position_key(self, time_step: int) -> tuple[float, float, float]:
        return (
            float(self.position.r[time_step].to_value(u.m)),
            float(self.position.lon[time_step].to_value(u.rad)),
            float(self.position.lat[time_step].to_value(u.rad)),
        )

    def _compute_fov_geometry(self, time_step):
        key = self._observer_position_key(time_step)
        if key == self._fov_geometry_key:
            # Observer has not moved: geometry from the previous call is still valid
            return None

        super()._compute_fov_geometry(time_step)
        self._fov_geometry_key = key
        # The mask depends on the geometry, so invalidate it
        self._fov_mask_cache = None
        self._fov_mask_model_id = None

        return None

    def _fov_mask(self, model):
        if (self._fov_mask_cache is not None) and (self._fov_mask_model_id == id(model)):
            return self._fov_mask_cache

        fov_mask = super()._fov_mask(model)
        self._fov_mask_cache = fov_mask
        self._fov_mask_model_id = id(model)

        return fov_mask


# Elongation range (deg) of the synthetic heliospheric imager used for the observation operator.
#  The upper limit is deliberately wider than the largest elongation assimilated (elong_max_obs = 30 deg
#  in da_surf_setup) so that a member whose CME runs slightly ahead of the real CME is still tracked at
#  the last observations, rather than having already left the imager's field of view (-> NaN).
#  The same limits set the SURF3d latitude range (from the imager field of view), so they must match.
IMAGER_ELON_MIN: float = 4.0
IMAGER_ELON_MAX: float = 35.0


def merge_tracked_features(
        cme_profile: dict,
        join_tolerance_deg: float = 0.5,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Combine the pieces of a CME's tracked J-map track into one time-elongation track.

    SURF's track_cmes often splits a CME's track into several features, named by size, so
    "feature_00" is the largest piece - frequently the later part of the track (e.g. 16-30 deg), while
    the early part (6-10 deg) is "feature_01". Using feature_00 alone made every observation outside
    that piece NaN even though the CME was tracked there.

    Features can also be unrelated ridges running alongside the CME front (seen in the diagnostics as
    a feature at 24-26 deg overlapping in time with the front at 17-25 deg), so features are not simply
    pooled. Starting from the largest, each further feature is added only if
      - it does not overlap in time with any feature already accepted (sharing a single boundary
        frame is allowed, see below), and
      - elongation still increases across every join between consecutive pieces
        (end of the earlier piece <= start of the later piece + join_tolerance_deg).
    Two pieces can meet at one model frame where the edge detector found two edge points (e.g. 6.40 and
    7.35 deg at the same time). Previously this counted as an overlap, so the smaller piece was dropped
    even when it held the front (here 6.05 -> 7.35 -> 8.65 deg in equal steps, versus 6.05 -> 6.40 ->
    8.65 deg with the lagging point). Such a shared frame is now accepted, and at any frame with two
    values the one nearest the straight line through the neighbouring frames is kept.
    Gaps between accepted pieces are bridged by the linear interpolation that follows.
    :param cme_profile: {feature_name: {"t": times (days), "e": elongations (deg), ...}}
    :param join_tolerance_deg: Allowed elongation decrease across a join (tracking noise), in deg
    :return: (t, e) arrays sorted by time; empty arrays if nothing was tracked
    """
    pieces = []
    for feat in (cme_profile or {}).values():
        t = np.atleast_1d(np.asarray(feat["t"], dtype=float))
        e = np.atleast_1d(np.asarray(feat["e"], dtype=float))
        good = np.isfinite(t) & np.isfinite(e)
        if np.any(good):
            order = np.argsort(t[good])
            pieces.append((t[good][order], e[good][order]))

    if not pieces:
        return np.array([]), np.array([])

    # Largest piece first (track_cmes names them by region area; use point count here)
    pieces.sort(key=lambda p: p[0].size, reverse=True)

    def joins_ok(candidate_set):
        segs = sorted(candidate_set, key=lambda p: p[0][0])
        for (t1, e1), (t2, e2) in zip(segs[:-1], segs[1:]):
            if t1[-1] > t2[0] + 1e-9:
                return False  # overlap in time (more than a single shared boundary frame)
            end_prev = np.median(e1[-2:])
            start_next = np.median(e2[:2])
            if end_prev > start_next + join_tolerance_deg:
                return False  # elongation would decrease across the join
        return True

    accepted = [pieces[0]]
    for piece in pieces[1:]:
        if joins_ok(accepted + [piece]):
            accepted.append(piece)

    t_all = np.concatenate([p[0] for p in accepted])
    e_all = np.concatenate([p[1] for p in accepted])
    order = np.argsort(t_all, kind="stable")
    t_all, e_all = t_all[order], e_all[order]

    # Frames with two values (pieces sharing a boundary frame): keep the value nearest the straight
    #  line through the neighbouring frames (or, with only one neighbour, nearest that neighbour)
    t_u = np.unique(t_all)
    if t_u.size == t_all.size:
        return t_all, e_all
    e_u = np.empty(t_u.size)
    for i, t in enumerate(t_u):
        vals = e_all[np.isclose(t_all, t)]
        if vals.size == 1:
            e_u[i] = vals[0]
            continue
        nb_t, nb_e = [], []
        for j in (i - 1, i + 1):
            if 0 <= j < t_u.size:
                nb_t.append(t_u[j])
                nb_e.append(float(np.median(e_all[np.isclose(t_all, t_u[j])])))
        if len(nb_t) == 2:
            pred = np.interp(t, nb_t, nb_e)
        elif len(nb_t) == 1:
            pred = nb_e[0]
        else:
            pred = np.max(vals)
        e_u[i] = vals[np.argmin(np.abs(vals - pred))]

    return t_u, e_u


# Write per-member diagnostics from ObserverSynthHI (tracked CME features, observation times and,
#  when a member's elongation is NaN, its J-map) to <OUT_BASE_DIR>/obsop_diagnostics.
#  Set to False once the observation operator is behaving.
WRITE_OBSOP_DIAGNOSTICS: bool = True


def obsop_diagnostics_dir() -> str:
    base = os.getenv("OUT_BASE_DIR") or "."
    out_dir = os.path.join(base, "obsop_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


###############################################################################################
# Members whose CME has not yet reached the model's inner boundary (or has not yet been picked up
#  by the J-map tracker) at an observation time used to return NaN, which the filter treats as zero
#  likelihood. That kills late/slow members outright, however small the delay, and biases the
#  posterior towards early, fast CMEs. With EXTRAPOLATE_BEFORE_TRACK the elongation at such times
#  is instead extrapolated backwards from the first tracked model point with a self-similar
#  expansion (SSE) geometry and constant-speed kinematics inside r_min (the same assumption used
#  to move t_init from cme_init_rad to r_min), scaled to match the model at the first tracked
#  point so the profile is continuous. Times after the CME leaves the field of view stay NaN.
###############################################################################################
EXTRAPOLATE_BEFORE_TRACK: bool = False
# Allowed range of (model elongation / SSE elongation) at the first tracked point. Outside this
#  range the SSE shape is not trusted for that member and its values stay NaN.
EXTRAP_ANCHOR_RATIO_BOUNDS: tuple[float, float] = (0.5, 2.0)
# Only fill a time if the extrapolated elongation is at most this far (deg) above the smallest
#  elongation the model domain can show (arcsin(r_min / r_obs), ~6 deg for STEREO-A). This limits
#  the extrapolation to its purpose (CME not yet in, or just inside, the model domain). A missing
#  value further out means the tracker failed to pick up a CME that is in the model; that is left
#  as NaN (and reported in the diagnostics) rather than hidden behind an extrapolated value.
EXTRAP_MAX_ABOVE_MODEL_MIN_DEG: float = 2.0


def sse_flank_elongation(r_apex, d_obs, dlon, beta, half_width):
    """
    Elongation (rad) of the outer tangent to a self-similar-expansion (SSE) CME front, seen from an
    observer, in the observer's imaging plane.
    The CME front is a sphere tangent to the cone of half-angle `half_width` with its apex at
    heliocentric distance `r_apex` (SSE geometry, Davies et al. 2012). The front is cut by the plane
    containing the Sun and the observer (offset in latitude by `beta` from the CME axis), giving a circle
    whose outer tangent elongation is returned.
    :param r_apex: Heliocentric distance of the CME apex (same units as d_obs)
    :param d_obs: Heliocentric distance of the observer
    :param dlon: Longitude of the CME apex minus longitude of the observer (rad)
    :param beta: Latitude of the CME axis relative to the imaging plane (rad)
    :param half_width: CME angular half-width (rad)
    :return: Elongation in rad, NaN where the front does not intersect the plane or encloses the observer
    """
    r_apex, d_obs, dlon, beta, half_width = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (r_apex, d_obs, dlon, beta, half_width)])
    sin_lam = np.sin(half_width)
    d_c = r_apex / (1.0 + sin_lam)              # Sun to centre of the front sphere
    r_c = r_apex * sin_lam / (1.0 + sin_lam)    # radius of the front sphere
    big_d = d_c * np.cos(beta)                  # centre projected into the imaging plane
    rho2 = r_c ** 2 - (d_c * np.sin(beta)) ** 2  # radius^2 of the in-plane circle
    with np.errstate(invalid="ignore"):
        rho = np.sqrt(rho2)
        cx = big_d * np.cos(dlon) - d_obs
        cy = big_d * np.sin(dlon)
        ell = np.hypot(cx, cy)                  # observer to circle centre
        eps_c = np.arccos(np.clip(-cx / ell, -1.0, 1.0))
        delta = np.arcsin(np.clip(rho / ell, -1.0, 1.0))
        eps = eps_c + delta
    bad = (rho2 <= 0) | (ell <= rho) | ~np.isfinite(eps)
    return np.where(bad, np.nan, eps)


###############################################################################################
# Search band for SURF's J-map CME tracker
#  SyntheticImager3DOrig.track_cmes only looks for the CME front inside a band of elongations at
#  each time (cme_mask). SURF sets the band by putting the CME nose in the plane of the sky,
#  e = arctan(r / r_obs), moving at 0.75-1.25 v. For a CME heading well away from the plane of the
#  sky (e.g. 09A: ~29 deg in longitude from STEREO-A's Sun direction) the visible front is at lower
#  elongation than that, so the band sat above the front and nothing was tracked below ~20 deg.
#  With USE_WIDE_SEARCH_BAND the same plane-of-sky conversion is kept but the band is widened: a
#  lower slow-speed factor, and the lower edge moved down by a further fixed angle (the front at
#  6-9 deg, just after the CME enters the model, is not covered by a smaller speed factor alone).
#  The defaults contain the CME front in every frame of all 20 members of the 09A test run with at
#  least 1.5 deg to spare (SURF's band: 240 of 520 frames).
###############################################################################################
USE_WIDE_SEARCH_BAND: bool = True
SEARCH_BAND_SPEED_FACTORS: tuple[float, float] = (0.5, 1.25)  # (slow, fast); SURF uses (0.75, 1.25)
SEARCH_BAND_LOWER_MARGIN_DEG: float = 2.0  # lower edge moved down by this angle; SURF uses 0
SEARCH_BAND_UPPER_MARGIN_DEG: float = 2.0  # upper edge moved up by this angle; SURF uses 0
SEARCH_BAND_TIME_PAD_HOURS: float = 2.0  # as in SURF


class WideBandSyntheticImager3D(Sim.SyntheticImager3DOrig):
    """
    SURF's SyntheticImager3DOrig with a wider search band for the CME tracker (see
    USE_WIDE_SEARCH_BAND). The band still uses SURF's plane-of-sky conversion; only the speed
    factors and angular margins differ. Everything else (J-map, ridge and edge detection, feature
    extraction) is unchanged.
    """

    def _search_band(self, cme, times):
        """
        Lower and upper elongation of the tracker's search band at each model output time.
        Nose distances r = initial height + f x v x (t - t_launch -/+ pad), with f the slow/fast
        factors of SEARCH_BAND_SPEED_FACTORS, are converted with the plane-of-sky approximation
        e = arctan(r / r_obs) (as in SURF), then the lower edge is moved down by
        SEARCH_BAND_LOWER_MARGIN_DEG and the upper edge up by SEARCH_BAND_UPPER_MARGIN_DEG.
        Both edges are capped at the imager's maximum elongation.
        :param cme: Solved ConeCME
        :param times: Model output times (days from model start)
        :return: (e_slow, e_fast) arrays in deg; NaN where there is no band (before launch, or after
            the CME has left the field of view)
        """
        f_slow, f_fast = SEARCH_BAND_SPEED_FACTORS
        dt_pad = SEARCH_BAND_TIME_PAD_HOURS * 3600.0
        t_launch = cme.t_launch.to(u.day).value
        r_init = cme.initial_height.to(u.m).value
        v = cme.v.to(u.m / u.s).value
        e_max = self.e_max.to(u.deg).value

        e_slow_all = np.full(len(times), np.nan)
        e_fast_all = np.full(len(times), np.nan)
        for id_t, t in enumerate(times):
            travel_time = (t - t_launch) * 86400
            if travel_time < -dt_pad:
                continue

            r_obs = self.position.r[id_t].to(u.m).value
            r_nose_slow = max(r_init + f_slow * v * (travel_time - dt_pad), 0.0)
            r_nose_fast = r_init + f_fast * v * (travel_time + dt_pad)

            e_slow = np.rad2deg(np.arctan(r_nose_slow / r_obs)) - SEARCH_BAND_LOWER_MARGIN_DEG
            e_fast = np.rad2deg(np.arctan(r_nose_fast / r_obs)) + SEARCH_BAND_UPPER_MARGIN_DEG
            e_slow = min(e_slow, e_max)
            e_fast = min(e_fast, e_max)

            if e_slow == e_fast:
                # Leave this loop, as CME has almost certainly left the field of view.
                break
            e_slow_all[id_t], e_fast_all[id_t] = e_slow, e_fast

        return e_slow_all, e_fast_all

    def track_cmes(self, model, djmap):
        """
        Use image processing techniques to track the CME front in the differenced Jmap.
        Copy of SURF's SyntheticImager3DOrig.track_cmes (surf_imaging.py) except that the
        search band (cme_mask) comes from self._search_band (wider band, see USE_WIDE_SEARCH_BAND).
        """
        # Check that the model is compatible with this FoV.
        self._check_latitude_compatibility(model)

        # Check a ConeCME object exists.
        if not model.SURFlat[0].cmes:
            raise ValueError(
                "model.cmes is empty. Solve the model with at least one ConeCME before calling "
                "track_cmes.")
        if not all(isinstance(cme, S.ConeCME) for cme in model.SURFlat[0].cmes):
            raise TypeError(
                f"All entries in model.cmes must be instances of S.ConeCME. "
                f"Got types: {[type(cme).__name__ for cme in model.SURFlat[0].cmes]}."
            )

        times = model.SURFlat[0].time_out.to(u.day).value
        elons = self.e.to(u.deg).value

        cmap = plt.cm.gray.copy()
        cmap.set_bad(color='midnightblue')

        """fig, ax = plt.subplots(figsize=(10, 10))
        vmin, vmax = np.nanpercentile(djmap, [1, 99])
        ax.pcolormesh(times, elons, djmap, cmap=cmap, vmin=vmin, vmax=vmax)"""

        # Clip and scale the jmap. Find ridges.
        djmap = np.nan_to_num(djmap, nan=0.0, posinf=0.0, neginf=0.0)

        # This test is shit and needs fixing.
        # if np.allclose(djmap, 0.0):
        #    print('No structure in the jmap. Returning empty list of profiles.')
        #    return []

        vmin, vmax = np.nanpercentile(np.abs(djmap), [0, 100])
        djmap_clipped = np.clip(djmap, vmin, vmax)
        djmap_norm = (djmap_clipped - vmin) / (vmax - vmin)

        ridge = ski.filters.frangi(djmap_norm, sigmas=[1], black_ridges=False)
        ridge[ridge > np.percentile(ridge, 95)] = 1
        ridge[ridge < 1] = 0
        # Now do edge detection on the ridge.
        grad_t = ski.filters.sobel_v(djmap_norm)
        pos_grad = grad_t > 0


        # Build a mask to limit the search region for this CME's t-e profile (see _search_band).
        #  Only the band differs from SURF's track_cmes.
        cme_profiles = []
        for cme in model.SURFlat[0].cmes:
            e_slow_all, e_fast_all = self._search_band(cme, times)
            cme_mask = np.zeros(djmap.shape, dtype=bool)
            for id_t in range(times.size):
                if np.isnan(e_slow_all[id_t]):
                    continue
                id_good = (elons > e_slow_all[id_t]) & (elons < e_fast_all[id_t])
                cme_mask[id_good, id_t] = True

            edges = ski.feature.canny(ridge, sigma=1, mask=cme_mask) & pos_grad
            label = ski.measure.label(edges)
            regions = ski.measure.regionprops(label)
            # Sort regions from largest to smallest (by number of coordinates)
            # This increases the chance the t-e profiles are in the correct order.
            regions = sorted(regions, key=lambda r: r.area, reverse=True)

            # For each region convert pixel coords to map coords and average multiple elons at fixed
            # times
            profiles = {}
            for id_r, region in enumerate(regions):

                # Do not include small regions
                if region.area < 5:
                    continue

                c = np.array(region.coords)
                # Get map coords from pixel coords.
                t_pix = times[c[:, 1]]
                e_pix = elons[c[:, 0]]
                # Now average the e_pix values for each unique t_pix value.
                t_pix_unique = np.unique(t_pix)
                e_pix_mean = np.zeros(t_pix_unique.shape)
                for id_tu, tu in enumerate(t_pix_unique):
                    id_t = np.where(t_pix == tu)[0]
                    e_pix_mean[id_tu] = np.nanmean(e_pix[id_t])

                t_real = t_pix_unique + model.SURFlat[0].time_init.jd
                profiles[f"feature_{id_r:02d}"] = {'t': t_pix_unique, 't_real': t_real,
                                                   'e': e_pix_mean}

            cme_profiles.append(profiles)

        return cme_profiles



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

            # If none of the CME points are in the observer's field of view, there is no flank to
            #  measure (np.argmax would raise "attempt to get argmax of an empty sequence")
            if np.size(e_obs) == 0:
                flank.loc[i, ['lon', 'r', 'el']] = np.nan
                continue

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
            diag_info: dict | None = None,
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
        self.surf_model_init_time: datetime.datetime = surf_model_init_time
        self.obs_time_sec: npt.NDArray[Quantity[u.s]] = np.array(obs_time_from_start_sec) * u.s #ert_ephem.time
        self.obs_time_day: npt.NDArray[Quantity[u.day]] = self.obs_time_sec.to(u.day)
        self.n_obs = len(self.obs_time)
        self.el_min: float = el_min
        self.el_max: float = el_max
        self.psi_ert_avg: float = psi_ert_avg

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
            # Combine all tracked pieces of the CME's track (previously only "feature_00", the
            #  largest piece, was used; see merge_tracked_features)
            synth_imager_time, synth_imager_elon = merge_tracked_features(cme_profile)

        elif use_model == "compress_surf3d":
            # Cached-geometry imager: identical output, but the LOS geometry is not rebuilt at
            #  every one of the model's output steps when the observer does not move
            # SURF's imager with a wider search band for the CME tracker (see
            #  WideBandSyntheticImager3D / USE_WIDE_SEARCH_BAND)
            imager_cls = WideBandSyntheticImager3D if USE_WIDE_SEARCH_BAND else Sim.SyntheticImager3DOrig
            synth_imager_class = imager_cls(
                observer=self.craft_observer,
                pa=self.psi_ert_avg,
                elon_min=self.el_min,
                elon_max=self.el_max
            )
            jmap, djmap = synth_imager_class.compute_jmap(model)
            cme_profile = synth_imager_class.track_cmes(model, djmap)[0]
            #fig, ax = synth_imager_class.plot_jmap(model, jmap, djmap)
            #plt.show()

            # Extract time and elongation profiles of synthetic HI-data
            # Combine all tracked pieces of the CME's track (previously only "feature_00", the
            #  largest piece, was used; see merge_tracked_features)
            synth_imager_time, synth_imager_elon = merge_tracked_features(cme_profile)

        # Interpolate the synthetic elongations to observation times
        #  - NaN pairs are dropped (np.interp does not handle NaNs and returns garbage around them)
        #  - points are sorted by time (np.interp requires increasing x)
        #  - observation times outside the tracked feature's time range return NaN rather than being
        #    clamped to the first/last tracked elongation (np.interp's default), which previously gave
        #    ensemble members whose CME was not yet / no longer in the field of view a fabricated value
        synth_imager_time = np.atleast_1d(np.asarray(synth_imager_time, dtype=float))
        synth_imager_elon = np.atleast_1d(np.asarray(synth_imager_elon, dtype=float))
        valid = np.isfinite(synth_imager_time) & np.isfinite(synth_imager_elon)
        if np.any(valid):
            t_valid = synth_imager_time[valid]
            e_valid = synth_imager_elon[valid]
            order = np.argsort(t_valid)
            synth_imager_elon_interp = np.interp(
                self.obs_time_day.value, t_valid[order], e_valid[order],
                left=np.nan, right=np.nan
            )
        else:
            synth_imager_elon_interp = np.nan * np.ones(self.n_obs)

        # Fill observation times before the first tracked point (CME not yet in the model domain
        #  or not yet tracked) by SSE extrapolation; see EXTRAPOLATE_BEFORE_TRACK
        self.extrap_mask = np.zeros(self.n_obs, dtype=bool)
        self.extrap_ratio = np.nan
        self.n_extrap_refused = 0
        if EXTRAPOLATE_BEFORE_TRACK:
            try:
                synth_imager_elon_interp = self._extrapolate_before_track(
                    model, synth_imager_time, synth_imager_elon, synth_imager_elon_interp
                )
            except Exception as err:
                print(f"Warning: elongation extrapolation before the tracked feature failed: {err}")

        flank_data = {
            "time": obs_time,
            "el": synth_imager_elon_interp,
            "r": np.nan * np.ones(self.n_obs),
            "lon": np.nan * np.ones(self.n_obs),
            # True where "el" was extrapolated before the first tracked point
            "extrapolated": np.asarray(self.extrap_mask, dtype=bool),
        }

        self.model_flank: pd.DataFrame = pd.DataFrame(flank_data)

        if WRITE_OBSOP_DIAGNOSTICS and (diag_info is not None):
            try:
                self._write_diagnostics(
                    diag_info, model, synth_imager_class, jmap, djmap, cme_profile,
                    synth_imager_elon_interp, track_t=synth_imager_time, track_e=synth_imager_elon
                )
            except Exception as err:
                print(f"Warning: could not write observation-operator diagnostics: {err}")

    def _extrapolate_before_track(self, model, track_t_day, track_e_deg, elon_interp):
        """
        Extrapolate the modelled elongation to observation times *before* the first tracked point
        of the CME, e.g. when the CME has not yet crossed the model's inner boundary r_min.
        The apex distance is advanced at constant speed from r_min (r = r_min + v (t - t_launch)),
        converted to elongation with the SSE geometry (sse_flank_elongation) for the observer's
        position at that time, and scaled by the ratio of the model to SSE elongation at the first
        tracked point, so the extrapolated profile joins the modelled one continuously.
        Only times where the extrapolated elongation is within EXTRAP_MAX_ABOVE_MODEL_MIN_DEG of the
        model's minimum visible elongation are filled; later gaps (a tracker failure, not a CME
        outside the model domain) stay NaN.
        :param model: Solved SURF / SURF3d model (provides the CME and r_min)
        :param track_t_day: Times of the merged tracked feature (days from model start)
        :param track_e_deg: Elongations of the merged tracked feature (deg)
        :param elon_interp: Elongations interpolated to the observation times (NaN outside the track)
        :return: Copy of elon_interp with the pre-track NaNs filled where possible
        """
        elon_out = np.array(elon_interp, dtype=float, copy=True)
        track_t_day = np.atleast_1d(np.asarray(track_t_day, dtype=float))
        track_e_deg = np.atleast_1d(np.asarray(track_e_deg, dtype=float))
        valid = np.isfinite(track_t_day) & np.isfinite(track_e_deg)
        if not np.any(valid):
            return elon_out

        i0 = int(np.argmin(np.where(valid, track_t_day, np.inf)))
        t0_day, e0_deg = track_t_day[i0], track_e_deg[i0]
        obs_day = np.atleast_1d(self.obs_time_day.value)
        need = (~np.isfinite(elon_out)) & (obs_day < t0_day)
        if not np.any(need):
            return elon_out

        surf_run = model.SURFlat[0] if self.use_model == "compress_surf3d" else model
        if not surf_run.cmes:
            return elon_out
        cme = surf_run.cmes[0]
        r_min_rs = float(surf_run.r[0].to(u.solRad).value)
        v_rs_per_day = float((cme.v * u.day).to(u.solRad).value)
        t_launch_day = float(cme.t_launch.to(u.day).value)
        half_width = float((cme.width / 2.0).to(u.rad).value)
        cme_lon = float(cme.longitude.to(u.rad).value)
        cme_lat = float(cme.latitude.to(u.rad).value)

        obs_jd = np.asarray(self.craft_observer.time.jd, dtype=float)
        init_jd = Time(self.surf_model_init_time).jd

        def sse_elon_deg(t_day):
            t_day = np.atleast_1d(np.asarray(t_day, dtype=float))
            idx = np.array([int(np.argmin(np.abs(obs_jd - (init_jd + t)))) for t in t_day])
            d_obs = self.craft_observer.r[idx].to(u.solRad).value
            dlon = cme_lon - self.craft_observer.lon[idx].to(u.rad).value
            beta = cme_lat - self.craft_observer.lat[idx].to(u.rad).value
            r_apex = r_min_rs + v_rs_per_day * (t_day - t_launch_day)
            eps = np.rad2deg(sse_flank_elongation(np.maximum(r_apex, 1e-6), d_obs, dlon, beta, half_width))
            # Before the CME has left the Sun the elongation is zero
            return np.where(r_apex <= 0.0, 0.0, eps)

        eps0 = float(sse_elon_deg(t0_day)[0])
        ratio = e0_deg / eps0 if (np.isfinite(eps0) and eps0 > 0) else np.nan
        lo, hi = EXTRAP_ANCHOR_RATIO_BOUNDS
        if not (np.isfinite(ratio) and lo <= ratio <= hi):
            print(f"Warning: not extrapolating elongation before the tracked feature "
                  f"(model/SSE ratio at first tracked point = {ratio})")
            return elon_out

        e_extrap = ratio * sse_elon_deg(obs_day[need])

        # Smallest elongation the model domain can show from the observer at each of these times
        idx_obs = np.array([int(np.argmin(np.abs(obs_jd - (init_jd + t)))) for t in obs_day[need]])
        r_obs_rs = self.craft_observer.r[idx_obs].to(u.solRad).value
        elon_min_model = np.rad2deg(np.arcsin(np.clip(r_min_rs / r_obs_rs, 0.0, 1.0)))
        within_limit = e_extrap <= elon_min_model + EXTRAP_MAX_ABOVE_MODEL_MIN_DEG
        self.n_extrap_refused = int(np.sum(np.isfinite(e_extrap) & ~within_limit))

        fill = np.isfinite(e_extrap) & within_limit
        idx_need = np.flatnonzero(need)[fill]
        elon_out[idx_need] = e_extrap[fill]
        self.extrap_mask[idx_need] = True
        self.extrap_ratio = ratio
        return elon_out

    def _write_diagnostics(self, diag_info, model, imager, jmap, djmap, cme_profile, elon_interp,
                           track_t=None, track_e=None):
        """
        Record why this member's observation operator did or did not return an elongation.
        Writes <OUT_BASE_DIR>/obsop_diagnostics/obs_<time>_ens<NNN>.json for every member, and the
        J-map (.npz) as well when any returned elongation is NaN.
        """
        import json

        surf_run = model.SURFlat[0] if self.use_model == "compress_surf3d" else model
        frame_days = surf_run.time_out.to(u.day).value
        elons_deg = imager.e.to(u.deg).value
        obs_days = np.atleast_1d(self.obs_time_day.value)

        # Observer position and the smallest elongation the model domain can show
        r_obs = float(imager.position.r[0].to(u.solRad).value)
        r_min = float(surf_run.r[0].to(u.solRad).value)
        elon_min_model = float(np.rad2deg(np.arcsin(min(1.0, r_min / r_obs))))

        features = []
        for name, feat in (cme_profile or {}).items():
            t = np.asarray(feat["t"], dtype=float)
            e = np.asarray(feat["e"], dtype=float)
            features.append({
                "name": name, "n_points": int(t.size),
                "t_min_day": float(np.nanmin(t)) if t.size else None,
                "t_max_day": float(np.nanmax(t)) if t.size else None,
                "e_min_deg": float(np.nanmin(e)) if e.size else None,
                "e_max_deg": float(np.nanmax(e)) if e.size else None,
            })

        elon_interp = np.atleast_1d(np.asarray(elon_interp, dtype=float))
        # Classify against the merged track actually used for the interpolation
        if track_t is None or track_e is None:
            track_t, track_e = merge_tracked_features(cme_profile)
        track_t = np.asarray(track_t, dtype=float)
        track_e = np.asarray(track_e, dtype=float)
        reasons = []
        for k, t_obs in enumerate(obs_days):
            if np.isfinite(elon_interp[k]) and bool(getattr(self, "extrap_mask", np.zeros(len(obs_days), bool))[k]):
                reasons.append("extrapolated_before_track")
            elif np.isfinite(elon_interp[k]):
                reasons.append("ok")
            elif track_t.size == 0:
                reasons.append("no_feature_tracked")
            elif t_obs < track_t.min():
                # (with the extrapolation limit, this means the CME should be in view but was not
                #  tracked: compare model_elongation=NaN with elon_min_model_deg)
                reasons.append("obs_before_tracked_feature")
            elif t_obs > track_t.max():
                reasons.append("obs_after_tracked_feature")
            else:
                reasons.append("nan_inside_feature")

        cme = surf_run.cmes[0] if surf_run.cmes else None
        record = {
            "ens_no": diag_info.get("ens_no"),
            "obs_time_utc": [str(t) for t in np.atleast_1d(self.obs_time)],
            "obs_time_day": obs_days.tolist(),
            "model_elongation": elon_interp.tolist(),
            "reason": reasons,
            "model_init_time": str(surf_run.time_init.isot),
            "n_frames": int(frame_days.size),
            "frame_spacing_min": float(np.median(np.diff(frame_days)) * 1440) if frame_days.size > 1 else None,
            "cme_t_launch_day": float(cme.t_launch.to(u.day).value) if cme is not None else None,
            "cme_v_kms": float(cme.v.to(u.km / u.s).value) if cme is not None else None,
            "observer_r_rs": r_obs,
            "observer_lon_deg": float(imager.position.lon[0].to(u.deg).value),
            "observer_lat_deg": float(imager.position.lat[0].to(u.deg).value),
            "imager_pa_deg": float(imager.pa.to(u.deg).value) if hasattr(imager, "pa") else None,
            "model_r_min_rs": r_min,
            "elon_min_model_deg": elon_min_model,
            "extrap_ratio": float(getattr(self, "extrap_ratio", np.nan)),
            "n_extrap_refused": int(getattr(self, "n_extrap_refused", 0)),
            "search_band": (
                f"wide: speed x{SEARCH_BAND_SPEED_FACTORS}, lower -{SEARCH_BAND_LOWER_MARGIN_DEG} deg, "
                f"upper +{SEARCH_BAND_UPPER_MARGIN_DEG} deg"
                if isinstance(imager, WideBandSyntheticImager3D) else "surf (speed x(0.75, 1.25))"
            ),
            "jmap_finite_fraction": float(np.isfinite(jmap).mean()),
            "jmap_frames_with_data": int(np.isfinite(jmap).any(axis=0).sum()),
            "features": features,
            "merged_track": {
                "n_points": int(track_t.size),
                "t_min_day": float(track_t.min()) if track_t.size else None,
                "t_max_day": float(track_t.max()) if track_t.size else None,
                "e_min_deg": float(track_e.min()) if track_e.size else None,
                "e_max_deg": float(track_e.max()) if track_e.size else None,
            },
            "imager_elon_range_deg": [float(imager.e.to(u.deg).value.min()), float(imager.e.to(u.deg).value.max())],
            "model_lat_range_deg": (
                [float(model.lat.min().to(u.deg).value), float(model.lat.max().to(u.deg).value)]
                if self.use_model == "compress_surf3d" else None
            ),
        }
        record.update({k: v for k, v in diag_info.items() if k != "ens_no"})

        out_dir = obsop_diagnostics_dir()
        t_tag = str(np.atleast_1d(self.obs_time)[0]).replace(":", "").replace(" ", "T").replace("-", "")[:15]
        stem = os.path.join(out_dir, f"obs_{t_tag}_ens{int(diag_info.get('ens_no') or 0):03d}")
        with open(stem + ".json", "w") as f:
            json.dump(record, f, indent=1, default=str)

        if not np.all(np.isfinite(elon_interp)):
            np.savez_compressed(
                stem + "_jmap.npz", jmap=jmap, djmap=djmap, frame_days=frame_days,
                elons_deg=elons_deg, obs_time_day=obs_days,
                **{f"{f['name']}_t": np.asarray(cme_profile[f['name']]['t'], float) for f in features},
                **{f"{f['name']}_e": np.asarray(cme_profile[f['name']]['e'], float) for f in features},
            )


class ObservationOperator:
    def __init__(
            self,
            use_model: str,
            cme_par_dict: CmeParEns,
            obs_time_in_datetime: list[datetime.datetime] | datetime.datetime,
            obs_cov: npt.NDArray[float] | float,
            craft: str | None = None,
            obs: list[float] | float | None = None,
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
            write_diagnostics: bool = True,
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

        # Observation value(s), if provided (only used for printing). Set for every craft, including
        #  twin mode (it was previously only set for a real craft, so get_cme_flanks raised
        #  AttributeError in twin mode).
        self.obs = obs
        self.write_diagnostics: bool = write_diagnostics

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
                    # Must match the member imagers (IMAGER_ELON_MIN/MAX): the SURF3d latitude range is
                    #  taken from this field of view
                    elon_min=IMAGER_ELON_MIN,
                    elon_max=IMAGER_ELON_MAX
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
                el_min=IMAGER_ELON_MIN,
                el_max=IMAGER_ELON_MAX,
                # (no diagnostics for e.g. full elongation profiles, where NaNs before the CME
                #  appears and after it leaves the field of view are expected)
                diag_info={
                    "ens_no": ens_no,
                    "cme_lon_deg": float(cme_member.longitude.to(u.deg).value),
                    "cme_lat_deg": float(cme_member.latitude.to(u.deg).value),
                    "cme_width_deg": float(cme_member.width.to(u.deg).value),
                } if self.write_diagnostics else None,
            )
            cme_flank: pd.DataFrame = obs_synth_HI_class.model_flank

            if self.write_diagnostics and np.isnan(cme_flank['el'].values).any():
                print("=====================================================")
                print(f"psi_ert_avg = {self.psi_ert_avg}")
                print(f"Observation operator failed: ens_no = {ens_no}")
                print(f"obs_time = {self.obs_time_in_datetime}")
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

        # An obs_lon array holds one longitude per *observation*, not per ensemble member, so it must not
        #  be indexed with the ensemble number (previously self.obs_lon[i] paired member i with
        #  observer longitude i, and raised IndexError when n_members > len(obs_lon)). The flank
        #  calculation uses a single observer longitude, so collapse an array of identical values to one.
        if isinstance(self.obs_lon, np.ndarray) and np.ndim(self.obs_lon) > 0:
            obs_lon_vals = np.ravel(self.obs_lon.to(u.deg).value)
            if np.allclose(obs_lon_vals, obs_lon_vals[0]):
                self.obs_lon = float(obs_lon_vals[0]) * u.deg
            else:
                raise NotImplementedError(
                    "get_cme_flanks: observer longitude varies between observation times "
                    f"({obs_lon_vals}); the flank calculation supports a single observer longitude. "
                    "Call the observation operator once per observation time instead."
                )

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

            # Use no more workers than there are ensemble members (extra workers would sit idle),
            #  and leave one core free for the main process.
            n_proc = max(1, min(int((os.cpu_count() or 2) - 1), int(self.n_members)))

            cme_flanks: list[DataFrame] | list[None] = [None] * self.n_members

            # Reuse the run-wide worker pool rather than spawning a new one for every call
            pool = get_worker_pool(n_proc)
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

            # (The pool is no longer closed/joined here; .get() below waits for each member's result.
            #  The pool stays alive for the next call and is shut down by shutdown_worker_pool().)
            for i, _ in enumerate(cme_objects):
                cme_flanks[i] = obj_res[i].get()

            if self.obs is not None:
                print(f"Obs = {self.obs}")

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
        # Straight line through (x1, bias_term_5rs) and (x2, bias_term_21rs) with x1 = 5, x2 = 21.
        #  Previously m = (b5 - b21)/15 and c = (21*b5 - 5*b21)/15: the slope had the wrong sign and
        #  both used 15 instead of x2 - x1 = 16, so the line passed through neither calibration point
        #  (e.g. b5 = b21 = 2.5 gave a constant bias of 2.67 instead of 2.5).
        x1: float = 5.0
        x2: float = 21.0
        m: float = (self.bias_term_21rs - self.bias_term_5rs) / (x2 - x1)
        c: float = self.bias_term_5rs - (m * x1)

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

        # Observation-bias term: h' = h + beta * p(h) for each member (see obs_bias.py). beta is 0 unless
        #  the bias is estimated or fixed (and, in twin experiments, for the synthetic truth it is the
        #  injected bias), so this leaves the elongations unchanged by default.
        beta = np.asarray(self.cme_par_dict.get("bias", 0.0), dtype=float)
        if np.any(beta != 0):
            obs_op = apply_bias(obs_op, np.broadcast_to(np.atleast_1d(beta), (self.n_members,)))

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