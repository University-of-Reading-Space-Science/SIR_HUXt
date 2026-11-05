"""
CME arrival at a body (e.g. Earth) from a solved SURF / SURF3d / HUXt model.

Used by osse_ana.py and by the observation operator (sir_observation_operator.py), whose model runs
for the elongation profiles also give each member's arrival, so the two need not be run separately.
"""

from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np

import surf.surf as S


def get_arrival_cme(model, use_model, cme_hit_object="EARTH"):
    """
    Return the solved ConeCME used for arrival statistics.
    For compress_surf3d the model is a stack of 2D SURF runs, one per latitude. The CME must be
    taken from the latitude slice nearest the target body's latitude; SURFlat[0] is the most
    southerly slice (e.g. -7.7 deg), not the one containing Earth.
    :param model: Solved SURF/SURF3d/HUXt model
    :param use_model: Model name
    :param cme_hit_object: Body the CME arrival is computed for
    :return: ConeCME object of the relevant (latitude slice of the) model
    """
    if use_model in ["compress_surf3d"]:
        times = model.SURFlat[0].time_init + model.SURFlat[0].time_out
        body = S.Observer(cme_hit_object.upper(), times)
        body_lat = np.nanmean(body.lat.to(u.deg).value)
        slice_lats = np.array([sl.latitude.to(u.deg).value for sl in model.SURFlat])
        i_lat = int(np.argmin(np.abs(slice_lats - body_lat)))
        return model.SURFlat[i_lat].cmes[0]
    return model.cmes[0]


def get_arrival_speed_from_front(cme, stats, n_side=2):
    """
    Estimate the CME front speed at arrival as dr/dt of the front along the hit longitude.
    Needed for the compressible SURF solver, which tracks the CME front radius but never fills
    the tracer-particle speeds (cme_particles_v stays NaN), so compute_arrival_at_body returns
    v = NaN for every member.
    :param cme: Solved ConeCME
    :param stats: Output of cme.compute_arrival_at_body
    :param n_side: Number of output time steps either side of the arrival step to fit
    :return: Arrival speed in km/s (float), NaN if it cannot be estimated
    """
    if not stats["hit"]:
        return np.nan
    hit_id = int(stats["hit_id"])
    hit_lon = stats["lon"].to(u.rad).value
    t_list, r_list = [], []
    for i in range(hit_id - n_side, hit_id + n_side + 1):
        coord = cme.coords.get(i)
        if (coord is None) or (len(coord["r"]) == 0):
            continue
        front = np.asarray(coord["front_id"].value) == 1.0
        if np.sum(front) < 2:
            continue
        # longitudes relative to the CME nose, in (-pi, pi], so the front is monotonic in lon
        lon_rel = np.angle(
            np.exp(
                1j
                * (coord["lon"].to(u.rad).value[front] - cme.longitude.to(u.rad).value)
            )
        )
        target = np.angle(np.exp(1j * (hit_lon - cme.longitude.to(u.rad).value)))
        order = np.argsort(lon_rel)
        r_at = np.interp(
            target,
            lon_rel[order],
            coord["r"].to(u.km).value[front][order],
            left=np.nan,
            right=np.nan,
        )
        if np.isfinite(r_at):
            t_list.append(coord["time"].jd * 86400.0)
            r_list.append(r_at)
    if len(t_list) < 2:
        return np.nan
    slope = np.polyfit(np.array(t_list) - t_list[0], np.array(r_list), 1)[0]
    return float(slope)


def get_arrival_time_and_speed(model, use_model, cme_hit_object):
    """
    Solve-independent helper: compute arrival stats for the relevant CME, with a speed fallback
    :return: (arrival datetime or NaN, arrival speed km/s or NaN, hit count 0/1, stats dict)
    """
    cme = get_arrival_cme(model, use_model, cme_hit_object)
    stats = cme.compute_arrival_at_body(cme_hit_object)
    if stats["hit"]:
        arr_time = stats["t_arrive"].datetime
        arr_speed = (stats["v"].to(u.km / u.s)).value
        if not np.isfinite(arr_speed):
            arr_speed = get_arrival_speed_from_front(cme, stats)
        return arr_time, arr_speed, 1, stats
    return np.nan, np.nan, 0, stats


def body_in_model_grid(model, use_model: str, cme_hit_object: str = "EARTH") -> bool:
    """
    Whether the body's latitude is inside the model's latitude grid. Always True except for
    compress_surf3d, whose latitude slices may not reach the body (e.g. a grid set from an imager's
    field of view); the nearest slice would then misplace the arrival.
    """
    if use_model not in ["compress_surf3d"]:
        return True
    times = model.SURFlat[0].time_init + model.SURFlat[0].time_out
    body = S.Observer(cme_hit_object.upper(), times)
    body_lat = float(np.nanmean(body.lat.to(u.deg).value))
    slice_lats = np.sort([float(sl.latitude.to(u.deg).value) for sl in model.SURFlat])
    half_step = (
        0.5 * float(np.median(np.diff(slice_lats))) if len(slice_lats) > 1 else 0.0
    )
    return bool(slice_lats[0] - half_step <= body_lat <= slice_lats[-1] + half_step)


def member_arrival(
    model, use_model: str, cme_hit_object: str = "EARTH"
) -> dict[str, Any]:
    """
    Arrival of the CME of a solved one-CME model at cme_hit_object
    :return: {"time": arrival datetime or None, "speed": km/s or NaN, "hit": bool,
        "available": False if the body is outside the model's latitude grid (no arrival computed)}
    """
    if not body_in_model_grid(model, use_model, cme_hit_object):
        return {"time": None, "speed": np.nan, "hit": False, "available": False}
    arr_time, arr_speed, hit, _ = get_arrival_time_and_speed(
        model, use_model, cme_hit_object
    )
    return {
        "time": arr_time if hit else None,
        "speed": float(arr_speed) if hit else np.nan,
        "hit": bool(hit),
        "available": True,
    }
