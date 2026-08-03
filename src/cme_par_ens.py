"""
cme_par_ens.py
A module that defines the classes required for non-standard type-hints
"""
import datetime

from mypy.build import TypedDict

import astropy.units as u
from astropy.units import Quantity


class CmeParEns(TypedDict):
    surf_init_time: datetime.datetime
    t_init: list[datetime.datetime] | datetime.datetime
    v: list[Quantity[u.km / u.s]] | Quantity[u.km / u.s]
    width: list[Quantity[u.deg]] | Quantity[u.deg]
    lon: list[Quantity[u.deg]] | Quantity[u.deg]
    lat: list[Quantity[u.deg]] | Quantity[u.deg]
    thick: list[Quantity[u.solRad]] | Quantity[u.solRad]
    v_transit: list[float] | float
    v_hit: list[float] | float
    likelihood: list[float] | float
    weight: list[float]
    log_weight: list[float]
    n_members: int