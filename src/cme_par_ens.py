"""
cme_par_ens.py

Type definitions used for type hints throughout the SIR/Aux-PF code.

``CmeParEns`` describes the "CME parameter ensemble dictionary" (``cme_par_dict``) that is passed
between the data-assimilation modules. Each CME parameter key holds either one value per ensemble
member (a list / Quantity array) or a single value (e.g. for the "truth" CME in an OSSE).

Layout used throughout the code:
    surf_init_time : datetime the SURF/HUXt simulation starts at
    t_init         : CME launch time(s) at ``cme_init_rad``
    v, width, lon, lat, thick : cone-CME speed, angular width, longitude, latitude and thickness
    weight / log_weight       : particle weights (linear and logarithmic)
    n_members      : number of ensemble members (particles)
"""
import datetime

from typing import TypedDict

import astropy.units as u
from astropy.units import Quantity


class CmeParEns(TypedDict):
    # NB: the annotations below *are* the definition of the dictionary keys (TypedDict)
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
