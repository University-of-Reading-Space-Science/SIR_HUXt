"""
Module to make a CME parameter dictionary in the correct format
"""
import datetime
import sys
from astropy.units import Quantity

#sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
from init_sir import initialise_cme_parameter_ensemble_dict
from cme_par_ens import CmeParEns
from cme_par_dict_structure import required_dict_keys

class CmeParDict:
    def __init__(
            self,
            n_members: int,
            huxt_init_time: datetime.datetime,
            t_init: list[datetime.datetime],
            v: list[Quantity],
            width: list[Quantity],
            lon: list[Quantity],
            lat: list[Quantity],
            thick: list[Quantity]
    ):
        self.n_members = n_members
        self.huxt_init_time = huxt_init_time
        self.t_init = t_init
        self.v = v
        self.width = width
        self.lon = lon
        self.lat = lat
        self.thick = thick


    def make_par_dict(self) -> CmeParEns:
        # Define a parameter dictionary
        par_dict = initialise_cme_parameter_ensemble_dict(
            n_ensemble=self.n_members, huxt_init_time=self.huxt_init_time
        )
        par_dict['t_init'] = self.t_init
        par_dict['v'] = self.v
        par_dict['width'] = self.width
        par_dict['lon'] = self.lon
        par_dict['lat'] = self.lat
        par_dict['thick'] = self.thick

        return par_dict