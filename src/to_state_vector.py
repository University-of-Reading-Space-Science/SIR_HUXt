"""
A module containing the class ToStateVector that converts from the cme_par_dictionary to the
cme_par_array and then onto a state vector to be input into the data assimilation
"""

import numpy as np
import numpy.typing as npt

import astropy.units as u
from astropy.units import Quantity

from cme_par_ens import CmeParEns
from cme_par_dict_structure import (
    required_dict_keys,
    cme_par_get_keys,
    cme_par_key_to_index,
    state_par_keys,
)


class ToStateVector:
    def __init__(
        self,
        cme_par_dict: CmeParEns,
        cme_par_array: npt.NDArray[np.float64] | None = None,
        pars_in_state_vector: list[str] | None = None,
    ) -> None:

        self.cme_par_dict: CmeParEns = cme_par_dict

        if pars_in_state_vector is None:
            self.pars_in_state_vector: list[str] = [
                "t_init",
                "v",
                "width",
                "lon",
                "lat",
                "thick",
            ]
        else:
            self.pars_in_state_vector: list[str] = pars_in_state_vector

        self.n_ensemble: int = self.cme_par_dict["n_members"]
        self.n_pars: int = len(self.pars_in_state_vector)

        if cme_par_array is None:
            self.cme_par_array: npt.NDArray[np.float64] = np.zeros(
                (self.n_ensemble, len(cme_par_key_to_index()))
            )
        else:
            self.cme_par_array: npt.NDArray[np.float64] = cme_par_array

        self.state_ens: npt.NDArray[np.float64] = np.zeros(
            (self.n_ensemble, self.n_pars)
        )

    def convert_t_init_dict_to_array(self) -> npt.NDArray[np.float64]:
        """
        Function to convert datetime t_init to seconds since surf_init_time
        :param cme_par_dict: Dictionary that contains all CME parameters required
            (must contain "surf_init_time" and "t_init")
        :return: t_init_arr: Array of t_init values in seconds since surf_init_time
        """
        assert all(
            p in self.cme_par_dict.keys()
            for p in ["surf_init_time", "t_init", "n_members"]
        )

        t_init_arr: npt.NDArray[np.float64] = np.array(
            [
                (
                    self.cme_par_dict["t_init"][i] - self.cme_par_dict["surf_init_time"]
                ).total_seconds()
                for i in range(self.n_ensemble)
            ]
        )

        return t_init_arr

    def convert_v_dict_to_array(self) -> npt.NDArray[np.float64]:
        """
        Function to convert cme_speed (v) to km/s and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "v")
        :return: v_arr: Array of v values in km/s
        """
        assert all(p in self.cme_par_dict.keys() for p in ["v"])

        v_arr: npt.NDArray[np.float64] = np.array(
            [v.to(u.km / u.s).value for v in self.cme_par_dict["v"][:]]
        )

        return v_arr

    def convert_width_dict_to_array(self) -> npt.NDArray[np.float64]:
        """
        Function to convert width to deg and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "width")
        :return: width_arr: Array of width values in deg
        """
        assert all(p in self.cme_par_dict.keys() for p in ["width"])

        width_arr: npt.NDArray[np.float64] = np.array(
            [w.to(u.deg).value for w in self.cme_par_dict["width"][:]]
        )

        return width_arr

    def convert_lon_dict_to_array(self) -> npt.NDArray[np.float64]:
        """
        Function to convert lon from 0 to 360 degrees into -180 to 180 degrees
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "lon")
        :return: lon_arr: Array of longitude values converted to -180 to 180 degrees
        """
        assert "lon" in self.cme_par_dict.keys()

        lon_arr: npt.NDArray[np.float64] = np.array(
            [lon.to(u.deg).value for lon in self.cme_par_dict["lon"][:]]
        )
        lon_arr[lon_arr > 180] -= 360

        return lon_arr

    def convert_lat_dict_to_array(self) -> npt.NDArray[np.float64]:
        """
        Function to convert lat to deg and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "lat")
        :return: lat_arr: Array of lat values in deg
        """
        assert all(p in self.cme_par_dict.keys() for p in ["lat"])

        lat_arr: npt.NDArray[np.float64] = np.array(
            [lat.to(u.deg).value for lat in self.cme_par_dict["lat"][:]]
        )

        return lat_arr

    def convert_thick_dict_to_array(self) -> npt.NDArray[np.float64]:
        """
        Function to convert thickness to solar radii and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "thick")
        :return: thick_arr: Array of thickness values in solar radii
        """
        assert all(p in self.cme_par_dict.keys() for p in ["thick"])

        thick_arr: npt.NDArray[np.float64] = np.array(
            [thick.to(u.solRad).value for thick in self.cme_par_dict["thick"][:]]
        )

        return thick_arr

    def convert_bias_dict_to_array(self) -> npt.NDArray[np.float64]:
        """
        Function to put the observation-bias coefficients (deg, no units) in an array. Optional key:
         a dictionary without "bias" gives zeros, and a single value is used for every member.
        :return: bias_arr: Array of bias coefficients
        """
        if "bias" not in self.cme_par_dict:
            return np.zeros(self.n_ensemble)
        bias: npt.NDArray[np.float64] = np.atleast_1d(
            np.asarray(self.cme_par_dict["bias"], dtype=float)
        )
        return (
            np.broadcast_to(bias, (self.n_ensemble,)).copy() if bias.size == 1 else bias
        )

    def cme_par_dict_to_cme_par_array(self) -> npt.NDArray[np.float64]:
        """
        Extract the relevant cme parameters from the cme parameter dictionary, remove the units
            and place them into an array for use in the DA
        :param cme_par_dict: Dictionary that contains all CME parameters required
        :return: cme_par_array: Array of CME parameters (with no units) in
            an (nEns, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
                cme_par_array[:, 6] = cme_par_dict["bias"] (observation-bias coefficient, deg; 0 if absent)
        """
        req_keys: set[str] = required_dict_keys() | {"surf_init_time", "n_members"}
        assert all(p in self.cme_par_dict.keys() for p in req_keys)

        # Each column is filled by convert_<key>_dict_to_array, which standardises the units and
        #  removes them
        for ind in range(len(cme_par_key_to_index())):
            self.cme_par_array[:, ind] = getattr(
                self, f"convert_{cme_par_get_keys(ind)}_dict_to_array"
            )()

        return self.cme_par_array

    def cme_par_dict_to_state_vector(self) -> npt.NDArray[np.float64]:
        """
        Function to convert a set of parameter keys to a single array
        :return: state_ens: (self.n_ensemble, self.n_pars) array containing an
          ensemble of parameter values specified by self.pars_in_state_vector

        """
        assert all(p in state_par_keys() for p in self.pars_in_state_vector)

        for i, ip in enumerate(self.pars_in_state_vector):
            self.state_ens[:, i] = getattr(self, f"convert_{ip}_dict_to_array")()

        return self.state_ens
