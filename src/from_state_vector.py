"""
A module containing the class FromStateVector that converts from the state vector to the cme_par_array
 to update the cme_par_dict with the results of the data assimilation.
"""

import numpy as np
import numpy.typing as npt
import datetime

import astropy.units as u
from astropy.units import Quantity

from cme_par_ens import CmeParEns
from cme_par_dict_structure import (
    required_dict_keys,
    cme_par_get_indices,
    state_par_keys,
)


class FromStateVector:
    def __init__(
        self,
        cme_par_array: npt.NDArray[np.float64],
        cme_par_dict: CmeParEns,
        state_ens: npt.NDArray[np.float64],
        pars_in_state_vector: list[str] | None = None,
    ) -> None:
        self.cme_par_array: npt.NDArray[np.float64] = cme_par_array
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

        self.n_pars: int = len(self.pars_in_state_vector)

        self.n_ensemble: int = np.shape(cme_par_array)[0]

        assert np.shape(state_ens) == (self.n_ensemble, self.n_pars)
        self.state_ens: npt.NDArray[np.float64] = state_ens

    def state_vector_to_cme_par_array(self) -> npt.NDArray[np.float64]:
        """
        Function to input the contents of a state_vector after DA into a cme_par_array
        :param state_ens Ensemble of state vectors
        :param par_req List containing the names of all the required parameters from cme_par_dict
        :return cme_par_array Array that contains all CME parameters required
        """
        assert all(p in state_par_keys() for p in self.pars_in_state_vector)

        for i, ip in enumerate(self.pars_in_state_vector):
            self.cme_par_array[:, cme_par_get_indices(ip)] = self.state_ens[:, i]

        return self.cme_par_array

    def convert_t_init_array_to_dict(self) -> list[datetime.datetime]:
        """
        Convert t_init from seconds since surf_init_time to datetimes for cme_par_dict
        (column layout of cme_par_array: see cme_par_array_to_cme_par_dict)
        :return: t_init_dict: List of launch datetimes
        """
        assert all(
            p in self.cme_par_dict.keys() for p in ["surf_init_time", "n_members"]
        )

        n_ens: int = np.shape(self.cme_par_array)[0]

        # Get the required cme_par_array index for t_init
        ind_req: int = cme_par_get_indices("t_init")

        # Calculate number of seconds that need to be added to surf_init_time
        t_init_dict: list[datetime.datetime] = [
            self.cme_par_dict["surf_init_time"]
            + datetime.timedelta(seconds=self.cme_par_array[i, ind_req])
            for i in range(n_ens)
        ]

        return t_init_dict

    def convert_v_array_to_dict(self) -> list[Quantity[u.km / u.s]]:
        """
        Convert the CME speeds (km/s) to a Quantity list for cme_par_dict
        (column layout of cme_par_array: see cme_par_array_to_cme_par_dict)
        :return: v_dict: CME speeds (km/s)
        """
        ind_req: int = cme_par_get_indices("v")

        v_dict: list[Quantity[u.km / u.s]] = (
            list(self.cme_par_array[:, ind_req]) * u.km / u.s
        )

        return v_dict

    def convert_width_array_to_dict(self) -> list[Quantity[u.deg]]:
        """
        Convert the CME widths (deg) to a Quantity list for cme_par_dict
        (column layout of cme_par_array: see cme_par_array_to_cme_par_dict)
        :return: width_dict: CME widths (deg)
        """
        ind_req: int = cme_par_get_indices("width")

        width_dict: list[Quantity[u.deg]] = list(self.cme_par_array[:, ind_req]) * u.deg

        return width_dict

    def convert_lon_array_to_dict(self) -> list[Quantity[u.deg]]:
        """
        Convert the CME longitudes from -180-180 to 0-360 deg as a Quantity list for cme_par_dict
        (column layout of cme_par_array: see cme_par_array_to_cme_par_dict)
        :return: lon_dict: CME longitudes (deg, 0-360)
        """
        ind_req: int = cme_par_get_indices("lon")

        # Copy, so that the 0-360 conversion doesn't modify self.cme_par_array (kept in +/-180) in place
        lon_arr: npt.NDArray[np.float64] = np.copy(self.cme_par_array[:, ind_req])

        lon_cond: npt.NDArray[np.bool_] = lon_arr < 0
        lon_arr[lon_cond]: npt.NDArray[np.float64] = lon_arr[lon_cond] + 360

        lon_dict: list[Quantity[u.deg]] = list(lon_arr) * u.deg

        return lon_dict

    def convert_lat_array_to_dict(self) -> list[Quantity[u.deg]]:
        """
        Convert the CME latitudes (deg) to a Quantity list for cme_par_dict
        (column layout of cme_par_array: see cme_par_array_to_cme_par_dict)
        :return: lat_dict: CME latitudes (deg)
        """
        ind_req: int = cme_par_get_indices("lat")

        lat_dict: list[Quantity[u.deg]] = list(self.cme_par_array[:, ind_req]) * u.deg

        return lat_dict

    def convert_thick_array_to_dict(self) -> list[Quantity[u.solRad]]:
        """
        Convert the CME thicknesses (solar radii) to a Quantity list for cme_par_dict
        (column layout of cme_par_array: see cme_par_array_to_cme_par_dict)
        :return: thick_dict: CME thicknesses (solar radii)
        """
        ind_req: int = cme_par_get_indices("thick")

        thick_dict: list[Quantity[u.solRad]] = (
            list(self.cme_par_array[:, ind_req]) * u.solRad
        )

        return thick_dict

    def convert_bias_array_to_dict(self) -> list[float]:
        """
        Function to take the observation-bias coefficients (deg) from cme_par_array for cme_par_dict
        :return: bias_dict: List of bias coefficients (0 if cme_par_array has no bias column)
        """
        ind_req: int = cme_par_get_indices("bias")
        if self.cme_par_array.shape[1] <= ind_req:
            return [0.0] * self.cme_par_array.shape[0]
        return [float(b) for b in self.cme_par_array[:, ind_req]]

    def cme_par_array_to_cme_par_dict(self) -> CmeParEns:
        """
        Extract the relevant cme parameters from the cme parameter array, add the units back
            and place them into the CME parameter dictionary
        :param cme_par_array: Array that contains all CME parameters required,
            an (nEns, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
                cme_par_array[:, 6] = cme_par_dict["bias"] (observation-bias coefficient, deg)
        :param cme_par_dict: Dictionary that contains all CME parameters required
        :return: cme_par_dict: Dictionary of CME parameters (with units added back in)
        """
        req_keys: set[str] = required_dict_keys() | {"surf_init_time", "n_members"}
        assert all(p in self.cme_par_dict.keys() for p in req_keys)

        # Add the units back
        self.cme_par_dict["t_init"]: list[
            datetime.datetime
        ] = self.convert_t_init_array_to_dict()
        self.cme_par_dict["v"]: list[
            Quantity[u.km / u.s]
        ] = self.convert_v_array_to_dict()
        self.cme_par_dict["width"]: list[
            Quantity[u.deg]
        ] = self.convert_width_array_to_dict()
        self.cme_par_dict["lon"]: list[
            Quantity[u.deg]
        ] = self.convert_lon_array_to_dict()
        self.cme_par_dict["lat"]: list[
            Quantity[u.deg]
        ] = self.convert_lat_array_to_dict()
        self.cme_par_dict["thick"]: list[
            Quantity[u.solRad]
        ] = self.convert_thick_array_to_dict()
        self.cme_par_dict["bias"]: list[float] = self.convert_bias_array_to_dict()

        return self.cme_par_dict
