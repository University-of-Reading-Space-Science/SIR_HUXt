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
from cme_par_dict_structure import required_dict_keys, cme_par_get_indices

class FromStateVector:
    def __init__(
            self,
            cme_par_array: npt.NDArray[float],
            cme_par_dict: CmeParEns,
            state_ens: npt.NDArray[float],
            pars_in_state_vector=None,
            weights=None,
            x_mean=None,
            x_std=None
    ) -> None:
        self.cme_par_array: npt.NDArray[float] = cme_par_array
        self.cme_par_dict: CmeParEns = cme_par_dict

        if pars_in_state_vector is None:
            self.pars_in_state_vector:list[str] = ["t_init", "v", "width", "lon", "lat", "thick"]
        else:
            self.pars_in_state_vector: list[str] = pars_in_state_vector

        self.n_pars = len(self.pars_in_state_vector)

        self.n_ensemble = np.shape(cme_par_array)[0]

        if weights is None:
            self.weights = [-np.log(self.n_ensemble) for _ in range(self.n_ensemble)]
        else:
            self.weights = weights

        #print(f"np.shape(state_ens) = {np.shape(state_ens)}")
        #print(f"(self.n_ensemble, self.n_pars) = {(self.n_ensemble, self.n_pars)}")

        assert np.shape(state_ens) == (self.n_ensemble, self.n_pars)
        self.state_ens: npt.NDArray[float] = state_ens

        self.zscore_bool = (
            (x_mean is not None)
            and (x_std is not None)
        )

        if self.zscore_bool:
            self.x_mean = x_mean
            self.x_std = x_std


    def state_vector_to_cme_par_array(self) -> npt.NDArray[float]:
        """
        Function to input the contents of a state_vector after DA into a cme_par_array
        :param state_ens Ensemble of state vectors
        :param par_req List containing the names of all the required parameters from cme_par_dict
        :return cme_par_array Array that contains all CME parameters required
        """
        assert all(p in required_dict_keys() for p in self.pars_in_state_vector)

        #f"state_ens={self.state_ens}")
        for i, ip in enumerate(self.pars_in_state_vector):
            if ip in required_dict_keys():
                cme_par_ind_req = cme_par_get_indices(ip)
                #print(f"i={i}, ip={ip}: cme_par_ind_req={cme_par_ind_req}")
                self.cme_par_array[:, cme_par_ind_req] = self.state_ens[:, i]

        return self.cme_par_array


    def z_state_vector_to_cme_par_array(self) -> npt.NDArray[float]:
        """
        Function to convert the state vector back into the required CME parameters
        :param z_ens (nEns, nPar)-array containing the state ensemble converted to z-scores
        :param x_mean (nPar)-array containing the mean of the ensemble parameters
        :param x_std (nPar)-array containing the standard deviation of the ensemble parameters
        :param weights (nPar)-array containing the weights of the ensemble parameters
        :param par_req List containing the names of all the required parameters from cme_par_dict
        :param cme_par_dict Array that contains all CME parameters required
        :return cme_par_dict Updated parameter dictionary
        """
        # Check if all z_score values have been loaded into the class, if not raise an exception
        if not self.zscore_bool:
            raise Exception(
                f"Missing zscore values input into FromStateVector."
                f" z_ens={self.state_ens}, x_mean={self.x_mean}, x_std={self.x_std}"
            )
        assert all(p in required_dict_keys() for p in self.pars_in_state_vector)

        # Extract all required parameters, put them in an ensemble matrix, then calculate the zscores for output
        for i, ip in enumerate(self.pars_in_state_vector):
            if ip in required_dict_keys():
                cme_par_ind_req = cme_par_get_indices(ip)
                self.cme_par_array[:, cme_par_ind_req] = sir.inv_zscore(
                    self.state_ens[:, i], self.x_mean[i], self.x_std[i]
                )

        return self.cme_par_array


    def convert_t_init_array_to_dict(self) -> list[datetime.datetime]:
        """
        Function to convert t_init from seconds since surf_init_time to datetime for insertion back into a dictionary
        :param cme_par_dict: Dictionary containing CME parameters with following keys:
            ['t_init', 'v', 'width', 'lon', 'lat', 'thick', 'weight', 'surf_init_time', 'n_members']
        :param cme_par_array: Array that contains all CME parameters required,
            an (n_ens, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
        :return: t_init_arr: Array of t_init values in seconds since surf_init_time
        """
        assert all(p in self.cme_par_dict.keys() for p in ["surf_init_time", "n_members"])

        n_ens: int = np.shape(self.cme_par_array)[0]

        # Get the required cme_par_array index for t_init
        ind_req = cme_par_get_indices("t_init")

        # Calculate number of seconds that need to be added to surf_init_time
        t_init_dict: list[datetime.datetime] = [
            self.cme_par_dict["surf_init_time"] + datetime.timedelta(seconds=self.cme_par_array[i, ind_req])
            for i in range(n_ens)
        ]

        return t_init_dict

    def convert_v_array_to_dict(self) -> list[Quantity[u.km / u.s]]:
        """
        Function to convert cme_speed (v) from an (nEns, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
        to a list of velocities with units km/s for placing into cme_par_dict
        :param cme_par_array: An (nEns, nPar) array that contains all CME parameters required
        :return: v_dict: Array of v values in km/s (with units)
        """
        # Get the required cme_par_array index for v
        ind_req = cme_par_get_indices("v")

        v_dict: list[Quantity[u.km / u.s]] = list(self.cme_par_array[:, ind_req]) * u.km / u.s

        return v_dict

    def convert_width_array_to_dict(self) -> list[Quantity[u.deg]]:
        """
        Function to convert width from an (nEns, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
        to a list of widths with units degrees for placing into cme_par_dict
        :param cme_par_array: An (nEns, nPar) array that contains all CME parameters required
        :return: width_dict: Array of width values in deg
        """
        # Get the required cme_par_array index for t_init
        ind_req = cme_par_get_indices("width")

        width_dict: list[Quantity[u.deg]] = list(self.cme_par_array[:, ind_req]) * u.deg

        return width_dict

    def convert_lon_array_to_dict(self) -> list[Quantity[u.deg]]:
        """
        Function to convert lon from an (nEns, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
        to a list of longitudes between 0-360 with units degrees for placing into cme_par_dict
        :param cme_par_array: Array that contains all CME parameters required,
        :return: lon_dict: Array of longitude values converted to 0-360 degrees
        """
        # Get the required cme_par_array index for t_init
        ind_req = cme_par_get_indices("lon")

        lon_arr: npt.NDArray[Quantity[None]] = self.cme_par_array[:, ind_req]

        lon_cond: list[bool] = list(lon_arr < 0)
        lon_arr[lon_cond]: npt.NDArray[Quantity[None]] = lon_arr[lon_cond] + 360

        lon_dict: list[Quantity[u.deg]] = list(lon_arr) * u.deg

        return lon_dict

    def convert_lat_array_to_dict(self) -> list[Quantity[u.deg]]:
        """
        Function to convert lat from an (nEns, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
        to a list of latitudes with units degrees for placing into cme_par_dict
        :param cme_par_array: Array that contains all CME parameters required
        :return: lat_dict: Array of lat values in deg
        """
        # Get the required cme_par_array index for t_init
        ind_req = cme_par_get_indices("lat")

        lat_dict: list[Quantity[u.deg]] = list(self.cme_par_array[:, ind_req]) * u.deg

        return lat_dict

    def convert_thick_array_to_dict(self) -> list[Quantity[u.solRad]]:
        """
        Function to convert thickness to solar radii and place in array,
            an (nEns, nPar) array with the following entries:
                cme_par_array[:, 0] = cme_par_dict["t_init"] in seconds from cme_par_dict["surf_init_time"]
                cme_par_array[:, 1] = cme_par_dict["v"] in km/s
                cme_par_array[:, 2] = cme_par_dict["width"] in deg
                cme_par_array[:, 3] = cme_par_dict["lon"] in deg (between +/- 180)
                cme_par_array[:, 4] = cme_par_dict["lat"] in deg
                cme_par_array[:, 5] = cme_par_dict["thick"] in solar radii
        :param cme_par_array: Dictionary that contains all CME parameters required
        :return: thick_dict: Array of thickness values in solar radii
        """
        # Get the required cme_par_array index for t_init
        ind_req = cme_par_get_indices("thick")

        thick_dict: list[Quantity[u.solRad]] = list(self.cme_par_array[:, ind_req]) * u.solRad

        return thick_dict

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
        :param cme_par_dict: Dictionary that contains all CME parameters required
        :return: cme_par_dict: Dictionary of CME parameters (with units added back in)
        """
        # Get the required keys and add the 'surf_init_time' and 'n_members' keys as requirements for this function
        req_keys = required_dict_keys()
        additional_keys_req = ["surf_init_time", "n_members"]
        req_keys.update(additional_keys_req)

        assert all(p in self.cme_par_dict.keys() for p in req_keys)

        # Standardise the units and then remove the astropy units
        self.cme_par_dict["t_init"]: list[datetime.datetime] = self.convert_t_init_array_to_dict()
        self.cme_par_dict["v"]: list[Quantity[u.km / u.s]] = self.convert_v_array_to_dict()
        self.cme_par_dict["width"]: list[Quantity[u.deg]] = self.convert_width_array_to_dict()
        self.cme_par_dict["lon"]: list[Quantity[u.deg]] = self.convert_lon_array_to_dict()
        self.cme_par_dict["lat"]: list[Quantity[u.deg]] = self.convert_lat_array_to_dict()
        self.cme_par_dict["thick"]: list[Quantity[u.solRad]] = self.convert_thick_array_to_dict()

        return self.cme_par_dict

