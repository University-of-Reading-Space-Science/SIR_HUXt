"""
A module containing the class ToStateVector that converts from the cme_par_dictionary to the
cme_par_array and then onto a state vector to be input into the data assimilation
"""
import numpy as np
import numpy.typing as npt

import astropy.units as u
from astropy.units import Quantity

from cme_par_ens import CmeParEns
from cme_par_dict_structure import required_dict_keys, cme_par_get_keys

class ToStateVector:
    def __init__(
        self,
        cme_par_dict,
        cme_par_array=None,
        pars_in_state_vector=None
    ) -> None:

        self.cme_par_dict: CmeParEns = cme_par_dict

        if pars_in_state_vector is None:
            self.pars_in_state_vector:list[str] = ["t_init", "v", "width", "lon", "lat", "thick"]
        else:
            self.pars_in_state_vector: list[str] = pars_in_state_vector

        self.n_ensemble: int = self.cme_par_dict["n_members"]
        #print(self.pars_in_state_vector)
        self.n_pars: int = len(self.pars_in_state_vector)

        if cme_par_array is None:
            self.cme_par_array: npt.NDArray[float] = np.zeros((self.n_ensemble, 6))
        else:
            self.cme_par_array: npt.NDArray[float] = cme_par_array

        # Initialise ToStateVector outputs
        self.state_ens = np.zeros((self.n_ensemble, self.n_pars))

        self.z_ens: npt.NDArray[float] = np.zeros((self.n_ensemble, self.n_pars))
        self.x_mean: npt.NDArray[float] = np.zeros(self.n_pars)
        self.x_std: npt.NDArray[float] = np.zeros(self.n_pars)


    def convert_t_init_dict_to_array(self) -> npt.NDArray[float]:
        """
        Function to convert datetime t_init to seconds since surf_init_time
        :param cme_par_dict: Dictionary that contains all CME parameters required
            (must contain "surf_init_time" and "t_init")
        :return: t_init_arr: Array of t_init values in seconds since surf_init_time
        """
        assert all(p in self.cme_par_dict.keys() for p in ["surf_init_time", "t_init", "n_members"])

        print(f"cme_par_dict[t_init] = {self.cme_par_dict['t_init']}")
        print(f"cme_par_dict[surf_init_time] = {self.cme_par_dict['surf_init_time']}")
        t_init_arr: npt.NDArray[float] = np.array([
            (self.cme_par_dict["t_init"][i] - self.cme_par_dict["surf_init_time"]).total_seconds()
            for i in range(self.n_ensemble)
        ])

        return t_init_arr


    def convert_v_dict_to_array(self) -> npt.NDArray[float]:
        """
        Function to convert cme_speed (v) to km/s and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "v")
        :return: v_arr: Array of v values in km/s
        """
        assert all(p in self.cme_par_dict.keys() for p in ["v"])

        v_arr: npt.NDArray[float] = np.array([
            v.to(u.km / u.s).value for v in self.cme_par_dict["v"][:]
        ])

        return v_arr


    def convert_width_dict_to_array(self) -> npt.NDArray[float]:
        """
        Function to convert width to deg and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "width")
        :return: width_arr: Array of width values in deg
        """
        assert all(p in self.cme_par_dict.keys() for p in ["width"])

        width_arr: npt.NDArray[float] = np.array([
            w.to(u.deg).value for w in self.cme_par_dict["width"][:]
        ])

        return width_arr


    def convert_lon_dict_to_array(self) -> npt.NDArray[float]:
        """
        Function to convert lon from 0 to 360 degrees into -180 to 180 degrees
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "lon")
        :return: lon_arr: Array of longitude values converted to -180 to 180 degrees
        """
        assert "lon" in self.cme_par_dict.keys()

        lon_arr: npt.NDArray[Quantity[None]] = np.array([
            lon.to(u.deg).value for lon in self.cme_par_dict["lon"][:]
        ])
        lonCond: list[bool] = list(lon_arr > 180)
        lon_arr[lonCond]: Quantity[None] = lon_arr[lonCond] - 360

        return lon_arr


    def convert_lat_dict_to_array(self) -> npt.NDArray[float]:
        """
        Function to convert lat to deg and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "lat")
        :return: lat_arr: Array of lat values in deg
        """
        assert all(p in self.cme_par_dict.keys() for p in ["lat"])

        lat_arr: npt.NDArray[float] = np.array([
            lat.to(u.deg).value for lat in self.cme_par_dict["lat"][:]
        ])

        return lat_arr


    def convert_thick_dict_to_array(self) -> npt.NDArray[float]:
        """
        Function to convert thickness to solar radii and place in array
        :param cme_par_dict: Dictionary that contains all CME parameters required (must contain "thick")
        :return: thick_arr: Array of thickness values in solar radii
        """
        assert all(p in self.cme_par_dict.keys() for p in ["thick"])

        thick_arr: npt.NDArray[float] = np.array([
            thick.to(u.solRad).value for thick in self.cme_par_dict["thick"][:]
        ])

        return thick_arr


    def cme_par_dict_to_cme_par_array(self) -> npt.NDArray[float]:
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
        """
        # Get the required keys and add the 'surf_init_time' and 'n_members' keys as requirements for this function
        req_keys = required_dict_keys()
        additional_keys_req = ["surf_init_time", "n_members"]
        req_keys.update(additional_keys_req)

        assert all(p in self.cme_par_dict.keys() for p in req_keys)

        # Initialise array to hold the CME parameters
        #cme_par_array: npt.NDArray = np.zeros((self.n_ensemble, 6))

        # Standardise the units and then remove the astropy units
        # The following for loop below is equivalent to writing, but allows for greater flexibility
        #  if cme_par_array's structure changes in the future:
        #   cme_par_array[:, 0]: npt.NDArray[float] = self.convert_t_init_dict_to_array()
        #   cme_par_array[:, 1]: npt.NDArray[float] = self.convert_v_dict_to_array()
        #   cme_par_array[:, 2]: npt.NDArray[float] = self.convert_width_dict_to_array()
        #   cme_par_array[:, 3]: npt.NDArray[float] = self.convert_lon_dict_to_array()
        #   cme_par_array[:, 4]: npt.NDArray[float] = self.convert_lat_dict_to_array()
        #   cme_par_array[:, 5]: npt.NDArray[float] = self.convert_thick_dict_to_array()
        for ind in range(6):
            # Get the key that corresponds to the current index required
            req_key = cme_par_get_keys(ind)

            # Get the required function for this key
            func_name = f"convert_{req_key}_dict_to_array"
            func = getattr(self, func_name)

            # Call the function and place it's output into cme_par_array
            self.cme_par_array[:, ind] = func()

        return self.cme_par_array


    def cme_par_dict_to_state_vector(self) -> npt.NDArray[float]:
        """
        Function to convert a set of parameter keys to a single array
        :return: state_ens: (self.n_ensemble, self.n_pars) array containing an
          ensemble of parameter values specified by self.pars_in_state_vector

        """
        assert all(p in required_dict_keys() for p in self.pars_in_state_vector)

        # Extract all required parameters, and put them in a state ensemble matrix
        # Below if loop is equivalent to
        #    if ip == "t_init":
        #        state_ens[:, i]: npt.NDArray[float] = self.convert_t_init_dict_to_array()
        #    elif ip == "v":
        #        state_ens[:, i]: npt.NDArray[float] = self.convert_v_dict_to_array()
        #    elif ip == "width":
        #        state_ens[:, i]: npt.NDArray[float] = self.convert_width_dict_to_array()
        #    elif ip == "lon":
        #        state_ens[:, i]: npt.NDArray[float] = self.convert_lon_dict_to_array()
        #    elif ip == "lat":
        #        state_ens[:, i]: npt.NDArray[float] = self.convert_lat_dict_to_array()
        #    elif ip == "thick":
        #        state_ens[:, i]: npt.NDArray[float] = self.convert_thick_dict_to_array()

        for i, ip in enumerate(self.pars_in_state_vector):
            if ip in required_dict_keys():
                # Get required function for key ip
                func_name = f"convert_{ip}_dict_to_array"
                func = getattr(self, func_name)

                self.state_ens[:, i] = func()
            else:
                raise Exception(f"Unrecognised CME parameter key: {ip}")

        return self.state_ens


    def cme_par_to_z_state_vector(self) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
        """
        Function to convert the CME parameters into a state vector
        :param cme_par_dict: Dictionary that contains all CME parameters required
        :param par_req: List containing the names of all the required parameters from cme_par_dict
        :return z_ens: Ensemble of z-scores
        :return x_mean: Mean of required CME parameters
        :return x_std: Standard deviation of required CME parameters
        """
        assert all(p in required_dict_keys() for p in self.pars_in_state_vector)

        # Extract weights of particles
        self.cme_par_dict = remove_infinite_weights(self.cme_par_dict)

        # Extract all required parameters, put them in an ensemble matrix, then calculate the zscores for output
        state_mat: npt.NDArray[float] = self.cme_par_dict_keys_to_array()

        for i in range(self.n_pars):
            # Convert to z_scores
            self.z_ens[:, i], self.x_mean[i], self.x_std[i] = sir.zscore(state_mat[:, i])

        return self.z_ens, self.x_mean, self.x_std