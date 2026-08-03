"""
Module to make a CME parameter array with the correct format
"""
import numpy as np
import datetime
import numpy.typing as npt

class CmeParArray:
    def __init__(
            self,
            n_members: int,
            surf_init_time: datetime.datetime,
            t_init: npt.NDArray[float],
            v: npt.NDArray[float],
            width: npt.NDArray[float],
            lon: npt.NDArray[float],
            lat: npt.NDArray[float],
            thick: npt.NDArray[float]
    ):
        self.n_members: int = n_members
        self.surf_init_time: datetime.datetime = surf_init_time
        self.t_init: npt.NDArray[float] = t_init
        self.v: npt.NDArray[float] = v
        self.width: npt.NDArray[float] = width
        self.lon: npt.NDArray[float] = lon
        self.lat: npt.NDArray[float] = lat
        self.thick: npt.NDArray[float] = thick

    def make_par_array(self) -> npt.NDArray[float]:
        # Define a parameter dictionary
        par_array: npt.NDArray[float] = np.zeros((self.n_members, 6))

        par_array[:, 0] = self.t_init
        par_array[:, 1] = self.v
        par_array[:, 2] = self.width
        par_array[:, 3] = self.lon
        par_array[:, 4] = self.lat
        par_array[:, 5] = self.thick

        return par_array
