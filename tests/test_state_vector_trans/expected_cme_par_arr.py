import numpy as np
import datetime
import sys

import numpy.typing as npt

from SIR_HUXt.tests.cme_par_array import CmeParArray

#sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
from SIR_HUXt.code.cme_par_dict_structure import required_dict_keys, cme_par_get_indices

class ExpCmeParArray:
    def __init__(self):
        self.huxt_init_time: datetime.datetime = datetime.datetime(
            year=2021, month=1, day=1, hour=18, minute=0, second=0
        )
        self.n_members: int = 5

        self.exp_t_init: npt.NDArray[float] = np.array([
            43200, 46800, 50400, 64800, 55800
        ])

        self.exp_v: npt.NDArray[float] = np.array([
            500, 550, 750, 740, 1200
        ])

        self.exp_width: npt.NDArray[float] = np.array([
            50, 30, 48, 56, 12
        ])

        self.exp_lon: npt.NDArray[float] = np.array([
            -34, -120, 49, -120, -175
        ])

        self.exp_lat: npt.NDArray[float] = np.array([
            2, 4, -49, -12, 8
        ])

        self.exp_thick: npt.NDArray[float] = np.array([
            3, 12, 5, 7, 8
        ])

    def expected_huxt_init_time(self) -> datetime.datetime:
        return self.huxt_init_time

    def expected_t_init(self) -> npt.NDArray[float]:
        return self.exp_t_init

    def expected_v(self) -> npt.NDArray[float]:
        return self.exp_v

    def expected_width(self) -> npt.NDArray[float]:
        return self.exp_width

    def expected_lon(self) -> npt.NDArray[float]:
        return self.exp_lon

    def expected_lat(self) -> npt.NDArray[float]:
        return self.exp_lat

    def expected_thick(self) -> npt.NDArray[float]:
        return self.exp_thick

    def expected_cme_par_array(self) -> npt.NDArray[float]:
        par_arr_class = CmeParArray(
            n_members=self.n_members,
            huxt_init_time=self.huxt_init_time,
            t_init=self.exp_t_init,
            v=self.exp_v,
            width=self.exp_width,
            lon=self.exp_lon,
            lat=self.exp_lat,
            thick=self.exp_thick
        )

        return par_arr_class.make_par_array()


class ExpStateVector(ExpCmeParArray):
    def __init__(self, pars_in_state_vector: list[str]):
        super().__init__()

        assert all(p in required_dict_keys() for p in pars_in_state_vector)
        self.pars_in_state_vector: list[str] = pars_in_state_vector

        self.n_pars = len(pars_in_state_vector)


    def expected_state_vector(self) -> npt.NDArray[float]:
        # Initialise an empty array
        exp_state_vector: npt.NDArray[float] = np.zeros((self.n_members, self.n_pars))

        for i, ip in enumerate(self.pars_in_state_vector):
            # Get the required function parameter 'ip'
            func_name = f"expected_{ip}"
            func = getattr(self, func_name)

            # Call the function and place it's output into cme_par_array
            exp_state_vector[:, i] = func()

        return exp_state_vector


class ExpCmeArrayFromStateVector(ExpCmeParArray):
    def __init__(self, pars_in_state_vector: list[str]):
        super().__init__()
        assert all(p in required_dict_keys() for p in pars_in_state_vector)

        self.pars_in_state_vector: list[str] = pars_in_state_vector
        self.n_pars = len(pars_in_state_vector)

        self.state_t_init = np.array([
            49200, 50800, 53400, 49000, 66000
        ])
        self.state_v_init = np.array([
            550, 1000, 940, 800, 650
        ])
        self.state_width: npt.NDArray[float] = np.array([
            40, 38, 34, 53, 27
        ])
        self.state_lon: npt.NDArray[float] = np.array([
            -30, -100, 84, -90, -74
        ])
        self.state_lat: npt.NDArray[float] = np.array([
            3, 5, -45, -26, 11
        ])
        self.state_thick: npt.NDArray[float] = np.array([
            7, 11, 6, 19, 2
        ])

    def exp_state_t_init(self) -> npt.NDArray[float]:
        return self.state_t_init

    def exp_state_v(self) -> npt.NDArray[float]:
        return self.state_v_init

    def exp_state_width(self) -> npt.NDArray[float]:
        return self.state_width

    def exp_state_lon(self) -> npt.NDArray[float]:
        return self.state_lon

    def exp_state_lat(self) -> npt.NDArray[float]:
        return self.state_lat

    def exp_state_thick(self) -> npt.NDArray[float]:
        return self.state_thick

    def expected_state_to_array(self) -> npt.NDArray[float]:
        # Get original cme_par_array
        exp_cme_par_array: npt.NDArray[float] = self.expected_cme_par_array()

        for i, ip in enumerate(self.pars_in_state_vector):
            func_name: str = f"exp_state_{ip}"
            func = getattr(self, func_name)

            # Get the required cme_par_array index for ip
            ind_req: int = cme_par_get_indices(ip)

            exp_cme_par_array[:, ind_req] = func()

        return exp_cme_par_array