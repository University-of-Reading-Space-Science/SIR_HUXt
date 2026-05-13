import astropy.units as u
import datetime

from astropy.units import Quantity

from SIR_HUXt.tests.cme_par_dict import CmeParDict


class ExpCmeParDict:
    def __init__(self):
        self.huxt_init_time: datetime.datetime = datetime.datetime(
            year=2021, month=1, day=1, hour=0, minute=0, second=0
        )
        self.n_members: int = 5

        self.exp_t_init: list[datetime.datetime] = [
            datetime.datetime(year=2021, month=1, day=1, hour=12, minute=0, second=0),
            datetime.datetime(year=2021, month=1, day=1, hour=13, minute=0, second=0),
            datetime.datetime(year=2021, month=1, day=1, hour=14, minute=0, second=0),
            datetime.datetime(year=2021, month=1, day=1, hour=18, minute=0, second=0),
            datetime.datetime(year=2021, month=1, day=1, hour=15, minute=30, second=0)
        ]

        self.exp_v: list[Quantity[u.km / u.s]] = [
            500, 550, 750, 740, 1200
        ] * u.km / u.s

        self.exp_width: list[Quantity[u.deg]] = [
            50, 30, 48, 56, 12
        ] * u.deg

        self.exp_lon: list[Quantity[u.deg]] = [
            326, 240, 49, 240, 185
        ] * u.deg

        self.exp_lat: list[Quantity[u.deg]] = [
            2, 4, -49, -12, 8
        ] * u.deg

        self.exp_thick: list[Quantity[u.solRad]] = [
            3, 12, 5, 7, 8
        ] * u.solRad

    def expected_huxt_init_time(self) -> datetime.datetime:
        return self.huxt_init_time

    def expected_t_init(self) -> list[datetime.datetime]:
        return self.exp_t_init

    def expected_v(self) -> list[Quantity[u.km / u.s]]:
        return self.exp_v

    def expected_width(self) -> list[Quantity[u.deg]]:
        return self.exp_width

    def expected_lon(self) -> list[Quantity[u.deg]]:
        return self.exp_lon

    def expected_lat(self) -> list[Quantity[u.deg]]:
        return self.exp_lat

    def expected_thick(self) -> list[Quantity[u.solRad]]:
        return self.exp_thick

    def expected_cme_par_dict(self):
        par_class = CmeParDict(
            n_members=self.n_members,
            huxt_init_time=self.huxt_init_time,
            t_init=self.exp_t_init,
            v=self.exp_v,
            width=self.exp_width,
            lon=self.exp_lon,
            lat=self.exp_lat,
            thick=self.exp_thick
        )

        return par_class.make_par_dict()