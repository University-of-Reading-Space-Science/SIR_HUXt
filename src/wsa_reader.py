import os
from datetime import datetime
from pathlib import Path
import re

import huxt.huxt_inputs as Hin
import astropy.units as u

import requests
from bs4 import BeautifulSoup


class ReadWSAFiles():
    def __init__(self, date_required, output_dir=None):
        self.base_url = (
            "https://iswa.ccmc.gsfc.nasa.gov/"
            "iswa_data_tree/model/solar/WSA4.5/"
            "WSA_VEL_GONG"
        )

        self.date_required = date_required

        self.wsa_url, self.wsa_url_prev_month = self.get_wsa_url()

        if output_dir is None:
            self.output_dir="."
        else:
            self.output_dir = output_dir

        # Make output directory if it doesn't already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def get_wsa_url(self):
        year_req = self.date_required.year
        month_req = self.date_required.month

        print(f"year_req={year_req:04d}, month_req={month_req:02d}")
        wsa_url = f"{self.base_url}/{year_req:04d}/{month_req:02d}"

        wsa_url_prev_month = f"{self.base_url}/{year_req:04d}/{month_req - 1:02d}"
        print(f"wsa_url={wsa_url}")

        return wsa_url, wsa_url_prev_month


    def list_files(self):
        """
        Get all files in the directory and extract timestamps.
        """

        response = requests.get(self.wsa_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        files = []

        for link in soup.find_all("a"):

            href = link.get("href", "")

            if not href.endswith(".fits"):
                continue

            filename = href.split("/")[-1]

            # Match:
            # vel_201209110504R000_gong.fits
            match = re.match(
                r"vel_(\d{12})R\d{3}_gong\.fits",
                filename,
            )

            if match is None:
                continue

            file_time = datetime.strptime(
                match.group(1),
                "%Y%m%d%H%M"
            )

            files.append(
                {
                    "datetime": file_time,
                    "filename": filename,
                    "url": f"{self.wsa_url}/{href}",
                }
            )

        files.sort(key=lambda x: x["datetime"])

        return files


    def find_file(self):
        """
        Find file at or immediately before self.date_required.
        """

        files = self.list_files()

        candidates = [
            f for f in files
            if f["datetime"] <= self.date_required
        ]

        if not candidates:
            raise ValueError(
                f"No file found before {self.date_required}"
            )

        return candidates[-1]


    def download_file(self):
        """
        Download the closest file at or before self.date_required.
        """

        file_info = self.find_file()

        print(
            f"Selected file: {file_info['filename']}\n"
            f"Timestamp: {file_info['datetime']}"
        )

        output_path = (
            Path(self.output_dir) / file_info["filename"]
        )

        response = requests.get(
            file_info["url"],
            stream=True,
            timeout=300,
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded to {output_path}")

        return output_path


if __name__ == "__main__":

    requested_time = datetime(
        2012, 9, 11, 6, 0
    )

    parent_dir = Path(__file__).parent
    output_dir = os.path.abspath(
        os.path.join(parent_dir, "..", "WSA_fits_files")
    )
    readWSAClass = ReadWSAFiles(requested_time, output_dir=output_dir)
    output_path = readWSAClass.download_file()

    # Get input speeds for all longitudes for input into huxt
    vr = Hin.get_WSA_long_profile(output_path, lat=0.0 * u.deg)
    print(f"vr = {vr}")
    print(len(vr))
