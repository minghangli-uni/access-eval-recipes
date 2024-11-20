import os
import re
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class TruncationFilesReader(object):
    """
    A class to read, process and organise truncation files.
    This class provides methods to:
        Loads static grid coordinates.
        Parses and extract data from truncation files.
        Organises truncation data by specified variables (e.g., time or spatial attributes).

    Attributes:
        static: Dict[str, Any]: contains the static grid configuration
        archive_path: str: Path to the directory containing archived output files for processing.
        xh (np.ndarray): Longitude coordinates of the grid.
        yh (np.ndarray): Latitude coordinates of the grid.
        geolon (np.ndarray): Longitude values in the grid.
        geolat (np.ndarray): Latitude values in the grid.

    Methods:
        load_coords():
            Loads and returns the grid coordinates from the static configuration.
        process_truncation_files():
            Searches the archive directory for truncation files, parses them, and returns a list of truncation data.
        parse_truncation_files(truncation_files_path: str):
            Parses truncation files and extracts truncation data using regex.
        _extract_truncation_data(match):
            Extracts truncation details such as datetime, velocity type, processor adn etc.
        find_indices(target_lon, target_lat):
            Finds the closest grid indices and coordinates for a given longitude and latitude.
        convert_to_date_time(year, yearday, time):
            Converts year, yearday, and fractional time into a `datetime` object.
        truncation_organise_by_variable(truncation_data, trunc_variable):
            Organises truncation data by a specified variable (e.g., month, year, processor).
    """

    def __init__(self, static, archive_path):
        self.static = static
        self.archive_path = archive_path
        self.xh, self.yh, self.geolon, self.geolat = self.load_coords()

    def load_coords(self):
        return (
            self.static["xh"],
            self.static["yh"],
            self.static["geolon"],
            self.static["geolat"],
        )

    def process_truncation_files(self):
        all_truncation_data = []

        for output_dir in os.listdir(self.archive_path):
            output_dir_path = os.path.join(self.archive_path, output_dir)
            if os.path.isdir(output_dir_path) and output_dir.startswith("output"):
                for filename in os.listdir(output_dir_path):
                    if filename.startswith(
                        "U_velocity_truncations"
                    ) or filename.startswith("V_velocity_truncations"):
                        truncation_files_path = os.path.join(output_dir_path, filename)
                        truncation_data = self.parse_truncation_files(
                            truncation_files_path
                        )
                        if truncation_data:
                            all_truncation_data.append(truncation_data)
        if not all_truncation_data:
            logging.warning(f"No velocity truncations exist in {self.archive_path}.")
        return all_truncation_data

    def parse_truncation_files(self):
        trunc_pattern = re.compile(
            r"Time\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([UV])-velocity violation at\s+(\d+):\s+\d+\s+\d+\s+\(\s*([-+]?\d*\.\d+)\s+[E]?\s+([-+]?\d*\.\d+)\s+[N]?\)\s+Layers\s+(\d+)\s+to\s+(\d+)\.\s+dt\s+=\s+(\d+)"
        )
        truncations = []

        try:
            with open(self.truncation_files_path, "r") as f:
                file_read = f.read()
            for match in trunc_pattern.finditer(file_read):
                truncations.append(_extract_truncation_data(match))
        except FileNotFoundError:
            logging.error(f"Files not found: {truncation_file_path}")
        except Exception as e:
            logging.error(f"Error processing file {truncation_file_path}: {e}")

    def _extract_truncation_data(self, match):
        year = int(match.group(1))
        yearday = int(match.group(2))
        time_of_day = float(match.group(3))
        velocity_type = match.group(4) + "-velocity"
        processor = int(match.group(5))
        lon = float(match.group(6))
        lat = float(match.group(7))
        layer_start = int(match.group(8))
        layer_end = int(match.group(9))
        dt = int(match.group(10))

        (
            nearest_idx,
            nearest_lon_idx,
            nearest_lat_idx,
            nearest_lon,
            nearest_lat,
            nearest,
        ) = self.find_indices(lon, lat)
        datetime_value = self.convert_to_date_time(year, yearday, time_of_day)

        return {
            "datetime": datetime_value,
            "velocity_type": velocity_type,
            "processor": processor,
            "longitude": lon,
            "latitude": lat,
            "longitude_index": nearest_lon_idx,
            "latitude_index": nearest_lat_idx,
            "eval_longitude": nearest_lon,
            "eval_latitude": nearest_lat,
            "layers_start": layer_start,
            "layers_end": layer_end,
            "dt": dt,
        }

    def find_indices(self, target_lon, target_lat):

        if hasattr(self.xh, "values"):
            xh = self.xh.values
        if hasattr(self.xh, "values"):
            yh = self.xh.values

        lon_diff = np.abs(xh - target_lon)
        lat_diff = np.abs(yh - target_lat)

        nearest_lon_idx = np.argmin(lon_diff)
        nearest_lat_idx = np.argmin(lat_diff)
        return (
            (nearest_lat_idx, nearest_lon_idx),
            nearest_lon_idx,
            nearest_lat_idx,
            xh[nearest_lon_idx],
            yh[nearest_lat_idx],
            (yh[nearest_lat_idx], xh[nearest_lon_idx]),
        )

    def convert_to_date_time(self, year, yearday, time):
        yearday = max(1, yearday)
        tmp = datetime(year, 1, 1)
        target_date = tmp + timedelta(days=yearday - 1)

        hours = int(time)
        minutes = int((time - hours) * 60)
        seconds = int((((time - hours) * 60) - minutes) * 60)
        return target_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)

    def truncation_organise_by_variable(self, truncation_data, trunc_variable):
        group_data = defaultdict(list)
        for trunc in truncation_data:
            if trunc_variable == "trunc_month":
                v = trunc["datetime"].month
            elif trunc_variable == "trunc_year":
                v = trunc["datetime"].year
            else:
                v = trunc.get(trunc_variable)
            group_data[v].append(trunc)
        trunc_keys = list(group_data.keys())
        logging.info(f"Available truncation keys: {list(group_data.keys())}")
        return dict(group_data)
