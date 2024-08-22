import os
import glob
import json
import numpy as np
import pandas as pd


def scan_files(directory):
    """
    Scan the directory for CSV and geojson files.
    Returns a list of CSV and geojson file paths.
    """
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    json_files = glob.glob(os.path.join(directory, "*.geojson"))
    return csv_files, json_files


def process_csv(file_path):
    """
    Process a CSV file to generate the SRF as a Gaussian.
    Returns an interpolated array of shape (n_bands, 2101).
    """
    data = pd.read_csv(file_path)
    center_wavelengths = data.iloc[:, 0].values
    fwhms = data.iloc[:, 1].values

    wavelength_range = np.arange(400, 2501, 1)  # 1 nm step from 400 to 2500 nm
    n_bands = len(center_wavelengths)
    srf_array = np.zeros((n_bands, len(wavelength_range)))

    for i, (center, fwhm) in enumerate(zip(center_wavelengths, fwhms)):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        response = np.exp(-0.5 * ((wavelength_range - center) / sigma) ** 2)
        srf_array[i, :] = response

    return srf_array


def process_json(file_path):
    """
    Process a geojson file to generate the SRF.
    Returns an interpolated array of shape (n_bands, 2101).
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    bands = data["features"]
    wavelength_range = np.arange(400, 2501, 1)  # 1 nm step from 400 to 2500 nm
    n_bands = len(bands)
    srf_array = np.zeros((n_bands, len(wavelength_range)))

    for i, band in enumerate(bands):
        wavelengths = np.array(band["properties"]["response"]["x"])
        responses = np.array(band["properties"]["response"]["y"])
        interpolated_response = np.interp(
            wavelength_range, wavelengths, responses
        )
        srf_array[i, :] = interpolated_response

    return srf_array


def main(directory):
    """
    Main function to process the files and store the results.
    """
    csv_files, json_files = scan_files(directory)
    sensors_srf_data = {}

    for csv_file in csv_files:
        sensor_name = os.path.basename(csv_file).replace(".csv", "")
        srf_array = process_csv(csv_file)
        sensors_srf_data[sensor_name] = srf_array

    for json_file in json_files:
        sensor_name = os.path.basename(json_file).replace(".geojson", "")
        srf_array = process_json(json_file)
        sensors_srf_data[sensor_name] = srf_array

    return sensors_srf_data


# Example usage:
directory = "."
srf_data = main(directory)

# You can now access the SRF data for each sensor by sensor name.
for sensor, srf_array in srf_data.items():
    print(f"Sensor: {sensor}, SRF array shape: {srf_array.shape}")

# Dump SRF data to a npz file in disk
np.savez("sensor_srf.npz", **srf_data)
