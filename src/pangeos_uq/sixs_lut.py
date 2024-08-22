"""This module contains functions to generate a Look-Up Table (LUT)
using the sixs model. The model is called externally using subprocess and
the output is stored in an HDF5 file file. The LUT is generated using either a
LHS or Sobol sampling scheme.

NOTE: This code has strong dependencies on the 6S model and Python's parallel
processing libraries, so probably won't run on Windows."""

import subprocess
import re
from typing import List, Dict, Tuple
import numpy as np
from scipy.spatial import KDTree
import h5py
from multiprocessing import Pool
from scipy.stats.qmc import Sobol
import importlib.resources as resources

__author__ = "Jose Gomez-Dans"
__copyright__ = "Copyright 2024, Jose Gomez-Dans"
__license__ = "MIT"


SENTINEL_2_CENTER_WAVELENGTHS = [
    0.4925,
    0.5600,
    0.6650,
    0.7050,
    0.7400,
    0.7825,
    0.8325,
    0.8650,
    1.6125,
    2.2025,
]


class LUTQuery:
    def __init__(self, hdf5_path: str | None = None, k: int = 1):
        """
        Initialize the LUTQuery object.

        Parameters:
        hdf5_path (str | None): Path to the HDF5 file containing the LUT. By
            default, it will use the small LUT provided with the package.
        k (int): Number of nearest neighbors to return.
        """
        if hdf5_path is None:
            self.hdf5_path = str(
                resources.files("pangeos_uq.data")
                / "lut_s2_small_compressed.h5"
            )

        else:
            self.hdf5_path = hdf5_path
        self.k = k
        self._load_lut()
        self._load_or_build_index()

    def _load_lut(self):
        """
        Load the LUT data from the HDF5 file into memory.
        """
        # Define the weights for each feature based on their range
        self.weights = np.array(
            [
                1.0 / 20,  # vza: normalized assuming max range of 20
                1.0 / 60,  # sza: normalized assuming max range of 60
                1.0 / 180,  # raa: normalized assuming max range of 180
                1.0 / 1.5,  # aot: normalized assuming max range of 1.5
                1.0 / 10,  # tcwv: normalized assuming max range of 10
            ]
        )  # wv: normalized assuming max range of 5

        with h5py.File(self.hdf5_path, "r") as f:
            self.lut_data = {
                "vza": f["vza"][:] * self.weights[0],
                "sza": f["sza"][:] * self.weights[1],
                "raa": f["raa"][:] * self.weights[2],
                "aot": f["aot"][:] * self.weights[3],
                "tcwv": f["tcwv"][:] * self.weights[4],
                "wv": f["wv"][:],
                "xap": f["xap"][:],
                "xb": f["xb"][:],
                "xc": f["xc"][:],
            }

        # Precompute and store indices where wv is blue
        first_wv = SENTINEL_2_CENTER_WAVELENGTHS[0]

        self.zero_wv_indices = np.where(self.lut_data["wv"] == first_wv)[0]

    def _load_or_build_index(self):
        """
        Load the index from a file or build it if it doesn't exist.
        """
        self._build_index()

    def _build_index(self):
        """
        Build the kdtree index for the LUT data.
        """
        # Combine the five search dimensions into a single matrix
        indices_for_tree = self.zero_wv_indices

        search_data = np.vstack(
            (
                self.lut_data["vza"][indices_for_tree],
                self.lut_data["sza"][indices_for_tree],
                self.lut_data["raa"][indices_for_tree],
                self.lut_data["aot"][indices_for_tree],
                self.lut_data["tcwv"][indices_for_tree],
                # self.lut_data["wv"],
            )
        ).T.astype(np.float32)

        # Build the kdtree index
        self.index = KDTree(search_data)

    def query(
        self,
        vza: float,
        sza: float,
        raa: float,
        aot: float,
        tcwv: float,
    ) -> Dict[str, np.ndarray]:
        """
        Query the LUT based on the provided angles, AOT, and TCWV.

        Parameters:
        vza (float): View Zenith Angle in degrees.
        sza (float): Solar Zenith Angle in degrees.
        raa (float): Relative solar-view Azimuth Angle in degrees.
        aot (float): Aerosol Optical Thickness at 550 nm.
        tcwv (float): Total Column Water Vapor in g/cm².

        Returns:
        Dict[str, np.ndarray]: A dictionary containing the 'wv', 'xap', 'xb',
        and 'xc' arrays for the k-nearest neighbors.
        """

        # Create the query point and apply weights
        query_point = (
            np.array([[vza, sza, raa, aot, tcwv]], dtype=np.float32)
            * self.weights
        )
        n_wavelengths = len(SENTINEL_2_CENTER_WAVELENGTHS)
        # Perform the KNN search
        distances, indices = self.index.query(query_point, k=self.k)

        # Extract the corresponding data for the nearest neighbors
        idx = indices[0][0]

        # Use precomputed indices to find the start index
        start_idx = self.zero_wv_indices[idx]
        end_idx = start_idx + n_wavelengths  # Select N rows starting from

        result = {
            "wv": self.lut_data["wv"][start_idx:end_idx],
            "xap": self.lut_data["xap"][start_idx:end_idx],
            "xb": self.lut_data["xb"][start_idx:end_idx],
            "xc": self.lut_data["xc"][start_idx:end_idx],
        }

        return result


def interpolate_results(
    results: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Process the results of LUT lookupto set specific wv ranges to np.nan and
    perform a linear interpolation between 0.4 and 2.5 um every 0.001 um.

    Parameters:
    results (Dict[str, np.ndarray]): A dictionary containing 'wv', 'xap', 'xb'
                                    and 'xc' arrays.

    Returns:
    Dict[str, np.ndarray]: A dictionary with 'wv', 'xap', 'xb', 'xc' arrays
                    interpolated and with specified ranges set to np.nan.
    """
    # Extract the wv array from results
    wv = results["wv"]

    # Set specified ranges to np.nan
    # These are strong absorption features
    mask = (
        ((wv >= 1.100) & (wv <= 1.140))
        | ((wv >= 1.325) & (wv <= 1.475))
        | ((wv >= 1.800) & (wv <= 2.000))
    )
    results["wv"][mask] = np.nan
    results["xap"][mask] = np.nan
    results["xb"][mask] = np.nan
    results["xc"][mask] = np.nan

    # Create an array for interpolated wv values
    wv_interpolated = np.arange(0.4, 2.5 + 0.001, 0.001)

    # Initialize interpolated arrays
    xap_interpolated = np.full(wv_interpolated.shape, np.nan)
    xb_interpolated = np.full(wv_interpolated.shape, np.nan)
    xc_interpolated = np.full(wv_interpolated.shape, np.nan)

    # Get the valid indices for interpolation
    valid_indices = ~np.isnan(wv)
    valid_wv = wv[valid_indices]
    if valid_wv.size > 0:
        # Linear interpolation for non-nan values
        xap_interpolated = np.interp(
            wv_interpolated,
            valid_wv,
            results["xap"][valid_indices],
            # left=np.nan,
            # right=np.nan,
        )
        xb_interpolated = np.interp(
            wv_interpolated,
            valid_wv,
            results["xb"][valid_indices],
            # left=np.nan,
            # right=np.nan,
        )
        xc_interpolated = np.interp(
            wv_interpolated,
            valid_wv,
            results["xc"][valid_indices],
            # left=np.nan,
            # right=np.nan,
        )

        mask = (
            ((wv_interpolated >= 1.100) & (wv_interpolated <= 1.140))
            | ((wv_interpolated >= 1.325) & (wv_interpolated <= 1.475))
            | ((wv_interpolated >= 1.800) & (wv_interpolated <= 2.000))
        )
        mask_arr = np.ones_like(wv_interpolated)
        mask_arr[mask] = np.nan
    # Replace the original arrays with the interpolated ones
    return {
        "wv": wv_interpolated * mask_arr,
        "xap": xap_interpolated * mask_arr,
        "xb": xb_interpolated * mask_arr,
        "xc": xc_interpolated * mask_arr,
    }


def generate_lut_hdf5(
    parameter_bounds: List[Tuple[float, float]],
    n_samples: int,
    hdf5_path: str,
    sampling_method: str = "lhs",
    lower_wv: float = 0.38,
    upper_wv: float = 2.5,
) -> None:
    """
    Generates a LUT using parallel processing and stores it in an HDF5 file.

    Parameters:
    parameter_bounds (List[Tuple[float, float]]): Min and max values for
                                                  each parameter.
    n_samples (int): Number of samples to generate.
    hdf5_path (str): Path to the output HDF5 file.
    sampling_method (str): Sampling method to use, either 'lhs' or 'sobol'.
    lower_wv (float): Lower wavelength limit in microns.
    upper_wv (float): Upper wavelength limit in microns.

    Returns:
    None
    """
    # Calculate the number of wavelengths in the output
    # n_wavelengths = calculate_wavelength_count(lower_wv, upper_wv)
    # n_wavelengths = len(SENTINEL_2_CENTER_WAVELENGTHS)

    # Choose the sampling strategy
    if sampling_method == "lhs":
        samples = generate_lhs_samples(parameter_bounds, n_samples)
    elif sampling_method == "sobol":
        samples = generate_sobol_samples(parameter_bounds, n_samples)
    else:
        raise ValueError("sampling_method must be 'lhs' or 'sobol'.")

    # Prepare to accumulate valid rows
    valid_data = {
        "vza": [],
        "sza": [],
        "raa": [],
        "aot": [],
        "tcwv": [],
        "wv": [],
        "xap": [],
        "xb": [],
        "xc": [],
    }

    # Use multiprocessing.Pool for parallel processing
    with Pool() as pool:
        # Process each sample in parallel
        for result_list in pool.imap_unordered(
            process_sample_wrapper,
            [(sample, SENTINEL_2_CENTER_WAVELENGTHS) for sample in samples],
        ):
            for result in result_list:
                # Check for NaN values and skip them
                if (
                    np.isnan(result[6])  # xap
                    or np.isnan(result[7])  # xb
                    or np.isnan(result[8])  # xc
                ):
                    continue

                # Accumulate valid data
                valid_data["vza"].append(result[0])
                valid_data["sza"].append(result[1])
                valid_data["raa"].append(result[2])
                valid_data["aot"].append(result[3])
                valid_data["tcwv"].append(result[4])
                valid_data["wv"].append(result[5])
                valid_data["xap"].append(result[6])
                valid_data["xb"].append(result[7])
                valid_data["xc"].append(result[8])

    # Convert lists to numpy arrays
    for key in valid_data:
        valid_data[key] = np.array(valid_data[key])

    # Write all valid data to HDF5
    with h5py.File(hdf5_path, "w") as f:
        for key, data in valid_data.items():
            f.create_dataset(key, data=data, dtype="f")


def process_sample_wrapper(args):
    """
    Wrapper to allow passing of multiple arguments to process_sample
    through multiprocessing.Pool.
    """
    sample, n_wavelengths = args
    return process_sample(sample, n_wavelengths)


def generate_sobol_samples(
    parameter_bounds: List[Tuple[float, float]], n_samples: int
) -> np.ndarray:
    """
    Generates samples using Sobol sequences.

    Parameters:
    parameter_bounds (List[Tuple[float, float]]): Min and max values for
                                                  each parameter.
    n_samples (int): Number of samples to generate.

    Returns:
    np.ndarray: Array of sample points.
    """
    n_params = len(parameter_bounds)
    sobol_engine = Sobol(d=n_params, scramble=True)
    sobol_samples = sobol_engine.random(n=n_samples)
    samples = np.zeros_like(sobol_samples)

    for i, (low, high) in enumerate(parameter_bounds):
        samples[:, i] = low + sobol_samples[:, i] * (high - low)

    return samples


def process_sample(
    sample: Tuple[float, ...], wavelengths: List[float]
) -> List[Tuple[float, ...]]:
    """
    Generates LUT entries for a given sample, accounting for multiple
    wavelength outputs.

    Parameters:
    sample (Tuple[float, ...]): A tuple containing the input parameters.
    n_wavelengths (int): Number of wavelengths to expect from the 6S output.

    Returns:
    List[Tuple[float, ...]]: A list of tuples with the sample's inputs
                             and the corresponding outputs from the 6S model
                             for each wavelength.
    """
    vza, sza, raa, aot, tcwv = sample
    config = generate_6s_config(vza, sza, raa, aot, tcwv)
    output = run_6s_model(config)
    params_list = extract_parameters_from_output(output)
    results = []
    results = []
    for wv in wavelengths:
        # Extract parameters for the specific wavelength
        params = next((p for p in params_list if p["wavelength"] == wv), None)
        if params:
            xap = params["xap"]
            xb = params["xb"]
            xc = params["xc"]
            results.append((vza, sza, raa, aot, tcwv, wv, xap, xb, xc))

    return results


def generate_lhs_samples(
    parameter_bounds: List[Tuple[float, float]], n_samples: int
) -> np.ndarray:
    """
    Generates samples using Latin Hypercube Sampling (LHS).

    Parameters:
    parameter_bounds (List[Tuple[float, float]]): Min and max values for
                                                  each parameter.
    n_samples (int): Number of samples to generate.

    Returns:
    np.ndarray: Array of sample points.
    """
    n_params = len(parameter_bounds)

    # Generate the intervals for each parameter
    intervals = np.linspace(0, 1, n_samples + 1)

    # Randomly permute the intervals for each parameter
    lhs_samples = np.zeros((n_samples, n_params))
    for i in range(n_params):
        lhs_samples[:, i] = (
            np.random.permutation(intervals[:-1] + intervals[1:]) / 2.0
        )

    # Scale the samples to the given parameter bounds
    samples = np.zeros_like(lhs_samples)
    for i, (low, high) in enumerate(parameter_bounds):
        samples[:, i] = low + lhs_samples[:, i] * (high - low)

    return samples


def generate_6s_config(
    vza: float,
    sza: float,
    raa: float,
    aot: float,
    tcwv: float,
    lower_wv: float = 0.400,
    upper_wv: float = 2.500,
    path_to_sixs: str = "sixsV2.1",
) -> str:
    """Generate a 6S configuration and run the model.

    Parameters:
    vza (float): View Zenith Angle in degrees.
    sza (float): Solar Zenith Angle in degrees.
    raa (float): Relatiive view-solar Azimuth Angle in degrees.
    aot (float): Aerosol Optical Thickness at 550 nm.
    tcwv (float): Total Column Water Vapor in g/cm².
    lower_wv (float): Lower wavelength in µm.
    upper_wv (float): Upper wavelength in µm.
    path_to_sixs (str): Path to the 6S executable.

    Returns:
    str: The output from the 6S model as a string.
    """

    # Define the 6S configuration template
    config_template = f"""
    0  # User defined
    {sza} {0} {vza} {raa} 1 1  # sza saa vza vaa month day
    8  # User defined H2O, O3
    {tcwv}, 0.4  # H2O and ozone
    1  # Aerosol model (Continental)
    0
    {aot}  # AOT at 550 nm
    0.00  # Target level km above sea level
    -1000.00  # Sensor level satellite
    -2  # Lower and upper spectral range report each step
    {lower_wv} {upper_wv}  # Lower and upper spectral range
    0  # Homogeneous surface
    0  # No directional effects
    0
    0
    0
    -1  # No atmospheric corrections selected
    """
    return config_template


def run_6s_model(
    config: str, path_to_sixs: str = "/home/jose/python/sixs/sixsv2.1/sixsV2.1"
) -> str:
    """
    Run the 6S model with the given configuration string and return the output.
    """
    result = subprocess.run(
        [path_to_sixs],
        input=config,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def extract_parameters_from_output(output: str) -> List[Dict[str, float]]:
    """
    Extract xap, xb, and xc for each wavelength from the 6S output.

    Parameters:
    output (str): The output string from the 6S model.

    Returns:
    List[Dict[str, float]]: A list of dictionaries, each containing
                            'wavelength', 'xap', 'xb', and 'xc' keys
                            with their respective float values.
    """
    # Split the output by lines
    lines = output.splitlines()

    # List to hold the result dictionaries
    result = []

    # Parse through lines to find the table and extract relevant data
    for line in lines:
        # Look for lines that start with a wavelength value
        if re.match(r"^\*\s*\d+\.\d+", line):
            # Extract the numeric values from the line
            values = list(map(float, re.findall(r"\d+\.\d+", line)))

            if len(values) >= 6:
                wavelength = values[0]
                total_gas_trans = values[1]
                total_scat_down = values[2]
                total_scat_up = values[3]
                atm_intr_refl = values[4]
                total_spheri_albedo = values[5]

                # Calculate xap, xb, and xc
                try:
                    xap = 1 / (
                        total_gas_trans * total_scat_up * total_scat_down
                    )
                except ZeroDivisionError:
                    xap = np.nan
                try:
                    xb = atm_intr_refl / (
                        total_gas_trans * total_scat_up * total_scat_down
                    )
                except ZeroDivisionError:
                    xb = np.nan
                xc = total_spheri_albedo

                # Append the result as a dictionary
                result.append(
                    {"wavelength": wavelength, "xap": xap, "xb": xb, "xc": xc}
                )

    return result


def calculate_wavelength_count(lower_wv: float, upper_wv: float) -> int:
    """
    Calculate the number of wavelengths between lower_wv and upper_wv.

    Parameters:
    lower_wv (float): Lower wavelength limit in microns.
    upper_wv (float): Upper wavelength limit in microns.

    Returns:
    int: Number of wavelengths.
    """
    iinf = int((lower_wv - 0.25) / 0.0025 + 1.5)
    isup = int((upper_wv - 0.25) / 0.0025 + 1.5)
    return isup - iinf + 1


if __name__ == "__main__":
    generate_lut_hdf5(
        parameter_bounds=[
            (0, 60),  # VZA
            (0, 20),  # VAA
            (0, 180),  # RAA
            (0, 0.5),  # AOT
            (0, 10),  # TCWV
        ],
        n_samples=50_000,
        hdf5_path="small_lut.h5",
        sampling_method="lhs",
    )
