from .prosail_funcs import call_prosail
from .sixs_lut import LUTQuery
from .mcmc import generate_samples
import numpy as np
from collections import namedtuple
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from scipy.stats import gaussian_kde
from typing import Tuple

geometry = namedtuple("geometry", ["sza", "vza", "raa"])
atmospheric_parameters = namedtuple(
    "atmospheric_parameters", ["LUT", "AOT", "TCWV", "AOT_unc", "TCWV_unc"]
)


class BiophysicalRetrieval:
    """
    This class manages the overall biophysical parameter retrieval simulation.

    The class basically:
    1. Simulates a canopy reflectance using PROSAIL.
    2. Adds some noise and integrates the reflectance over the spectral of
    (say a satellite) sensor.
    3. Propagates the simulated reflectance to the top-of-atmosphere (TOA)
    4. Corrects the TOA reflectance to bottom-of-atmosphere (BOA), but does so
    with uncertainty in AOT and TCWV (so it returns an ensemble of corrected
    BOA reflectances).
    5. Runs a Markov Chain Monte Carlo (MCMC) sampler to infer the biophysical
    parameters from the ensemble of BOA reflectances.
    6. Visualizes the results in two panels: Observations (spectgral plots,
    residuals, etc) and Parameters (prior and posterior pdfs/histograms).

    Attributes:
        x (np.ndarray): User-given parameters.
        AOT (float): Aerosol Optical Thickness (user-provided).
        TCWV (float): Total Column Water Vapor (user-provided).
        toa_reflectance (np.ndarray): Simulated top-of-atmosphere reflectance.
        boa_reflectance_ensemble (np.ndarray): Simulated ensemble of
                    bottom-of-atmosphere reflectance.
        prior (callable): User-defined prior distribution for parameters.
    """

    def __init__(
        self,
        parameters: dict,
        srf: np.ndarray,
        lut: LUTQuery,
        prior: callable,
    ):
        """Initializes the BiophysicalParameterRetrieval class.

        Args:
            parameters (dict): Dictionary of biophysical parameters.
            srf (np.ndarray): Spectral response function.
            prior (callable): Prior distribution function for MCMC.
        """
        # parameters are stored in a dictionary, which we convert to a list
        # Note that Car is Cab/4, anthocyanin is 0, and hotspots is 0.01
        self.x = [
            parameters["N"],
            parameters["Cab"],
            parameters["Cab"] * 0.25,
            0.0,
            parameters["Cbrown"],
            parameters["Cw"],
            parameters["Cm"],
            parameters["LAI"],
            parameters["ALA"],
            parameters["psoil"],
            parameters["rsoil"],
            0.01,
            parameters["sza"],
            parameters["vza"],
            parameters["raa"],
        ]
        # Store geometry into a named tuple
        self.geometry = geometry(
            parameters["sza"], parameters["vza"], parameters["raa"]
        )
        # Could put all this into a namedtuple, too...
        self.atmospheric_parameters = atmospheric_parameters(
            lut,
            parameters["AOT"],
            parameters["TCWV"],
            parameters["AOT_unc"],
            parameters["TCWV_unc"],
        )
        print(self.atmospheric_parameters)

        # Store the spectral response function
        self.srf = srf

        self.prior = prior

        # Initialize TOA and BOA reflectance properties
        self.toa_reflectance = None
        self.boa_reflectance_ensemble = None
        self.boa_reflectance_mean = None
        self.inv_obs_cov = None

    def simulate_reflectance(
        self, x: np.ndarray | None = None, cov_matrix: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Simulates spectral reflectance with optional noise.

        Args:
            cov_matrix (np.ndarray, optional): Covariance matrix for noise.

        Returns:
            np.ndarray: Simulated reflectance.
        """
        self.rho_simulated = simulate_spectral_reflectance(self.x, cov_matrix)
        self.wvc, self.boa_reflectance_sim = integrate_spectral_reflectance(
            self.rho_simulated, self.srf
        )

    def cost_function(self, x: np.ndarray) -> float:
        """
        Computes samples for the (log) posterior calculations.

        Args:
            x (np.ndarray): Biophysical parameters to invert.

        Returns:
            float: The computed cost value.
        """
        x_full = np.array(self.x.copy())
        posns = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10])
        x_full[posns] = x

        rho_canopy_sim = simulate_spectral_reflectance(x_full, cov_matrix=None)
        _, rho_canopy_sim = integrate_spectral_reflectance(
            rho_canopy_sim, self.srf
        )
        diff = self.boa_reflectance_mean - rho_canopy_sim
        obs_cost = -0.5 * diff.T @ self.inv_obs_cov @ diff
        prior_cost = self.prior(x)
        return prior_cost + obs_cost

    def propagate_to_toa(self) -> np.ndarray:
        """
        Propagates simulated reflectance to TOA using the look-up table (LUT).
        """
        toa_reflectance = propagate_to_toa(
            self.boa_reflectance_sim,
            self.geometry,
            self.atmospheric_parameters,
        )
        sigma = toa_reflectance * 0.03  # 3% error
        self.toa_reflectance = (
            toa_reflectance + np.random.randn(*toa_reflectance.shape) * sigma
        )

    def correct_to_boa(self) -> np.ndarray:
        """
        Corrects TOA reflectance to BOA using the LUT and error sampling.
        """
        self.boa_reflectance_ensemble = uncertain_correct_to_boa(
            self.toa_reflectance,
            self.geometry,
            self.atmospheric_parameters,
        )
        self.boa_reflectance_mean = self.boa_reflectance_ensemble.mean(axis=0)
        self.obs_cov = np.cov(
            self.boa_reflectance_ensemble - self.boa_reflectance_mean,
            rowvar=False,
        ) + np.diag(np.ones(len(self.wvc)) * 0.05**2)
        self.obs_corr = np.corrcoef(
            (self.boa_reflectance_ensemble - self.boa_reflectance_mean).T,
        )
        L = np.linalg.cholesky(self.obs_cov)
        # First solve L * y = I, where I is the identity matrix
        identity_matrix = np.eye(L.shape[0])
        inv_L = np.linalg.solve(L, identity_matrix)

        # Compute the inverse of the covariance matrix
        # inv(L * L^T) = inv(L^T) * inv(L)
        self.inv_obs_cov = inv_L.T @ inv_L

    def run_mcmc(self, n_samples: int) -> np.ndarray:
        """
        Runs the MCMC sampling based on the prior and the simulated data.
        """
        initial_value = self.prior.mean
        self.posterior_samples = np.array(
            list(
                generate_samples(initial_value, n_samples, self.cost_function)
            )
        )

    def plot_posterior(self, output_panel: widgets.Output) -> None:
        """
        Visualizes the results in two panels: Observations and Parameters.
        """
        testme = np.random.random((500, 10))
        visualize_panels(
            self.toa_reflectance,
            self.boa_reflectance_sim,
            self.boa_reflectance_ensemble,
            testme,
            output_panel,
        )


def simulate_spectral_reflectance(
    x: np.ndarray, cov_matrix: np.ndarray | None = None
) -> np.ndarray:
    """
    Simulates spectral reflectance with optional noise, assumed to be zero-mean
    Gaussian with a covariance matrix `cov_matrix`.

    Args:
        x (np.ndarray): Input parameters for simulation.
        cov_matrix (np.ndarray, optional): Covariance matrix for noise.

    Returns:
        np.ndarray: Simulated spectral reflectance with noise.
    """
    # Create noise
    if cov_matrix is not None:
        nnoise = np.random.multivariate_normal(np.zeros(2101), cov=cov_matrix)
    else:
        nnoise = 0.0
    # PROSAIL simulation
    return call_prosail(*x) + nnoise


def integrate_spectral_reflectance(
    reflectance: np.ndarray, srf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrates spectral reflectance over the spectral response function (SRF).

    Args:
        reflectance (np.ndarray): Simulated reflectance.
        srf (np.ndarray): Spectral response function for the sensor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Wavelength centers and integrated
                    reflectance.
    """
    # Make sure the SRF are normalised....
    srf = srf / np.sum(srf, axis=1, keepdims=True)
    wv = np.linspace(400, 2500, 2101)
    wvc = np.sum(wv * srf, axis=1) / np.sum(srf, axis=1)
    rho_canopy_sensor = np.nansum(srf[:, :] * reflectance[None, :], axis=1)
    return (wvc, rho_canopy_sensor)


def propagate_to_toa(
    reflectance: np.ndarray,
    geometry: geometry,
    atmos_params: atmospheric_parameters,
) -> np.ndarray:
    """
    Propagates simulated reflectance to TOA using the look-up table (LUT).

    Args:
        reflectance (np.ndarray): Simulated bottom-of-atmosphere reflectance.
        geometry (geometry): Viewing and solar geometry parameters.
        atmos_params (atmopsheric_parameters): Atmospheric parameters.
    Returns:
        np.ndarray: Simulated top-of-atmosphere reflectance.
    """
    # Atmospheric lookup
    to_toa_dict = atmos_params.LUT.query(
        geometry.vza,
        geometry.sza,
        geometry.raa,
        atmos_params.AOT,
        atmos_params.TCWV,
    )
    # Remove band 6
    xap = np.delete(to_toa_dict["xap"], 6)
    xb = np.delete(to_toa_dict["xb"], 6)
    xc = np.delete(to_toa_dict["xc"], 6)
    # Lambertian coupling correction using 6s coefficients
    numerator = reflectance + xb * (1 - reflectance * xc)
    denominator = xap * (1 - reflectance * xc)
    return numerator / denominator


def uncertain_correct_to_boa(
    toa_reflectance: np.ndarray,
    geometry: geometry,
    atmos_params: atmospheric_parameters,
    n_ensemble: int = 15,  # 12 samples are enough for everyone! ;)
) -> np.ndarray:
    """Corrects the TOA reflectance to BOA using LUT and uncertainty sampling.

    Args:
        toa_reflectance (np.ndarray): TOA reflectance.
        geometry (geometry): Viewing and solar geometry.
        atmos_params (atmospheric_parameters): Atmospheric parameters.
        n_ensemble (int): Number of ensemble samples.

    Returns:
        np.ndarray: Ensemble of BOA reflectance values.
    """
    ens_aot = (
        np.random.randn(n_ensemble) * atmos_params.AOT_unc + atmos_params.AOT
    )
    ens_tcwv = (
        np.random.randn(n_ensemble) * atmos_params.TCWV_unc + atmos_params.TCWV
    )
    rho_boa_ensemble = []
    for aot, tcwv in zip(ens_aot, ens_tcwv):
        atmos_paramsx = atmos_params.LUT.query(
            geometry.vza, geometry.sza, geometry.raa, aot, tcwv
        )
        # Remove band 6
        xap = np.delete(atmos_paramsx["xap"], 6)
        xb = np.delete(atmos_paramsx["xb"], 6)
        xc = np.delete(atmos_paramsx["xc"], 6)
        y = xap * toa_reflectance - xb
        rho_boa_corr = np.clip(y / (1.0 + xc * y), 0, 1)
        rho_boa_ensemble.append(rho_boa_corr)
    return np.array(rho_boa_ensemble)


def visualize_panels(
    toa_reflectance: np.ndarray,
    boa_reflectance_sim: np.ndarray,
    boa_reflectance_ensemble: np.ndarray,
    posterior_samples: np.ndarray,
    output_panel: widgets.Output,
) -> None:
    """
    Visualizes the observations and parameter distributions in two
    panels within a tabbed view.

    Args:
        toa_reflectance (np.ndarray): Top-of-atmosphere reflectance.
        boa_reflectance_ensemble (np.ndarray): Ensemble of BOA reflectance.
        posterior_samples (np.ndarray): Posterior parameter samples.
    """

    # Create the Observations Panel
    def create_observations_panel():
        """
        Creates the observations panel with TOA reflectance, BOA reflectance,
        correlation matrix of BOA reflectance errors, and residuals.

        Returns:
            widget: A widget containing the Observations panel.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # 1. Plot TOA reflectance
        axes[0, 0].plot(toa_reflectance, label="TOA Reflectance")

        axes[0, 0].set_title("Top-of-Atmosphere Reflectance")
        axes[0, 0].set_xlabel("Band")
        axes[0, 0].set_ylabel("Reflectance")
        axes[0, 0].legend()

        # 2. Plot BOA reflectance (mean of ensemble)
        boa_mean = np.mean(np.atleast_2d(boa_reflectance_ensemble), axis=0)
        axes[0, 1].plot(
            boa_reflectance_ensemble.T,
            color="orange",
            alpha=0.5,
        )
        axes[0, 1].plot(
            boa_reflectance_sim,
            color="green",
            label="Simulated BOA Reflectance",
        )

        axes[0, 1].set_title("Bottom-of-Atmosphere Reflectance")
        axes[0, 1].set_xlabel("Band")
        axes[0, 1].set_ylabel("Reflectance")
        axes[0, 1].legend()

        # 3. Correlation matrix of BOA reflectance errors
        boa_errors = boa_reflectance_ensemble - boa_mean
        boa_corr = np.corrcoef(boa_errors.T)
        im = axes[1, 0].imshow(
            boa_corr, cmap="coolwarm", vmin=-1, vmax=1, interpolation="none"
        )
        axes[1, 0].set_title("Correlation Matrix of BOA Reflectance Errors")
        fig.colorbar(im, ax=axes[1, 0])

        # 4. Residuals (mean BOA reflectance - TOA reflectance)
        residuals = boa_mean - toa_reflectance
        axes[1, 1].plot(residuals, label="Residuals (BOA - TOA)", color="red")
        axes[1, 1].set_title("Residuals")
        axes[1, 1].set_xlabel("Band")
        axes[1, 1].set_ylabel("Residual Reflectance")
        axes[1, 1].legend()

        plt.tight_layout()
        observations_panel = widgets.Output()
        with observations_panel:
            plt.show()

        return observations_panel

    # Create the Parameters Panel
    def create_parameters_panel():
        """
        Creates the parameters panel with posterior distributions, scatter
        plots, and correlations.

        Returns:
            widget: A widget containing the Parameters panel.
        """
        n_params = posterior_samples.shape[1]
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))

        # Loop through each parameter and create diagonal histograms
        # (posterior distributions) and off-diagonal scatter plots
        for i in range(n_params):
            for j in range(n_params):
                if i == j:
                    # Diagonal: Histogram and KDE of posterior samples
                    # for each parameter
                    axes[i, j].hist(
                        posterior_samples[:, i],
                        bins=20,
                        density=True,
                        alpha=0.5,
                    )
                    kde = gaussian_kde(posterior_samples[:, i])
                    x_vals = np.linspace(
                        np.min(posterior_samples[:, i]),
                        np.max(posterior_samples[:, i]),
                        100,
                    )
                    axes[i, j].plot(
                        x_vals,
                        kde(x_vals),
                        label="Posterior KDE",
                        color="blue",
                    )
                    axes[i, j].set_title(f"Posterior Distribution Param {i+1}")
                else:
                    # Off-diagonal: Scatter plot of posterior samples
                    axes[i, j].scatter(
                        posterior_samples[:, i],
                        posterior_samples[:, j],
                        alpha=0.5,
                        s=5,
                    )
                    axes[i, j].set_title(f"Scatter Param {i+1} vs Param {j+1}")
                if i == n_params - 1:
                    axes[i, j].set_xlabel(f"Param {j+1}")
                if j == 0:
                    axes[i, j].set_ylabel(f"Param {i+1}")

        plt.tight_layout()
        parameters_panel = widgets.Output()
        with parameters_panel:
            plt.show()

        return parameters_panel

    # Create the two panels as separate tabs
    observations_tab = create_observations_panel()
    parameters_tab = create_parameters_panel()

    # Create the tabbed interface
    tab = widgets.Tab([observations_tab, parameters_tab])
    tab.set_title(0, "Observations")
    tab.set_title(1, "Parameters")
    # Remove prior tabs
    output_panel.clear_output()
    # Add the tab widget to the output panel
    with output_panel:
        display(tab)
