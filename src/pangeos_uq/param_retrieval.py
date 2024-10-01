from .prosail_funcs import call_prosail
from .sixs_lut import LUTQuery
from .mcmc import generate_samples
import datetime as dt
import numpy as np
import importlib
from collections import namedtuple
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import seaborn as sns
from scipy import stats

from typing import Tuple

geometry = namedtuple("geometry", ["sza", "vza", "raa"])
atmospheric_parameters = namedtuple(
    "atmospheric_parameters", ["LUT", "AOT", "TCWV", "AOT_unc", "TCWV_unc"]
)
# We define min and max values for the parameters in a single place
# order is 'N', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown', 'psoil', 'rsoil'
min_vals = np.array([1.1, 5, 0.001, 0.001, 0.01, 0, 0, 0, 0])
max_vals = np.array([2.9, 100, 0.03, 0.06, 8, 90, 1, 1, 1])


with importlib.resources.path(
    "pangeos_uq.data", "archetype_statistics.npz"
) as f:
    ff = np.load(f)
    prior_means = {
        stage: ff[f"{stage}_mean"] for stage in ["early", "mid", "late", "all"]
    }
    prior_cov = {
        stage: ff[f"{stage}_covariance"]
        for stage in ["early", "mid", "late", "all"]
    }
    prior_inv_cov = {
        stage: ff[f"{stage}_inv_covariance"]
        for stage in ["early", "mid", "late", "all"]
    }
    param_names = list(ff["param_names"])


def uniform_prior(x: np.ndarray) -> float:
    uniform_prior.mean = (min_vals + max_vals) / 2.0
    uniform_prior.name = "uniform"
    if np.any(x < min_vals) or np.any(x > max_vals):
        return -np.inf
    else:
        # Product of uniform priors
        return -np.sum(np.log(max_vals - min_vals))


def normal_prior_func(
    mean: np.ndarray, inv_cov: np.ndarray, stage
) -> callable:
    def normal_prior(x: np.ndarray) -> float:
        # n, lai, ala, cab, cw, cm, cbrown, psoil, rsoil
        # prior from arc is N', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown'
        normal_prior_func.mean = (min_vals + max_vals) / 2.0
        normal_prior_func.name = stage
        if np.any(x < min_vals) or np.any(x > max_vals):
            return -np.inf
        else:
            # Ignore soil parameters
            diff = x[:-2] - mean
            return -0.5 * diff @ inv_cov @ diff

    return normal_prior


def get_priors():
    prior_funcs = {}
    prior_funcs["uniform"] = uniform_prior
    prior_funcs["uniform"].mean = (min_vals + max_vals) / 2.0

    for stage in ["early", "mid", "late", "all"]:
        prior_funcs[stage] = normal_prior_func(
            prior_means[stage], prior_inv_cov[stage], stage
        )
        prior_funcs[stage].mean = np.concatenate(
            (prior_means[stage], [0.5, 0.5])
        )
        prior_funcs[stage].name = stage
    return prior_funcs


def sample_prior_distribution(
    prior_name: str, n_samples: int = 3000
) -> pd.DataFrame:
    """Returns a DataFrame of samples from the selected prior distribution."""

    PARAMETER_NAMES = param_names + ["psoil", "rsoil"]
    vals = list(prior_means.keys()) + ["uniform"]
    # Validate the prior name
    if prior_name not in vals:
        raise ValueError(
            f"Invalid prior name '{prior_name}'. ",
            f"Choose from: {list(prior_means.keys())}",
        )

    if prior_name != "uniform":
        # Draw samples from the selected prior
        mean = prior_means[prior_name]
        cov = prior_cov[prior_name]
        samples = np.random.multivariate_normal(mean, cov, size=n_samples)
        samples_df = pd.DataFrame(samples, columns=param_names)
        # Add rsoil and psoil as uniform [0,1] random variables
        samples_df["rsoil"] = np.random.uniform(0, 1, size=n_samples)
        samples_df["psoil"] = np.random.uniform(0, 1, size=n_samples)
        return samples_df

    else:
        samples = np.random.uniform(
            low=min_vals, high=max_vals, size=(n_samples, len(min_vals))
        )
        samples_df = pd.DataFrame(samples, columns=PARAMETER_NAMES)
        return samples_df


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
        wvs: np.ndarray,
    ):
        """Initializes the BiophysicalParameterRetrieval class.

        Args:
            parameters (dict): Dictionary of biophysical parameters.
            srf (np.ndarray): Spectral response function.
            prior (callable): Prior distribution function for MCMC.
        """
        # parameters are stored in a dictionary, which we convert to a list
        # Note that Car is Cab/4, anthocyanin is 0, and hotspots is 0.01

        # N', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown
        # 0, 1, 6, 5, 7, 8, 4, 9, 10
        self.wvs = wvs
        self.x = [
            parameters["N"],  # 0
            parameters["Cab"],  # 1
            parameters["Cab"] * 0.25,  # 2
            0.0,  # 3
            parameters["Cbrown"],  # 4
            parameters["Cw"],  # 5
            parameters["Cm"],  # 6
            parameters["LAI"],  # 7
            parameters["ALA"],  # 8
            parameters["psoil"],  # 9
            parameters["rsoil"],  # 10
            0.01,
            parameters["sza"],
            parameters["vza"],
            parameters["raa"],
        ]
        self.parameters = parameters
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
        # N', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown'
        # posns = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10])
        posns = np.array([0, 1, 6, 5, 7, 8, 4, 9, 10])
        x_full[posns] = x
        prior_cost = self.prior(x)
        if np.isneginf(prior_cost):
            return -np.inf
        rho_canopy_sim = simulate_spectral_reflectance(x_full, cov_matrix=None)
        _, rho_canopy_sim = integrate_spectral_reflectance(
            rho_canopy_sim, self.srf
        )
        diff = self.boa_reflectance_mean - rho_canopy_sim
        obs_cost = -0.5 * diff.T @ self.inv_obs_cov @ diff
        cost = prior_cost + obs_cost
        return cost

    def propagate_to_toa(self) -> np.ndarray:
        """
        Propagates simulated reflectance to TOA using the look-up table (LUT).
        """
        toa_reflectance = propagate_to_toa(
            self.boa_reflectance_sim,
            self.geometry,
            self.atmospheric_parameters,
        )
        sigma = 2 * toa_reflectance * self.parameters["noise_unc"] / 100.0
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
        sigma_obs = (
            2.0 * self.toa_reflectance * self.parameters["noise_unc"] / 100
        )
        self.obs_cov = np.cov(
            self.boa_reflectance_ensemble - self.boa_reflectance_mean,
            rowvar=False,
        ) + np.diag(np.ones(len(self.wvc)) * sigma_obs**2)
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
        if hasattr(self.prior, "mean"):
            initial_value = np.array(self.prior.mean)
        else:
            initial_value = min_vals + 0.5 * (max_vals - min_vals)

        if len(initial_value) != 9:
            initial_value = np.concatenate((initial_value, [0.5, 0.5]))
        posns = np.array([0, 1, 6, 5, 7, 8, 4, 9, 10])
        # initial_value = np.array(self.x.copy())[posns]
        trans_vector = (max_vals - min_vals) / 100.0
        self.posterior_samples = np.array(
            list(
                generate_samples(
                    initial_value, n_samples, self.cost_function, trans_vector
                )
            )
        )
        parameter_names = param_names + ["psoil", "rsoil"]
        fname_out = f"{dt.datetime.now().isoformat()}_posterior_samples.npz"
        true_values = np.array(self.x.copy())[posns]
        np.savez(
            fname_out,
            posterior_samples=self.posterior_samples,
            parameter_names=parameter_names,
            true_values=true_values,
        )
        print(f"Saved posterior samples to {fname_out}")

    def plot_posterior(self, output_panel: widgets.Output) -> None:
        """
        Visualizes the results in two panels: Observations and Parameters.
        """
        prior_samples = sample_prior_distribution(
            self.prior.name, n_samples=300
        )
        posns = np.array([0, 1, 6, 5, 7, 8, 4, 9, 10])
        true_values = np.array(self.x.copy())[posns]
        x_mode, x_samples = modal_and_random_samples(
            self.posterior_samples, 100
        )
        this_x = np.array(self.x.copy())
        this_x[posns] = x_mode
        rho_canopy_sim_tmp = simulate_spectral_reflectance(
            this_x, cov_matrix=None
        )
        _, rho_canopy_sim_tmp = integrate_spectral_reflectance(
            rho_canopy_sim_tmp, self.srf
        )

        rho_canopy_sim = [rho_canopy_sim_tmp]
        for i in range(x_samples.shape[0]):
            this_x[posns] = x_mode
            rho_canopy_sim_tmp = simulate_spectral_reflectance(
                this_x, cov_matrix=None
            )
            _, rho_canopy_sim_tmp = integrate_spectral_reflectance(
                rho_canopy_sim_tmp, self.srf
            )

            rho_canopy_sim.append(rho_canopy_sim_tmp)

        visualize_panels(
            self.wvs,
            true_values,
            self.toa_reflectance,
            self.boa_reflectance_sim,
            self.boa_reflectance_ensemble,
            self.posterior_samples,
            rho_canopy_sim,
            output_panel,
            prior_samples=prior_samples,
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
    n_ensemble: int = 50,  # 12 samples are enough for everyone! ;)
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
        xap = xap + 0.005 * np.random.randn(len(xap))
        xb = xb + 0.005 * np.random.randn(len(xap))
        xc = xc + 0.005 * np.random.randn(len(xap))

        y = xap * toa_reflectance - xb
        rho_boa_corr = np.clip(y / (1.0 + xc * y), 0, 1)
        rho_boa_ensemble.append(rho_boa_corr)
    return np.array(rho_boa_ensemble)


def visualize_panels(
    wvs: np.ndarray,
    true_values: np.ndarray,
    toa_reflectance: np.ndarray,
    boa_reflectance_sim: np.ndarray,
    boa_reflectance_ensemble: np.ndarray,
    posterior_samples: np.ndarray,
    rho_canopy_sim: np.ndarray,
    output_panel: widgets.Output,
    prior_samples: np.ndarray | None = None,
) -> None:
    """
    Visualizes the observations and parameter distributions in two
    panels within a tabbed view.

    Args:
        toa_reflectance (np.ndarray): Top-of-atmosphere reflectance.
        boa_reflectance_ensemble (np.ndarray): Ensemble of BOA reflectance.
        posterior_samples (np.ndarray): Posterior parameter samples.
    """
    rho_canopy_sim = np.array(rho_canopy_sim)

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
        axes[0, 0].plot(wvs, toa_reflectance, label="TOA Reflectance")

        axes[0, 0].set_title("Top-of-Atmosphere Reflectance")
        axes[0, 0].set_xlabel("Wavelength [nm]")
        axes[0, 0].set_ylabel("Reflectance")
        axes[0, 0].legend()

        # 2. Plot BOA reflectance (mean of ensemble)
        boa_mean = np.mean(np.atleast_2d(boa_reflectance_ensemble), axis=0)
        axes[0, 1].plot(
            wvs,
            boa_reflectance_ensemble.T,
            color="orange",
            alpha=0.5,
        )
        axes[0, 1].plot(
            wvs,
            boa_reflectance_sim,
            color="green",
            label="Simulated BOA Reflectance",
        )

        axes[0, 1].set_title("Bottom-of-Atmosphere Reflectance")
        axes[0, 1].set_xlabel("Wavelength [nm]")
        axes[0, 1].set_ylabel("Reflectance")
        axes[0, 1].legend()

        # 3. Correlation matrix of BOA reflectance errors
        boa_errors = boa_reflectance_ensemble - boa_mean
        boa_corr = np.corrcoef(boa_errors.T)
        im = axes[1, 0].imshow(
            boa_corr, cmap="coolwarm", vmin=-1, vmax=1, interpolation="none"
        )
        axes[1, 0].set_title("Correlation Matrix of BOA Reflectance Errors")
        axes[1, 0].set_xticks(ticks=range(len(wvs)), labels=wvs, rotation=90)
        axes[1, 0].set_yticks(ticks=range(len(wvs)), labels=wvs)
        fig.colorbar(im, ax=axes[1, 0])

        # 4. Residuals (mean BOA reflectance - TOA reflectance)
        residuals = boa_mean - rho_canopy_sim[0]
        axes[1, 1].plot(
            wvs,
            residuals,
            label="Residuals (BOA meas - sim mod posterior)",
            color="red",
        )
        for i in range(1, len(rho_canopy_sim)):
            residuals = boa_mean - rho_canopy_sim[0]
            axes[1, 1].plot(
                wvs,
                residuals,
                label=None,
                lw=0.5,
                color="0.8",
            )
        # Make plot symmetric around 0
        (a, b) = axes[1, 1].get_ylim()

        lim = max(np.abs(a), np.abs(b))

        axes[1, 1].set_ylim(-lim - 0.1, lim + 0.1)
        ticks = np.linspace(-lim, lim, num=5)

        axes[1, 1].set_yticks(ticks)
        axes[1, 1].axhline(0, color="0.8", lw=1, linestyle="--")
        axes[1, 1].set_title("Residuals")
        axes[1, 1].set_xlabel("Wavelength [nm]")
        axes[1, 1].set_ylabel("Residual Reflectance")
        axes[1, 1].legend()

        plt.tight_layout()
        observations_panel = widgets.Output()
        with observations_panel:
            plt.show()

        fname_out = f"{dt.datetime.now().isoformat()}_observation_plots.pdf"
        fig.savefig(fname_out, dpi=300, bbox_inches="tight")

        return observations_panel

    # Create the Parameters Panel
    def create_parameters_panel():
        """
        Creates the parameters panel with posterior distributions, scatter
        plots, and correlations.

        Returns:
            widget: A widget containing the Parameters panel.
        """
        PARAMETERS_NAME = param_names + ["psoil", "rsoil"]
        posterior_df = pd.DataFrame(posterior_samples, columns=PARAMETERS_NAME)

        # fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
        g = sns.pairplot(
            posterior_df.iloc[-2000::5, :],
            kind="hist",
            diag_kind="kde",
        )
        for i, ax in enumerate(g.diag_axes):
            ax.axvline(
                true_values[i],
                color="0.8",
                lw=2,
            )
            sns.kdeplot(
                prior_samples.iloc[:, i], ax=ax, alpha=0.5, linestyle="--"
            )

        plt.tight_layout()

        parameters_panel = widgets.Output()
        with parameters_panel:
            plt.show()
        fname_out = f"{dt.datetime.now().isoformat()}_posterior_plots.pdf"
        g.savefig(fname_out, dpi=300, bbox_inches="tight")

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


def modal_and_random_samples(
    posterior_samples: np.ndarray, N: int
) -> Tuple[float, np.ndarray]:
    """
    Estimate the mode of the posterior distribution and sample N random values.

    Parameters:
    posterior_samples (np.ndarray): Array of posterior samples.
    N (int): Number of random samples to draw from the posterior.

    Returns:
    Tuple[float, np.ndarray]: The mode of the posterior and an array of
                                N random samples.
    """
    # Estimate the mode of the posterior distribution
    mode = stats.mode(posterior_samples, axis=0)[0]

    # Randomly sample N values from the posterior samples
    random_samples = posterior_samples[
        np.random.choice(posterior_samples.shape[0], N, replace=False), :
    ]
    return mode, random_samples
