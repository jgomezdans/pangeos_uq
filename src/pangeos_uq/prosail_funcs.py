#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy.optimize as opt
import scipy.linalg as sl
import ipywidgets as widgets
from IPython.display import display, clear_output

from typing import List

plt.ion()

try:
    import prosail
except ImportError:
    print("We need to have the prosail Python bindings installed")
    print("Available from http://github.com/jgomezdans/prosail")

__author__ = "J Gomez-Dans"
__version__ = "1.0 (09.12.2019)"
__email__ = "j.gomez-dans@ucl.ac.uk"


def pretty_axes(ax=None):
    """This function takes an axis object ``ax``, and makes it purrty.
    Namely, it removes top and left axis & puts the ticks at the
    bottom and the left"""
    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    loc = plt.MaxNLocator(6)
    ax.yaxis.set_major_locator(loc)
    # Make text output also look sane...
    np.set_printoptions(suppress=True, precision=3)

    ax.tick_params(
        axis="both",
        which="both",
        bottom="on",
        top="off",
        labelbottom="on",
        left="on",
        right="off",
        labelleft="on",
    )


def call_prospect_5(n, cab, car, cbrown, cw, cm):
    """A wrapper for PROSPECT that does away with some numerical
    instabilities by interpolating over spectrally."""
    x, r, t = prosail.run_prospect(
        n, cab, car, cbrown, cw, cm, prospect_version="5"
    )
    rpass = np.isfinite(r)
    tpass = np.isfinite(t)
    ri = np.interp(x, x[rpass], r[rpass])
    ti = np.interp(x, x[tpass], t[tpass])
    return np.c_[ri, ti]


def call_prospect_d(n, cab, car, cbrown, cw, cm, ant):
    """A wrapper for PROSPECT that does away with some numerical
    instabilities by interpolating over spectrally."""
    x, r, t = prosail.run_prospect(
        n, cab, car, cbrown, cw, cm, ant=ant, prospect_version="D"
    )
    rpass = np.isfinite(r)
    tpass = np.isfinite(t)
    ri = np.interp(x, x[rpass], r[rpass])
    ti = np.interp(x, x[tpass], t[tpass])
    return np.c_[ri, ti]


def prospect_sensitivity_ssa(
    leaf_n, leaf_cab, leaf_car, leaf_cbrown, leaf_cw, leaf_cm
):
    """Local approximation (around ``x0``) of the sensitivity of the PROSPECT
    model to each input parameter. You can set up the point around which the
    sensitivity is calculated (``x0``), the value of the finite difference
    (``epsilon``).

    Parameters
    -------------
    x0: array
        A size 6 array, with the center point around which the partial
        derivatives will be calculated. The different elements are in PROSPECT
        order, e.g. N, Cab, Car, Cbrown, Cw, and Cm
    epsilon: float
        The finite difference amount. If you get NaN, make it a bit larger.

    """

    # Clear previous output
    clear_output(wait=True)

    epsilon = 1e-5
    x0 = np.array([leaf_n, leaf_cab, leaf_car, leaf_cbrown, leaf_cw, leaf_cm])
    sensitivity = np.zeros((7, 2101))
    span = np.array([1.5, 80.0, 20.0, 1.0, 0.0439 - 0.0043, 0.0152 - 0.0017])
    rr = call_prospect_5(*x0)
    r0 = rr.sum(axis=1)
    for i in range(6):
        xp = x0 * 1.0
        xp[i] = x0[i] + epsilon * span[i]
        r1 = call_prospect_5(*xp).sum(axis=1)
        sensitivity[i, :] = (r0 - r1) / epsilon

    # Plotting the sensitivity
    wv = np.arange(400, 2501)

    # First figure
    fig1, axs = plt.subplots(
        figsize=(9.5, 4.5), nrows=2, ncols=3, sharex=True, sharey=True
    )
    axs = axs.flatten()
    for i, input_parameter in enumerate(
        ["n", "cab", "car", "cbrown", "cw", "cm"]
    ):
        axs[i].plot(wv, sensitivity[i, :], "-", lw=2)
        axs[i].set_title(input_parameter)
        axs[i].set_xlim(400, 2500)
        if i in [0, 3]:
            axs[i].set_ylabel(r"$\partial f/\partial \mathbf{x}$")
        if i > 2:
            axs[i].set_xlabel("Wavelength [nm]")
    fig1.tight_layout()
    plt.close(fig1)
    # Second figure
    fig2, ax2 = plt.subplots(figsize=(6.47, 4))
    for i, input_parameter in enumerate(
        ["n", "cab", "car", "cbrown", "cw", "cm"]
    ):
        ax2.plot(wv, sensitivity[i, :], lw=2, label=input_parameter)

    ax2.set_xlim(400, 2500)
    ax2.set_ylabel(r"$\partial f/\partial \mathbf{x}$")
    ax2.set_xlabel("Wavelength [nm]")
    ax2.legend(loc="best", frameon=False, fontsize=10)
    fig2.tight_layout()
    plt.close(fig2)
    # Third figure
    fig3, ax3 = plt.subplots(figsize=(6.47, 4))
    ax3.plot(wv, rr[:, 0], lw=2, label="Leaf Reflectance")
    ax3.plot(wv, rr[:, 1], lw=2, label="Leaf Transmittance")
    ax3.plot(wv, rr.sum(axis=1), lw=2, label=r"Leaf $\omega$")

    ax3.set_xlim(400, 2500)
    ax3.set_ylabel(r"$\rho,\;\tau\;\omega$")
    ax3.set_xlabel("Wavelength [nm]")
    ax3.legend(loc="best", frameon=False, fontsize=10)
    fig3.tight_layout()
    plt.close(fig3)

    display(fig1)  # Display fig3 below
    display(fig2)  # Display second figure
    display(fig3)  # Display third figure below


def sensitivity_prospect():
    epsilon = 1e-5  # noqa

    _ = widgets.interact_manual(
        prospect_sensitivity_ssa,
        leaf_n=widgets.FloatSlider(min=1, max=3, value=1.8),
        leaf_cab=widgets.FloatSlider(min=0, max=100, value=40.0),
        leaf_car=widgets.FloatSlider(min=0, max=25, value=10.0),
        leaf_cbrown=widgets.FloatSlider(min=0, max=1, value=0.1),
        leaf_cw=widgets.FloatLogSlider(min=-3, max=-1.0, value=0.0133),
        leaf_cm=widgets.FloatLogSlider(min=-3, max=-1.0, value=0.0053),
    )


def prospect_sensitivity_n(
    n_samples=20,
    spaces=10,
    minvals={"n": 1, "cab": 10, "car": 0, "cbrown": 0, "cw": 0.0, "cm": 0.0},
    maxvals={
        "n": 3.5,
        "cab": 280,
        "car": 200,
        "cbrown": 1,
        "cw": 0.4,
        "cm": 0.5,
    },
    do_plots=True,
):
    """Visualise the effect of the PROSPECT parameter $N$, the number of
    layers. The aim of this is for you to view how the response of the leaves
    changes as a function of this parameter, while the other
    parameters are modified randomly (within the parameter spans provided).

    Parameters
    ------------
    n_samples: int
        The number of random model realisations for each bin of N.
    spaces: int
        The number of bins in which to divide the span of N
    minvals: dict
        A dictionary of parameter minimum values in the adequate units.
    maxvals: dict
        A dictionary of parameter minimum values in the adequate units.
    do_plots: bool
        Whether to do plots or not

    Returns
    -----------
    xleafn, reflectance, transmittance: The value of N used, as well as the
            reflectance and transmittance.
    """

    xbase = np.zeros(6)
    span = np.zeros(6)
    for i, param in enumerate(["n", "cab", "car", "cbrown", "cw", "cm"]):
        xbase[i] = minvals[param]
        span[i] = (maxvals[param] - minvals[param]) * 0.5
    spaces = 10
    t = []
    r = []
    xleafn = []
    delta = (maxvals["n"] - minvals["n"]) / spaces
    for s in np.linspace(minvals["n"], maxvals["n"], num=spaces):
        for i in range(n_samples):
            x = xbase + span * np.random.rand(6)
            tx = s + delta * (np.random.rand() - 0.5)
            x[0] = tx
            rr = call_prospect_5(*x)
            r.append(rr[:, 0])
            t.append(rr[:, 1])
            xleafn.append(tx)
    r = np.array(r)
    t = np.array(t)
    xleafn = np.array(xleafn)
    return xleafn, r, t


def visualise_effect_nlayers(
    n_samples: int = 50,
    bands: List[int] = [
        560,
        665,
        705,
        740,
        783,
        800,
        842,
        865,
        945,
        1121,
        1610,
        2090,
    ],
) -> None:
    """
    Visualises the effect of varying leaf structure (N layers) on the ratio
    of reflectance to transmittance across different wavelength bands. Fits a
    linear regression to the ratio for each band.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples for the leaf structure parameter N. Default is 50.
    bands : List[int], optional
        List of wavelength bands (in nm) to visualize. Default includes 12
        common bands.

    Returns
    -------
    None
        Displays the plot with subplots for each band, showing the
        reflectance/transmittance ratio vs. leaf structure.
    """
    # Generate wavelength array and simulate data for reflectance and
    # transmittance
    wv = np.arange(400, 2501)
    xleafn, refl, trans = prospect_sensitivity_n(n_samples)

    # Create the subplots: 4 rows, 3 columns
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(8, 8), sharex=True)
    axs = axs.flatten()

    # Loop through each band and plot the ratio of reflectance to transmittance
    for i, band in enumerate(bands):
        # Calculate reflectance/transmittance ratio for the current band
        ratio = np.mean(refl[:, wv == band] / trans[:, wv == band], axis=1)
        ratio_clipped = np.clip(ratio, 0, 4)

        # Scatter plot of the ratio against the leaf structure parameter (N)
        axs[i].plot(
            xleafn, ratio_clipped, "o", markerfacecolor="none", label="Data"
        )

        # Perform linear regression (1st degree polynomial) and plot the
        # regression line
        p = np.polyfit(xleafn, ratio_clipped, 1)
        axs[i].plot(xleafn, np.polyval(p, xleafn), "-", label="Fit")

        # Set plot title and axis limits
        axs[i].set_title(f"Band {band} nm {p[0]:.2f}")
        axs[i].set_ylim(0, 4)
        axs[i].set_ylabel(r"$\rho(N)/\tau(N)$", fontsize=9)
        axs[i].set_xlabel(r"$N$ layers", fontsize=9)
    #        axs[i].legend()
    axs[-1].set_visible(False)
    # Adjust layout to prevent overlap of subplots
    fig.tight_layout()

    # Display the final plot
    plt.show()


def veg_index_playground():
    """Sets up an interactive widget to explore vegetation index calculations
    with PROSPECT model."""
    np.seterr(divide="ignore", invalid="ignore")

    # Create an output widget to update plots
    output_plot = widgets.Output()

    _ = widgets.interact_manual(
        do_index,
        mode=widgets.Dropdown(options=["ssa", "refl", "trans"], value="ssa"),
        sweep_param=widgets.Dropdown(
            options=["n", "cab", "car", "cbrown", "cw", "cm"], value="cw"
        ),
        band1=widgets.IntSlider(min=400, max=2500, value=1610),
        band2=widgets.IntSlider(min=400, max=2500, value=865),
        bwidth1=widgets.IntSlider(min=1, max=100, value=5),
        bwidth2=widgets.IntSlider(min=1, max=100, value=5),
        n_samples=widgets.fixed(50),
        spaces=widgets.fixed(25),
        minvals=widgets.fixed(
            {"n": 0, "cab": 0, "car": 0, "cbrown": 0, "cw": 0.0, "cm": 0.0}
        ),
        maxvals=widgets.fixed(
            {
                "n": 3.5,
                "cab": 80,
                "car": 200,
                "cbrown": 1,
                "cw": 0.4,
                "cm": 0.5,
            }
        ),
        do_plots=widgets.fixed(True),
        output_plot=widgets.fixed(
            output_plot
        ),  # Pass the output_plot as a parameter
    )

    # Display the output widget where plots will appear
    display(output_plot)


def do_index(
    mode="refl",
    sweep_param="cab",
    band1=650,
    band2=850,
    bwidth1=5,
    bwidth2=5,
    n_samples=1,
    spaces=20,
    minvals={"n": 0, "cab": 0, "car": 0, "cbrown": 0, "cw": 0.0, "cm": 0.0},
    maxvals={
        "n": 3.5,
        "cab": 80,
        "car": 200,
        "cbrown": 1,
        "cw": 0.4,
        "cm": 0.5,
    },
    do_plots=True,
    output_plot=None,  # Output widget passed as an argument
):
    """
    Runs PROSPECT model for a specific parameter and calculates a
    normalized vegetation index.
    """
    if output_plot is not None:
        # Clear previous output from the output widget
        with output_plot:
            output_plot.clear_output(wait=True)

            xbase = np.zeros(6)
            span = np.zeros(6)
            pars = ["n", "cab", "car", "cbrown", "cw", "cm"]
            for i, param in enumerate(pars):
                xbase[i] = minvals[param]
                span[i] = maxvals[param] - minvals[param]

            r = []
            t = []
            wv = np.arange(400, 2501)
            ss = []
            delta = (maxvals[sweep_param] - minvals[sweep_param]) / spaces
            for s in np.linspace(
                minvals[sweep_param], maxvals[sweep_param], num=spaces
            ):
                for i in range(n_samples):
                    x = xbase + span * np.random.rand(6)
                    tx = s + delta * (np.random.rand() - 0.5)
                    x[pars.index(sweep_param)] = tx
                    rr = call_prospect_5(
                        *x
                    )  # Call PROSPECT model with parameters
                    r.append(rr[:, 0])
                    t.append(rr[:, 1])
                    ss.append(tx)
            r = np.array(r)
            t = np.array(t)
            s = np.array(ss)

            # Select wavelengths within the bands
            isel1 = np.logical_and(
                wv >= band1 - bwidth1, wv <= band1 + bwidth1
            )
            isel2 = np.logical_and(
                wv >= band2 - bwidth2, wv <= band2 + bwidth2
            )

            # Calculate the normalized vegetation index based on the
            # selected mode
            if mode == "refl":
                b1 = r[:, isel1]
                b2 = r[:, isel2]
            elif mode == "trans":
                b1 = t[:, isel1]
                b2 = t[:, isel2]
            elif mode == "ssa":
                b1 = r[:, isel1] + t[:, isel1]
                b2 = r[:, isel2] + t[:, isel2]
            else:
                raise ValueError("`mode` can only 'refl', 'trans' or 'ssa'!")

            # Normalized vegetation index calculation
            vi = (b2.mean(axis=1) - b1.mean(axis=1)) / (
                b2.mean(axis=1) + b1.mean(axis=1)
            )

            # Plot the results if required
            if do_plots:
                fig = plt.figure(figsize=(6.47, 4))
                ax = plt.gca()
                ax.plot(vi, s, "o", markerfacecolor="none")
                ax.set_ylabel(sweep_param)
                ax.set_xlabel("VI")
                ax.set_title(f"B1: {band1}, B2: {band2}")
                fig.tight_layout()
                plt.show()

            return s, vi


def red_edge_playground():
    """Sets up an interactive widget to explore red edge calculations
    with PROSPECT model."""
    np.seterr(divide="ignore", invalid="ignore")

    # Create an output widget to update plots
    output_plot = widgets.Output()

    _ = widgets.interact_manual(
        red_edge,
        mode=widgets.Dropdown(options=["ssa", "refl", "trans"], value="refl"),
        sweep_param=widgets.Dropdown(
            options=["n", "cab", "car", "cbrown", "cw", "cm"], value="cab"
        ),
        band1=widgets.IntSlider(min=400, max=2500, value=670),
        band2=widgets.IntSlider(min=400, max=2500, value=780),
        n_samples=widgets.fixed(150),
        spaces=widgets.fixed(7),
        minvals=widgets.fixed(
            {"n": 0.8, "cab": 0, "car": 0, "cbrown": 0, "cw": 0.0, "cm": 0.0}
        ),
        maxvals=widgets.fixed(
            {
                "n": 3.5,
                "cab": 80,
                "car": 200,
                "cbrown": 1,
                "cw": 0.4,
                "cm": 0.5,
            }
        ),
        do_plots=widgets.fixed(True),
        output_plot=widgets.fixed(
            output_plot
        ),  # Pass the output_plot as a parameter
    )

    # Display the output widget where plots will appear
    display(output_plot)


def red_edge(
    mode="refl",
    sweep_param="cab",
    band1=670,
    band2=780,
    n_samples=150,
    spaces=7,
    minvals={
        "n": 1.0,
        "cab": 15,
        "car": 10,
        "cbrown": 0,
        "cw": 0.0,
        "cm": 0.0,
    },
    maxvals={
        "n": 2.5,
        "cab": 120,
        "car": 20,
        "cbrown": 1,
        "cw": 0.4,
        "cm": 0.5,
    },
    do_plots=True,
    output_plot=None,  # Output widget passed as an argument
):
    """
    Runs PROSPECT model for a specific parameter and calculates the red edge
    turning point.
    """
    if output_plot is not None:
        # Clear previous output from the output widget
        with output_plot:
            output_plot.clear_output(wait=True)

            xbase = np.zeros(6)
            span = np.zeros(6)
            pars = ["n", "cab", "car", "cbrown", "cw", "cm"]
            for i, param in enumerate(pars):
                xbase[i] = minvals[param]
                span[i] = maxvals[param] - minvals[param]

            wv = np.arange(400, 2501)
            isel = np.logical_and(wv >= band1, wv <= band2)
            wv = np.arange(band1, band2 + 1)

            ss = []
            delta = (maxvals[sweep_param] - minvals[sweep_param]) / spaces

            red_edge = []
            fig, axs = plt.subplots(
                nrows=1, ncols=2, figsize=(6.47 * 1.62, 6.47)
            )
            for ii, s in enumerate(
                np.linspace(
                    minvals[sweep_param], maxvals[sweep_param], num=spaces
                )
            ):
                yy = np.zeros((isel.sum(), n_samples))
                for i in range(n_samples):
                    x = xbase + span * np.random.rand(6)
                    x = np.array([1.5, 40.0, 5.0, 0.0, 0.0113, 0.0053])
                    tx = s + delta * (np.random.rand() - 0.5)
                    x[pars.index(sweep_param)] = tx
                    rr = call_prospect_5(*x)  # Call the PROSPECT model
                    if mode == "refl":
                        y = rr[isel, 0]
                    elif mode == "trans":
                        y = rr[isel, 1]
                    elif mode == "ssa":
                        y = rr[isel, 0] + rr[isel, 1]
                    yy[:, i] = y

                # Fit a cubic curve to the data
                pp, cov = curve_fit(
                    lambda x, a, b, c, d: a
                    + b * x
                    + c * x * x
                    + d * x * x * x,
                    wv,
                    yy.mean(axis=1),
                )
                ss.append(s)
                red_edge.append(-pp[2] / (3 * pp[3]))  # Red edge turning point
                if do_plots:
                    iloc = np.argmin(np.abs(wv - -pp[2] / (3 * pp[3])))
                    if np.abs(wv - -pp[2] / (3 * pp[3]))[iloc] <= 5:
                        axs[0].plot(wv[iloc], yy.mean(axis=1)[iloc], "ko")
                    axs[0].plot(wv, yy.mean(axis=1), "-")
                    axs[0].set_xlabel("Wavelength [nm]")
                    axs[0].set_ylabel(mode)

            if do_plots:
                axs[1].plot(ss, red_edge, "o-")
                axs[1].set_xlabel(sweep_param)
                axs[1].set_ylabel(r"$\lambda_{red\, edge}\;\left[nm\right]$")
                axs[1].set_ylim(band1 - band1 * 0.1, band2 + band2 * 0.1)
            fig.tight_layout()
            plt.show()  # Display the plot

            return np.array(ss), np.array(red_edge)


def read_lopex_sample(sample_no=23, do_plot=True, db_path="../data/LOPEX93/"):
    """Reads a sample from the LOPEX'93 database, and optionally, plots the
    reflectance and transmittance. By default, there's an oak leaf.

    Parameters
    ------------
    sample_no: int
        Integer to the sample number in the database
    do_plot: bool
        Whether to do the plot or not
    db_path: str
        A path for the database

    Returns the reflectance and transmittance associated with the sample,
    typically there are 5 replicates, so two arrays of size [5, 2101], over
    400 to 2500 nm.
    """
    if sample_no < 1 or sample_no > 116:
        raise ValueError("There are only 116 samples in this database")
    try:
        refl = np.loadtxt("%s/refl.%03d.dat" % (db_path, sample_no)).reshape(
            (5, 2101)
        )
    except IOError:
        print("This sample is not available in the database!")
        read_lopex_sample.data = (None, None)
        return

    trans = np.loadtxt("%s/trans.%03d.dat" % (db_path, sample_no)).reshape(
        (5, 2101)
    )

    wv = np.arange(400, 2501)

    if do_plot:
        plt.close("all")
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 4))
        for nsample in range(5):
            axs[0].plot(wv, refl[nsample, :])
            axs[1].plot(wv, trans[nsample, :])
        axs[0].set_ylabel("Reflectance")
        axs[1].set_ylabel("Transmittance")
        axs[1].set_xlabel("Wavelength [nm]")
        fig.tight_layout()
    read_lopex_sample.data = (refl, trans)
    return


def the_cost_function(x, refl, trans):
    """A standard cost function. Returns the cost for an input vector ``x``"""
    retval = call_prospect_5(*x)
    if refl is None:
        cost_refl = 0.0
    else:
        cost_refl = (refl - retval[:, 0]) ** 2
    if trans is None:
        cost_trans = 0.0
    else:
        cost_trans = (trans - retval[:, 1]) ** 2
    return np.sum(cost_refl + cost_trans)


def calculate_mean_and_covariance(
    refl: np.ndarray, trans: np.ndarray
) -> tuple:
    """
    Calculate the mean and combined covariance matrix for reflectance and
    transmittance replicates.

    Parameters:
    refl : np.ndarray
        Reflectance array of shape (5, 2101), where 5 is the number of
        replicates.
    trans : np.ndarray
        Transmittance array of shape (5, 2101), where 5 is the number
        of replicates.

    Returns:
    tuple:
        mean_refl : np.ndarray
            Mean reflectance across replicates, shape (2101,).
        mean_trans : np.ndarray
            Mean transmittance across replicates, shape (2101,).
        combined_cov : np.ndarray
            Combined covariance matrix, shape (4202, 4202), capturing
            covariances between
            reflectance and transmittance, as well as their internal
            correlations.
        inv_cov : np.ndarray
            Inverse of the combined covariance matrix, shape (4202, 4202). This
            is an approximation often calculated using SVD.
        correlation_matrix : np.ndarray
            The correlation matrix, shape (4202, 4202), with the full
            correlation structure between reflectance and
            transmittance measurements.
    """
    # Calculate the mean reflectance and transmittance
    mean_refl = np.mean(refl, axis=0)  # Shape: (2101,)
    mean_trans = np.mean(trans, axis=0)  # Shape: (2101,)

    # Stack reflectance and transmittance replicates into a single matrix
    stacked_data = np.hstack((refl, trans))  # Shape: (5, 4202)

    #  Calculate the covariance matrix (combined for reflectance
    # and transmittance)
    combined_cov = np.cov(
        stacked_data - np.hstack((mean_refl, mean_trans)), rowvar=False
    )  # Shape: (4202, 4202)
    # Perform Cholesky decomposition to get lower triangular matrix L
    try:
        L = np.linalg.cholesky(combined_cov)
        # Invert the covariance matrix using forward and backward substitution
        # L * L^T = combined_cov, so inv(combined_cov) = inv(L^T) * inv(L)
        # First solve for inv(L)
        inv_L = sl.solve_triangular(L, np.eye(L.shape[0]), lower=True)
        inv_cov = inv_L.T @ inv_L
    except np.linalg.LinAlgError:
        print("⚠️ Cholesky decomposition failed. Using SVD instead....⚠️")
        print("⚠️ This will take a while...⚠️")

        U, S, Vt = np.linalg.svd(combined_cov)
        # Invert the diagonal matrix of singular values, ensuring small
        # values are thresholded
        S_inv = np.diag([1 / s if s > 1e-8 else 0 for s in S])
        inv_cov = Vt.T @ S_inv @ U.T

    # Then, calculate inv(combined_cov) = inv(L^T) @ inv(L)

    # Correlation matrix:
    correlation_matrix = combined_cov / np.outer(
        np.sqrt(np.diag(combined_cov)), np.sqrt(np.diag(combined_cov))
    )

    return mean_refl, mean_trans, combined_cov, inv_cov, correlation_matrix


def the_cost_function_covariance(
    x: np.ndarray, refl: np.ndarray, trans: np.ndarray, inv_cov: np.ndarray
) -> float:
    """
    Cost function that computes the Mahalanobis distance between modeled
    and observed reflectance and transmittance, accounting for their
    uncertainties via an inverse covariance matrix.

    Parameters:
    x : np.ndarray
        Input vector of parameters to invert the model.
    refl : np.ndarray
        Measured reflectance vector (2101 elements).
    trans : np.ndarray
        Measured transmittance vector (2101 elements).
    inv_cov : np.ndarray
        Inverse covariance matrix, should be of shape (4202, 4202) if stacked
        reflectance and transmittance.

    Returns:
    float
        The cost associated with the given parameters.
    """
    # Run the model with the parameters in x
    retval = call_prospect_5(*x)

    # Extract modeled reflectance and transmittance
    modeled_refl = retval[:, 0]  # Reflectance model output
    modeled_trans = retval[:, 1]  # Transmittance model output

    # Stack reflectance and transmittance vectors
    if refl is None:
        refl_diff = np.zeros_like(modeled_refl)
    else:
        refl_diff = refl - modeled_refl

    if trans is None:
        trans_diff = np.zeros_like(modeled_trans)
    else:
        trans_diff = trans - modeled_trans

    # Create the residual vector (stacked reflectance and
    # transmittance differences)
    residual = np.hstack((refl_diff, trans_diff))

    # Compute the Mahalanobis distance using the inverse covariance matrix
    return -0.5 * residual.T @ inv_cov @ residual


def optimise_random_starts(
    refl: np.ndarray,
    trans: np.ndarray,
    cost_function: callable = the_cost_function,
    n_tries: int = 10,
    lobound: np.ndarray = np.array([0.8, 0, 0, 0, 0.0043, 0.0017]),
    hibound: np.ndarray = np.array([2.8, 80, 20, 1, 0.0439, 0.0152]),
    verbose: bool = True,
    do_plots: bool = True,
) -> tuple:
    """Minimise a cost function with either reflectance and/or
    transmittance measurements. To account for the local nature of the gradient
    descent minimiser, the optimisation is started from 10 (or `n_tries`)
    random points within the parameter boundaries. Additional options let you
    change the parameter boundaries,as well as the amount of information it
    reports back, and comparison "goodness of fit" plot.

    Parameters
    ------------
    refl: arr
        A reflectance array measured between 400 and 2500 nm at 1nm intervals
    trans: arr
        A transmittance array measured between 400 and 2500 nm at 1nm intervals
    lobound: arr
        A 6-element array with lower boundaries for the 6 PROSPECT5b parameters
    hibound: arr
        A 6-element array with upper boundaries for the 6 PROSPECT5b parameters
    verbose: bool
        Be verbose
    do_plots: bool
        Do a plot

    Returns
    ----------
    A pandas Dataframe with the minimisation solution per try (plus mean and
    standard deviation) and forward modelled reflectances and transmittances.
    """
    wv = np.arange(400, 2501)

    bounds = [[lobound[i], hibound[i]] for i in range(6)]

    store = []

    for tries in range(n_tries):
        x0 = lobound + np.random.rand(6) * (hibound - lobound)
        retval = opt.minimize(
            the_cost_function,
            x0,
            args=(refl, trans),
            jac=False,
            bounds=bounds,
            options={"disp": False},
        )
        store.append(retval)
        if verbose:
            xd = "".join(
                [
                    f"{par}:{retval.x[i]:6.4g} "
                    for i, par in enumerate(
                        ["n", "cab", "car", "cbrown", "cw", "cm"]
                    )
                ]
            )
            print(
                f"Start {tries+1} => \tCost {retval.fun:g} & Parameters: {xd}"
            )
    i = np.argmin([res.fun for res in store])
    fwd = call_prospect_5(*store[i].x)
    fwd_refl = fwd[:, 0]
    fwd_trans = fwd[:, 1]

    if verbose:
        print(store[i])

    if do_plots:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 4), sharex=True)
        axs = axs.flatten()
        if refl is not None:
            (l1,) = axs[0].plot(wv, refl, "--", label="Measurements")
        (l2,) = axs[0].plot(
            wv,
            call_prospect_5(*store[i].x)[:, 0],
            "-",
            lw=2,
            label="Fitted PROSPECT",
        )
        axs[0].legend(loc="best", fontsize=8)

        if trans is not None:
            (l3,) = axs[1].plot(wv, trans, "--", label="Measurements")

        (l4,) = axs[1].plot(
            wv,
            call_prospect_5(*store[i].x)[:, 1],
            "-",
            lw=2,
            label="Fitted PROSPECT",
        )
        axs[0].set_ylabel("Reflectance", fontsize=8)
        axs[1].set_ylabel("Transmittance", fontsize=8)
        axs[1].set_xlabel("Wavelength [nm]", fontsize=8)
        axs[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
    all_cost = [x.fun for x in store]
    all_x = [x.x for x in store]
    df = pd.DataFrame(all_x, columns=["n", "cab", "car", "cbrown", "cw", "cm"])
    df["cost"] = all_cost
    df = pd.concat(
        [
            df,
            df.mean().to_frame().T.rename(index={0: "Mean"}),
            df.std().to_frame().T.rename(index={0: "Std"}),
        ]
    )
    return (df, fwd_refl, fwd_trans)


def call_prosail(
    n,
    cab,
    car,
    ant,
    cbrown,
    cw,
    cm,
    lai,
    lidf,
    rsoil,
    psoil,
    hspot,
    sza,
    vza,
    vaa,
):
    """Helper function"""
    r = prosail.run_prosail(
        n,
        cab,
        car,
        cbrown,
        cw,
        cm,
        lai,
        lidf,
        hspot,
        sza,
        vza,
        vaa,
        ant=ant,
        rsoil=rsoil,
        psoil=psoil,
        prospect_version="D",
    )
    rpass = np.isfinite(r)
    x = np.arange(400, 2501)
    ri = np.interp(x, x[rpass], r[rpass])
    return ri


def prosail_sensitvitiy_gui():
    parameters = [
        "n",
        "cab",
        "car",
        "ant",
        "cbrown",
        "cw",
        "cm",
        "lai",
        "lidf",
        "rsoil",
        "psoil",
        "hspot",
    ]
    minvals = {
        "n": 1.0,
        "cab": 15.0,
        "car": 10.0,
        "ant": 0.0,
        "cbrown": 0.0,
        "cw": 0.001,
        "cm": 0.0,
        "lai": 0.5,
        "lidf": 0.0,
        "rsoil": 0.0,
        "psoil": 0.0,
        "hspot": 0.0001,
    }

    maxvals = {
        "n": 2.5,
        "cab": 80.0,
        "car": 20.0,
        "ant": 40,
        "cbrown": 1.0,
        "cw": 0.04,
        "cm": 0.5,
        "lai": 8.05,
        "lidf": 90.0,
        "rsoil": 2.0,
        "psoil": 2.0,
        "hspot": 0.5,
    }

    vals = {
        "n": 2.1,
        "cab": 30.0,
        "car": 16.0,
        "ant": 8,
        "cbrown": 0.0,
        "cw": 0.013,
        "cm": 0.0053,
        "lai": 3,
        "lidf": 45.0,
        "rsoil": 1.0,
        "psoil": 1.0,
        "hspot": 0.01,
    }
    leaf_w = {
        k: widgets.FloatSlider(
            min=minvals[k], max=maxvals[k], value=vals[k], description=k
        )
        for k in parameters[:6]
    }

    canopy_w = {
        k: widgets.FloatSlider(
            min=minvals[k], max=maxvals[k], value=vals[k], description=k
        )
        for k in parameters[6:]
    }
    acq_w = {
        "sza": widgets.FloatSlider(min=0, max=90, value=0, description="SZA"),
        "vza": widgets.FloatSlider(min=0, max=90, value=30, description="VZA"),
        "raa": widgets.FloatSlider(min=0, max=180, value=0, description="RAA"),
    }
    r1 = list(leaf_w.values())
    r2 = list(canopy_w.values())
    r3 = list(acq_w.values())

    leaf_w.update(canopy_w)
    leaf_w.update(acq_w)

    gui = widgets.VBox(
        [
            widgets.HBox(r1[:3]),
            widgets.HBox(r1[3:]),
            widgets.HBox(r2[:3]),
            widgets.HBox(r2[3:]),
            widgets.HBox(r3),
        ]
    )

    def on_toggle(**values):
        xx = [v for k, v in values.items()]
        prosail_sensitivity(*xx, epsilon=1e-5, do_plots=True)

    interact_out = widgets.interactive_output(on_toggle, leaf_w)
    display(gui, interact_out)


def prosail_sensitivity(
    leaf_n,
    leaf_cab,
    leaf_car,
    leaf_ant,
    leaf_cbrown,
    leaf_cw,
    leaf_cm,
    lai,
    ala,
    rsoil,
    psoil,
    hspot,
    sza=0.0,
    vza=30.0,
    raa=0.0,
    epsilon=1e-5,
    do_plots=True,
):
    """Local approximation (around ``x0``) of the sensitivity of the PROSAIL
    model to each input parameter. YOu can set up the point around which the
    sensitivity is calculated (``x0``), the value of the finite difference
    (``epsilon``).

    Parameters
    -------------
    x0: array
        A size 11 array, with the centre point around which the partial
        derivatives will be calculated. The different elements are in PROSAIL
        order, e.g. N, Cab, Car, Cbrown, Cw, Cm, LAI, average leaf angle
        (degrees), rsoil, psoil, hotspot.
    vza: float
        The view zenith angle in degrees
    sza: float
        The solar zenith angle in degrees
    raa: float
        The relative azimuth angle (e.g. vaa - saa) in degrees.
    epsilon: float
        The finite difference amount. If you get NaN, make it a bit larger.
    do_plots: bool
        Whether to do some pretty plots

    Returns
    ----------

    """
    ########################################################
    x0 = np.array(
        [
            leaf_n,
            leaf_cab,
            leaf_car,
            leaf_ant,
            leaf_cbrown,
            leaf_cw,
            leaf_cm,
            lai,
            ala,
            rsoil,
            psoil,
            hspot,
        ]
    )
    sensitivity = np.zeros((12, 2101))
    span = np.array(
        [
            1.5,
            20.0,
            20.0,
            40.0,
            1.0,
            0.0439 - 0.0043,
            0.0152 - 0.0017,
            8.0,
            10.0,
            2.0,
            2,
            0.2,
        ]
    )
    r0 = call_prosail(*(x0.tolist() + [sza, vza, raa]))
    for i in range(12):
        xp = x0 * 1.0
        xp[i] = x0[i] + epsilon * span[i]
        r1 = call_prosail(*(xp.tolist() + [sza, vza, raa]))
        sensitivity[i, :] = ((r0 - r1) / epsilon) * (xp[i] / r1[:])
    #################################################################

    if do_plots:
        wv = np.arange(400, 2501)
        fig1, axs = plt.subplots(
            nrows=4, ncols=3, figsize=(6.47, 6.47), sharex=True
        )
        axs = axs.flatten()
        for i, input_parameter in enumerate(
            [
                "n",
                "cab",
                "car",
                "Anth",
                "cbrown",
                "cw",
                "cm",
                "LAI",
                "LIDF",
                "RSOIL",
                "PSOIL",
                "HOTSPOT",
            ]
        ):
            axs[i].plot(wv, sensitivity[i, :], "-", lw=2)
            axs[i].set_title(f"{input_parameter}: {x0[i]}")
            axs[i].set_xlim(400, 2500)
            axs[i].axhline(0, color="0.8")
            if i in [0, 3]:
                axs[i].set_ylabel(r"$\partial f/\partial \mathbf{x}$")
            if i > 2:
                axs[i].set_xlabel("Wavelength [nm]")
            # pretty_axes ( axs[i] )
            # axs[i].set_ylim(-0.5, 0.5)
        fig1.tight_layout()
        fig2 = plt.figure(figsize=(6.47, 4))
        for i, input_parameter in enumerate(
            [
                "n",
                "cab",
                "car",
                "Anth",
                "cbrown",
                "cw",
                "cm",
                "LAI",
                "LIDF",
                "RSOIL",
                "PSOIL",
                "HOTSPOT",
            ]
        ):
            plt.plot(
                wv, np.abs(sensitivity[i, :]), lw=2, label=input_parameter
            )

        plt.xlim(400, 2500)
        plt.ylabel(r"$\partial f/\partial \mathbf{x}$")
        plt.xlabel("Wavelength [nm]")
        plt.legend(loc="best", frameon=False, fontsize=8)
    fig2.tight_layout()
    display(fig1.canvas)
    display(fig2.canvas)
    return wv, sensitivity


def hspot(h, theta):
    retval1 = []
    retval2 = []
    wv = np.arange(400, 2501)
    for theta_v in np.arange(-80, 80, 5):
        if theta_v < 0:
            raa = -180
            t = -theta_v
        else:
            raa = 0
            t = theta_v
        r = call_prosail(
            2.0,
            40,
            10.0,
            0.0,
            0.2,
            0.0143,
            0.007,
            4,
            45,
            0.1,
            1,
            h,
            theta,
            t,
            raa,
        )
        retval1.append(r[wv == 865])
        retval2.append(r[wv == 650])
    plt.figure(figsize=(5, 5))
    plt.plot(
        np.arange(-80, 80, 5),
        np.array(retval1).squeeze(),
        "-",
        label=f"{h:5g} @865nm",
    )
    plt.plot(
        np.arange(-80, 80, 5),
        np.array(retval2).squeeze(),
        "-",
        label=f"{h:5g} @650nm",
    )
    plt.xlabel("VZA [deg]")
    plt.ylabel("Reflectance")
    plt.legend(loc="best", frameon=False, fontsize=10)


def canopy_mtci_exp_gui():
    parameters = [
        "n",
        "cab",
        "car",
        "ant",
        "cbrown",
        "cw",
        "cm",
        "lai",
        "lidf",
        "rsoil",
        "psoil",
        "hspot",
    ]
    # parameters = ["leaf_n", "leaf_cab", "leaf_car", "leaf_ant",
    #                   "leaf_cbrown", "leaf_cw", "leaf_cm", "lai", "ala",
    #                   "rsoil", "psoil", "hspot", "sza", "vza", "raa"]
    some_buttons = {
        k: widgets.ToggleButton(description=k, value=False)
        for k in parameters
        if k != "cab"
    }
    some_buttons["cab"] = widgets.ToggleButton(description="cab", value=True)

    some_pars = {
        "SZA": widgets.FloatSlider(min=0, max=90, value=0, description="SZA"),
        "VZA": widgets.FloatSlider(min=0, max=90, value=30, description="VZA"),
        "RAA": widgets.FloatSlider(min=0, max=180, value=0, description="RAA"),
        "obs_unc": widgets.FloatLogSlider(
            min=-4, max=-0.2, value=0.001, description="Uncertainty"
        ),
        "bwidth": widgets.IntSlider(
            min=1, max=50, value=5, description="Bandwidth"
        ),
    }

    some_buttons.update(some_pars)

    gui_disp = widgets.VBox(
        [
            widgets.HBox(
                [some_buttons["n"], some_buttons["cab"], some_buttons["car"]]
            ),
            widgets.HBox(
                [
                    some_buttons["cbrown"],
                    some_buttons["cw"],
                    some_buttons["cm"],
                ]
            ),
            widgets.HBox(
                [
                    some_buttons["lai"],
                    some_buttons["lidf"],
                    some_buttons["hspot"],
                ]
            ),
            widgets.HBox([some_buttons["rsoil"], some_buttons["psoil"]]),
            widgets.HBox(
                [some_pars["SZA"], some_pars["VZA"], some_pars["RAA"]]
            ),
            widgets.HBox([some_pars["obs_unc"], some_pars["bwidth"]]),
        ]
    )

    def on_toggle(**values):
        nuisance = [p for p in parameters if values[p]]
        obs_noise = values["obs_unc"]
        sza = values["SZA"]
        vza = values["VZA"]
        raa = values["RAA"]
        bwidth = values["bwidth"]
        _ = mtci_experiment(
            nuisance=nuisance,
            noise_level=obs_noise,
            sza=sza,
            vza=vza,
            raa=raa,
            bwidth=bwidth,
        )

    interact_out = widgets.interactive_output(on_toggle, some_buttons)
    display(gui_disp, interact_out)


def mtci_experiment(
    x0=np.array(
        [1.5, 40.0, 5.0, 8.0, 0.0, 0.0113, 0.0053, 6, 30.0, 1, 1, 0.01]
    ),
    minvals={
        "n": 1.0,
        "cab": 15.0,
        "car": 10.0,
        "ant": 0.0,
        "cbrown": 0.0,
        "cw": 0.001,
        "cm": 0.0,
        "lai": 0.5,
        "lidf": 0.0,
        "rsoil": 0.0,
        "psoil": 0.0,
        "hspot": 0.0001,
    },
    maxvals={
        "n": 2.5,
        "cab": 80.0,
        "car": 20.0,
        "ant": 40,
        "cbrown": 1.0,
        "cw": 0.04,
        "cm": 0.5,
        "lai": 8.05,
        "lidf": 90.0,
        "rsoil": 2.0,
        "psoil": 2.0,
        "hspot": 0.5,
    },
    bwidth=10,
    nuisance=None,
    sza=0.0,
    vza=30.0,
    raa=0.0,
    do_plots=True,
    n_tries=100,
    noise_level=0,
):
    """An experiment in exploring the relationship between MTCI and
    chlorophyll concentration using PROSAIL This function allows one to forward
    model MTCI using the MERIS centre wavelength as a function of
    chlorophyll, while keeping the rest of the parameters set to ``x0``. If
    the ``nuisance`` option is set, then the reflectances will be calculated by
    randomly varying the nuisance parameter between the bounds given in
    ``minvals`` and ``maxvals``. You can specify the acquisition and
    illumination geometry, the additive noise level and the number of tries
    with their respective options. The function also fits a fist order
    polynomial to the data, and optionally does plots.

    Parameters
    --------------
    x0: array
        A size 11 array, with the centre point around which the partial
        derivatives will be calculated. The different elements are in PROSAIL
        order, e.g. N, Cab, Car, Cbrown, Cw, Cm, LAI, average leaf angle
        (degrees), rsoil, psoil, hotspot.
    minvals: dict
        Dictionary with minimum values for each parameter
    maxvals: dict
        Dictionary with maximum values for each parameter
    nuisance: str or iter
        A single parameter name (or a list of names) that indicate
        which parameter(s) will be varied between the boundaries set in
        ``minvals`` and ``maxvals``
    vza: float
        The view zenith angle in degrees
    sza: float
        The solar zenith angle in degrees
    raa: float
        The relative azimuth angle (e.g. vaa - saa) in degrees.
    epsilon: float
        The finite difference amount. If you get NaN, make it a bit larger.
    do_plots: bool
        Whether to do some pretty plots
    n_tries: int
        The number of realisations of the simulation
    noise_level: float
        Standard deviation of the additive noise (in units of reflectance).

    Returns
    ---------
    The values of chlorophyll concentration, MTCI, and the two parametes of
    the fitted polynomial.
    """
    param_position = [
        "n",
        "cab",
        "car",
        "ant",
        "cbrown",
        "cw",
        "cm",
        "lai",
        "lidf",
        "rsoil",
        "psoil",
        "hspot",
    ]
    if isinstance(nuisance, str) and (nuisance in param_position):
        nuisance = list(nuisance)
    elif isinstance(nuisance, (list, tuple)):
        for x in nuisance:
            if x not in param_position:
                raise ValueError("%s is not a PROSAIL parameter!" % x)

    wv = np.arange(400, 2501)
    band8_pass = np.logical_and(wv >= (681 - bwidth), wv <= (681 + bwidth))
    band9_pass = np.logical_and(wv >= (709 - bwidth), wv <= (709 + bwidth))
    band10_pass = np.logical_and(wv >= (865 - bwidth), wv <= (865 + bwidth))
    MTCI = []
    xp = x0 * 1.0
    cab_axis = []
    for n_tries in range(n_tries):
        for cab in np.arange(20, 100, 5):
            cab = cab + 5.0 * (np.random.rand() - 0.5)
            if nuisance is None:  # No flipping parameters around
                xp[1] = cab
                r = call_prosail(*(xp.tolist() + [sza, vza, raa]))
                r8 = r[band8_pass].mean() + np.random.randn() * noise_level
                r9 = r[band9_pass].mean() + np.random.randn() * noise_level
                r10 = r[band10_pass].mean() + np.random.randn() * noise_level
                mtci = (r10 - r9) / (r9 - r8)
                MTCI.append(mtci)
                cab_axis.append(cab)
            else:
                for nuisance_parameter in nuisance:
                    delta = (
                        maxvals[nuisance_parameter]
                        - minvals[nuisance_parameter]
                    )
                    ip = param_position.index(nuisance_parameter)

                    s = minvals[nuisance_parameter] + np.random.rand() * delta
                    xp[ip] = s

                xp[1] = cab
                r = call_prosail(*(xp.tolist() + [sza, vza, raa]))
                r8 = r[band8_pass].sum() + np.random.randn() * noise_level
                r9 = r[band9_pass].sum() + np.random.randn() * noise_level
                r10 = r[band10_pass].sum() + np.random.randn() * noise_level
                mtci = (r10 - r9) / (r9 - r8)
                MTCI.append(mtci)
                cab_axis.append(cab)
    MTCI = np.array(MTCI)
    mtci_passer = np.logical_and(MTCI > 0.01, MTCI < 50)
    MTCI = MTCI[mtci_passer]
    cab_axis = np.array(cab_axis)
    cab_axis = cab_axis[mtci_passer]
    p = np.polyfit(MTCI, cab_axis, 1)
    rr = np.corrcoef(cab_axis, np.polyval(p, MTCI))[0, 1]

    if do_plots:
        fig1 = plt.figure(figsize=(6.47, 4))
        plt.plot(MTCI, cab_axis, "o", markerfacecolor="none")
        plt.ylabel("Leaf chlorophyll concentration")
        plt.xlabel("MTCI")
        if nuisance is None:
            plt.title(
                "Fit: %g*MTCI +%g; R2=%g" % (p[0], p[1], rr * rr), fontsize=10
            )
        else:
            plt.title(
                "Fit: %g*MTCI +%g; R2=%g" % (p[0], p[1], rr * rr)
                + "\nNuisance: %s" % (" ".join(nuisance)),
                fontsize=10,
            )

        plt.plot([0, 10], np.polyval(p, [0, 10]), "--")
    fig1.tight_layout()
    display(fig1.canvas)
    return cab_axis, np.array(MTCI), p[0], p[1]


def plot_vi_space(v, r, n, the_vi):
    """Plots vegetation index RED/NIR space for a particular VI of those
    considered (NDVI, SAVI and OSAVI), and plot a colour shaded scatter plot of
    a set of points over RED/NIR space, the colour given by the
    value of `v`.

    Parameters
    ---------------
    v: array
        The value associated to each `r` and ``n`` (e.g. LAI)
    r: array
        Reflectance in the red band
    n: array
        Reflectance in the nir band
    the_vi: str
        The vegetation index to calculate. MUST be one of NDVI, SAVI or OSAVI,
        else raises Exception

    Returns
    --------
    Nothing, just does some plots!

    """
    L = 0.5
    red = np.arange(0.01, 0.8, 0.05)
    nir = np.arange(0.01, 0.8, 0.05)
    R, N = np.meshgrid(red, nir)
    if the_vi == "NDVI":
        vi = (N - R) / (N + R)
        vix = (n - r) / (n + r)
    elif the_vi == "SAVI":
        vi = (1 + L) * (N - R) / (N + R + L)
        vix = (1 + L) * (n - r) / (n + r + L)
    elif the_vi == "OSAVI":
        vi = 1.16 * (N / R) / (R + N + 0.16)
        vix = 1.16 * (n / r) / (r + n + 0.16)
    else:
        raise ValueError("We only deal with NDVI, SAVI or OSAVI")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 4))
    axs = axs.flatten()
    CS = axs[0].contour(R, N, vi, 6, colors="k")
    axs[0].clabel(CS, fontsize=9, inline=1)
    axs[0].scatter(r, n, c=v, cmap=plt.cm.Greens)

    axs[1].plot(v, vix, "o")
    axs[1].set_xlabel("value")
    axs[1].set_ylabel(the_vi)


def canopy_vi_exp_gui():
    parameters = [
        "n",
        "cab",
        "car",
        "ant",
        "cbrown",
        "cw",
        "cm",
        "lai",
        "lidf",
        "rsoil",
        "psoil",
        "hspot",
    ]
    # parameters = ["leaf_n", "leaf_cab", "leaf_car", "leaf_ant",
    #                   "leaf_cbrown", "leaf_cw", "leaf_cm", "lai", "ala",
    #                   "rsoil", "psoil", "hspot", "sza", "vza", "raa"]
    some_buttons = {
        k: widgets.ToggleButton(description=k, value=False)
        for k in parameters
        if k != "lai"
    }
    some_buttons["lai"] = widgets.ToggleButton(description="LAI", value=True)

    some_pars = {
        "SZA": widgets.FloatSlider(min=0, max=90, value=0, description="SZA"),
        "VZA": widgets.FloatSlider(min=0, max=90, value=30, description="VZA"),
        "RAA": widgets.FloatSlider(min=0, max=180, value=0, description="RAA"),
        "obs_unc": widgets.FloatLogSlider(
            min=-4, max=-0.2, value=0.001, description="Uncertainty"
        ),
        "index": widgets.Dropdown(
            options=["NDVI", "SAVI", "OSAVI"], value="NDVI"
        ),
    }

    some_buttons.update(some_pars)

    gui_disp = widgets.VBox(
        [
            widgets.HBox(
                [some_buttons["n"], some_buttons["cab"], some_buttons["car"]]
            ),
            widgets.HBox(
                [
                    some_buttons["cbrown"],
                    some_buttons["cw"],
                    some_buttons["cm"],
                ]
            ),
            widgets.HBox(
                [
                    some_buttons["lai"],
                    some_buttons["lidf"],
                    some_buttons["hspot"],
                ]
            ),
            widgets.HBox([some_buttons["rsoil"], some_buttons["psoil"]]),
            widgets.HBox(
                [some_pars["SZA"], some_pars["VZA"], some_pars["RAA"]]
            ),
            widgets.HBox([some_pars["index"], some_pars["obs_unc"]]),
        ]
    )

    def on_toggle(**values):
        nuisance = [p for p in parameters if values[p]]
        obs_noise = values["obs_unc"]
        sza = values["SZA"]
        vza = values["VZA"]
        raa = values["RAA"]
        veg_index = values["index"]

        _ = canopy_vi_expt(
            nuisance=nuisance,
            obs_noise=obs_noise,
            sza=sza,
            vza=vza,
            raa=raa,
            vin=veg_index,
        )

    interact_out = widgets.interactive_output(on_toggle, some_buttons)
    display(gui_disp, interact_out)


def canopy_vi_expt(
    x0=np.array(
        [1.5, 40.0, 5.0, 0.0, 0.0, 0.0113, 0.0053, 1, 30.0, 4, 0, 0.01]
    ),
    nuisance=["lai"],
    obs_noise=1e-4,
    bwidth_r=10,
    bwidth_n=10,
    minvals={
        "n": 1.0,
        "cab": 15.0,
        "car": 5.0,
        "ant": 0.0,
        "cbrown": 0.0,
        "cw": 0.001,
        "cm": 0.0,
        "lai": 0.5,
        "lidf": 0.0,
        "rsoil": 0.0,
        "psoil": 0.0,
        "hspot": 0.0001,
    },
    maxvals={
        "n": 2.5,
        "cab": 80.0,
        "car": 20.0,
        "ant": 40,
        "cbrown": 0.1,
        "cw": 0.04,
        "cm": 0.5,
        "lai": 8.05,
        "lidf": 90.0,
        "rsoil": 5.0,
        "psoil": 2.0,
        "hspot": 0.5,
    },
    sza=0.0,
    vza=30.0,
    raa=10.0,
    n_tries=5000,
    do_plots=True,
    vin="NDVI",
):
    """Experiments with VIs in the red/nir region. This function performs
    some experiments using simulations of the red and nir reflectance (around
    650 and 865nm, respectively). The two bands have a bandwidth controlled
    by ``bwidth_r`` and ``bwidth_n``. The geometry can be set up by the usual
    options of ``sza``, ``vza`` and ``raa``. Additive random independent
    Gaussian noise is added by the two-element ``obs_noise`` (red and nir bands
    noise, resp.). Additionally, we have ``nuisance`` parameters that will
    be randomly set between the span given by ``minvals`` and ``maxvals``.

    Parameters
    -------------
    x0: array
        An 12-element array with the different parameters in their usual
        PROSAIL positions. If a parameter is not in the ``nuisance`` list,
        its value will be set from here.
    nuisance: list
        A list of parameters to be randomly varied in the simulations between
        their corresponding entries in ``minvals`` and ``maxvals``.
    obs_noise: 2-element array
        A 2 element array with the variance of the additive noise for the red
        and nir bands resp.
    bwidth_r: scalar
        The bandwith (centered at 650nm) of the red band
    bwidth_n: scalar
        The bandwith (centered at 865nm) of the nir band
    minvals: dict
        Dictionary with minimum values for each parameter
    maxvals: dict
        Dictionary with maximum values for each parameter
    vza: float
        The view zenith angle in degrees
    sza: float
        The solar zenith angle in degrees
    raa: float
        The relative azimuth angle (e.g. vaa - saa) in degrees.
    n_tries: int
        The number of realisations of the simulation
    do_plots: bool
        Whether to do plots or not
    vi: str
        VI to calculate for plots: NDVI, SAVI or OSAVI
    Returns
    ----------
    A tuple containing the input parameters, and the red and nir reflectances
    """
    if not isinstance(nuisance, list):
        nuisance = list(nuisance)
    wv = np.arange(400, 2501)
    obs_noise = [obs_noise, obs_noise]
    red_pass = np.logical_and(
        wv >= (650 - bwidth_r), wv <= (650 + bwidth_r)
    )  # (wv == 650 )
    nir_pass = np.logical_and(wv >= (865 - bwidth_r), wv <= (865 + bwidth_r))
    param_names = [
        "n",
        "cab",
        "car",
        "ant",
        "cbrown",
        "cw",
        "cm",
        "lai",
        "lidf",
        "rsoil",
        "psoil",
        "hspot",
    ]
    xbase = np.zeros(12)
    span = np.zeros(12)
    for i, param in enumerate(param_names):
        xbase[i] = minvals[param]
        if param in nuisance:
            span[i] = maxvals[param] - minvals[param]
    x = []
    red = []
    nir = []
    for n_tries in range(n_tries):
        xp = x0 * 1.0
        for i, p in enumerate(param_names):
            if p in nuisance:
                xp[i] = xbase[i] + np.random.rand() * span[i]
        r = call_prosail(*(xp.tolist() + [sza, vza, raa]))
        red.append(r[red_pass].mean())
        nir.append(r[nir_pass].mean())
        x.append(xp)
    x = np.array(x)
    red = np.array(red)
    nir = np.array(nir)
    red += np.random.randn(red.shape[0]) * obs_noise[0]
    nir += np.random.randn(red.shape[0]) * obs_noise[1]
    red = np.clip(red, 0.01, 1)
    nir = np.clip(nir, 0.01, 1)
    if do_plots:
        p = param_names.index(nuisance[0])
        plot_vi_space(x[:, p], red, nir, vin)
    return x, red, nir
