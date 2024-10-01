import ipywidgets as widgets
from IPython.display import display
from pangeos_uq import BiophysicalRetrieval, spectral_srf, LUTQuery
from pangeos_uq.param_retrieval import get_priors
import numpy as np


# Set up... Get Sentinel 2 SRFs and wavelengths...
srf = spectral_srf["sentinel2a_msi"]
central_wavelengths = (
    np.round((srf @ np.arange(400, 2500 + 1, 1) / np.sum(srf, axis=1)) / 2.5)
    * 2.5
)
srf = np.delete(srf, [0, 6, 9, 10], axis=0)
wvs = np.delete(central_wavelengths, [0, 6, 9, 10]) / 1000.0

# set up the LUT query object
lut = LUTQuery(k=10)
# set up the prior functions
prior_funcs = get_priors()


def run_simulation(sliders, floattext_widgets, output_widget):
    """
    Function to handle the simulation when the button is clicked.

    Parameters:
    - sliders: Dictionary of slider widgets.
    - floattext_widgets: Dictionary of FloatText widgets.
    - output_widget: Output widget to display plots.
    """
    parameters = {name: slider.value for name, slider in sliders.items()}
    parameters.update(
        {
            name: text_widget.value
            for name, text_widget in floattext_widgets.items()
        }
    )

    srf = np.delete(spectral_srf["sentinel2a_msi"], [0, 6, 9, 10], axis=0)
    srf = srf / np.sum(srf, axis=1, keepdims=True)
    the_prior = prior_funcs[parameters["prior_selection"]]
    biophys = BiophysicalRetrieval(parameters, srf, lut, the_prior, wvs)
    biophys.simulate_reflectance()
    biophys.propagate_to_toa()
    biophys.correct_to_boa()
    biophys.run_mcmc(n_samples=parameters["n_iterations"])
    biophys.plot_posterior(output_widget)


def create_prosail_gui(boundaries: dict):
    """
    Creates a GUI for parameterizing the PROSAIL model using ipywidgets.

    Parameters:
    - boundaries: A dictionary where keys are parameter names and values are
        tuples defining (min, max) values for sliders.

    """

    # Helper function to create sliders dynamically
    def create_slider(param_name, param_bounds):
        slider_type = (
            widgets.IntSlider
            if isinstance(param_bounds[0], int)
            else widgets.FloatSlider
        )
        return slider_type(
            value=(param_bounds[0] + param_bounds[1]) / 2,
            min=param_bounds[0],
            max=param_bounds[1],
            description=param_name,
            step=0.001 if slider_type is widgets.FloatSlider else 1,
            readout_format=".3f"
            if slider_type is widgets.FloatSlider
            else None,
        )

    # Create sliders from boundaries
    sliders = {
        name: create_slider(name, bounds)
        for name, bounds in boundaries.items()
        if name != "AOT"
    }
    sliders["AOT"] = widgets.FloatLogSlider(
        value=0.15,  # Default selected value
        base=10,  # Logarithmic base
        min=-2,  # Corresponds to 0.01 (log10 of 0.01 = -2)
        max=0,  # Corresponds to 1 (log10 of 1 = 0)
        step=0.01,  # Step size for the slider
        description="AOT",  # Label for the slider
        readout=True,  # Display the numeric value
        readout_format=".2f",  # Format the readout to 2 decimal places
    )

    sliders["noise_unc"] = widgets.FloatSlider(
        value=2,
        min=0,
        max=10,
        step=0.5,
        description="Thermal Noise in percent:",
        readout=True,
        readout_format=".1f",
    )
    sliders["n_iterations"] = widgets.IntSlider(
        value=10_000,
        min=5_000,
        max=100_000,
        step=1000,
        description="Number of MCMC iterations:",
        readout=True,
        readout_format="d",
    )
    # FloatText widgets for uncertainties
    floattext_widgets = {
        "AOT_unc": widgets.FloatText(
            value=sliders["AOT"].value * 0.3, description="AOT_unc", step=0.01
        ),
        "TCWV_unc": widgets.FloatText(
            value=sliders["TCWV"].value * 0.1,
            description="TCWV_unc",
            step=0.01,
        ),
    }

    # Link the FloatText widgets to the sliders
    def create_update_function_aot(slider, text_widget):
        def update_unc(change):
            if not text_widget.disabled:
                # Using Eq. 6 of Gorroño et al. (2024)
                text_widget.value = (
                    change["new"] * 0.1
                    + 0.03
                    + np.abs(-0.46 * change["new"] + 0.07)
                )

        return update_unc

    def create_update_function_tcwv(slider, text_widget):
        def update_unc(change):
            if not text_widget.disabled:
                # Using Eq. 9 of Gorroño et al. (2024)
                text_widget.value = (
                    change["new"] * 0.1
                    + 0.2
                    + np.abs(-0.1 * change["new"] + 0.03)
                )

        return update_unc

    sliders["AOT"].observe(
        create_update_function_aot(
            sliders["AOT"], floattext_widgets["AOT_unc"]
        ),
        names="value",
    )
    sliders["TCWV"].observe(
        create_update_function_tcwv(
            sliders["TCWV"], floattext_widgets["TCWV_unc"]
        ),
        names="value",
    )

    # Output widget for plots
    output_widget = widgets.Output()

    # Simulate button
    button_simulate = widgets.Button(
        description="Simulate", button_style="danger", icon="check"
    )
    button_box = widgets.HBox(
        [button_simulate], layout=widgets.Layout(justify_content="flex-end")
    )

    # Link the button to the external run_simulation function
    button_simulate.on_click(
        lambda b: run_simulation(sliders, floattext_widgets, output_widget)
    )

    sliders["prior_selection"] = widgets.Dropdown(
        options=[
            ("Uniform", "uniform"),
            ("Crop", "all"),
            ("Early", "early"),
            ("Mid", "mid"),
            ("Late", "late"),
        ],
        value="all",
        description="Prior type:",
        disabled=False,
    )
    # Organize the widgets into columns and display
    column1 = widgets.VBox(
        [
            sliders[name]
            for name in [
                "N",
                "LAI",
                "ALA",
                "Cab",
                "Cw",
                "Cm",
                "Cbrown",
                "psoil",
                "rsoil",
            ]
        ]
    )
    column2 = widgets.VBox(
        [sliders[name] for name in ["sza", "vza", "raa", "AOT", "TCWV"]]
    )

    # Define a Box with flexible space to align the button properly
    spacer = widgets.Box(
        layout=widgets.Layout(flex="1 1 auto")
    )  # Flexible space to push the button down

    # Define the third column with a spacer before the button
    column3 = widgets.VBox(
        [
            floattext_widgets["AOT_unc"],
            floattext_widgets["TCWV_unc"],
            sliders["noise_unc"],
            sliders["prior_selection"],
            sliders["n_iterations"],
            spacer,  # This spacer will take up the remaining space
            button_box,
        ],
        layout=widgets.Layout(
            align_items="stretch", height="auto"
        ),  # Adjust 'height' to auto to allow flexible space
    )

    # Combine the output and the GUI layout
    ui = widgets.VBox(
        [output_widget, widgets.HBox([column1, column2, column3])]
    )
    display(ui)
