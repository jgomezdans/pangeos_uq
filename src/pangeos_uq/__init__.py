"""
PANGEOS Uncertainty Quantification training
"""

__author__ = """Jose Gomez-Dans"""
__email__ = "jose.gomez-dans@kcl.ac.uk"
__version__ = "0.1.0"
import importlib.resources
import numpy as np
from .prosail_funcs import call_prosail  # noqa F401
from .mcmc import generate_samples  # noqa F401
from .sixs_lut import LUTQuery  # noqa F401
from .param_retrieval import BiophysicalRetrieval  # noqa F401

with importlib.resources.path("pangeos_uq.data", "sensor_srf.npz") as f:
    tmp_file = np.load(f)
    spectral_srf = {sensor: arr for sensor, arr in tmp_file.items()}
__all__ = ["spectral_srf"]
