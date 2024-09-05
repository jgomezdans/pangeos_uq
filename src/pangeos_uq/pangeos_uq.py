#!/usr/bin/env python3

"""
pangeos_uq.py

PANGEOS uncertainty quantification school
"""

import numpy as np

__author__ = "Jose Gomez-Dans"
__copyright__ = "Copyright 2024, Jose Gomez-Dans"
__license__ = "MIT"

__version__ = "0.1.0"
__maintainer__ = "Jose Gomez-Dans"
__email__ = "jose.gomez-dans@kcl.ac.uk"
__status__ = "Development"


def to_toa(
    rho_s: float | np.ndarray,
    xap: float | np.ndarray,
    xb: float | np.ndarray,
    xc: float | np.ndarray,
) -> float | np.ndarray:
    """Convert surface reflectance to top of atmosphere reflectance.

    Uses 6s xap, xb and xc coefficients to convert surface reflectance to top
    of atmosphere reflectance.

    Args:
        rho_s (float | np.ndarray): surface reflectance
        xap (float | np.ndarray): xap coefficient (inverse trasmitances)
        xb (float | np.ndarray): xb coefficient (path radiance)
        xc (float | np.ndarray): xc coefficient (multiple scattering)

    Returns:
        float: top of atmosphere reflectance
    """
    numerator = rho_s + xb * (1 - rho_s * xc)
    denominator = xap * (1 - rho_s * xc)
    return np.clip(numerator / denominator, 0, 1)


def to_boa(
    rho_toa: float | np.ndarray,
    xap: float | np.ndarray,
    xb: float | np.ndarray,
    xc: float | np.ndarray,
) -> float | np.ndarray:
    """Convert top of atmosphere reflectance to surface reflectance.

    Uses 6s xap, xb and xc coefficients to  convert top of atmosphere
    reflectance to surface reflectance to .

    Args:
        rho_toa (float | np.ndarray): toa reflectance
        xap (float | np.ndarray): xap coefficient
        xb (float | np.ndarray): xb coefficient
        xc (float | np.ndarray): xc coefficient

    Returns:
        float | np.ndarray: bottom of atmosphere reflectance
    """
    y = xap * (rho_toa) - xb
    return y / (1.0 + xc * y)
