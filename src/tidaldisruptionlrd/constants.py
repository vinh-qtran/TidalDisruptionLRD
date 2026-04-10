import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import quad

c = 299792.458  # km/s

G = 4.30071e-6  # kpc * (km/s)^2 / M_sun

Rsun = 2.2555823856078e-11  # kpc

z45_shell_volume = (Planck18.comoving_volume(5) - Planck18.comoving_volume(4)).value
z56_shell_volume = (Planck18.comoving_volume(6) - Planck18.comoving_volume(5)).value


def dilation_integrand(z):
    return 1 / (1 + z) * 4 * np.pi * Planck18.differential_comoving_volume(z).value


z45_dilation_factor = quad(dilation_integrand, 4, 5)[0] / z45_shell_volume
z56_dilation_factor = quad(dilation_integrand, 5, 6)[0] / z56_shell_volume

G25_number_density = 1.06145e-05 + 7.61923e-05 + 3.27835e-05

all_sky_deg_sqr = 41252.96
