from astropy.cosmology import Planck18

c = 299792.458  # km/s

G = 4.30071e-6  # kpc * (km/s)^2 / M_sun

Rsun = 2.2555823856078e-11  # kpc

z45_shell_volume = (Planck18.comoving_volume(5) - Planck18.comoving_volume(4)).value
z56_shell_volume = (Planck18.comoving_volume(6) - Planck18.comoving_volume(5)).value

G25_number_density = 1.06145e-05 + 7.61923e-05 + 3.27835e-05
