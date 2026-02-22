import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from tidaldisruptionlrd.constants import G


class BaseProfile:
    def __init__(self, r_bin_min, r_bin_max, N_bins, M_bh=0):
        """
        Initialize the base profile class.

        Parameters:
        ----------
        r_bin_min: float
            Minimum profile radius of the halo in kpc.
        r_bin_max: float
            Maximum profile radius of the halo in kpc.
        N_bins: int
            Number of bins to use for the profiles.
        """

        self._r_bin_min = r_bin_min
        self._r_bin_max = r_bin_max

        self._N_bins = N_bins

        self.M_bh = M_bh

        (
            self.r_bins,
            self.stellar_rho_bins,
            self.stellar_mass_bins,
            self.stellar_phi_bins,
            self.phi_bins,
            self.eta_bins,
            self.f_eta_bins,
        ) = self._get_profiles()

    def _get_interp(self, x_bins, y_bins):
        """
        Get the interpolated profiles of the halo.

        Parameters:
        ----------
        x_bins: array
            Array of the x-param.
        y_bins: array
            Array of the y-param.

        Returns:
        -------
        interp: CubicSpline
            Interpolated profile of the halo.
        """

        x_order = np.argsort(x_bins)
        x_increasing_mask = np.append([True], np.diff(x_bins[x_order]) > 0)

        x_bins = x_bins[x_order][x_increasing_mask]
        y_bins = y_bins[x_order][x_increasing_mask]

        finite_mask = np.logical_and(np.isfinite(x_bins), np.isfinite(y_bins))

        return CubicSpline(x_bins[finite_mask], y_bins[finite_mask])

    def _get_stellar_rho_bins(self, r_bins):
        """
        Get the stellar density profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.

        Returns:
        -------
        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        """

        raise NotImplementedError("Not implemented in base class.")  # noqa: EM101

    def _get_stellar_mass_bins(self, r_bins, stellar_rho_bins):
        """
        Get the stellar mass profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        stellar_rho_bins: array
            Array of stellar density bins in M_sun / kpc^3.

        Returns:
        -------
        stellar_mass_bins: array
            Array of stellar mass bins in M_sun.
        """

        _zero_mass = 4 / 3 * np.pi * r_bins[0] ** 3 * stellar_rho_bins[0]
        _mass_integrand = 4 * np.pi * r_bins**2 * stellar_rho_bins
        return cumulative_trapezoid(_mass_integrand, r_bins, initial=0) + _zero_mass

    def _get_stellar_phi_bins(self, r_bins, stellar_mass_bins):
        """
        Get the stellar potential profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        stellar_mass_bins: array
            Array of stellar mass bins in M_sun.

        Returns:
        -------
        stellar_phi_bins: array
            Array of potential bins in (km/s)^2.
        """

        _delta_phi_integrand = G * stellar_mass_bins / r_bins**2
        _delta_phi_bins = cumulative_trapezoid(_delta_phi_integrand, r_bins, initial=0)

        return _delta_phi_bins - _delta_phi_bins[-1]

    def _get_Eddington_bins(self, rho_bins, phi_bins):
        """
        Get the Eddington inversion bins.

        Parameters:
        ----------
        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        phi_bins: array
            Array of potential bins in (km/s)^2.

        Returns:
        -------
        eta_bins: array
            Array of eta bins in (km/s)^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (km/s)^-2.
        """

        _psi_bins = np.flip(-phi_bins)
        _d2rho_dpsi2_bins = np.flip(
            rho_bins
            * (
                np.gradient(np.log(rho_bins), phi_bins) ** 2
                + np.gradient(np.gradient(np.log(rho_bins), phi_bins), phi_bins)
            )
        )

        _lin_log_d2rho_dpsi2_interp = self._get_interp(
            _psi_bins[1:], np.log(_d2rho_dpsi2_bins[1:])
        )

        _eta_bins = _psi_bins
        _f_eta_bins = [0]
        for _i in tqdm(range(1, self._N_bins), desc="Eddington's inversion"):

            def _f_eta_integrand(psi):
                return (
                    1
                    / np.sqrt(_psi_bins[_i] - psi)
                    * np.exp(_lin_log_d2rho_dpsi2_interp(psi))
                )

            _f_eta_bins.append(
                quad(
                    _f_eta_integrand,
                    0,
                    _psi_bins[_i],
                )[0]
            )

        return _eta_bins, 1 / np.sqrt(8) / np.pi**2 * np.array(_f_eta_bins)

    def _get_profiles(self):
        """
        Get the profiles of the halo.

        Returns:
        -------
        r_bins: array
            Array of radius bins in kpc.

        stellar_rho_bins: array
            Array of stellar density bins in M_sun / kpc^3.
        stellar_mass_bins: array
            Array of stellar mass bins in M_sun.
        stellar_phi_bins: array
            Array of stellar potential bins in (km/s)^2.

        phi_bins: array
            Array of potential bins in (km/s)^2.

        eta_bins: array
            Array of eta bins in (km/s)^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (km/s)^-2.
        """

        r_bins = np.logspace(
            np.log10(self._r_bin_min),
            np.log10(self._r_bin_max),
            self._N_bins,
            dtype=np.float64,
        )
        stellar_rho_bins = self._get_stellar_rho_bins(r_bins)
        stellar_mass_bins = self._get_stellar_mass_bins(r_bins, stellar_rho_bins)
        stellar_phi_bins = self._get_stellar_phi_bins(r_bins, stellar_mass_bins)

        phi_bins = stellar_phi_bins - G * self.M_bh / r_bins

        eta_bins, f_eta_bins = self._get_Eddington_bins(stellar_rho_bins, phi_bins)

        return (
            r_bins,
            stellar_rho_bins,
            stellar_mass_bins,
            stellar_phi_bins,
            phi_bins,
            eta_bins,
            f_eta_bins,
        )

    def reconstruct_stellar_rho_bins(self):
        """
        Reconstruct the density profile from the potential and Eddington distribution.

        Returns:
        -------
        reconstructed_rho_bins: array
            Array of reconstructed density bins in M_sun / kpc^3.
        """

        _lin_log_eddington_interp = self._get_interp(
            self.eta_bins, np.log(self.f_eta_bins)
        )

        def _rho_integrand(v, psi):
            return 4 * np.pi * v**2 * np.exp(_lin_log_eddington_interp(psi - v**2 / 2))

        _reconstructed_rho_bins = []
        for phi in tqdm(self.phi_bins, desc="Reconstructing densities"):
            _reconstructed_rho_bins.append(  # noqa: PERF401
                quad(_rho_integrand, 0, np.sqrt(-2 * phi), args=(-phi,))[0]
            )

        return np.array(_reconstructed_rho_bins)
