import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from tqdm import tqdm

from tidaldisruptionlrd.utils import get_interp


class BaseProfile:
    def __init__(self, r_bin_min, r_bin_max, N_bins):
        """
        Initialize the base profile class. The profile is defined in scale parameters,
        with the scale mass being the mass of the central black hole and the scale length
        being the radius of influence of the black hole. The scale velocity is then
        defined as sqrt(G * scale_mass / scale_length), i.e. the velocity dispersion of
        the stars at the radius of influence.

        Parameters:
        ----------
        r_bin_min: float
            Minimum profile radius of the halo in scale_length.
        r_bin_max: float
            Maximum profile radius of the halo in scale_length.
        N_bins: int
            Number of bins to use for the profiles.
        """

        self._r_bin_min = r_bin_min
        self._r_bin_max = r_bin_max

        self._N_bins = N_bins

        self._get_profiles()

    def _get_stellar_rho_bins(self, r_bins):
        """
        Get the stellar density profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in scale_length.

        Returns:
        -------
        rho_bins: array
            Array of density bins in scale_mass / scale_length^3.
        """

        raise NotImplementedError("Not implemented in base class.")  # noqa: EM101

    def _get_stellar_mass_bins(self, r_bins, stellar_rho_bins):
        """
        Get the stellar mass profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in scale_length.
        stellar_rho_bins: array
            Array of stellar density bins in scale_mass / scale_length^3.

        Returns:
        -------
        stellar_mass_bins: array
            Array of stellar mass bins in scale_mass.
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
            Array of radius bins in scale_length.
        stellar_mass_bins: array
            Array of stellar mass bins in scale_mass.

        Returns:
        -------
        stellar_phi_bins: array
            Array of potential bins in scale_velocity^2.
        """

        _delta_phi_integrand = stellar_mass_bins / r_bins**2
        _delta_phi_bins = cumulative_trapezoid(_delta_phi_integrand, r_bins, initial=0)

        return _delta_phi_bins - _delta_phi_bins[-1]

    def _get_Eddington_bins(self, rho_bins, phi_bins):
        """
        Get the Eddington inversion bins.

        Parameters:
        ----------
        rho_bins: array
            Array of density bins in scale_mass / scale_length^3.
        phi_bins: array
            Array of potential bins in scale_velocity^2.

        Returns:
        -------
        eta_bins: array
            Array of eta bins in scale_velocity^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (scale_velocity)^-2.
        """

        _psi_bins = np.flip(-phi_bins)

        _rho_interp_bins = np.flip(rho_bins)[1:]
        _psi_interp_bins = _psi_bins[1:]

        _d2rho_dpsi2_bins = (
            _rho_interp_bins
            / _psi_interp_bins**2
            * (
                np.gradient(np.log(_rho_interp_bins), np.log(_psi_interp_bins)) ** 2
                + np.gradient(
                    np.gradient(np.log(_rho_interp_bins), np.log(_psi_interp_bins)),
                    np.log(_psi_interp_bins),
                )
                - np.gradient(np.log(_rho_interp_bins), np.log(_psi_interp_bins))
            )
        )

        _log_log_d2rho_dpsi2_interp = get_interp(
            np.log(_psi_interp_bins), np.log(_d2rho_dpsi2_bins)
        )

        _eta_bins = _psi_bins
        _f_eta_bins = [0]
        for i in tqdm(range(1, self._N_bins), desc="Eddington's inversion"):

            def _f_eta_integrand(psi):
                return (
                    1
                    / np.sqrt(_psi_bins[i] - psi)  # noqa: B023
                    * np.exp(_log_log_d2rho_dpsi2_interp(np.log(psi)))
                )

            _f_eta_bins.append(
                quad(
                    _f_eta_integrand,
                    _psi_bins[0],
                    _psi_bins[i],
                )[0]
            )

        return _eta_bins, 1 / np.sqrt(8) / np.pi**2 * np.array(_f_eta_bins)

    def _get_profiles(self):
        """
        Get the profiles of the halo.

        Returns:
        -------
        r_bins: array
            Array of radius bins in scale_length.

        stellar_rho_bins: array
            Array of stellar density bins in scale_mass / scale_length^3.
        stellar_mass_bins: array
            Array of stellar mass bins in scale_mass.
        stellar_phi_bins: array
            Array of stellar potential bins in scale_velocity^2.

        phi_bins: array
            Array of potential bins in scale_velocity^2.

        eta_bins: array
            Array of eta bins in scale_velocity^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (scale_velocity)^-2.
        """

        self.r_bins = np.logspace(
            np.log10(self._r_bin_min),
            np.log10(self._r_bin_max),
            self._N_bins,
            dtype=np.float64,
        )
        self.stellar_rho_bins = self._get_stellar_rho_bins(self.r_bins)
        self.stellar_mass_bins = self._get_stellar_mass_bins(
            self.r_bins, self.stellar_rho_bins
        )
        self.stellar_phi_bins = self._get_stellar_phi_bins(
            self.r_bins, self.stellar_mass_bins
        )

        self.mass_bins = self.stellar_mass_bins + 1
        self.phi_bins = self.stellar_phi_bins - 1 / self.r_bins

        self.eta_bins, self.f_eta_bins = self._get_Eddington_bins(
            self.stellar_rho_bins, self.phi_bins
        )

    def reconstruct_stellar_rho_bins(self, psi_bins, eta_bins, f_eta_bins):
        """
        Reconstruct the density profile from the potential and Eddington distribution.

        Parameters:
        ----------
        psi_bins: array
            Array of negative potential bins in scale_velocity^2 = G scale_mass / scale_length.
        eta_bins: array
            Array of eta bins in scale_velocity^2 = G scale_mass / scale_length.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (scale_velocity)^-2 = (G scale_mass / scale_length)^-1.

        Returns:
        -------
        reconstructed_rho_bins: array
            Array of reconstructed density bins in scale_mass / scale_length^3.
        """

        _log_log_eddington_interp = get_interp(np.log(eta_bins), np.log(f_eta_bins))

        def _rho_integrand(eta_prime, psi):
            return (
                4
                * np.pi
                * np.sqrt(2 * (psi - eta_prime))
                * np.exp(_log_log_eddington_interp(np.log(eta_prime)))
            )

        _reconstructed_rho_bins = []
        for _psi in tqdm(psi_bins, desc="Reconstructing densities"):
            _reconstructed_rho_bins.append(  # noqa: PERF401
                quad(_rho_integrand, 0, _psi, args=(_psi,))[0]
            )

        return np.array(_reconstructed_rho_bins)
