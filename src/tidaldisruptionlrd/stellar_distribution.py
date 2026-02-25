import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, cumulative_trapezoid, quad
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)

from tidaldisruptionlrd.utils import get_interp  # noqa: E402


class BaseProfile:
    def __init__(
        self,
        r_bin_min=1e-4,
        r_bin_max=1e4,
        N_bins=10000,
        reduce_factor=10,
        N_trapz_bins=1000,
    ):
        """
        Initialize the base profile class. The profile is defined in scale parameters,
        with the scale mass being the mass of the central black hole and the scale length
        being the radius of influence of the black hole. The scale velocity is then
        defined as sqrt(G * scale_mass / scale_length), i.e. the velocity dispersion of
        the stars at the radius of influence.

        Parameters:
        ----------
        r_bin_min: float, optional
            Minimum profile radius of the halo in scale_length. Default is 1e-4.
        r_bin_max: float, optional
            Maximum profile radius of the halo in scale_length. Default is 1e4.
        N_bins: int, optional
            Number of bins to use for the profiles. Default is 10000.

        reduce_factor: int, optional
            Factor by which to reduce the number of bins for the loss cone calculations. Default is 10.
        N_trapz_bins: int, optional
            Number of bins to use for the trapezoidal integration in the loss cone calculations. Default is 1000.
        """

        self._r_bin_min = r_bin_min
        self._r_bin_max = r_bin_max
        self._N_bins = N_bins

        self._reduce_factor = reduce_factor
        self._N_reduced_bins = self._N_bins // self._reduce_factor

        self._N_trapz_bins = N_trapz_bins

        self._get_base_profiles()

        self._get_reduced_profiles()

    # BASE PROFILES CALCULATIONS
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
        epsilon_bins: array
            Array of epsilon bins in scale_velocity^2.
        g_epsilon_bins: array
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

        _epsilon_bins = _psi_bins
        _g_epsilon_bins = [0]
        for i in tqdm(range(1, self._N_bins), desc="Calculating g(eps)"):

            def _g_epsilon_integrand(psi):
                return (
                    1
                    / np.sqrt(_psi_bins[i] - psi)  # noqa: B023
                    * np.exp(_log_log_d2rho_dpsi2_interp(np.log(psi)))
                )

            _g_epsilon_bins.append(
                quad(
                    _g_epsilon_integrand,
                    _psi_bins[0],
                    _psi_bins[i],
                )[0]
            )

        return _epsilon_bins, 1 / np.sqrt(8) / np.pi**2 * np.array(_g_epsilon_bins)

    def _get_base_profiles(self):
        """
        Get the base profiles of the halo.

        Set Attributes:
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

        epsilon_bins: array
            Array of epsilon bins in scale_velocity^2.
        g_epsilon_bins: array
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

        self.epsilon_bins, self.g_epsilon_bins = self._get_Eddington_bins(
            self.stellar_rho_bins, self.phi_bins
        )

    # SELF-CONSISTENCY CHECKS
    def reconstruct_stellar_rho_bins(self, psi_bins, epsilon_bins, g_epsilon_bins):
        """
        Reconstruct the density profile from the potential and Eddington distribution.

        Parameters:
        ----------
        psi_bins: array
            Array of negative potential bins in scale_velocity^2.
        epsilon_bins: array
            Array of epsilon bins in scale_velocity^2.
        g_epsilon_bins: array
            Array of Eddington distribution probability bins in (scale_velocity)^-2.

        Returns:
        -------
        reconstructed_rho_bins: array
            Array of reconstructed density bins in scale_mass / scale_length^3.
        """

        _log_log_eddington_interp = get_interp(
            np.log(epsilon_bins), np.log(g_epsilon_bins)
        )

        def _rho_integrand(epsilon_prime, psi):
            return (
                4
                * np.pi
                * np.sqrt(2 * (psi - epsilon_prime))
                * np.exp(_log_log_eddington_interp(np.log(epsilon_prime)))
            )

        _reconstructed_rho_bins = []
        for _psi in tqdm(psi_bins, desc="Reconstructing densities"):
            _reconstructed_rho_bins.append(  # noqa: PERF401
                quad(_rho_integrand, 0, _psi, args=(_psi,))[0]
            )

        return np.array(_reconstructed_rho_bins)

    # LOSS CONE PROFILES CALCULATIONS
    def _get_Jc_sqr_bins(self, epsilon_bins):
        """
        Get the square angular momentum of a circular orbit bins.

        Parameters:
        ----------
        epsilon_bins: array
            Array of epsilon bins in scale_velocity^2.

        Returns:
        -------
        reduced_Jc_sqr_bins: array
            Array of Jc square bins in scale_length^2 * scale_velocity^2.
        """
        _orbit_Vc_bins = np.sqrt(self.mass_bins / self.r_bins)

        _orbit_epsilon_bins = -(self.phi_bins + 0.5 * _orbit_Vc_bins**2)
        _orbit_Jc_sqr_bins = self.r_bins**2 * _orbit_Vc_bins**2

        _log_log_Jc_sqr_interp = get_interp(
            np.log(_orbit_epsilon_bins), np.log(_orbit_Jc_sqr_bins)
        )

        return np.exp(_log_log_Jc_sqr_interp(np.log(epsilon_bins)))

    def _get_r_interps(self):
        """
        Get the interpolators for the r bins-related calculation.

        Returns:
        -------
        log_log_r_sqr_interp: function
            Interpolator for (natural log) r^2 as a function of (natural log) psi.
        log_log_neg_dr_dpsi_interp: function
            Interpolator for (natural log) - dr / dpsi as a function of (natural log) psi.
        """

        log_log_r_sqr_interp = get_interp(
            np.log(-self.phi_bins), 2 * np.log(self.r_bins)
        )

        _dr_dpsi_bins = (
            self.r_bins
            / (-self.phi_bins)
            * np.gradient(np.log(self.r_bins), np.log(-self.phi_bins))
        )

        log_log_neg_dr_dpsi_interp = get_interp(
            np.log(-self.phi_bins),
            np.log(-_dr_dpsi_bins),
        )

        return (log_log_r_sqr_interp, log_log_neg_dr_dpsi_interp)

    def _get_epsilon_interps(self):
        """
        Get the interpolators for epsilon functions.

        Returns:
        -------
        log_log_g_epsilon_interp: function
            Interpolator for (natural log) g as a function of (natural log) epsilon.
        log_log_I_0_bar_interp: function
            Interpolator for (natural log) I_0_bar as a function of (natural log) epsilon.
        """

        log_log_g_epsilon_interp = get_interp(
            np.log(self.epsilon_bins), np.log(self.g_epsilon_bins)
        )

        _I_0_bar_bins = cumulative_trapezoid(
            self.g_epsilon_bins, self.epsilon_bins, initial=0
        )
        log_log_I_0_bar_interp = get_interp(
            np.log(self.epsilon_bins), np.log(_I_0_bar_bins)
        )

        return (
            log_log_g_epsilon_interp,
            log_log_I_0_bar_interp,
        )

    def _get_h_epsilon_bins(self, epsilon_bins, max_psi):
        """
        Get the h bins for the loss cone calculation.

        Parameters:
        ----------
        epsilon_bins: array
            Array of epsilon bins in scale_velocity^2.
        max_psi: float
            Maximum psi value to integrate to in scale_velocity^2.

        Returns:
        -------
        reduced_h_epsilon_bins: array
            Array of h bins in scale_length^3 / scale_velocity.
        """

        (
            _log_log_r_sqr_interp,
            _log_log_neg_dr_dpsi_interp,
        ) = self._get_r_interps()

        (
            _log_log_g_epsilon_interp,
            _log_log_I_0_bar_interp,
        ) = self._get_epsilon_interps()

        def _I_n_bar_integrand(epsilon_prime, psi, n):
            return (psi - epsilon_prime) ** (n / 2) * np.exp(
                _log_log_g_epsilon_interp(np.log(epsilon_prime))
            )

        h_bins = []
        for i in tqdm(range(len(epsilon_bins)), desc="Calculating h(eps)"):
            _epsilon = epsilon_bins[i]

            def _h_integrand(psi):
                _r_sqr = np.exp(_log_log_r_sqr_interp(np.log(psi)))
                _neg_dr_dpsi = np.exp(_log_log_neg_dr_dpsi_interp(np.log(psi)))

                _epsilon_prime_bins = np.linspace(_epsilon, psi, self._N_trapz_bins)  # noqa: B023
                _I_1_bar = np.trapezoid(
                    _I_n_bar_integrand(_epsilon_prime_bins, psi, 1), _epsilon_prime_bins
                )
                _I_3_bar = np.trapezoid(
                    _I_n_bar_integrand(_epsilon_prime_bins, psi, 3), _epsilon_prime_bins
                )
                _I_0_bar = np.exp(_log_log_I_0_bar_interp(np.log(_epsilon)))  # noqa: B023

                return (
                    _r_sqr
                    * _neg_dr_dpsi
                    * (
                        3 * (psi - _epsilon) ** (-1) * _I_1_bar  # noqa: B023
                        - (psi - _epsilon) ** (-2) * _I_3_bar  # noqa: B023
                        + 2 * (psi - _epsilon) ** (-1 / 2) * _I_0_bar  # noqa: B023
                    )
                )

            h_bins.append(quad(_h_integrand, _epsilon, max_psi)[0])

        return np.array(h_bins)

    def _get_reduced_profiles(self):
        _mask = np.arange(0, self._N_bins, self._reduce_factor)

        self.reduced_epsilon_bins = self.epsilon_bins[_mask]
        self.reduced_g_epsilon_bins = self.g_epsilon_bins[_mask]

        self.reduced_Jc_sqr_bins = self._get_Jc_sqr_bins(self.reduced_epsilon_bins)

        self.reduced_h_epsilon_bins = self._get_h_epsilon_bins(
            self.reduced_epsilon_bins, -self.phi_bins[0]
        )


class SingularIsothermalSphereProfile(BaseProfile):
    def _get_stellar_rho_bins(self, r_bins):
        """
        Get the stellar density profile of the singular isothermal sphere halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in scale_length.

        Returns:
        -------
        rho_bins: array
            Array of density bins in scale_mass / scale_length^3.
        """

        return 1 / (2 * np.pi * r_bins**2)


class PlummerCuspProfile(BaseProfile):
    def __init__(self, M_s, *args, **kwargs):
        """
        Initialize the Plummer cusp profile class. The Plummer cusp profile is defined
        as a Plummer profile with a Bahcall-Wolf cusp inside the radius of influence.
        The scale length is [fill in info]

        Parameters:
        ----------
        M_s: float
            The total stellar mass of the halo in scale_mass (i.e. the mass of the central black hole).
        """

        self.M_s = M_s

        self._mu = 1 / self.M_s
        self._a = np.sqrt(
            (1 - self._mu ** (2 / 3)) / self._mu ** (2 / 3)
        )  # np.sqrt(13/7)

        super().__init__(*args, **kwargs)

    def _get_Plummer_rho_bins(self, r_bins):
        """
        Get the stellar density profile of the Plummer halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in scale_length.

        Returns:
        -------
        rho_bins: array
            Array of density bins in scale_mass / scale_length^3.
        """

        return (
            3
            / self._mu
            / (4 * np.pi)
            / self._a**3
            * (1 + r_bins**2 / self._a**2) ** (-5 / 2)
        )

    def _get_Bahcall_Wolf_rho_bins(self, r_bins):
        """
        Get the stellar density profile of the Bahcall-Wolf cusp halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in scale_length.

        Returns:
        -------
        rho_bins: array
            Array of density bins in scale_mass / scale_length^3.
        """

        _rho0 = self._get_Plummer_rho_bins(1)

        return _rho0 * r_bins ** (-7 / 4)

    def _get_stellar_rho_bins(self, r_bins):
        """
        Get the stellar density profile of the Plummer cusp halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in scale_length.

        Returns:
        -------
        rho_bins: array
            Array of density bins in scale_mass / scale_length^3.
        """

        return np.where(
            r_bins < 1,
            self._get_Bahcall_Wolf_rho_bins(r_bins),
            self._get_Plummer_rho_bins(r_bins),
        )


class HernquistProfile(BaseProfile):
    def __init__(self, M_s, *args, **kwargs):
        """
        Initialize the Hernquist profile class. The scale length (i.e. the influent radius
        of the central black hole) is defined as the radius where the enclosed stellar mass
        is equal to the mass of the central black hole, i.e. M_s(<r_h) = M_s.
        This gives a stellar scale radius of a = sqrt(M_s) - 1 in scale length.
        """

        self.M_s = M_s
        self.a = np.sqrt(M_s) - 1

        super().__init__(*args, **kwargs)

    def _get_stellar_rho_bins(self, r_bins):
        """
        Get the stellar density profile of the Hernquist halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in scale_length.

        Returns:
        -------
        rho_bins: array
            Array of density bins in scale_mass / scale_length^3.
        """

        return self.M_s / (2 * np.pi) * self.a / r_bins / (r_bins + self.a) ** 3
