import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from tqdm import tqdm

from tidaldisruptionlrd.constants import G
from tidaldisruptionlrd.utils import get_interp


class DiffusionCoefficient:
    def __init__(
        self,
        r_bins,
        stellar_mass_bins,
        M_bh,
        phi_bins,
        eta_bins,
        f_eta_bins,
        reduce_factor=10,
        N_bins=1000,
        G=G,
    ):
        """
        Initialize the diffusion coefficient class.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        stellar_mass_bins: array
            Array of stellar mass bins in M_sun.
        M_bh: float
            Black hole mass in M_sun.
        phi_bins: array
            Array of potential bins in (km/s)^2.
        eta_bins: array
            Array of eta bins.
        f_eta_bins: array
            Array of distribution function bins in M_sun / (kpc^3 * (km/s)^3).

        reduce_factor: int, optional
            Factor by which to reduce the number of bins for the diffusion coefficient calculation. Default is 10.
        N_bins: int, optional
            Number of bins to use for the diffusion coefficient calculation. Default is 1000.

        G: float, optional
            Gravitational constant in kpc^3 / (M_sun * Gyr^2). Default is G from tidaldisruptionlrd.constants.
        """

        self._reduce_factor = reduce_factor
        self._G = G

        self.r_bins = self._reduce_bins(r_bins)
        self.mass_bins = self._reduce_bins(stellar_mass_bins + M_bh)
        self.M_bh = M_bh
        self.psi_bins = -self._reduce_bins(phi_bins)
        self.eta_bins = self._reduce_bins(eta_bins)
        self.f_eta_bins = self._reduce_bins(f_eta_bins)

        self.Jc_sqr_bins = self._get_Jc_sqr_bins()

        self._N_bins = N_bins

        self.scaled_diff_coeff_bins = self._get_scaled_diff_coeff_bins()

    def _reduce_bins(self, bins):
        """
        Reduce the number of bins by the reduce factor.

        Parameters:
        ----------
        bins: array
            Array of bins to reduce.

        Returns:
        -------
        reduced_bins: array
            Array of reduced bins.
        """

        _mask = np.arange(0, bins.shape[0], self._reduce_factor)

        return bins[_mask]

    def _get_r_interps(self):
        """
        Get the interpolated profiles of the halo as a function of radius.

        Returns:
        -------
        lin_log_r_interp: CubicSpline
            Interpolated profile of the (log) radius as a function of negative potential.
        log_log_neg_dr_dpsi_interp: CubicSpline
            Interpolated profile of the (negative log) derivative of radius with respect to negative potential as a function of negative potential
        """

        lin_log_r_interp = get_interp(self.psi_bins, np.log(self.r_bins))

        _dlnr_dpsi_bins = np.gradient(np.log(self.r_bins), self.psi_bins)

        lin_log_neg_dr_dpsi_interp = get_interp(
            self.psi_bins,
            np.log(-self.r_bins * _dlnr_dpsi_bins),
        )

        return lin_log_r_interp, lin_log_neg_dr_dpsi_interp

    def _get_Jc_sqr_bins(self):
        """
        Get the square angular momentum of a circular orbit bins.

        Returns:
        -------
        Jc_sqr_bins: array
            Array of Jc square bins in (kpc * km/s)^2.
        """
        _orbit_Vc_bins = np.sqrt(self._G * self.mass_bins / self.r_bins)

        _orbit_eta_bins = self.psi_bins - 0.5 * _orbit_Vc_bins**2
        _orbit_Jc_sqr_bins = self.r_bins**2 * _orbit_Vc_bins**2

        _lin_log_Jc_sqr_interp = get_interp(_orbit_eta_bins, np.log(_orbit_Jc_sqr_bins))

        return np.exp(_lin_log_Jc_sqr_interp(self.eta_bins))

    def _get_lin_log_f_eta_interp(self):
        """
        Get the interpolated profile of the (log) distribution function as a function of eta.

        Returns:
        -------
        log_log_f_eta_interp: CubicSpline
            Interpolated profile of the (log) distribution function as a function of eta.
        """

        return get_interp(self.eta_bins, np.log(self.f_eta_bins))

    def _get_I0_bins(self):
        """
        Get the I0 bins.

        Returns:
        -------
        I0_bins: array
            Array of I0 bins.
        """

        return cumulative_trapezoid(self.f_eta_bins, self.eta_bins, initial=0)

    def _get_scaled_diff_coeff_bins(self):
        """
        Get the scaled orbit-averaged diffusion coefficient bins.

        Returns:
        -------
        scaled_diff_coeff_bins: array
            Array of scaled orbit-averaged diffusion coefficient bins in (km/s)^2.
        """

        _lin_log_r_interp, _lin_log_neg_dr_dpsi_interp = self._get_r_interps()
        _lin_log_f_eta_interp = self._get_lin_log_f_eta_interp()

        _I0_bins = self._get_I0_bins()

        def _I_n_integrand(eta_prime, psi, n):
            return (2 * (psi - eta_prime)) ** (n / 2) * np.exp(
                _lin_log_f_eta_interp(eta_prime)
            )

        scaled_diff_coeff_bins = []
        for i in tqdm(
            range(self.eta_bins.shape[0]), desc="Computing diffusion coefficients"
        ):
            _eta = self.eta_bins[i]

            def _w_bar_integrand(psi):
                _r = np.exp(_lin_log_r_interp(psi))
                _dr_dpsi = -np.exp(_lin_log_neg_dr_dpsi_interp(psi))

                _eta_prime_bins = np.linspace(_eta, psi, self._N_bins)  # noqa: B023
                _I_1 = (2 * (psi - _eta)) ** (-1 / 2) * np.trapezoid(  # noqa: B023
                    _I_n_integrand(_eta_prime_bins, psi, 1), _eta_prime_bins
                )
                _I_3 = (2 * (psi - _eta)) ** (-3 / 2) * np.trapezoid(  # noqa: B023
                    _I_n_integrand(_eta_prime_bins, psi, 3), _eta_prime_bins
                )

                _w_integrand = _r**2 * (
                    3 * _I_1 - _I_3 + 2 * _I0_bins[i]  # noqa: B023
                )

                return 2 * _dr_dpsi * (2 * (psi - _eta)) ** (-1 / 2) * _w_integrand  # noqa: B023

            scaled_diff_coeff_bins.append(
                1
                / self.Jc_sqr_bins[i]
                * quad(
                    _w_bar_integrand,
                    np.max(self.psi_bins),
                    _eta,
                )[0]
            )

        return np.array(scaled_diff_coeff_bins)
