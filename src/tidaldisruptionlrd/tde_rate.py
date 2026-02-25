import numpy as np
from astropy import units as u
from tqdm import tqdm

from tidaldisruptionlrd.constants import G, Rsun


class BaseTDERate:
    def __init__(self, dimensionless_profile, m_s_bins, M_bhs, r_hs, eta=0.844):
        """
        Initialize the base TDE rate class.

        Parameters:
        ----------
        dimensionless_profile: Profile
            The dimensionless profile of the stellar distribution.

        m_s_bins: array
            Array of stellar mass bins in M_sun.

        M_bhs: array
            Array of black hole masses in M_sun.
        r_hs: array
            Array of black hole influence radii in pc.

        eta: float, optional
            The tidal disruption parameter, eta=0.844 for n=3 polytrope. Default is 0.844.
        """

        self._read_profile(dimensionless_profile)

        self.M_bhs = M_bhs
        self.r_hs = r_hs

        self._eta = eta

        self._m_s_bins = m_s_bins
        self._mass_func_bins = self._get_mass_function_bins(m_s_bins)
        self._m_avg, self._m_sqr_avg = self._get_mass_moment(
            self._m_s_bins, self._mass_func_bins
        )

        self._timescales = (
            r_hs / np.sqrt(G * M_bhs / r_hs) * (1 * u.kpc / (u.km / u.s)).to_value(u.yr)
        )

        _N_TDEs = []
        for _i in tqdm(range(len(self.M_bhs)), desc="Calculating TDE rates"):
            _M_bh = self.M_bhs[_i]
            _r_h = self.r_hs[_i]
            _timescale = self._timescales[_i]

            _N_TDE_bins = self._get_N_TDE_bins(_M_bh, _r_h)
            _N_TDE = (
                np.trapezoid(_N_TDE_bins * self._mass_func_bins, self._m_s_bins)
                / _timescale
            )

            _N_TDEs.append(_N_TDE)
        self.N_TDEs = np.array(_N_TDEs)

    def _read_profile(self, dimensionless_profile):
        """
        Read the dimensionless profile stored in the reduced data bins of the Profile class.

        Set Attributes:
        ----------
        _epsilon_bar_bins: array
            Array of dimensionless epsilon bins.
        _g_bar_bins: array
            Array of g(epsilon) bins.
        _h_bar_bins: array
            Array of h(epsilon) bins.
        _Jc_sqr_bar_bins: array
            Array of dimensionless Jc^2 bins.
        """

        self._epsilon_bar_bins = dimensionless_profile.reduced_epsilon_bins
        self._g_bar_bins = dimensionless_profile.reduced_g_epsilon_bins
        self._h_bar_bins = dimensionless_profile.reduced_h_epsilon_bins
        self._Jc_sqr_bar_bins = dimensionless_profile.reduced_Jc_sqr_bins

    def _get_mass_function_bins(self, m_s_bins):
        """
        Get the mass function bins for the given stellar mass bins.

        Parameters:
        ----------
        m_s_bins: array
            Array of stellar mass bins in M_sun.
        """

        raise NotImplementedError("Not implemented in base class.")  # noqa: EM101

    def _get_mass_moment(self, m_s_bins, mass_func_bins):
        """
        Get the mass moments for the given stellar mass bins and mass function bins.

        Parameters:
        ----------
        m_s_bins: array
            Array of stellar mass bins in M_sun.
        mass_func_bins: array
            Array of mass function bins corresponding to the stellar mass bins.

        Returns:
        -------
        m_avg: float
            The average stellar mass in M_sun.
        m_sqr_avg: float
            The average of the square of the stellar mass in M_sun^2.
        """

        _normalizer = 1 / np.trapezoid(mass_func_bins, m_s_bins)

        m_avg = _normalizer * np.trapezoid(m_s_bins * mass_func_bins, m_s_bins)
        m_sqr_avg = _normalizer * np.trapezoid(m_s_bins**2 * mass_func_bins, m_s_bins)

        return m_avg, m_sqr_avg

    def _get_r_t_bins(self, m_s_bins, M_bh):
        """
        Get the tidal radius bins for the given stellar mass bins and black hole mass.

        Parameters:
        ----------
        m_s_bins: array
            Array of stellar mass bins in M_sun.
        M_bh: float
            Black hole mass in M_sun.

        Returns:
        -------
        r_t_bins: array
            Array of tidal radius bins in pc.
        """

        return Rsun * self._eta ** (2 / 3) * (m_s_bins / M_bh) ** 0.467 * M_bh**0.8

    def _get_q_bins(self, M_bh, r_t, r_h):
        """
        Get the q bins for the TDE rate calculation.

        Parameters:
        ----------
        M_bh: float
            Black hole mass in M_sun.
        r_t: float
            Tidal radius in pc.
        r_h: float
            Influence radius in pc.

        Returns:
        -------
        q_bins: array
            Array of q bins.
        """

        _Lambda = 0.4 * M_bh / self._m_avg

        _scaler = (
            32
            * np.pi**2
            / (3 * np.sqrt(2))
            * np.log(_Lambda)
            * (self._m_sqr_avg / M_bh / self._m_avg)
            * (r_t / r_h) ** (-2)
        )

        return (
            _scaler * self._h_bar_bins / ((r_t / r_h) ** (-1) - self._epsilon_bar_bins)
        )

    def _get_ln_R0_bins(self, q_bins, r_t, r_h):
        """
        Get the natural log R0 bins for the TDE rate calculation.

        Parameters:
        ----------
        q_bins: array
            Array of q bins.
        r_t: float
            Tidal radius in pc.
        r_h: float
            Influence radius in pc.

        Returns:
        -------
        ln_R0_bins: array
            Array of natural log R0 bins.
        """

        _scaler = 2 * (r_t / r_h) ** 2

        return np.log(
            _scaler
            * ((r_t / r_h) ** (-1) - self._epsilon_bar_bins)
            * self._h_bar_bins
            / self._Jc_sqr_bar_bins
        ) + np.where(
            q_bins > 1,
            -q_bins,
            -0.186 * q_bins - 0.824 * np.sqrt(q_bins),
        )

    def _get_F_bar_bins(self, ln_R0_bins, M_bh):
        """
        Get the F_bar bins for the TDE rate calculation.

        Parameters:
        ----------
        ln_R0_bins: array
            Array of natural log R0 bins.
        M_bh: float
            Black hole mass in M_sun.

        Returns:
        -------
        F_bar_bins: array
            Array of F_bar bins.
        """

        _Lambda = 0.4 * M_bh / self._m_avg

        _scaler = (
            256
            * np.pi**4
            / 3
            / np.sqrt(2)
            * np.log(_Lambda)
            * (self._m_sqr_avg / self._m_avg**2)
        )

        return _scaler / (-ln_R0_bins) * self._h_bar_bins * self._g_bar_bins

    def _get_N_TDE_bins(self, M_bh, r_h):
        """
        Get the N_TDE bins for the TDE rate calculation.

        Parameters:
        ----------
        M_bh: float
            Black hole mass in M_sun.
        r_h: float
            Influence radius in pc.

        Returns:
        -------
        N_TDE_bins: array
            Array of N_TDE bins.
        """

        _r_t_bins = self._get_r_t_bins(self._m_s_bins, M_bh)

        _N_TDE_bins = []
        for _r_t in _r_t_bins:
            _q_bins = self._get_q_bins(M_bh, _r_t, r_h)
            _ln_R0_bins = self._get_ln_R0_bins(_q_bins, _r_t, r_h)
            _F_bar_bins = self._get_F_bar_bins(_ln_R0_bins, M_bh)

            _N_TDE_bins.append(np.trapezoid(_F_bar_bins, self._epsilon_bar_bins))

        return np.array(_N_TDE_bins)


class SingleMassTDERate(BaseTDERate):
    """
    Class for calculating the TDE rate for a single mass stellar population.
    """

    def _get_mass_function_bins(self, m_s_bins, m_s0=1):
        _m_s_bin_centers = 0.5 * (m_s_bins[1:] + m_s_bins[:-1])

        _m_s0_idx = np.argmin(np.abs(_m_s_bin_centers - m_s0))

        _mass_function_bins = np.zeros_like(m_s_bins)
        _mass_function_bins[_m_s0_idx] = 2 / (
            _m_s_bin_centers[_m_s0_idx + 1] - _m_s_bin_centers[_m_s0_idx - 1]
        )

        return _mass_function_bins
