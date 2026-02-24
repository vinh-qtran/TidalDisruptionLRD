import numpy as np
from astropy import units as u

from tidaldisruptionlrd.constants import G, Rsun


class TDRRate:
    def __init__(
        self,
        eta_ast_bins,
        g_ast_bins,
        h_ast_bins,
        Jc_sqr_ast_bins,
        # P_ast_bins,
        M_bh,
        sigma,
        m_s,
        r_s,
        eta=0.844,
    ):
        """
        Initialize the TDR rate class.

        Parameters:
        ----------
        eta_ast_bins: array
            Array of eta_ast bins.
        g_ast_bins: array
            Array of g_ast bins.
        h_ast_bins: array
            Array of h_ast bins.
        Jc_sqr_ast_bins: array
            Array of Jc_sqr_ast bins.
        #P_ast_bins: array
        #    Array of P_ast bins.

        M_bh: array or float
            Black hole mass in M_sun.
        sigma: array or float
            Velocity dispersion of the host galaxy in km/s.
        m_s: array or float
            Mass of the stars in M_sun.
        r_s: array or float
            Radius of the stars in R_sun.

        eta: float, optional
            The tidal disruption parameter, eta=0.844 for n=3 polytrope. Default is 0.844.
        """

        self._eta_ast_bins = eta_ast_bins
        self._g_ast_bins = g_ast_bins
        self._h_ast_bins = h_ast_bins
        self._Jc_sqr_ast_bins = Jc_sqr_ast_bins
        # self._P_ast_bins = P_ast_bins

        self.M_bh = M_bh
        self.sigma = sigma
        self.m_s = m_s
        self.r_s = r_s * Rsun
        self.eta = eta

        self._timescale = (
            G * M_bh / sigma**3 * (1 * u.kpc / (u.km / u.s)).to_value(u.yr)
        )

        self._m_s_ast_bins = m_s / M_bh
        self._r_t_ast_bins = (
            (G * self.m_s / self.r_s) ** (-1)
            * eta ** (2 / 3)
            * self.sigma**2
            * self._m_s_ast_bins ** (2 / 3)
        )

        self.N_rate = []
        for _m_s_ast, _r_t_ast, _timescale in zip(
            self._m_s_ast_bins, self._r_t_ast_bins, self._timescale, strict=False
        ):
            _N_rate = self._get_single_N_rate_ast(_m_s_ast, _r_t_ast) / _timescale
            self.N_rate.append(_N_rate)
        self.N_rate = np.array(self.N_rate)

    def _get_q_bins(self, m_s_ast, r_t_ast):
        """
        Get the q bins for the TDR rate calculation.

        Parameters:
        ----------
        m_s_ast: float
            Mass ratio of the stars to the black hole, m_s / M_bh.
        r_t_ast: float
            Tidal radius in units of the influence radius, r_t / r_h.

        Returns:
        -------
        q_bins: array
            Array of q bins.
        """

        _scaler = (
            32
            * np.pi**2
            / 3
            / np.sqrt(2)
            * np.log(0.4 / m_s_ast)
            * m_s_ast
            * r_t_ast ** (-2)
        )

        return _scaler * self._h_ast_bins / (r_t_ast ** (-1) - self._eta_ast_bins)

    def _get_ln_R0_bins(self, q_bins, r_t_ast):
        """
        Get the natural log R0 bins for the TDR rate calculation.

        Parameters:
        ----------
        q_bins: array
            Array of q bins.
        r_t_ast: float
            Tidal radius in units of the influence radius, r_t / r_h.

        Returns:
        -------
        ln_R0_bins: array
            Array of natural log R0 bins.
        """

        _scaler = 2 * r_t_ast**2

        return np.log(
            _scaler * (r_t_ast ** (-1) - self._eta_ast_bins) / self._Jc_sqr_ast_bins
        ) + np.where(
            q_bins > 1,
            -q_bins,
            -0.186 * q_bins - 0.824 * np.sqrt(q_bins),
        )

    def _get_F_ast_bins(self, ln_R0_bins, m_s_ast):
        """
        Get the F_ast bins for the TDR rate calculation.

        Parameters:
        ----------
        ln_R0_bins: array
            Array of natural log R0 bins.
        m_s_ast: float
            Mass ratio of the stars to the black hole, m_s / M_bh.

        Returns:
        -------
        F_ast_bins: array
            Array of F_ast bins.
        """

        _scaler = 256 * np.pi**4 / 3 / np.sqrt(2) * np.log(0.4 / m_s_ast)

        return (
            _scaler / (-ln_R0_bins) * self._g_ast_bins * self._h_ast_bins
        )  # / self._P_ast_bins

    def _get_single_N_rate_ast(self, m_s_ast, r_t_ast):
        """
        Get the TDR rate for a single set of parameters.

        Parameters:
        ----------
        m_s_ast: float
            Mass ratio of the stars to the black hole, m_s / M_bh.
        r_t_ast: float
            Tidal radius in units of the influence radius, r_t / r_h.

        Returns:
        -------
        N_rate: float
            Dimensionless TDR rate

        """

        q_bins = self._get_q_bins(m_s_ast, r_t_ast)
        ln_R0_bins = self._get_ln_R0_bins(q_bins, r_t_ast)
        F_ast_bins = self._get_F_ast_bins(ln_R0_bins, m_s_ast)

        return np.trapezoid(F_ast_bins, self._eta_ast_bins)
