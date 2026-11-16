import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from tqdm import tqdm

from tidaldisruptionlrd.constants import (
    G25_number_density,
    z45_dilation_factor,
    z45_shell_volume,
    z56_dilation_factor,
    z56_shell_volume,
)

repo_root = Path(__file__).resolve().parents[2]


class TDEGrid:
    def __init__(
        self,
        profile,
        tde,
        M_bhs,
        M_s_scalers,
        m_s_max=2,
        r_env=None,
        profile_params={},  # noqa: B006
        r_half_mass_params={"M0": 3.16e8, "r0": 0.381, "alpha": 0.27},  # noqa: B006
        r_half_mass_a_scaler=None,
    ):
        """
        Initialize the TDEGrid class, which computes TDE rates for a grid of black hole masses and stellar mass scalers.

        Parameters:
        ----------
        profile : callable
            The profile function to be used for computing TDE rates.
        tde : callable
            The TDE function to be used for computing TDE rates.

        M_bhs : array-like
            The black hole masses for which to compute TDE rates.
        M_s_scalers : array-like
            The stellar mass scalers for which to compute TDE rates.
        m_s_max : float, optional
            The maximum stellar mass for which to compute TDE rates. If not provided, it is set to 2 by default.

        r_env: array, optional
            Array of gas envelope radius in kpc. If given, it will be used instead of the tidal radius for the TDE rate calculation. Default is None.

        profile_params : dict, optional
            The parameters for the profile function, with keys depending on the specific profile used. If not provided, an empty dictionary is used.

        r_half_mass_params : dict, optional
            The parameters for the M-r_half_mass relation, with keys "M0", "r0", and "alpha". If not provided, default values are used.
        r_half_mass_a_scaler : float, optional
            The scaling factor for the half-mass radius in units of the scale radius. If not provided, it is computed from the profile.
        """

        self._profile_params = profile_params
        self._r_half_mass_params = r_half_mass_params
        self._r_half_mass_a_scaler = r_half_mass_a_scaler

        self._profile = profile
        self._tde = tde

        self.M_bhs = M_bhs
        self.M_s_scalers = M_s_scalers
        self._m_s_max = m_s_max

        self._r_env = r_env

        _N_TDEs = []
        for _M_s_scaler in tqdm(self.M_s_scalers):
            _N_TDEs.append(self._get_single_TDE_rates(_M_s_scaler))  # noqa: PERF401

        self.N_TDEs = np.vstack(_N_TDEs)

    def _get_r_half_mass(self, M_s, M0, r0, alpha):
        return r0 * (M_s / M0) ** alpha

    def _get_single_TDE_rates(self, M_s_scaler):
        _profile = self._profile(
            M_s_scaler,
            **self._profile_params,
            r_bin_min=1e-4,
            r_bin_max=1e6,
            N_bins=4000,
            reduce_factor=10,
            N_trapz_bins=100,
            show_progress=False,
        )

        _r_half_masses = self._get_r_half_mass(
            self.M_bhs * M_s_scaler, **self._r_half_mass_params
        )
        _r_half_mass_a_scaler = (
            self._r_half_mass_a_scaler
            or _profile.get_2D_half_mass_radius() / _profile.a
        )

        _as = _r_half_masses / _r_half_mass_a_scaler

        _r_hs = _as / _profile.a

        _tde = self._tde(
            m_s_max=self._m_s_max,
            dimensionless_profile=_profile,
            M_bhs=self.M_bhs,
            r_hs=_r_hs,
            m_s_bins=np.linspace(0.08, 10, 1000),
            eta=0.844,
            r_env=self._r_env,
            show_progress=False,
        )

        return _tde.N_TDEs


def KH13_logMs_from_logMbh(log_Mbh):
    """
    The M_bh-Ms relation from Kormendy & Ho (2013).
    """
    return 11 + (np.log10(1 / 0.49) + (log_Mbh - 9)) / 1.16


def RV15_logMs_from_logMbh(log_Mbh):
    """
    The M_bh-Ms relation from  Reines & Volonteri (2015).
    """
    return 11 + (log_Mbh - 7.45) / 1.05


def G20_logMs_from_logMbh(log_Mbh):
    """
    The M_bh-Ms relation from Greene (2020).
    """
    return np.log10(3) + 10 + (log_Mbh - 7.56) / 1.39


def P23_logMs_from_logMbh(log_Mbh):
    """
    The M_bh-Ms relation from Pacucci (2023).
    """
    return (2.43 + log_Mbh) / 1.06


def L25_logMs_from_logMbh(log_Mbh):
    """
    The M_bh-Ms relation from Li (2025).
    """
    return (3.32 + log_Mbh) / 1.02


class LRDNum:
    """
    Host population with scatter in M* at fixed M•: the number density is
    defined on a grid in log M• (``BHMF_type``), and log M* is Gaussian about
    the relation with width sigma_• / slope.
    """

    _logMs_from_logMbh_funcs = {  # noqa: RUF012
        "KH13": KH13_logMs_from_logMbh,
        "RV15": RV15_logMs_from_logMbh,
        "G20": G20_logMs_from_logMbh,
        "P23": P23_logMs_from_logMbh,
        "L25": L25_logMs_from_logMbh,
    }

    # Intrinsic scatter of each relation in log M• at fixed M* (dex).
    _log_Mbh_scatters = {  # noqa: RUF012
        "KH13": 0.29,
        "RV15": 0.55,
        "G20": 0.79,
        "P23": 0.69,
        "L25": 0.97,
    }

    _log_Ms_scatters = {  # noqa: RUF012
        "KH13": 0.29 / 1.16,
        "RV15": 0.55 / 1.05,
        "G20": 0.79 / 1.39,
        "P23": 0.69 / 1.06,
        "L25": 0.97 / 1.02,
    }

    _Inayoshi24_files = {  # noqa: RUF012
        "Inayoshi24_z4_Obscured": repo_root / "data/Inayoshi24/z4_Solid_Obscured.csv",
        "Inayoshi24_z4_Unobscured": repo_root
        / "data/Inayoshi24/z4_Dashed_Unobscured.csv",
        "Inayoshi24_z5_Obscured": repo_root / "data/Inayoshi24/z5_Solid_Obscured.csv",
        "Inayoshi24_z5_Unobscured": repo_root
        / "data/Inayoshi24/z5_Dashed_Unobscured.csv",
    }

    row_variable = "log_Mbh"

    def __init__(self, log_Mbh_bins, delta_log_Ms_scaler, Ms_Mbh_type, BHMF_type):

        self.log_Mbh_bins = log_Mbh_bins
        self.log_Ms_bins = self._logMs_from_logMbh_funcs[Ms_Mbh_type](log_Mbh_bins)

        self.delta_log_Ms_bins = (
            delta_log_Ms_scaler * self._log_Ms_scatters[Ms_Mbh_type]
        )

        if BHMF_type == "uniform":
            self.BHMF_bins = np.full_like(log_Mbh_bins, G25_number_density) / (
                log_Mbh_bins[-1] - log_Mbh_bins[0]
            )

        elif BHMF_type == "gaussian":
            self.BHMF_bins = (
                norm.pdf(log_Mbh_bins, loc=log_Mbh_bins.mean(), scale=1)
                * G25_number_density
                / (norm.cdf(1) - norm.cdf(-1))
            )

        elif BHMF_type == "Greene25":
            self._read_Greene25_DF()

            self.BHMF_bins = self._Greene25_BHMF_interp(log_Mbh_bins)

        elif BHMF_type.split("_")[0] == "Inayoshi24":
            self._read_Inayoshi24_DF(self._Inayoshi24_files[BHMF_type])
            self.BHMF_bins = self._Inayoshi24_BHMF_interp(log_Mbh_bins)

        else:
            raise ValueError("Invalid BHMF_type.")  # noqa: EM101, TRY003

        if Ms_Mbh_type == "L25":
            self.BHMF_bins /= 0.09

        self.p_delta_log_Ms_bins = (
            norm.pdf(delta_log_Ms_scaler) / self._log_Ms_scatters[Ms_Mbh_type]
        )

        # Common 2D interface read by AllSkyTDERate (rows: log M•, cols: scatter).
        _shape = (len(self.log_Mbh_bins), len(self.delta_log_Ms_bins))
        self.log_Mbh_grid = np.broadcast_to(
            np.asarray(self.log_Mbh_bins).reshape(-1, 1), _shape
        ).copy()
        self.log_Ms_grid = self.log_Ms_bins.reshape(-1, 1) + self.delta_log_Ms_bins
        self.p_grid = np.broadcast_to(
            self.p_delta_log_Ms_bins
            * (self.delta_log_Ms_bins[1] - self.delta_log_Ms_bins[0]),
            _shape,
        ).copy()
        self.row_number_density = self.BHMF_bins * (
            self.log_Mbh_bins[1] - self.log_Mbh_bins[0]
        )

    def _read_Greene25_DF(self):
        _G25_Lum = []
        _G25_LF = []

        with open(repo_root / "data/Greene25/z4-6_LF.csv") as f:  # noqa: PTH123
            for line in csv.reader(f):
                _G25_Lum.append(float(line[0]))
                _G25_LF.append(float(line[1]))

        self._G25_Lum = np.array(_G25_Lum)  # erg/s
        self._G25_LF = np.array(_G25_LF)  # Mpc^-3 dex^-1

        self._G25_BHMass = self._G25_Lum / 1.26e38  # M_sun
        self._G25_BHMF = self._G25_LF  # Mpc^-3 dex^-1

    def _Greene25_BHMF_interp(self, log_Mbh):
        _log_G25_BHMass = np.log10(self._G25_BHMass)
        _log_G25_BHMF = np.log10(self._G25_BHMF)

        return 10 ** np.where(
            log_Mbh < _log_G25_BHMass[1],
            (
                _log_G25_BHMF[1]
                + (_log_G25_BHMF[1] - _log_G25_BHMF[0])
                / (_log_G25_BHMass[1] - _log_G25_BHMass[0])
                * (log_Mbh - _log_G25_BHMass[1])
            ),
            (
                _log_G25_BHMF[1]
                + (_log_G25_BHMF[2] - _log_G25_BHMF[1])
                / (_log_G25_BHMass[2] - _log_G25_BHMass[1])
                * (log_Mbh - _log_G25_BHMass[1])
            ),
        )

    def _read_Inayoshi24_DF(self, BHMF_file):
        _I24_BHMass = []
        _I24_BHMF = []

        with open(BHMF_file) as f:  # noqa: PTH123
            for line in csv.reader(f):
                _I24_BHMass.append(float(line[0]))
                _I24_BHMF.append(float(line[1]))

        self._I24_BHMass = np.array(_I24_BHMass)  # M_sun
        self._I24_BHMF = np.array(_I24_BHMF)  # Mpc^-3 dex^-1

    def _Inayoshi24_BHMF_interp(self, log_Mbh):
        _log_I24_BHMass = np.log10(self._I24_BHMass)
        _log_I24_BHMF = np.log10(self._I24_BHMF)

        return 10 ** np.interp(log_Mbh, _log_I24_BHMass, _log_I24_BHMF)


class LRDNumScatterMbh:
    """
    Host population as described in the paper: the number density is
    distributed in log M* over ``log_Ms_bins`` (e.g. 7-9), and log M• is
    Gaussian about the relation at fixed M* with the relation's intrinsic
    scatter sigma_• (in dex of M•).

    Parameters
    ----------
    log_Ms_bins : array
        Grid in log M* [M_sun]. The full number density is spread over this range.
    delta_log_Mbh_scaler : array
        Offsets in units of sigma_• (e.g. np.linspace(-5, 5, N)).
    Ms_Mbh_type : str
        One of "KH13", "RV15", "G20", "P23", "L25".
    SMF_type : str
        "uniform" (flat in log M*) or "gaussian" (same form as LRDNum's, in log M*).
        Data-driven mass functions (Greene25, Inayoshi24) are defined in M•,
        so use LRDNum for those.
    """

    row_variable = "log_Ms"

    def __init__(self, log_Ms_bins, delta_log_Mbh_scaler, Ms_Mbh_type, SMF_type):
        _logMs_from_logMbh = LRDNum._logMs_from_logMbh_funcs[Ms_Mbh_type]
        # The relations are linear: log M* = c + s log M•  ->  invert exactly.
        _c = _logMs_from_logMbh(0.0)
        _s = _logMs_from_logMbh(1.0) - _c

        self.log_Ms_bins = np.asarray(log_Ms_bins, dtype=float)
        self.log_Mbh_bins = (self.log_Ms_bins - _c) / _s  # mean log M• at fixed M*

        _sigma = LRDNum._log_Mbh_scatters[Ms_Mbh_type]
        self.delta_log_Mbh_bins = np.asarray(delta_log_Mbh_scaler) * _sigma
        self.p_delta_log_Mbh_bins = norm.pdf(delta_log_Mbh_scaler) / _sigma

        if SMF_type == "uniform":
            self.SMF_bins = np.full_like(self.log_Ms_bins, G25_number_density) / (
                self.log_Ms_bins[-1] - self.log_Ms_bins[0]
            )
        elif SMF_type == "gaussian":
            self.SMF_bins = (
                norm.pdf(self.log_Ms_bins, loc=self.log_Ms_bins.mean(), scale=1)
                * G25_number_density
                / (norm.cdf(1) - norm.cdf(-1))
            )
        else:
            raise ValueError(  # noqa: TRY003
                f"SMF_type {SMF_type!r} not supported here; "  # noqa: EM102
                "mass functions defined in M• should use LRDNum."
            )

        if Ms_Mbh_type == "L25":
            self.SMF_bins /= 0.09

        # Common 2D interface read by AllSkyTDERate (rows: log M*, cols: scatter).
        _shape = (len(self.log_Ms_bins), len(self.delta_log_Mbh_bins))
        self.log_Ms_grid = np.broadcast_to(
            self.log_Ms_bins.reshape(-1, 1), _shape
        ).copy()
        self.log_Mbh_grid = self.log_Mbh_bins.reshape(-1, 1) + self.delta_log_Mbh_bins
        self.p_grid = np.broadcast_to(
            self.p_delta_log_Mbh_bins
            * (self.delta_log_Mbh_bins[1] - self.delta_log_Mbh_bins[0]),
            _shape,
        ).copy()
        self.row_number_density = self.SMF_bins * (
            self.log_Ms_bins[1] - self.log_Ms_bins[0]
        )


class AllSkyTDERate:
    def __init__(self, tde_grid, lrd_num_z4, lrd_num_z5=None):
        self._fill_tde_grid(tde_grid)
        self._get_tde_interp(tde_grid)

        if lrd_num_z5 is None:
            lrd_num_z5 = lrd_num_z4
        self._get_lrd_num(lrd_num_z4, lrd_num_z5)

        self._get_all_sky_tde_rate()

    def _fill_tde_grid(self, tde_grid):
        _M_s_scalers = tde_grid.M_s_scalers
        _M_bhs = tde_grid.M_bhs

        _dlog_M_s_scaler = np.log10(_M_s_scalers[1]) - np.log10(_M_s_scalers[0])
        _dlog_M_bh = np.log10(_M_bhs[1]) - np.log10(_M_bhs[0])

        _diff = _dlog_M_s_scaler / _dlog_M_bh
        _M_s_scaler_idx_step = np.round(1 / _diff).astype(int) if _diff < 1 else 1
        _M_bh_idx_step = np.round(_diff).astype(int) if _diff > 1 else 1

        _N_TDEs = tde_grid.N_TDEs

        for _i, _M_s_scaler in enumerate(_M_s_scalers):
            for _j, _M_bh in enumerate(_M_bhs):
                if _N_TDEs[_i, _j] == 0:
                    _j_ref = _j + _M_bh_idx_step
                    _i_ref = _i - _M_s_scaler_idx_step

                    if _j_ref < len(_M_bhs) and _i_ref >= 0:
                        _N_TDEs[_i, _j] = _N_TDEs[_i_ref, _j_ref]

    def _get_tde_interp(self, tde_grid):
        self._tde_interp = RegularGridInterpolator(
            (np.log10(tde_grid.M_s_scalers), np.log10(tde_grid.M_bhs)),
            np.log10(tde_grid.N_TDEs),
            bounds_error=False,
            fill_value=np.nan,
        )

    def _get_lrd_num(self, lrd_num_z4, lrd_num_z5):
        if lrd_num_z4.row_variable != lrd_num_z5.row_variable:
            raise ValueError(  # noqa: TRY003
                "The two LRD number objects use different scatter schemes"  # noqa: EM101
            )

        for _name in ("log_Ms_grid", "log_Mbh_grid", "p_grid"):
            if not np.array_equal(
                getattr(lrd_num_z4, _name), getattr(lrd_num_z5, _name)
            ):
                raise ValueError(  # noqa: TRY003
                    f"{_name} is not equal between the two LRD number objects"  # noqa: EM102
                )

        self._row_variable = lrd_num_z4.row_variable
        self._log_Ms_grid = lrd_num_z4.log_Ms_grid
        self._log_Mbh_grid = lrd_num_z4.log_Mbh_grid
        self._p_grid = lrd_num_z4.p_grid

        self._row_number_density = (
            lrd_num_z4.row_number_density * z45_shell_volume
            + lrd_num_z5.row_number_density * z56_shell_volume
        ) / (z45_shell_volume + z56_shell_volume)

    def _get_all_sky_tde_rate(self):
        _log_Ms_scaler = self._log_Ms_grid - self._log_Mbh_grid
        _log_tde_rate = self._tde_interp(
            (_log_Ms_scaler.flatten(), self._log_Mbh_grid.flatten())
        ).reshape(self._log_Mbh_grid.shape)

        # Mean rate per host in each row bin (rows are log M• for LRDNum,
        # log M* for LRDNumScatterMbh).
        self.tde_rate_bins = np.nansum(10**_log_tde_rate * self._p_grid, axis=1)
        if self._row_variable == "log_Mbh":
            self.tde_rate_Mbh_bins = self.tde_rate_bins
        else:
            self.tde_rate_Ms_bins = self.tde_rate_bins

        # Report how much of the population falls outside the TDE grid; those
        # cells are NaN and are silently dropped by nansum.
        _weight = self._row_number_density.reshape(-1, 1) * self._p_grid
        self.uncovered_weight_fraction = (
            _weight[np.isnan(_log_tde_rate)].sum() / _weight.sum()
        )
        if self.uncovered_weight_fraction > 1e-3:
            warnings.warn(
                f"{self.uncovered_weight_fraction:.1%} of the host population lies "
                "outside the TDE grid (M_s_scalers x M_bhs) and is dropped from "
                "the rate; extend the grid or truncate the scatter.",
                stacklevel=2,
            )

        self.all_sky_tde_rate = np.nansum(
            self.tde_rate_bins * self._row_number_density
        ) * (
            z45_shell_volume * z45_dilation_factor
            + z56_shell_volume * z56_dilation_factor
        )
