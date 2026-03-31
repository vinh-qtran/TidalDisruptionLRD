import csv
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from tqdm import tqdm

from tidaldisruptionlrd.constants import G, z45_shell_volume, z56_shell_volume
from tidaldisruptionlrd.stellar_distribution import PowerLawProfile

# 1. Get the root string
repo_root_str = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
).stdout.strip()

# 2. Convert to a Path object for easy joining
repo_root = Path(repo_root_str)


class TDEGrid:
    def __init__(
        self,
        profile,
        tde,
        M_bhs,
        M_s_scalers,
        m_s_max=1,
        a=None,
        sigma_coeff=None,
        sigma=None,
        sigma_params={"M200": 3.09e8, "p": 4.38},  # noqa: B006
        a_params={"M0": 1e9, "a0": 0.055, "alpha": 0.22},  # noqa: B006
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
            The maximum stellar mass for which to compute TDE rates. If not provided, it is set to 1 by default.
        a : float, optional
            The scale radii for which to compute TDE rates. If not provided, it is computed using the sigma coefficient and the M-sigma relation.
        sigma_coeff : float, optional
            The sigma coefficient to be used in computing the scale radius, defined as a = sigma_coeff * G * M_s / sigma^2.
            If not provided, a is computed using the M-a relation from van de Wel et al. (2014).
        sigma : float, optional
            The sigma value to be used in computing scale radius. If not provided, it is computed using the M-sigma relation.
        sigma_params : dict, optional
            The parameters for the M-sigma relation, with keys "M200" and "p". If not provided, default values are used.
        a_params : dict, optional
            The parameters for the M-a relation, with keys "M0", "a0", and "alpha". If not provided, default values are used.
        """

        self.sigma_params = sigma_params
        self.a_params = a_params

        self.profile = profile
        self.tde = tde

        self.M_bhs = M_bhs
        self.M_s_scalers = M_s_scalers
        self.m_s_max = m_s_max

        self.a = a

        self.sigma_coeff = sigma_coeff

        self.sigma = sigma or self._get_sigma(M_bhs, **sigma_params)

        _N_TDEs = []
        for _M_s_scaler in tqdm(self.M_s_scalers):
            _N_TDEs.append(self._get_single_TDE_rates(_M_s_scaler))  # noqa: PERF401

        self.N_TDEs = np.vstack(_N_TDEs)

    def _get_sigma(self, M_bh, M200, p):
        return 200 * (M_bh / M200) ** (1 / p)

    def _get_a_from_sigma(self, sigma_coeff, M_s, sigma):
        return sigma_coeff * G * M_s / sigma**2

    def _get_a_from_Mstar(self, M_s, M0, a0, alpha):
        return a0 * (M_s / M0) ** alpha

    def _get_single_TDE_rates(self, M_s_scaler):
        _profile = self.profile(
            M_s_scaler,
            r_bin_min=1e-4,
            r_bin_max=1e6,
            N_bins=1000,
            reduce_factor=10,
            N_trapz_bins=100,
            show_progress=False,
        )

        if self.a is not None:
            _as = np.full_like(self.M_bhs, self.a)
        elif self.sigma_coeff is not None:
            _as = self._get_a_from_sigma(
                self.sigma_coeff, self.M_bhs * M_s_scaler, self.sigma
            )
        else:
            _as = self._get_a_from_Mstar(self.M_bhs * M_s_scaler, **self.a_params)

        _r_hs = _as / _profile.a

        _tde = self.tde(
            m_s_max=self.m_s_max,
            dimensionless_profile=_profile,
            M_bhs=self.M_bhs,
            r_hs=_r_hs,
            m_s_bins=np.linspace(0.08, 10, 1000),
            eta=0.844,
            show_progress=False,
        )

        return _tde.N_TDEs


class IsothermalTDEGrid:
    def __init__(
        self,
        tde,
        M_bhs,
        M_s_scalers,
        m_s_max=1,
        sigma_params={"M200": 3.09e8, "p": 4.38},  # noqa: B006
    ):
        self.sigma_params = sigma_params

        self.tde = tde

        self.M_bhs = M_bhs
        self.M_s_scalers = M_s_scalers
        self.m_s_max = m_s_max

        self.sigma = self._get_sigma(M_bhs, **sigma_params)

        _N_TDEs = [self._get_single_TDE_rates()] * len(self.M_s_scalers)

        self.N_TDEs = np.vstack(_N_TDEs)

    def _get_sigma(self, M_bh, M200, p):
        return 200 * (M_bh / M200) ** (1 / p)

    def _get_single_TDE_rates(self):
        _profile = PowerLawProfile(
            gamma=2,
            r_bin_min=1e-4,
            r_bin_max=1e6,
            N_bins=1000,
            reduce_factor=10,
            N_trapz_bins=100,
            show_progress=True,
        )

        _r_hs = G * self.M_bhs / self.sigma**2 / 2

        _tde = self.tde(
            m_s_max=self.m_s_max,
            dimensionless_profile=_profile,
            M_bhs=self.M_bhs,
            r_hs=_r_hs,
            m_s_bins=np.linspace(0.08, 10, 1000),
            eta=0.844,
            show_progress=True,
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


class LRDNum:
    _logMs_from_logMbh_funcs = {  # noqa: RUF012
        "KH13": KH13_logMs_from_logMbh,
        "RV15": RV15_logMs_from_logMbh,
        "G20": G20_logMs_from_logMbh,
        "P23": P23_logMs_from_logMbh,
    }

    _log_Ms_scatters = {  # noqa: RUF012
        "KH13": 0.29 / 1.16,
        "RV15": 0.24 / 1.05,
        "G20": 0.79 / 1.39,
        "P23": 0.69 / 1.06,
    }

    _Inayoshi24_files = {  # noqa: RUF012
        "Inayoshi24_z4_Obscured": repo_root / "data/Inayoshi24/z4_Solid_Obscured.csv",
        "Inayoshi24_z4_Unobscured": repo_root
        / "data/Inayoshi24/z4_Dashed_Unobscured.csv",
        "Inayoshi24_z5_Obscured": repo_root / "data/Inayoshi24/z5_Solid_Obscured.csv",
        "Inayoshi24_z5_Unobscured": repo_root
        / "data/Inayoshi24/z5_Dashed_Unobscured.csv",
    }

    def __init__(self, log_Mbh_bins, delta_log_Ms_scaler, Ms_Mbh_type, BHMF_type):

        self.log_Mbh_bins = log_Mbh_bins
        self.log_Ms_bins = self._logMs_from_logMbh_funcs[Ms_Mbh_type](log_Mbh_bins)

        self.delta_log_Ms_bins = (
            delta_log_Ms_scaler * self._log_Ms_scatters[Ms_Mbh_type]
        )

        if BHMF_type in ["Greene25", "uniform"]:
            self._read_Greene25_DF()

            if BHMF_type == "uniform":
                self.BHMF_bins = np.trapezoid(
                    self._Greene25_BHMF_interp(log_Mbh_bins), log_Mbh_bins
                ) / np.full_like(log_Mbh_bins, log_Mbh_bins[-1] - log_Mbh_bins[0])
            else:
                self.BHMF_bins = self._Greene25_BHMF_interp(log_Mbh_bins)

        elif BHMF_type.split("_")[0] == "Inayoshi24":
            self._read_Inayoshi24_DF(self._Inayoshi24_files[BHMF_type])
            self.BHMF_bins = self._Inayoshi24_BHMF_interp(log_Mbh_bins)

        else:
            raise ValueError("Invalid BHMF_type.")  # noqa: EM101, TRY003

        self.p_delta_log_Ms_bins = (
            norm.pdf(delta_log_Ms_scaler) / self._log_Ms_scatters[Ms_Mbh_type]
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


class AllSkyTDERate:
    def __init__(self, tde_grid, lrd_num_z4, lrd_num_z5):
        self._get_tde_interp(tde_grid)
        self._get_lrd_num(lrd_num_z4, lrd_num_z5)

        self._get_all_sky_tde_rate()

    def _get_tde_interp(self, tde_grid):
        self._tde_interp = RegularGridInterpolator(
            (np.log10(tde_grid.M_s_scalers), np.log10(tde_grid.M_bhs)),
            np.log10(tde_grid.N_TDEs),
            bounds_error=False,
            fill_value=np.nan,
        )

    def _get_lrd_num(self, lrd_num_z4, lrd_num_z5):
        if np.equal(lrd_num_z4.log_Mbh_bins, lrd_num_z5.log_Mbh_bins).all():
            self._log_Mbh_bins = lrd_num_z4.log_Mbh_bins
        else:
            raise ValueError(  # noqa: TRY003
                "log_Mbh_bins are not equal between the two LRD number objects"  # noqa: EM101
            )

        if np.equal(lrd_num_z4.log_Ms_bins, lrd_num_z5.log_Ms_bins).all():
            self._log_Ms_bins = lrd_num_z4.log_Ms_bins
        else:
            raise ValueError(  # noqa: TRY003
                "log_Ms_bins are not equal between the two LRD number objects"  # noqa: EM101
            )

        if np.equal(lrd_num_z4.delta_log_Ms_bins, lrd_num_z5.delta_log_Ms_bins).all():
            self._delta_log_Ms_bins = lrd_num_z4.delta_log_Ms_bins
        else:
            raise ValueError(  # noqa: TRY003
                "delta_log_Ms_bins are not equal between the two LRD number objects"  # noqa: EM101
            )

        if np.equal(
            lrd_num_z4.p_delta_log_Ms_bins, lrd_num_z5.p_delta_log_Ms_bins
        ).all():
            self._p_delta_log_Ms_bins = lrd_num_z4.p_delta_log_Ms_bins
        else:
            raise ValueError(  # noqa: TRY003
                "p_delta_log_Ms_bins are not equal between the two LRD number objects"  # noqa: EM101
            )

        self._BHMF_bins = (
            lrd_num_z4.BHMF_bins * z45_shell_volume
            + lrd_num_z5.BHMF_bins * z56_shell_volume
        ) / (z45_shell_volume + z56_shell_volume)

    def _get_all_sky_tde_rate(self):
        _log_Mbh = np.repeat(self._log_Mbh_bins, len(self._delta_log_Ms_bins)).reshape(
            len(self._log_Mbh_bins), len(self._delta_log_Ms_bins)
        )
        _log_Ms = self._log_Ms_bins.reshape(-1, 1) + self._delta_log_Ms_bins

        _log_Ms_scaler = _log_Ms - _log_Mbh
        _log_tde_rate = self._tde_interp(
            (_log_Ms_scaler.flatten(), _log_Mbh.flatten())
        ).reshape(_log_Mbh.shape)

        self.tde_rate_Mbh_bins = np.nansum(
            10**_log_tde_rate
            * self._p_delta_log_Ms_bins.reshape(1, -1)
            * (self._delta_log_Ms_bins[1] - self._delta_log_Ms_bins[0]),
            axis=1,
        )

        self.all_sky_tde_rate = np.nansum(
            self.tde_rate_Mbh_bins
            * self._BHMF_bins
            * (self._log_Mbh_bins[1] - self._log_Mbh_bins[0])
        ) * (z45_shell_volume + z56_shell_volume)
