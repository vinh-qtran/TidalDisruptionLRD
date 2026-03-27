import numpy as np
import pandas as pd

halo_stellar_folder = "../data/LRD_Mbh_Ms/"
KH13_data_path = halo_stellar_folder + "KH13.csv"
Maiolino_data_path = halo_stellar_folder + "Maiolino+.csv"
Stone_data_path = halo_stellar_folder + "Stone+.csv"
Yue_data_path = halo_stellar_folder + "Yue+.csv"
Harikane_data_path = halo_stellar_folder + "Harikane+.csv"

KH13_data = pd.read_csv(KH13_data_path, header=None).sort_values(by=0)
Maiolino_data = pd.read_csv(Maiolino_data_path, header=None).sort_values(by=0)
Stone_data = pd.read_csv(Stone_data_path, header=None).sort_values(by=0)
Yue_data = pd.read_csv(Yue_data_path, header=None).sort_values(by=0)
Harikane_data = pd.read_csv(Harikane_data_path, header=None).sort_values(by=0)

# Yue
Yue_xerr_lower = [
    9.806907 - 9.5118,
    1.0466e1 - 10.145997,
    10.648352 - 1.0416e1,
    10.742543 - 1.0447e1,
    10.92467 - 1.0699e1,
    11.577708 - 1.1345e1,
]
Yue_xerr_upper = [
    9.806907 - 9.5118,
    1.0466e1 - 10.145997,
    0,
    10.742543 - 1.0447e1,
    0,
    0,
]
Yue_yerr_lower = [
    9.4628e0,
    9.4818e0,
    9.5687e0,
    1.0277e1,
    1.0025e1,
    1.0444e1,
] - Yue_data[1]
Yue_yerr_upper = Yue_yerr_lower

Yue_x_limit = [0, 0, 1, 0, 1, 1]

Yue_data["x_err_lower"] = Yue_xerr_lower
Yue_data["x_err_upper"] = Yue_xerr_upper
Yue_data["y_err_lower"] = Yue_yerr_lower
Yue_data["y_err_upper"] = Yue_yerr_upper
Yue_data["x_limit"] = Yue_x_limit

# Stone
Stone_xerr_lower = [
    10.001570 - 9.5620e0,
    10.302983 - 9.8634e0,
    10.340659 - 9.9137e0,
    10.403454 - 9.9765e0,
]
Stone_xerr_upper = [0] * len(Stone_data)
Stone_yerr_upper = [9.4816e0, 1.0233e1, 9.5867e0, 9.7837e0] - Stone_data[1]
Stone_yerr_lower = Stone_yerr_upper

Stone_x_limit = [1] * len(Stone_data)

Stone_data["x_err_lower"] = Stone_xerr_lower
Stone_data["x_err_upper"] = Stone_xerr_upper
Stone_data["y_err_lower"] = Stone_yerr_lower
Stone_data["y_err_upper"] = Stone_yerr_upper
Stone_data["x_limit"] = Stone_x_limit

# Maiolino
Maiolino_xerr_lower = Maiolino_data[0] - [
    7.4961e0,
    7.5965e0,
    7.3642e0,
    7.1947e0,
    7.9922e0,
    8.1429e0,
    8.2370e0,
    8.3878e0,
    8.2245e0,
    9.5871e0,
    1.0133e1,
    1.0560e1,
]
Maiolino_xerr_upper = Maiolino_xerr_lower
Maiolino_yerr_upper = [
    7.4474e0,
    7.1706e0,
    7.6570e0,
    7.5402e0,
    6.8694e0,
    7.7190e0,
    7.1591e0,
    7.6084e0,
    7.8121e0,
    8.0412e0,
    7.8695e0,
    8.2272e0,
] - Maiolino_data[1]
Maiolino_yerr_lower = Maiolino_yerr_upper

Maiolino_x_limit = [0] * len(Maiolino_data)

Maiolino_data["x_err_lower"] = Maiolino_xerr_lower
Maiolino_data["x_err_upper"] = Maiolino_xerr_upper
Maiolino_data["y_err_lower"] = Maiolino_yerr_lower
Maiolino_data["y_err_upper"] = Maiolino_yerr_upper
Maiolino_data["x_limit"] = Maiolino_x_limit

# Harikane
Harikane_xerr_lower = Harikane_data[0] - [
    7.6028e0,
    8.5761e0,
    8.4066e0,
    8.7582e0,
    8.7394e0,
    8.8587e0,
    9.1036e0,
    8.9152e0,
    8.4380e0,
    9.2480e0,
]
Harikane_xerr_upper = [
    9.2480e0 - 8.632653,
    0,
    9.2229e0 - 8.940345,
    0,
    9.4050e0 - 9.097331,
    0,
    0,
    9.7127e0 - 9.361068,
    1.0372e1 - 9.605965,
    1.0422e1 - 9.919937,
]
Harikane_yerr_upper = [
    8.1133e0,
    7.0490e0,
    7.8553e0,
    8.2800e0,
    7.2709e0,
    8.2925e0,
    8.2313e0,
    7.6713e0,
    8.6686e0,
    7.8999e0,
] - Harikane_data[1]
Harikane_yerr_lower = Harikane_yerr_upper

Harikane_x_limit = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0]

Harikane_data["x_err_lower"] = Harikane_xerr_lower
Harikane_data["x_err_upper"] = Harikane_xerr_upper
Harikane_data["y_err_lower"] = Harikane_yerr_lower
Harikane_data["y_err_upper"] = Harikane_yerr_upper
Harikane_data["x_limit"] = Harikane_x_limit


def plot_error_bar(ax, data_frame, name, color):
    x, y = data_frame[0], data_frame[1]
    xerr_lower, xerr_upper = data_frame["x_err_lower"], data_frame["x_err_upper"]
    yerr_lower, yerr_upper = data_frame["y_err_lower"], data_frame["y_err_upper"]

    for i in range(len(data_frame)):
        if not data_frame["x_limit"][i]:
            ax.errorbar(
                x[i],
                y[i],
                xerr=[[xerr_lower[i]], [xerr_upper[i]]],
                yerr=[[yerr_lower[i]], [yerr_upper[i]]],
                marker="o",
                ms=10,
                color=color,
                mec=color,
                capthick=2,
                lw=2,
                capsize=5,
                linestyle="none",
                label=name if i == 0 else "",
            )
        else:
            ax.errorbar(
                x[i],
                y[i],
                xerr=[[xerr_lower[i]], [xerr_upper[i]]],
                yerr=[[yerr_lower[i]], [yerr_upper[i]]],
                xuplims=True,
                capthick=2,
                lw=2,
                capsize=5,
                marker="o",
                ms=10,
                color=color,
                mec=color,
                linestyle="none",
                label=name if i == 0 else "",
            )


def show_observed_LRD(ax):
    x = np.linspace(7, 13, 100)
    y = 9 + np.log10(0.49) + 1.17 * (x - 11)
    (l1,) = ax.plot(x, y, lw=4, linestyle="--", dashes=(12, 6), color="navy")
    ax.fill_between(x, y - 0.28, y + 0.28, color="navy", alpha=0.3)

    x = np.linspace(7, 13, 100)
    y = 7.45 + 1.05 * (x - 11)
    (l2,) = ax.plot(x, y, lw=4, linestyle="--", dashes=(12, 6), color="grey")
    ax.fill_between(x, y - 0.24, y + 0.24, color="grey", alpha=0.3)

    leg1 = ax.legend([l1, l2], ["KH13 (local)", "RV15 (local)"], loc=4, fontsize=20)

    ax.errorbar(
        8.92,
        8.61,
        xerr=[[0.31], [0.30]],
        yerr=[[0.37], [0.38]],
        marker="*",
        linestyle="none",
        ms=18,
        color="gold",
        mec="gold",
        capthick=2,
        lw=2,
        capsize=5,
        label=r"$\rm Juodzbalis+24$",
    )

    plot_error_bar(ax, Stone_data, "Stone+24", "deeppink")
    plot_error_bar(ax, Yue_data, "Yue+24", "crimson")
    plot_error_bar(ax, Maiolino_data, "Maiolino+24", "darkorchid")
    plot_error_bar(ax, Harikane_data, "Harikane+23", "chocolate")

    x = np.array([np.log10(1.3e11), np.log10(3.4e10)])
    xloerr = x - np.array([np.log10(1.3e11 - 0.6e11), np.log10(3.4e10 - 1.9e10)])
    xhierr = np.array([np.log10(1.3e11 + 2.0e11), np.log10(3.4e10 + 7.6e10)]) - x
    y = np.array([np.log10(1.54e9), np.log10(2.02e8)])
    ax.errorbar(
        x,
        y,
        xerr=[xloerr, xhierr],
        yerr=0.4,
        marker="o",
        linestyle="none",
        ms=10,
        color="seagreen",
        mec="seagreen",
        capthick=2,
        lw=2,
        capsize=5,
        label=r"$\rm Ding+23$",
    )

    ax.legend(fontsize=20, loc=2)
    ax.add_artist(leg1)
