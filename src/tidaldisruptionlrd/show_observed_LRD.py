import numpy as np
import pandas as pd

halo_stellar_folder = "../data/LRD_Mbh_Ms/"
Maiolino_data_path = halo_stellar_folder + "Maiolino+.csv"
Stone_data_path = halo_stellar_folder + "Stone+.csv"
Yue_data_path = halo_stellar_folder + "Yue+.csv"
Harikane_data_path = halo_stellar_folder + "Harikane+.csv"

# Maiolino_data = pd.read_csv(Maiolino_data_path, header=None).sort_values(by=0)
Stone_data = pd.read_csv(Stone_data_path, header=None).sort_values(by=0)
Yue_data = pd.read_csv(Yue_data_path, header=None).sort_values(by=0)
# Harikane_data = pd.read_csv(Harikane_data_path, header=None).sort_values(by=0)

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
# Maiolino_xerr_lower = Maiolino_data[0] - [
#     7.4961e0,
#     7.5965e0,
#     7.3642e0,
#     7.1947e0,
#     7.9922e0,
#     8.1429e0,
#     8.2370e0,
#     8.3878e0,
#     8.2245e0,
#     9.5871e0,
#     1.0133e1,
#     1.0560e1,
# ]
# Maiolino_xerr_upper = Maiolino_xerr_lower
# Maiolino_yerr_upper = [
#     7.4474e0,
#     7.1706e0,
#     7.6570e0,
#     7.5402e0,
#     6.8694e0,
#     7.7190e0,
#     7.1591e0,
#     7.6084e0,
#     7.8121e0,
#     8.0412e0,
#     7.8695e0,
#     8.2272e0,
# ] - Maiolino_data[1]
# Maiolino_yerr_lower = Maiolino_yerr_upper

# Maiolino_x_limit = [0] * len(Maiolino_data)

# Maiolino_data["x_err_lower"] = Maiolino_xerr_lower
# Maiolino_data["x_err_upper"] = Maiolino_xerr_upper
# Maiolino_data["y_err_lower"] = Maiolino_yerr_lower
# Maiolino_data["y_err_upper"] = Maiolino_yerr_upper
# Maiolino_data["x_limit"] = Maiolino_x_limit

Maiolino_data = pd.DataFrame(
    {
        0: [8.92, 7.82, 8.24],
        1: [7.6, 6.81, 6.86],
        "x_err_lower": [0.76, 0.32, 0.32],
        "x_err_upper": [0.76, 0.32, 0.32],
        "y_err_lower": [0.07, 0.21, 0.11],
        "y_err_upper": [0.07, 0.18, 0.1],
        "x_limit": [0, 0, 0],
    }
)

# Maiolino et al. (2023), LRD=0
Maiolino_data_extra = pd.DataFrame(
    {
        0: [8.5, 8.54, 9.08, 8.34, 8.04],
        1: [7.01, 6.9, 6.97, 6.34, 6.51],
        "x_err_lower": [0.24, 0.72, 0.4, 0.31, 0.89],
        "x_err_upper": [0.24, 0.72, 0.0, 0.31, 0.89],
        "y_err_lower": [0.08, 0.13, 0.33, 0.19, 0.25],
        "y_err_upper": [0.08, 0.11, 0.25, 0.16, 0.2],
        "x_limit": [0, 0, 1, 0, 0],
    }
)

# Harikane
# Harikane_xerr_lower = Harikane_data[0] - [
#     7.6028e0,
#     8.5761e0,
#     8.4066e0,
#     8.7582e0,
#     8.7394e0,
#     8.8587e0,
#     9.1036e0,
#     8.9152e0,
#     8.4380e0,
#     9.2480e0,
# ]
# Harikane_xerr_upper = [
#     9.2480e0 - 8.632653,
#     0,
#     9.2229e0 - 8.940345,
#     0,
#     9.4050e0 - 9.097331,
#     0,
#     0,
#     9.7127e0 - 9.361068,
#     1.0372e1 - 9.605965,
#     1.0422e1 - 9.919937,
# ]
# Harikane_yerr_upper = [
#     8.1133e0,
#     7.0490e0,
#     7.8553e0,
#     8.2800e0,
#     7.2709e0,
#     8.2925e0,
#     8.2313e0,
#     7.6713e0,
#     8.6686e0,
#     7.8999e0,
# ] - Harikane_data[1]
# Harikane_yerr_lower = Harikane_yerr_upper

# Harikane_x_limit = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0]

# Harikane_data["x_err_lower"] = Harikane_xerr_lower
# Harikane_data["x_err_upper"] = Harikane_xerr_upper
# Harikane_data["y_err_lower"] = Harikane_yerr_lower
# Harikane_data["y_err_upper"] = Harikane_yerr_upper
# Harikane_data["x_limit"] = Harikane_x_limit

Harikane_data = pd.DataFrame(
    {
        0: [8.33],
        1: [6.93],
        "x_err_lower": [0.24],
        "x_err_upper": [0.24],
        "y_err_lower": [0.14],
        "y_err_upper": [0.13],
        "x_limit": [0],
    }
)

Harikane_data_extra = pd.DataFrame(
    {
        0: [9.21, 9.04],
        1: [7.34, 7.39],
        "x_err_lower": [0.33, 0.24],
        "x_err_upper": [0.33, 0.24],
        "y_err_lower": [0.09, 0.14],
        "y_err_upper": [0.08, 0.13],
        "x_limit": [0, 0],
    }
)

# Juodzbaliz
Juodzbaliz_data = pd.DataFrame(
    {
        0: [8.92],
        1: [8.61],
        "x_err_lower": [0.31],
        "x_err_upper": [0.30],
        "y_err_lower": [0.37],
        "y_err_upper": [0.38],
        "x_limit": [0],
    }
)

# Ding
Ding_data = pd.DataFrame(
    {
        0: [np.log10(1.3e11), np.log10(3.4e10)],
        1: [np.log10(1.54e9), np.log10(2.02e8)],
        "x_err_lower": [
            np.log10(1.3e11) - np.log10(1.3e11 - 0.6e11),
            np.log10(3.4e10) - np.log10(3.4e10 - 1.9e10),
        ],
        "x_err_upper": [
            np.log10(1.3e11 + 2.0e11) - np.log10(1.3e11),
            np.log10(3.4e10 + 7.6e10) - np.log10(3.4e10),
        ],
        "y_err_lower": [0.4, 0.4],
        "y_err_upper": [0.4, 0.4],
        "x_limit": [0, 0],
    }
)

# Perez-Gonzalez
PerezGonzalez_data = pd.DataFrame(
    {
        0: [8.1, 8.3, 8.4, 8.3],
        1: [6.6, 6.4, 6.4, 6.2],
        "x_err_lower": [0.3, 0.3, 0.3, 0.3],
        "x_err_upper": [0.3, 0.3, 0.3, 0.3],
        "y_err_lower": [0.3, 0.3, 0.4, 0.4],
        "y_err_upper": [0.3, 0.3, 0.4, 0.4],
        "x_limit": [1, 0, 0, 0],
    }
)

# Kocevski
Kocevski_data_23 = pd.DataFrame(
    {
        0: [7.99],
        1: [7.19],
        "x_err_lower": [0.37],
        "x_err_upper": [0.37],
        "y_err_lower": [0.13],
        "y_err_upper": [0.11],
        "x_limit": [0],
    }
)

Kocevski_data_23_extra = pd.DataFrame(
    {
        0: [9.53],
        1: [7.32],
        "x_err_lower": [0.4],
        "x_err_upper": [0.0],
        "y_err_lower": [0.16],
        "y_err_upper": [0.14],
        "x_limit": [1],
    }
)

Kocevski_data_25 = pd.DataFrame(
    {
        0: [8.42, 8.96, 8.71, 9.01, 9.05, 8.49, 8.68, 8.32, 8.7, 8.09, 8.16, 8.61],
        1: [7.18, 8.47, 7.58, 8.72, 8.55, 7.39, 7.0, 7.1, 6.74, 8.29, 7.26, 6.74],
        "x_err_lower": [
            0.37,
            0.46,
            0.46,
            0.35,
            0.52,
            0.29,
            0.46,
            0.36,
            0.36,
            0.51,
            0.43,
            0.48,
        ],
        "x_err_upper": [
            0.37,
            0.46,
            0.46,
            0.35,
            0.52,
            0.29,
            0.46,
            0.36,
            0.36,
            0.51,
            0.43,
            0.48,
        ],
        "y_err_lower": [
            0.08,
            0.02,
            0.03,
            0.09,
            0.12,
            0.04,
            0.18,
            0.04,
            0.18,
            0.06,
            0.12,
            0.21,
        ],
        "y_err_upper": [
            0.07,
            0.02,
            0.03,
            0.08,
            0.11,
            0.04,
            0.15,
            0.04,
            0.16,
            0.05,
            0.1,
            0.18,
        ],
        "x_limit": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
)

Kocevski_data_25_extra = pd.DataFrame(
    {
        0: [9.92],
        1: [7.36],
        "x_err_lower": [0.4],
        "x_err_upper": [0.0],
        "y_err_lower": [0.22],
        "y_err_upper": [0.18],
        "x_limit": [1],
    }
)

# Taylor

Taylor_data = pd.DataFrame(
    {
        0: [9.01, 8.65, 8.67, 8.22, 8.14, 8.29, 8.07, 8.65, 8.94, 8.23, 9.0, 8.34],
        1: [7.37, 7.56, 7.28, 7.17, 7.03, 7.52, 7.22, 7.86, 7.12, 7.63, 7.93, 7.19],
        "x_err_lower": [
            0.43,
            0.39,
            0.33,
            0.38,
            0.92,
            0.35,
            0.45,
            0.41,
            0.54,
            0.47,
            0.58,
            0.53,
        ],
        "x_err_upper": [
            0.43,
            0.39,
            0.33,
            0.38,
            0.92,
            0.35,
            0.45,
            0.41,
            0.54,
            0.47,
            0.58,
            0.53,
        ],
        "y_err_lower": [
            0.03,
            0.08,
            0.09,
            0.15,
            0.14,
            0.08,
            0.05,
            0.06,
            0.06,
            0.04,
            0.02,
            0.14,
        ],
        "y_err_upper": [
            0.03,
            0.08,
            0.08,
            0.13,
            0.13,
            0.07,
            0.05,
            0.06,
            0.06,
            0.04,
            0.02,
            0.12,
        ],
        "x_limit": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
)

Taylor_data_extra = pd.DataFrame(
    {
        0: [
            8.86,
            8.45,
            8.14,
            10.4,
            10.8,
            9.81,
            9.04,
            8.28,
            10.3,
            10.9,
            9.94,
            11.0,
            9.65,
            9.82,
            10.4,
            9.8,
            10.3,
            8.48,
            9.73,
            10.1,
            7.97,
            7.87,
            10.5,
            8.0,
            8.74,
            10.8,
            8.78,
            10.2,
            10.7,
            8.94,
            9.95,
        ],
        1: [
            6.21,
            6.41,
            6.72,
            8.08,
            6.23,
            7.04,
            6.98,
            7.52,
            6.53,
            7.65,
            7.08,
            7.16,
            6.82,
            6.76,
            6.71,
            6.53,
            7.03,
            6.122,
            6.71,
            6.67,
            6.35,
            6.32,
            6.79,
            6.58,
            6.76,
            7.12,
            6.93,
            7.42,
            6.8,
            7.18,
            7.16,
        ],
        "x_err_lower": [
            0.19,
            0.26,
            0.49,
            0.4,
            0.73,
            0.19,
            0.7,
            0.36,
            0.12,
            0.4,
            0.31,
            0.08,
            0.4,
            0.31,
            0.76,
            0.4,
            0.23,
            0.2,
            0.23,
            0.16,
            0.25,
            0.52,
            0.41,
            0.62,
            0.85,
            0.4,
            0.4,
            0.25,
            0.58,
            0.64,
            0.4,
        ],
        "x_err_upper": [
            0.19,
            0.26,
            0.49,
            0.0,
            0.73,
            0.19,
            0.7,
            0.36,
            0.12,
            0.0,
            0.31,
            0.08,
            0.0,
            0.31,
            0.76,
            0.0,
            0.23,
            0.2,
            0.23,
            0.16,
            0.25,
            0.52,
            0.41,
            0.62,
            0.85,
            0.0,
            0.0,
            0.25,
            0.58,
            0.64,
            0.0,
        ],
        "y_err_lower": [
            0.29,
            0.14,
            0.17,
            0.06,
            0.17,
            0.06,
            0.11,
            0.03,
            0.35,
            0.03,
            0.26,
            0.05,
            0.06,
            0.09,
            0.2,
            0.2,
            0.11,
            0.18,
            0.06,
            0.15,
            0.26,
            0.06,
            0.11,
            0.11,
            0.16,
            0.07,
            0.19,
            0.02,
            0.11,
            0.06,
            0.15,
        ],
        "y_err_upper": [
            0.23,
            0.12,
            0.14,
            0.05,
            0.14,
            0.06,
            0.1,
            0.03,
            0.27,
            0.03,
            0.21,
            0.05,
            0.06,
            0.08,
            0.17,
            0.17,
            0.01,
            0.15,
            0.06,
            0.13,
            0.21,
            0.06,
            0.1,
            0.1,
            0.14,
            0.07,
            0.16,
            0.02,
            0.1,
            0.06,
            0.13,
        ],
        "x_limit": [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
        ],
    }
)

# Rusakov
Rusakov_data = pd.DataFrame(
    {
        0: [10.9, 10.2, 9.59, 10.3, 10.62, 10.61, 10.09, 9.57, 8.92, 9.5, 9.44],
        1: [6.47, 7.88, 6.09, 7.58, 6.41, 7.5, 6.58, 5.63, 5.63, 5.91, 6.68],
        "x_err_lower": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "x_err_upper": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "y_err_lower": [0.2, 0.07, 0.06, 0.14, 0.62, 0.06, 0.16, 0.62, 0.31, 0.24, 0.2],
        "y_err_upper": [1, 0.06, 0.05, 0.11, 1.11, 0.05, 0.14, 2.51, 0.18, 0.22, 0.26],
        "x_limit": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
)


def plot_error_bar(ax, data_frame, color, marker):
    x, y = 10 ** data_frame[0], 10 ** data_frame[1]
    xerr_lower, xerr_upper = (
        10 ** data_frame[0] - 10 ** (data_frame[0] - data_frame["x_err_lower"]),
        10 ** (data_frame[0] + data_frame["x_err_upper"]) - 10 ** data_frame[0],
    )
    yerr_lower, yerr_upper = (
        10 ** data_frame[1] - 10 ** (data_frame[1] - data_frame["y_err_lower"]),
        10 ** (data_frame[1] + data_frame["y_err_upper"]) - 10 ** data_frame[1],
    )

    for i in range(len(data_frame)):
        if not data_frame["x_limit"][i]:
            ax.errorbar(
                x[i],
                y[i],
                xerr=[[xerr_lower[i]], [xerr_upper[i]]],
                yerr=[[yerr_lower[i]], [yerr_upper[i]]],
                marker=marker,
                ms=10,
                color=color,
                mec=color,
                capthick=2,
                lw=2,
                capsize=5,
                linestyle="none",
                # label=name if i == 0 else "",
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
                marker=marker,
                ms=10,
                color=color,
                mec=color,
                linestyle="none",
                # label=name if i == 0 else "",
            )


def show_observed_LRD(ax, legend_loc):
    for data in [
        Yue_data,
        Stone_data,
        Ding_data,
        # Maiolino_data_extra,
        # Harikane_data_extra,
        # Kocevski_data_23_extra,
        # Kocevski_data_25_extra,
        # Taylor_data_extra,
    ]:
        plot_error_bar(ax, data, "grey", "s")

    for data in [
        Maiolino_data,
        Harikane_data,
        Juodzbaliz_data,
        PerezGonzalez_data,
        Kocevski_data_23,
        Kocevski_data_25,
        Taylor_data,
        Rusakov_data,
    ]:
        plot_error_bar(ax, data, "maroon", "o")

    ax.errorbar(
        [1],
        [1],
        [1],
        [1],
        capthick=2,
        lw=2,
        capsize=5,
        ms=10,
        color="maroon",
        marker="o",
        label="LRDs",
    )
    ax.errorbar(
        [1],
        [1],
        [1],
        [1],
        capthick=2,
        lw=2,
        capsize=5,
        ms=10,
        color="grey",
        marker="s",
        label="Quasars",
    )

    ax.legend(
        loc=legend_loc,
        ncol=1,
        handletextpad=0.5,
        handlelength=1.0,
        columnspacing=0.5,
        labelspacing=0.25,
        # fontsize=24,
    )
    # ax.add_artist(leg1)
