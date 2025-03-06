import matplotlib.pyplot as plt
import numpy as np
from src.statistics.tests import ttest_rmse, ttest_rel, anova
from src.writer import save_unique_figure
from src.visualize import adjust_plotting


def plot_line_mean_bar(df, **kwargs):
    """
    Plot the rmse
    :param df:
    :return:
    """

    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Subject",
                "ylabel": "RMSE (MVC)",
                "location": "upper center",
                "width": 0.2,
                "rotation": 0.,
                # "location": "lower left",
                "annotate_base": 0.190,
                "domain": "f",
                "colors": ['#2066a8', '#ae282c', '#cde1ec', '#f6d6c2', "#ededed"],
                "labels": None,
                "suffix": "",
                "p_value_columns": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                }
    keywords.update(kwargs) if kwargs else keywords

    if isinstance(keywords["domain"], str):
        if keywords["domain"] == 'both':
            meanX = np.array([-1.8, -0.6, 0.6, 1.8]) * keywords["width"] + df.shape[1]
        elif keywords["domain"] == 'nf':
            df = df.loc[["baseline-NF", "proposed-NF"]]
            meanX = meanX = np.array([-0.6, 0.6]) * keywords["width"] + df.shape[1]
        elif keywords["domain"] == 'f':
            df = df.loc[["baseline-F", "proposed-F"]]
            meanX = np.array([-0.6, 0.6]) * keywords["width"] + df.shape[1]
        else:
            raise ValueError("Invalid domain")
    elif isinstance(keywords["domain"], list):
        df = df.loc[keywords["domain"]]
        meanX = np.linspace(-1.8, 1.8, df.shape[0]) * keywords["width"] + df.shape[1]
    else:
        raise ValueError("Invalid domain")

    fig, ax = plt.subplots(figsize=keywords["figsize"])
    rows = df.index
    marker = ['^', 'D', 's', 'd', 'D']
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    colors = keywords["colors"]
    print("Means: \n", means)
    print("Stds: \n", stds)

    for i, row in enumerate(rows):
        label = keywords["labels"][i] if keywords["labels"] else row
        ax.plot(df.loc[row], label=label, color=colors[i], marker=marker[i], fillstyle='none')

    add_means_bar(ax, meanX, means, stds, colors, marker)
    # for i, (p_1, p_2) in enumerate(keywords["p_value_columns"]):
    #     ttest_pvalue = ttest_rel(df.loc[rows[p_1]], df.loc[rows[p_2]])
    #     add_p_value_annotation(ax, meanX[p_1], meanX[p_2], keywords["annotate_base"], ttest_pvalue)

    ticks = [f"S{i}" for i in range(1, df.shape[1] + 1)]
    adjust_plotting(ax, ticks + ["Mean"], **keywords)
    plt.show()
    if keywords["save"]:
        save_path = rf"..\results\rmse_{keywords['suffix']}.png"

        save_unique_figure(fig, save_path)
    return fig

def plot_line_confidence_interval_mean_bar(df, **kwargs):
    """
    Plot the lines with confidence interval on the left side and mean bar on the right side
    :param df:
    :return:
    """

    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Subject",
                "ylabel": "RMSE (MVC)",
                "location": "upper center",
                "width": 0.2,
                "rotation": 0.,
                # "location": "lower left",
                "domain": "f",
                "colors": ['#2066a8', '#ae282c', '#cde1ec', '#f6d6c2', "#ededed"],
                "labels": None,
                "suffix": "",
                "p_value_columns": [(0, 1), (0, 2), (1, 2)],

                }
    keywords.update(kwargs) if kwargs else keywords

    if isinstance(keywords["domain"], str):
        if keywords["domain"] == 'both':
            meanX = np.array([-1.8, -0.6, 0.6, 1.8]) * keywords["width"] + df.shape[1]
        elif keywords["domain"] == 'nf':
            df = df.loc[["baseline-NF", "proposed-NF"]]
            meanX = meanX = np.array([-0.6, 0.6]) * keywords["width"] + df.shape[1]
        elif keywords["domain"] == 'f':
            df = df.loc[["baseline-F", "proposed-F"]]
            meanX = np.array([-0.6, 0.6]) * keywords["width"] + df.shape[1]
        else:
            raise ValueError("Invalid domain")
    elif isinstance(keywords["domain"], list):
        df = df.loc[keywords["domain"]]
        meanX = np.linspace(-0.6, 0.6, df.shape[0]) * keywords["width"] + df.shape[1]
    else:
        raise ValueError("Invalid domain")

    fig, ax = plt.subplots(figsize=keywords["figsize"])
    rows = df.index
    marker = ['o', 's', 'v', '^', 'D']
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    colors = keywords["colors"]
    print("Means: \n", means)
    print("Stds: \n", stds)

    for i, row in enumerate(rows):
        label = keywords["labels"][i] if keywords["labels"] else row
        ax.plot(df.loc[row], label=label, color=colors[i], marker=marker[i], fillstyle='none')

    add_means_bar(ax, meanX, means, stds, colors, marker)
    # p_value = ttest_rel(df.loc["baseline-F"], df.loc["baseline-NF"])
    # add_p_value_annotation(ax, meanX[0], meanX[1], 0.32, p_value)

    ticks = [f"S{i}" for i in range(1, df.shape[1] + 1)]
    adjust_plotting(ax, ticks + ["Mean"], **keywords)
    plt.show()
    if keywords["save"]:
        save_path = rf"D:\OneDrive - sjtu.edu.cn\Papers\EMG-based fatigue force estimation\
                            \Figures\Results\rmse_{keywords['suffix']}.png"

        save_unique_figure(fig, save_path)
    return fig


def plot_rmse_bars_ws(df, **kwargs):
    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Window Size (Frames)",
                "ylabel": "RMSE (MVC)",
                "width": 0.6,
                "colors": ['#2066a8', '#8ec1da', '#cde1ec', '#f6d6c2', '#d47264', '#ae282c'],
                "annotate_base": 0.080,
                "annotate_offset": 0.005,
                "legend": False,
                "rotation": 0,
                "p_value_columns": [(0, 1), (0, 2), (1, 2)],
                }
    keywords.update(kwargs)

    fig, ax = plt.subplots(figsize=keywords["figsize"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    rows = df.index

    means = df.mean(axis=1)
    stds = df.std(axis=1)
    print([f"{level}: {mean} ± {std}" for level, mean, std in zip(rows, means, stds)])

    meanX = np.arange(len(rows))
    yerr = [np.zeros_like(stds), stds]
    error_kw = dict(lw=1, capsize=3, capthick=1, zorder=1)
    ax.bar(meanX, means, keywords["width"], color=keywords["colors"],
           yerr=yerr, error_kw=error_kw, edgecolor=keywords["colors"], zorder=2)
    ttest = ttest_rmse(df, 512)
    ys = np.array([5, 3, 2, 0, 1, 4]) * keywords["annotate_offset"] + keywords["annotate_base"]
    for i, row in enumerate(rows):
        if row != 512:
            add_p_value_annotation(ax, meanX[i], meanX[3], ys[i], ttest[row].pvalue)

    adjust_plotting(ax, rows, **keywords)
    save_path = r"D:\OneDrive - sjtu.edu.cn\Papers\EMG-based fatigue force estimation\figures\Results\rmse_across_ws.png"
    save_unique_figure(fig, save_path)
    return fig


def plot_rmse_bars(df, **kwargs):
    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Force Level (MVC)",
                "ylabel": "RMSE (MVC)",
                "width": 0.6,
                "colors": ['#2066a8', '#8ec1da', '#cde1ec', '#f6d6c2', '#d47264', '#ae282c'],
                "annotate_base": 0.190,
                "annotate_offset": 0.008,
                "legend": False,
                "rotation": 0,
                "p_value_columns": [(0, 1), (0, 2), (1, 2)],
                "save": True,
                }
    keywords.update(kwargs)

    fig, ax = plt.subplots(figsize=keywords["figsize"])
    rows = df.index

    means = df.mean(axis=1)
    stds = df.std(axis=1)
    print([f"{level}: {mean} ± {std}" for level, mean, std in zip(rows, means, stds)])

    meanX = np.arange(len(rows))
    error_kw = dict(lw=1, capsize=3, capthick=2, zorder=2)
    ax.bar(meanX, means, keywords["width"], color=keywords["colors"],
           yerr=stds, error_kw=error_kw, edgecolor=keywords["colors"], zorder=2)

    for i, (p_1, p_2) in enumerate(keywords["p_value_columns"]):
        ttest_pvalue = ttest_rel(df.loc[rows[p_1]], df.loc[rows[p_2]])
        if "ys" in keywords:
            y = keywords["ys"][i] * keywords["annotate_offset"] + keywords["annotate_base"]
            add_p_value_annotation(ax, meanX[p_1], meanX[p_2], y, ttest_pvalue)
        else:
            add_p_value_annotation(ax, meanX[p_1], meanX[p_2], keywords["annotate_base"], ttest_pvalue)

    adjust_plotting(ax, rows, **keywords)
    plt.show()
    if keywords["save"]:
        save_path = r"..\results\rmse_across_levels.png"
        save_unique_figure(fig, save_path)
    return fig


def plot_violin(df, **kwargs):
    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Force Level (MVC)",
                "ylabel": "RMSE (MVC)",
                "width": 0.6,
                "colors": ['#2066a8', '#8ec1da', '#cde1ec', '#f6d6c2', '#d47264', '#ae282c'],
                "annotate_base": 0.190,
                "annotate_offset": 0.008,
                "legend": False,
                "rotation": 0,
                "p_value_columns": [(0, 1), (0, 2), (1, 2)],
                "save": True,
                "x_offset": 0.2,
                }
    keywords.update(kwargs)

    fig, ax = plt.subplots(figsize=keywords["figsize"])

    rows = df.index
    data = [df.loc[row].values for row in rows]

    for i, (color, value) in enumerate(zip(keywords["colors"], data)):
        parts = ax.violinplot(
            value, positions=[i + keywords["x_offset"]],
            showmeans=False, showextrema=False, showmedians=False,
            side="high")
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.5)

        jetter = np.random.uniform(0, 0.1, len(value))

        ax.scatter(
            np.full(value.shape, i + keywords["x_offset"]) + jetter,
            value, color=color, edgecolors="face", s=5,
            )

        ax.boxplot(
            value, positions=[i],
            widths=keywords["x_offset"] * 1.9, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, color=color, alpha=0.75),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            medianprops=dict(color="#f1f5f9")
        )

    means = df.mean(axis=1)
    stds = df.std(axis=1)
    print([f"{level}: {mean} ± {std}" for level, mean, std in zip(rows, means, stds)])

    meanX = np.arange(1, len(rows) + 1)
    for i, (p_1, p_2) in enumerate(keywords["p_value_columns"]):
        _, p_value = anova((df.loc[rows[p_1]], df.loc[rows[p_2]]))
        if "ys" in keywords:
            y = keywords["ys"][i] * keywords["annotate_offset"] + keywords["annotate_base"]
            add_p_value_annotation(ax, meanX[p_1], meanX[p_2], y, p_value)
        else:
            add_p_value_annotation(ax, meanX[p_1], meanX[p_2], keywords["annotate_base"], p_value)

    adjust_plotting(ax, rows, **keywords)
    # plt.show()
    if keywords["save"]:
        save_path = r"D:\OneDrive - sjtu.edu.cn\Papers\EMG-based fatigue force estimation\figures\Results\rmse_across_electrodes.png"
        save_unique_figure(fig, save_path)
    return fig


def add_p_value_annotation(ax, x1, x2, y, p_val):
    bar_height = y
    if p_val < 0.05:
        significance = "*"
        bar_line_height = bar_height * 1.02
        ax.plot([x1, x1], [bar_height, bar_line_height], c='black')
        ax.plot([x2, x2], [bar_height, bar_line_height], c='black')
        ax.plot([x1, x2], [bar_line_height, bar_line_height], color="black", linestyle='-', linewidth=1)
        ax.text((x1 + x2) * .5, bar_line_height, significance, ha='center', va='center', color='black')


def add_means_bar(ax, X, y, yerr, colors, markers):
    width = 0.2
    # yerr = [np.zeros_like(yerr), yerr]
    error_kw = dict(lw=1, capsize=3, capthick=1, zorder=2)
    ax.bar(X, y, width, color=colors, yerr=yerr, error_kw=error_kw, edgecolor=colors, zorder=2)
    for x, marker in zip(X, markers):
        ax.plot(x, 0.015, color='w', marker=marker, fillstyle='none')
    return ax
