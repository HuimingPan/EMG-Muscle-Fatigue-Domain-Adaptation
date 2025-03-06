import matplotlib.pyplot as plt
import numpy as np
from src.writer import save_unique_figure
from src.statistics.tests import anova
from src.visualize import adjust_plotting
from src.statistics.metrics import rmse_and_cc


def plot_fitting_curve(pred, target, **kwargs):
    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Time(s)",
                "ylabel": "Normalized Force(MVC)",
                "width": 0.6,
                "colors": ['#d47264', '#2066a8'],
                "legend": True,
                "xticks": np.arange(0, len(target), 1000),
                "xticklabels": np.arange(0, len(target), 1000) / 200,
                "xlim": [0, len(target)],
                "ylim": [0, 1],
                "rotation": 0,
                "location": "upper left",
                }
    keywords.update(kwargs)

    fig, ax = plt.subplots(figsize=keywords["figsize"])

    ax.plot(target,  label='Actual force', color=keywords["colors"][0])
    ax.plot(np.arange(0, len(pred), 5), pred[::5], linestyle=':', label='Estimated force',
            color=keywords["colors"][1], )
    # ax.plot(pred, linestyle=':', label='Estimated force',
    #         color=keywords["colors"][1], )

    rmse, cc = rmse_and_cc(pred, target)
    print(f"RMSE: {rmse}, CC: {cc}")
    adjust_plotting(ax, **keywords)
    plt.show()
    save_path = (f"E:/OneDrive - sjtu.edu.cn/Papers/EMG-based fatigue force estimation/figures/Results/"
                 f"fitting_curve.png")
    save_unique_figure(fig, save_path)
    return fig

def plotting_fitting_curve_with_confidence_interval(adver_pred, baseline_pred, target,
                                                    adver_pred_ci, baseline_pred_ci, target_ci,
                                                    **kwargs):
    """
    Plot the fitting curve with confidence interval
    :param adver_pred: Predicted force by adversarial model
    :param baseline_pred: Predicted force by baseline model
    :param target: Actual force
    :param adver_pred_ci: Confidence interval of predicted force by adversarial model
    :param baseline_pred_ci: Confidence interval of predicted force by baseline model
    :param target_ci: Confidence interval of actual force
    :param kwargs: Additional parameters
    """
    keywords = {"figsize": (8, 2.5),
                "xlabel": "Time(s)",
                "ylabel": "Normalized Force(MVC)",
                "width": 0.6,
                "colors": ['#d47264', '#2066a8'],
                "legend": True,
                "xticks": np.arange(0, len(target), 1000),
                "xticklabels": np.arange(0, len(target), 1000) / 200,
                "xlim": [0, len(target)],
                "ylim": [0, 1],
                "rotation": 0,
                "location": "upper left",
                }
    keywords.update(kwargs)

    fig, axes = plt.subplots(1,3, figsize=keywords["figsize"], sharey=True)

    for i, (ax, pred1, pred2, t, pred1_ci, pred2_ci, t_ci) in enumerate(zip(axes,
                                                            np.split(adver_pred, 3),
                                                            np.split(baseline_pred, 3),
                                                            np.split(target, 3),
                                                            np.split(adver_pred_ci, 3),
                                                            np.split(baseline_pred_ci, 3),
                                                            np.split(target_ci, 3))):
        ax.plot(t, linestyle='-', label='Actual force', color=keywords["colors"][0])
        ax.plot(pred1, linestyle='--', label='Adversarial force', color=keywords["colors"][1])
        ax.plot(pred2, linestyle='-.', label='Baseline force', color=keywords["colors"][2])
        ax.fill_between(np.arange(0, len(pred1)), pred1 - pred1_ci, pred1 + pred1_ci, color=keywords["colors"][0], alpha=0.2)
        ax.fill_between(np.arange(0, len(pred2)), pred2 - pred2_ci, pred2 + pred2_ci, color=keywords["colors"][1], alpha=0.2)
        ax.fill_between(np.arange(0, len(t)), t - t_ci, t + t_ci, color=keywords["colors"][2], alpha=0.2)
        ax.set_xlabel("Time(s)")
        ax.set_xticks(np.arange(0, len(t), 1000))
        ax.set_xticklabels(np.arange(0, len(t), 1000) / 200)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Super legend for the whole figure without background
    fig.legend(labels=['Actual force', 'Adversarial force', 'Baseline force'],
               loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3,
               frameon=False)

    # Set y label for the first subplot
    axes[0].set_ylabel("Normalized Force(MVC)")

    # Adjust the space between subplots

    plt.tight_layout()
    plt.show()

    save_path = f"D:/OneDrive - sjtu.edu.cn/Papers/EMG-based fatigue force estimation/figures/Results/fitting_curve.png"
    save_unique_figure(fig, save_path)
    return fig



def plot_performance(label, baseline_estimate, adver_estimate):
    fig, ax = plt.subplots()
    ax.plot(label, linestyle='--', label='Actual force', color="#d47264")
    ax.plot(baseline_estimate, linestyle=':', label='Baseline force', color="#2066a8")
    ax.plot(adver_estimate, linestyle='-.', label='Adversarial force', color="#8ec1da")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Force/MVC")
    ax.set_xticks(np.arange(0, len(label), 1000))
    ax.set_xticklabels(np.arange(0, len(label), 1000) / 200)

    ax.legend()
    plt.show()
    return fig


def plot_error_target(error, target, **kwargs):
    """
    Plot the error-target bar plot
    :param error:
    :param target:
    :param bins:
    :return:
    """
    keywords = {"figsize": (5.6, 4.2),
                "width": 1,
                "colors": "#2066a8",
                "bins": 10,
                "xlabel": "Force Range (%MVC)",
                "ylabel": "Error (%MVC)"
                }
    keywords.update(kwargs)
    fig, ax = plt.subplots(figsize=keywords["figsize"])

    edge_bins = np.linspace(0, 1, keywords["bins"] + 1)
    indices = np.digitize(target, edge_bins)
    error_bin = [error[indices == i] for i in range(1, len(edge_bins))]
    mean = [np.mean(bin) for bin in error_bin]
    std = [np.std(bin) for bin in error_bin]
    yerr = [np.zeros_like(std), std]

    print(anova(error_bin))

    error_kw = dict(linestyle='--', label='Error', color="#d47264")
    ax.bar(np.arange(1, len(edge_bins)), mean, keywords["width"], color=keywords["colors"],
           yerr=yerr, error_kw=error_kw, edgecolor="black", zorder=2)
    ax.set_xlabel(keywords["xlabel"])
    ax.set_ylabel(keywords["ylabel"])
    plt.show()
    return fig
