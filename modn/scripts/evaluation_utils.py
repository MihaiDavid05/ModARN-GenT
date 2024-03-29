import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def get_plots_path(model_name):
    """Define and create path for saving plots"""
    plots_path = os.path.join('plots', model_name.split('.')[0])
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


def plot(stages, results, plots_path, label, ylim):
    """Plot the metrics"""

    matplotlib.rcParams.update({'font.size': 17})

    if label == 'RMSE':
        point_label = 'min'
        best_idx_score = stages[np.nanargmin(results)]
        best_score = np.nanmin(results)
    else:
        point_label = 'max'
        best_idx_score = stages[np.nanargmax(results)]
        best_score = np.nanmax(results)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(stages, results)
    ax.plot(best_idx_score, best_score, 'o',
            label='{}: {:.2f}, timestep: {}'.format(point_label, best_score, best_idx_score),
            markersize=8, color='orange')
    point = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=5, label=point_label)
    ax.legend(handles=[point])
    ax.set(xlabel=r"Timestep", ylabel=label)
    ax.set_xlim(-2, len(stages))
    ax.set_ylim(0, ylim)
    ax.legend()
    plt.tight_layout()
    # plt.rcParams['figure.figsize'] = (10, 10)

    fig.savefig(plots_path)


def generate_plots(model, plots_path, feature_decoding, reset_state, loss_f1, stages, results):
    """Generate plots with metrics"""

    if feature_decoding:
        for f in model.feature_decoders_cont:
            rmse_res = [results[stage][f"{f}_rmse"]
                        if results[stage].get(f"{f}_rmse") else np.nan for stage in stages[:-1]]
            plots_name = "{}_rmse{}_best{}.png".format(f, '_reset_state' if reset_state else '', loss_f1)
            path = os.path.join(plots_path, plots_name)
            plot(stages[:-1], rmse_res, path, 'RMSE', 1.25)

        for f in model.feature_decoders_cat:
            f1_res = [results[stage][f"{f}_f1"] for stage in stages[:-1]]
            plots_name = "{}_f1{}_best{}.png".format(f, '_reset_state' if reset_state else '', loss_f1)
            path = os.path.join(plots_path, plots_name)
            plot(stages[:-1], f1_res, path, 'Macro F1', 1.1)

    for f in model.decoders:
        f1_res = [results[stage][f"{f}_f1"] for stage in stages]
        plots_name = "{}_f1{}_best{}.png".format(f, '_reset_state' if reset_state else '', loss_f1)
        path = os.path.join(plots_path, plots_name)
        plot(stages, f1_res, path, 'Macro F1', 1.1)
