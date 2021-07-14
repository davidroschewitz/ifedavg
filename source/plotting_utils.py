import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits import axes_grid1
import matplotlib as mpl


def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def _get_significant(m, std_sig=2.0, noise_std=0.05, row_std=2.0):
    sig = np.abs(m - m.mean(axis=0)) > (std_sig * m.std(axis=0))
    big_enough = np.repeat(
        (m.std(axis=0) > noise_std)[np.newaxis, :], m.shape[0], axis=0
    )
    return np.logical_and(sig, big_enough)


def _get_significant_cols(m, std_sig=2.0, noise_std=0.05, row_std=2.0):
    return np.repeat(
        ((m.std(axis=0) - np.mean(m.std(axis=0))) > row_std * (np.std(m.std(axis=0))))[
            np.newaxis, :
        ],
        m.shape[0],
        axis=0,
    )


def add_significant(ax, m, width, height):
    significant = _get_significant(m)
    for x in range(width):
        for y in range(height):
            if significant[y][x]:
                ax.annotate(
                    str("O"),
                    xy=(x, y),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    cols = _get_significant_cols(m)
    for x in range(width):
        for y in range(height):
            if cols[y][x]:
                ax.annotate(
                    str("x"),
                    xy=(x, y),
                    horizontalalignment="center",
                    verticalalignment="center",
                )


def plot_heatmap_colorbar(
    m, cmap, xlabels, ylabels, path, highlight_significant=True, vmin=-1.5, vmax=1.5,
):
    height, width = m.shape
    scaling = 0.8

    imheight = scaling * max(height, 12)
    imwidth = scaling * max(width, 8)

    sns.reset_orig()
    fig = plt.figure(figsize=(imwidth, imheight))
    plt.clf()
    ax = fig.gca()
    res = ax.imshow(
        np.array(m), cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax
    )
    _add_colorbar(res)

    plt.xticks(range(width), xlabels, rotation=90)
    plt.yticks(range(height), ylabels, rotation=0)
    plt.tight_layout()

    if highlight_significant:
        add_significant(ax, m, width, height)

    # plt.show()
    plt.savefig(path + ".png", dpi=100, format="png", bbox_inches="tight")
    plt.savefig(path + ".svg", format="svg")
    plt.savefig(path + ".pdf", format="pdf")
    plt.close()


def plot_heatmap_histogram(
    m, cmap, xlabels, ylabels, path, highlight_significant=True, vmin=-1.5, vmax=1.5,
):
    height, width = m.shape
    height = height * 4
    width = width * 4
    mpl.rcParams["image.composite_image"] = False

    plt.clf()
    offset = 6
    scaling = 0.18
    fig = plt.figure(figsize=(scaling * (width + offset), scaling * (height + offset)))
    gs = GridSpec(height + offset, width + offset)
    ax_im = fig.add_subplot(gs[offset : height + offset, 0:width])
    ax_top = fig.add_subplot(gs[0 : offset - 1, 0:width], sharex=ax_im)
    ax_side = fig.add_subplot(
        gs[offset : height + offset, width + 1 : width + offset], sharey=ax_im
    )

    height, width = m.shape

    # plotting
    ax_im.imshow(
        np.array(m),
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax_im.set(xticks=range(width), yticks=range(height))
    ax_im.set_xticklabels(xlabels, rotation=90)
    ax_im.set_yticklabels(ylabels, rotation=0)

    ax_top.bar(
        height=np.sum(np.abs(m - np.mean(m, axis=0)), axis=0),
        x=range(width),
        color="grey",
    )
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_ylabel("Cum. Deviations")

    ax_side.barh(
        width=np.sum(np.abs(m - np.mean(m, axis=0)), axis=1),
        y=range(height),
        color="grey",
    )  #
    plt.setp(ax_side.get_yticklabels(), visible=False)
    ax_side.set_xlabel("Cum. Deviations")

    # highlighting significant values
    if highlight_significant:
        add_significant(ax_im, m, width, height)

    plt.tight_layout()
    # plt.show()
    plt.savefig(path + ".png", dpi=100, format="png", bbox_inches="tight")
    plt.savefig(path + ".svg", format="svg")
    plt.savefig(path + ".pdf", format="pdf")
    plt.clf()
    plt.close()


def visualize_feature_hist(
    df,
    path,
    split_on,
    xlim,
    nbins=15,
    client_of_interest="5",
    feature_of_interest="0",
    color="red",
    othercolor="lightgrey",
    col=None,
):
    analysis = df.copy()
    analysis["Client"] = "Others"
    analysis.loc[analysis[split_on] == client_of_interest, "Client"] = str(
        client_of_interest
    )
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({"font.size": 14})

    palette = {"Others": othercolor, client_of_interest: color}

    if col is None:
        sns.histplot(
            data=analysis,
            x=feature_of_interest,
            hue="Client",
            stat="probability",
            bins=nbins,
            multiple="dodge",
            common_norm=False,
            palette=palette,
        )
        plt.xlim(xlim)
        plt.xlabel(None)
        plt.ylabel(None)
    else:
        g = sns.displot(
            data=analysis,
            x=feature_of_interest,
            hue="Client",
            col=col,
            multiple="dodge",
            stat="probability",
            common_norm=False,
            palette=palette,
            bins=nbins,
        )
        plt.xlim(xlim)
        g.set_xlabels(None)
        g.set_axis_labels(None, None)

    # plt.tight_layout()
    # plt.show()
    plt.savefig(path + ".svg", format="svg")
    plt.savefig(path + ".png", format="png", dpi=300)
    plt.clf()
    plt.close()


def visualize_participant_performance(
    df, metric, outdir, run_order,
):
    p_subset_melted = df.melt(id_vars=["Method"], var_name=[], value_vars=[metric])

    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(
        y="value",
        x="Method",
        data=p_subset_melted,
        order=run_order,
        color="lightgrey",
        bw=0.25,
    )
    if len(list(ax.get_xticklabels())) > 5:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.title(str("Participant Spread, Metric: " + metric))
    plt.tight_layout()
    # plt.show()
    plt.savefig(outdir + "participants-" + metric + ".jpg", dpi=300)
    plt.clf()
    plt.close()


def _get_y_label(metric):
    if metric == "roc_auc":
        return "ROC AUC Score"
    elif metric == "f1":
        return "F1 Score"
    elif metric == "balanced_accuracy":
        return "Balanced Accuracy"
    else:
        raise ValueError("Metric", metric, "is not known")


def visualize_participant_performance_multiseed(
    dataset_name, df_means, df_stds, metric, outdir, run_order,
):
    means_subset_melted = df_means.melt(
        id_vars=["Method"], var_name=[], value_vars=[metric]
    )
    mpl.rc("pdf", fonttype=42)

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        y="value",
        x="Method",
        data=means_subset_melted,
        order=run_order,
        color="lightgrey",
        bw=0.25,
    )
    plt.errorbar(
        x=np.linspace(0.12, len(run_order) + 0.12 - 1, len(run_order)),
        y=df_means.groupby("Method").median().loc[run_order].to_numpy().flatten(),
        yerr=df_stds.groupby("Method").mean().loc[run_order].to_numpy().flatten(),
        ecolor="darkred",
        barsabove=True,
        fmt="none",
        capsize=3.0,
    )
    plt.title(
        str(dataset_name) + str(": Client Performance Spread w/ Seed-SD"), fontsize=18
    )
    ylabel = _get_y_label(metric)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel("Method", fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(outdir + "spread-error-" + metric + ".jpg", dpi=300)
    plt.savefig(
        outdir + dataset_name.replace(" ", "-") + "-spread-" + metric + ".pdf",
        format="pdf",
    )
    plt.clf()
    plt.close()
