import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from mpl_toolkits import axes_grid1


def _im_plot(
    m,
    figsize,
    cmap,
    xlabels,
    ylabels,
    path,
    highlight_significant=False,
    vmin=None,
    vmax=None,
):
    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.gca()
    res = ax.imshow(
        np.array(m), cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax
    )
    _add_colorbar(res)

    height, width = m.shape
    plt.xticks(range(width), xlabels, rotation=90)
    plt.yticks(range(height), ylabels, rotation=0)
    plt.tight_layout()

    if highlight_significant:
        # highlighting values 1 std away from mean
        significant = np.array(
            ((m - m.mean(dim=0)).abs() > m.std(dim=0))
            & ((m - m.mean(dim=0)).abs() > 0.2)
        )
        for x in range(width):
            for y in range(height):
                if significant[y][x]:
                    ax.annotate(
                        str("O"),
                        xy=(x, y),
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

    plt.savefig(path, format="png")


def write_mixing_matrix(process):
    m = process.mixing_matrix
    client_names = [p.name for p in process.participants]

    _im_plot(
        m=m,
        figsize=(8, 8),
        cmap="BuGn",
        xlabels=client_names,
        ylabels=client_names,
        path=_get_path(process, "mixing_matrix"),
    )


def _get_layers_as_dict(process):
    local_weight_names = [
        x
        for x in process.participants[0]._model.state_dict().keys()
        if x in process.config["local_layers"]
    ]
    images = {}
    for weight_name in local_weight_names:
        list_of_images = []
        for p in process.participants:
            list_of_images.append(p._model.state_dict()[weight_name].cpu())
        img = torch.stack(list_of_images, dim=0).flatten(1)
        images[weight_name] = img
    return images


def _write_as_csv(weight_name, img, client_names, feature_names, logdir):
    df = pd.DataFrame(img.cpu().numpy(), columns=feature_names, index=client_names)

    df.to_csv(logdir + str(weight_name) + ".csv")


def write_local_weights(process):
    images = _get_layers_as_dict(process)

    for weight_name in images.keys():
        img = images[weight_name]

        if weight_name.endswith("_w"):  # multiplicative
            vmin = -1
            vmax = 3
            cmap = "PuOr"
        else:  # additive
            vmin = -1
            vmax = 1
            cmap = "PiYG"

        height, width = img.shape
        img_width = (width / 3) + 4
        img_height = (height / 3) + 3

        # axis labels
        client_names = [p.name for p in process.participants]
        feature_names = range(width)
        if width == len(process.fl_dataset.feature_names):
            feature_names = process.fl_dataset.feature_names

        _write_as_csv(
            weight_name, img, client_names, feature_names, process.config["logdir"]
        )

        _im_plot(
            m=img,
            figsize=(img_width, img_height),
            cmap=cmap,
            xlabels=feature_names,
            ylabels=client_names,
            path=_get_path(process, weight_name),
            highlight_significant=True,
            vmin=vmin,
            vmax=vmax,
        )


def _get_path(process, name):
    return (
        process.config["logdir"]
        + process.config["experiment_name"]
        + "-"
        + process.config["run_name"]
        + "-"
        + name
        + ".png"
    )


def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def visualize_features(process):
    input_images = _get_layers_as_dict(process)
    bias = input_images["feature_b"]
    mult = input_images["feature_w"]

    combined_df = None
    for i, train_data in enumerate(process.fl_dataset.fl_train_datasets):
        original = pd.DataFrame(
            train_data.x.numpy(), columns=process.fl_dataset.feature_names,
        )
        melted = original.melt(var_name="feature")
        melted["value"] = melted["value"] + np.random.normal(
            0.0, 0.01, melted["value"].shape
        )
        melted["name"] = process.fl_dataset.dataset_names[i]
        melted["type"] = "original"

        affine = (train_data.x.numpy() * mult[i].numpy()) + bias[i].numpy()
        affine = pd.DataFrame(affine, columns=process.fl_dataset.feature_names,)
        melted_affine = affine.melt(var_name="feature")
        melted_affine["value"] = melted_affine["value"] + np.random.normal(
            0.0, 0.01, melted_affine["value"].shape
        )
        melted_affine["name"] = process.fl_dataset.dataset_names[i]
        melted_affine["type"] = "transformed"

        combined = pd.concat([melted, melted_affine], axis=0)

        if combined_df is None:
            combined_df = combined
        else:
            combined_df = pd.concat([combined_df, combined], axis=0)

    plt.clf()
    sns.displot(
        combined_df,
        kind="kde",  # hist
        x="value",
        col="feature",
        row="name",
        hue="type",
        multiple="layer",  # fill, layer, stack, dodge
        height=2.5,
        aspect=1.0,
        bw_adjust=0.1,
        facet_kws={"sharex": "col", "sharey": False, "margin_titles": True},
    )
    plt.savefig(
        _get_path(process, "input-shift"), format="png",
    )


def write_n_samples(process):
    client_names = [p.name for p in process.participants]
    n_train_samples = [
        p.dataset_loader.train_loader.n_samples for p in process.participants
    ]
    n_test_samples = [
        p.dataset_loader.test_loader.n_samples for p in process.participants
    ]

    train = pd.DataFrame(n_train_samples, index=client_names, columns=["value"])
    train["Type"] = "Train"
    test = pd.DataFrame(n_test_samples, index=client_names, columns=["value"])
    test["Type"] = "Test"

    df = pd.concat([train, test], axis=0)

    plt.clf()
    sns.barplot(x="value", y=df.index, hue="Type", data=df, orient="h")
    plt.tight_layout()

    plt.savefig(
        _get_path(process, "samples"), format="png",
    )
