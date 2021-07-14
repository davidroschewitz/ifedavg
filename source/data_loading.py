import os
import zipfile
import requests

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm


def download_file(url, filename):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def load_VSN_data():
    data_dir = Path("./data") / "vehicle_sensor"

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    subdirs = [f for f in data_dir.iterdir() if f.is_file()]
    if not subdirs:
        url = "http://www.ecs.umass.edu/~mduarte/images/event.zip"
        zip_file = data_dir / "original_data.zip"
        download_file(url, zip_file)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    data_dir = data_dir / "events" / "runs"

    x = []
    y = []
    task_index = []
    for root, dir, file_names in os.walk(data_dir):
        if "acoustic" not in root and "seismic" not in root:
            x_tmp = []
            for file_name in file_names:
                if "feat" in file_name:
                    dt_tmp = pd.read_csv(
                        os.path.join(root, file_name),
                        sep=" ",
                        skipinitialspace=True,
                        header=None,
                    ).values[:, :50]
                    x_tmp.append(dt_tmp)
            if len(x_tmp) == 2:
                x_tmp = np.concatenate(x_tmp, axis=1)
                x.append(x_tmp)
                task_index.append(
                    int(os.path.basename(root)[1:]) * np.ones(x_tmp.shape[0])
                )
                y.append(
                    int("aav" in os.path.basename(os.path.dirname(root)))
                    * np.ones(x_tmp.shape[0])
                )

    x = np.concatenate(x)
    y = np.concatenate(y)
    task_index = np.concatenate(task_index)
    argsort = np.argsort(task_index)
    x = x[argsort]
    y = y[argsort]
    task_index = task_index[argsort]
    split_index = np.where(np.roll(task_index, 1) != task_index)[0][1:]
    x = np.split(x, split_index)
    y = np.split(y, split_index)

    df = pd.DataFrame()
    feature_cols = []
    for i, (p_x, p_y) in enumerate(zip(x, y)):
        p_df = pd.DataFrame(p_x)
        feature_cols = p_df.columns.astype(str)
        p_df["vehicle"] = str(i)
        p_df["y"] = p_y

        df = pd.concat([df, p_df], axis=0)

    df.columns = df.columns.astype(str)
    df = df.reset_index(drop=True)
    return df, feature_cols


def load_HAR_dataset():
    data_dir = Path("./data") / "human_activity_recognition"

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    subdirs = [f for f in data_dir.iterdir() if f.is_file()]
    if not subdirs:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240"
            "/UCI%20HAR%20Dataset.zip"
        )
        zip_file = data_dir / "original_data_har.zip"
        download_file(url, zip_file)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    data_dir = data_dir / "UCI HAR Dataset"
    data_dir_train = data_dir / "train"
    data_dir_test = data_dir / "test"

    # Training data
    train = pd.read_csv(
        data_dir_train / "X_train.txt", delim_whitespace=True, header=None
    )
    feature_cols = train.columns.astype(str)
    train["participant"] = pd.read_csv(
        data_dir_train / "subject_train.txt", delim_whitespace=True, header=None
    ).values
    train["y"] = pd.read_csv(
        data_dir_train / "y_train.txt", delim_whitespace=True, header=None
    ).values

    # Testing data
    test = pd.read_csv(data_dir_test / "X_test.txt", delim_whitespace=True, header=None)
    test["participant"] = pd.read_csv(
        data_dir_test / "subject_test.txt", delim_whitespace=True, header=None
    ).values
    test["y"] = pd.read_csv(
        data_dir_test / "y_test.txt", delim_whitespace=True, header=None
    ).values

    dataset = pd.concat([train, test], axis=0)

    dataset["y"] = dataset["y"] - 1

    dataset.columns = dataset.columns.astype(str)
    dataset = dataset.reset_index(drop=True)
    return dataset, feature_cols
