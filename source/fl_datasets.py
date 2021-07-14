import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from source.data_handlers import BaseDataset
from source.data_loading import load_VSN_data, load_HAR_dataset

pd.options.mode.chained_assignment = None


class BaseFLDataset:
    def __init__(self, config):
        self.config = config
        self.fl_train_datasets = None
        self.centralized_train_dataset = None
        self.fl_test_datasets = None
        self.dataset_names = None
        self.input_dim = None
        self.output_dim = None
        self.feature_names = None
        self.participant_normalizations = {}
        self.class_weights = {}
        self._initialize_participant_datasets(config)

        if self.fl_test_datasets is None:
            self.fl_test_datasets = [None for _ in range(len(self))]
            warnings.warn("It is not recommended to not define the test-set")

        if self.input_dim is None:
            raise RuntimeError("The input dimension must be defined")

        if self.output_dim is None:
            raise RuntimeError("The output dimension must be defined")

        if self.feature_names is None:
            raise RuntimeError("The feature names must be defined")

    def __getitem__(self, index):
        return (
            self.dataset_names[index],
            self.fl_train_datasets[index],
            self.fl_test_datasets[index],
        )

    def __len__(self):
        """
        Number of federated datasets which can be distributed to participants

        Returns: Integer

        """
        return len(self.fl_train_datasets)

    def _initialize_participant_datasets(self, config):
        raise NotImplementedError


class TabularDataset(BaseFLDataset):
    def __init__(self, config):
        super(TabularDataset, self).__init__(config)

    def _split_x_y(self, df: pd.DataFrame):
        raise NotImplementedError

    def _initialize_participant_datasets(self, config):
        self.fl_train_datasets = []
        self.fl_test_datasets = []
        self.dataset_names = []

    def _get_counts(self, preprocessed_df, split_column):
        counts = preprocessed_df[split_column].value_counts().reset_index()
        counts.columns = [split_column, "count"]
        counts = counts.sort_values(by=["count", split_column], ascending=False)
        counts = counts[counts["count"] >= 1.33 * self.config["test_min_samples"]]
        return counts

    def _fill_normalize_steps(
        self, dataset, numerical_columns, categorical_columns, split_column
    ):
        # fill missing values before normalization if enabled
        if self.config["fill_missing_before_normalization"]:
            dataset = dataset.fillna(float(self.config["missing_value_fill"]))

        # normalize
        if self.config["normalization_location"] == "global":
            num = dataset[numerical_columns]
            dataset[numerical_columns] = self._normalize(num)
        elif self.config["normalization_location"] == "local":
            for participant_name in list(dataset[split_column].unique()):
                row_idx = dataset[split_column] == participant_name
                num = dataset.loc[row_idx, numerical_columns]
                dataset.loc[row_idx, numerical_columns] = self._normalize(num)
        else:
            pass

        # compute the hot-start parameters for each client
        for participant_name in list(dataset[split_column].unique()):
            row_idx = dataset[split_column] == participant_name
            num = dataset.loc[row_idx, numerical_columns]
            self.participant_normalizations[participant_name] = {
                "means": num.mean().fillna(0),
                "stds": num.std().fillna(1),
            }

        # # fill remaining missing values
        preprocessed_dataset = dataset
        preprocessed_dataset[numerical_columns] = preprocessed_dataset[
            numerical_columns
        ].fillna(self.config["missing_value_fill"])
        preprocessed_dataset[categorical_columns] = preprocessed_dataset[
            categorical_columns
        ].fillna(self.config["missing_value_fill_binary"])

        if preprocessed_dataset.isna().any().any():
            raise ValueError("Missing values must be filled correctly")

        return preprocessed_dataset

    def _normalize(self, df):
        if self.config["normalization_mode"] == "standardization":
            num_normalized = (df - df.mean()) / df.std()
        else:
            num_normalized = df - df.min()
            num_normalized = num_normalized / num_normalized.max()
            # num_normalized = (num_normalized - 0.5) * 2
        if self.config["clip_standardized"]:
            num_normalized = num_normalized.clip(-1.5, 1.5)
        return num_normalized

    def _create_tabular_federated_dataset(self, df, splits, split_column):
        centralized_train_x = None
        centralized_train_y = None

        for i, x in splits.iterrows():
            participant_name = x[split_column]
            self.dataset_names.append(participant_name)

            subdataset = df[df[split_column] == participant_name]

            if len(subdataset) > self.config["max_samples"]:
                subdataset = subdataset.sample(self.config["max_samples"])
                warnings.warn(
                    "The number of available samples is reduced for participant "
                    + str(participant_name)
                )

            x, y = self._split_x_y(subdataset)
            self.input_dim = x.shape[1]
            self.feature_names = x.columns

            effective_test_size = int(
                max(
                    self.config["test_split"] * len(y), self.config["test_min_samples"],
                )
            )
            train_x, test_x, train_y, test_y = train_test_split(
                x,
                y,
                test_size=effective_test_size,
                stratify=y,
                random_state=self.config["seed"],
            )

            # set the class_weights
            self.class_weights[participant_name] = (
                (1 / np.array(train_y.value_counts().sort_index()))
                / np.sum(1 / (np.array(train_y.value_counts().sort_index())))
                * len(np.unique(train_y))
            )

            self.fl_train_datasets.append(
                BaseDataset(train_x.to_numpy(), train_y.to_numpy())
            )
            self.fl_test_datasets.append(
                BaseDataset(test_x.to_numpy(), test_y.to_numpy())
            )

            if centralized_train_x is None:
                centralized_train_x = train_x
                centralized_train_y = train_y
            else:
                centralized_train_x = pd.concat([centralized_train_x, train_x], axis=0)
                centralized_train_y = pd.concat([centralized_train_y, train_y], axis=0)

        self.class_weights["centralized"] = (
            (1 / np.array(centralized_train_y.value_counts().sort_index()))
            / np.sum(1 / (np.array(centralized_train_y.value_counts().sort_index())))
            * len(np.unique(centralized_train_y))
        )
        self.centralized_train_dataset = BaseDataset(
            centralized_train_x.to_numpy(), centralized_train_y.to_numpy()
        )

    def _experimental_modifications(
        self,
        df,
        split_column,
        label_column=None,
        target_flip_client=None,
        split_client_id=None,
    ):
        if (
            self.config["flip_target"]
            and label_column is not None
            and target_flip_client is not None
        ):
            df.loc[df[split_column] == target_flip_client, label_column] = np.abs(
                -(1 - df.loc[df[split_column] == target_flip_client, label_column,])
            )

        if self.config["split_client"] and split_column is not None and split_client_id:
            idx = df[df[split_column].astype(str) == str(split_client_id)].index
            part1, part2 = train_test_split(idx, test_size=0.5)
            df.loc[part1, split_column] = str(split_client_id) + "-1"
            df.loc[part2, split_column] = str(split_client_id) + "-2"

        return df


class HumanActivityRecognitionFLDataset(TabularDataset):
    def __init__(self, config):
        super(HumanActivityRecognitionFLDataset, self).__init__(config)

    def _split_x_y(self, df: pd.DataFrame):
        x = df[df.columns[~df.columns.isin(["y", "participant"])]]
        y = df["y"]
        return x, y

    def _initialize_participant_datasets(self, config):
        super(HumanActivityRecognitionFLDataset, self)._initialize_participant_datasets(
            config
        )
        split_column = "participant"
        self.output_dim = 6

        dataset, num_cols = load_HAR_dataset()

        if self.config["split_client"]:
            client_name = 17
            idx = dataset[dataset[split_column] == client_name].index
            part1, part2 = train_test_split(idx, test_size=0.5)
            dataset.loc[part1, split_column] = str(client_name) + "-1"
            dataset.loc[part2, split_column] = str(client_name) + "-2"

        dataset = self._fill_normalize_steps(dataset, num_cols, [], split_column)

        ordered_participant_names = self._get_counts(dataset, split_column)

        self._create_tabular_federated_dataset(
            df=dataset, splits=ordered_participant_names, split_column=split_column,
        )


class VehicleSensorNetworkFLDataset(TabularDataset):
    def __init__(self, config):
        super(VehicleSensorNetworkFLDataset, self).__init__(config)

    def _split_x_y(self, df: pd.DataFrame):
        x = df[df.columns[~df.columns.isin(["y", "vehicle"])]]
        y = df["y"]
        return x, y

    def _initialize_participant_datasets(self, config):
        super(VehicleSensorNetworkFLDataset, self)._initialize_participant_datasets(
            config
        )
        split_column = "vehicle"
        self.output_dim = 2

        dataset, num_cols = load_VSN_data()

        dataset = self._experimental_modifications(
            dataset,
            split_column,
            label_column="y",
            target_flip_client="10",
            split_client_id="17",
        )

        dataset = self._fill_normalize_steps(dataset, num_cols, [], split_column)

        ordered_participant_names = self._get_counts(dataset, split_column)

        self._create_tabular_federated_dataset(
            df=dataset, splits=ordered_participant_names, split_column=split_column,
        )
