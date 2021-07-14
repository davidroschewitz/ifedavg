import random
import warnings
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset


class FastTabularLoader:
    def __init__(self, x, y, batch_size=32, shuffle=False):
        """
        Inspired by https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
        """
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

        self.n_samples = self.x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.n_samples, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.n_samples)
            self.x = self.x[r]
            self.y = self.y[r]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_samples:
            raise StopIteration
        batch = (
            self.x[self.i : self.i + self.batch_size],
            self.y[self.i : self.i + self.batch_size],
        )
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class BaseDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.n_samples = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class BaseDataLoader:
    def __init__(
        self,
        config,
        train: BaseDataset,
        test: Optional[BaseDataset],
        test_split=0.3,
        shuffle_for_split=True,
        make_test: bool = True,
    ):
        self.config = config
        self.train_loader = None
        self.test_loader = None

        if test is None and make_test:
            warnings.warn(
                "No test dataset is defined - this is no recommended behavior"
            )

            indices = np.arange(len(train))
            if shuffle_for_split:
                indices = np.random.permutation(indices)
            split = int(np.floor(test_split * len(train)))
            train_sampler = SubsetRandomSampler(indices[split:])
            test_sampler = SubsetRandomSampler(indices[:split])
            self.train_loader = DataLoader(
                train,
                self.config["batch_size"],
                sampler=train_sampler,
                pin_memory=True,
            )
            self.test_loader = DataLoader(
                train, self.config["batch_size"], sampler=test_sampler, pin_memory=True,
            )
        else:
            self.train_loader = DataLoader(
                train, self.config["batch_size"], shuffle=True, pin_memory=True
            )
            self.test_loader = DataLoader(
                test, self.config["batch_size"], shuffle=False, pin_memory=True,
            )


class FastDataLoader:
    def __init__(
        self, config, train: BaseDataset, test: BaseDataset,
    ):
        self.config = config

        self.train_loader = FastTabularLoader(
            train.x, train.y, self.config["batch_size"], shuffle=True,
        )
        self.test_loader = FastTabularLoader(
            test.x, test.y, self.config["batch_size"], shuffle=False
        )
