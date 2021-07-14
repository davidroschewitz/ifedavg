import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, input_dim=64, output_dim=2, norm_params=None):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError


class MLPModel(BaseModel):
    def __init__(self, input_dim=64, output_dim=2, norm_params=None):
        super(MLPModel, self).__init__(input_dim, output_dim, norm_params)
        self.feature_w = torch.nn.Parameter(
            data=torch.Tensor(1, input_dim), requires_grad=True
        )
        if norm_params is not None and "stds" in norm_params:
            self.feature_w.data = torch.tensor(
                1 / norm_params["stds"].values, dtype=torch.float
            )
        else:
            self.feature_w.data.uniform_(1, 1)

        self.feature_b = torch.nn.Parameter(
            data=torch.Tensor(1, input_dim), requires_grad=True
        )
        if norm_params is not None and "means" in norm_params:
            self.feature_b.data = torch.tensor(
                -norm_params["means"].values, dtype=torch.float
            )
        else:
            self.feature_b.data.uniform_(0, 0)

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        # per-class target
        self.target_w = torch.nn.Parameter(
            data=torch.Tensor(1, output_dim), requires_grad=True
        )
        self.target_w.data.uniform_(1, 1)

        self.target_b = torch.nn.Parameter(
            data=torch.Tensor(1, output_dim), requires_grad=True
        )
        self.target_b.data.uniform_(0, 0)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = (x + self.feature_b) * self.feature_w
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = (x + self.target_b) * self.target_w
        output = F.log_softmax(x, dim=1)
        return output
