from typing import Union

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from modn.datasets import FeatureInfo
from torch import Tensor


class InitState(nn.Module):
    """Trainable initial state"""

    def __init__(self, state_size, random_state=False):
        super().__init__()
        self.state_size = state_size
        self.random_state = random_state
        self.state_value = None

        if not self.random_state:
            self.state_value = torch.nn.Parameter(torch.randn([1, state_size], requires_grad=True))

    def forward(self, n_data_points):
        if self.random_state:
            self.state_value = torch.empty(1, self.state_size, requires_grad=False)
            nn.init.kaiming_uniform_(self.state_value, a=math.sqrt(55))
            # state_value = torch.nn.Parameter(torch.normal(0, 0.5, size=(1, self.state_size)), requires_grad=False)

            init_tensor = torch.tile(self.state_value, [n_data_points, 1])
        else:
            init_tensor = torch.tile(self.state_value, [n_data_points, 1])

        return init_tensor


class Linear(nn.Linear):
    """Custom linear layer"""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, negative_slope=35):
        self.negative_slope = negative_slope
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in Kaiming_Uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(self.negative_slope))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class EpoctEncoder(nn.Module):
    """Basic feature encoder"""

    def __init__(
        self,
        state_size: int,
        feature_info: FeatureInfo,
        hidden_size: int = 32,
        add_state: bool = True,
        negative_slope: int = 5
    ):
        super().__init__()
        self.add_state = add_state
        self.negative_slope = negative_slope
        self.fc1 = (
            Linear(len(feature_info.possible_values) + state_size, hidden_size, negative_slope=self.negative_slope)
            if feature_info.type == "categorical"
            else Linear(1 + state_size, hidden_size,  negative_slope=self.negative_slope)
        )
        self.fc2 = Linear(hidden_size, state_size, negative_slope=self.negative_slope)
        self.feature_info = feature_info

    def forward(self, state: Tensor, x: Union[str, np.float64, Tensor], scale_skip: float = 1.0) -> Tensor:
        if self.feature_info.type == "categorical":
            if isinstance(x, Tensor):
                # Below, we convert x to a failsafe indexing tensor
                # Basically x, if x < num_classes, otherwise num_classes - 1
                x = x.to(torch.int64)
                # to indexing tensor
                x = x.view(-1)[0] if len(x.shape) > 1 else x[0]
                # if x < num_classes, x, else num_classes
                x = torch.max(
                    torch.min(
                        x,
                        # num_classes
                        torch.tensor(
                            len(self.feature_info.possible_values) - 1,
                            dtype=torch.int64
                        )),
                    torch.tensor(0, dtype=torch.int64)
                )
                # to one-hot
                x = (
                    F.one_hot(
                        x,
                        len(self.feature_info.possible_values)
                    )
                    .float()
                    .view(-1, len(self.feature_info.possible_values))
                )
            else:
                try:
                    x = (
                        F.one_hot(
                            self.feature_info.encoding_dict[x],
                            len(self.feature_info.possible_values),
                        )
                        .float()
                        .view(-1, len(self.feature_info.possible_values))
                    )
                except Exception as e:
                    print(self.feature_info)
                    raise e
        else:
            x = torch.tensor(np.array(x).reshape(1, -1), dtype=torch.float32)

        x = F.relu(self.fc1(torch.cat([x, state], axis=-1)))
        if self.add_state:
            return state * scale_skip + self.fc2(x)
        else:
            return self.fc2(x)


class EpoctBinaryDecoder(nn.Module):
    """Categorical individual disease decoder"""

    def __init__(self, state_size: int, negative_slope: int = 5):
        super().__init__()
        self.negative_slope = negative_slope
        self.fc1 = Linear(state_size, 2, bias=True, negative_slope=self.negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc1(x)


class EpoctCategoricalDecoder(nn.Module):
    """Categorical feature decoder"""

    def __init__(self, state_size: int, num_categories: int, hidden_size: int = 10, negative_slope: int = 5):
        super().__init__()
        self.negative_slope = negative_slope
        self.fc1 = Linear(state_size, hidden_size, negative_slope=self.negative_slope)
        self.fc2 = Linear(hidden_size, num_categories, bias=True, negative_slope=self.negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))

        # NOTE: Took F.log_softmax() later
        return self.fc2(x)


class EpoctDistributionDecoder(nn.Module):
    """Feature distribution decoder approximating mean and variance of distribution"""

    def __init__(self, state_size: int, hidden_size: int = 10, negative_slope: int = 5):
        super().__init__()
        self.negative_slope = negative_slope
        self.fc1_mu = Linear(state_size, hidden_size, bias=True, negative_slope=self.negative_slope)
        self.fc2_mu = Linear(hidden_size, 1, bias=True, negative_slope=self.negative_slope)
        self.fc1_sigma = Linear(state_size, hidden_size, bias=True, negative_slope=self.negative_slope)
        self.fc2_sigma = Linear(hidden_size, 1, bias=True, negative_slope=self.negative_slope)

    def forward(self, x):
        mu = self.fc2_mu(F.relu(self.fc1_mu(x)))
        log_sigma = self.fc2_sigma(F.relu(self.fc1_sigma(x)))
        return mu, log_sigma


class EpoctContinuousDecoder(nn.Module):
    """Continuous feature decoder"""

    def __init__(self, state_size: int, hidden_size: int = 10, negative_slope: int = 5):
        super().__init__()
        self.negative_slope = negative_slope
        self.fc1 = Linear(state_size, hidden_size, negative_slope=self.negative_slope)
        self.fc2 = Linear(hidden_size, 1, bias=True, negative_slope=self.negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
