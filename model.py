import warnings

import torch
import torch.nn as nn


class DeepCharacteristics(nn.Module):
    """ """

    def __init__(self, n_layers, in_channels, out_channels, features, activation_type="relu", bias=True):
        super().__init__()
        self.layers = []
        self.layers.append(
            DenseBlock(in_channels=in_channels, out_channels=features, activation_type=activation_type, bias=bias)
        )
        # intermediate layers
        for _ in range(n_layers - 2):
            self.layers.append(
                DenseBlock(in_channels=features, out_channels=features, activation_type=activation_type, bias=bias)
            )
        # last layer
        self.layers.append(
            DenseBlock(in_channels=features, out_channels=out_channels, activation_type=activation_type, bias=bias)
        )
        self.FC = nn.Sequential(*self.layers)

    def forward(self, Z):
        Y = self.FC(Z)
        return Y


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type="relu", bias=True):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        if activation_type == "linear":
            self.activate = nn.Identity()
        elif activation_type == "relu":
            self.activate = nn.ReLU(inplace=True)
        elif activation_type == "lrelu":
            self.activate = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(f"Not implemented activation function: " f"`{activation_type}`!")

    def forward(self, Z):

        Z = self.linear(Z)
        Z = self.activate(Z)
        N, M, K = Z.shape
        Z = Z.view(N * M, K)
        Z = self.bn(Z)
        return Z.view(N, M, K)


class SortedFactorModel(nn.Module):
    def __init__(
        self,
        n_layers,
        in_channels,
        features,
        n_deep_factors,
        n_BM_factors,
        n_portfolio,
        activation_type="relu",
        bias=True,
        ranking_method="softmax",
    ):
        super().__init__()
        self.DC_network = DeepCharacteristics(n_layers, in_channels, n_deep_factors, features, activation_type, bias)
        self.beta = nn.Parameter(torch.randn(n_portfolio, n_deep_factors), requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(n_portfolio, n_BM_factors), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros(n_portfolio, n_BM_factors), requires_grad=False)
        self.register_parameter(name="gamma", param=self.gamma)
        self.register_parameter(name="beta", param=self.beta)
        self.ranking_method = ranking_method

    def forward(self, Z, r, g):
        """
        Args:
            Z ([Tensor(T x M x K)]): firm characteristics
            r ([Tensor(T x M)]): firm returns
            g ([Tensor(T x D)]): benchmark factors
        """
        if len(Z.size()) == 2:
            Z = Z[None, :, :]
        Y = self.DC_network(Z)
        W = rank_weight(Y, method=self.ranking_method)  # T x M x P \ P := n_deep_factors
        f = torch.matmul(W.transpose(1, 2), r.unsqueeze(dim=-1)).squeeze(dim=-1)  # T x P

        R = torch.matmul(self.beta, f.transpose(0, 1))
        R += torch.matmul(self.gamma, g.transpose(0, 1))
        R = R.transpose(0, 1)
        return R, f


def rank_weight(Y, method="softmax"):
    """Applies the rank weight operation

    Args:
        Y      ([Tensor(T x M x P)])
        method (string)
    """
    eps = 1 - 6
    mean = torch.mean(Y, axis=1)
    std = torch.std(Y, axis=1)

    normalised_data = (Y - mean[:, None, :]) / (std[:, None, :] + eps)
    if method == "softmax":
        y_p = -50 * torch.exp(-5 * normalised_data)
        y_n = -50 * torch.exp(5 * normalised_data)
        softmax = nn.Softmax(dim=1)
        W = softmax(y_p) - softmax(y_n)
    elif method == "equal_ranks":
        T, M, P = Y.size()
        uniform_weight = 1 / (M // 3)
        _, indices = torch.sort(Y, dim=2)
        W = torch.zeros(Y.size())
        for t in range(T):
            for i in range(P):
                W[t, indices[t, 2 * M // 3:, i], i] = uniform_weight
                W[t, indices[t, M // 3: 2 * M // 3, i], i] = 0
                W[t, indices[t, : M // 3, i], i] = -uniform_weight
    else:
        warnings.warn(f"{method} not implemented yet. Softmax ranking will be applied.")
        return rank_weight(Y, method="softmax")

    return W
