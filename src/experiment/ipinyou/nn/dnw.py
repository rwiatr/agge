import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from ftrl import FTRL


class DeepAndWide(nn.Module):

    def __init__(self, input_sizes, hidden_layers_sizes, adagrad_props, ftrl_params,
                 bias_embeddings=False, relu_embeddings=True):
        super(DeepAndWide, self).__init__()

        self.ftrl_params = ftrl_params
        self.adagrad_props = adagrad_props

        self.features = input_sizes[0]
        self.cross_features = input_sizes[1]

        # deep
        deep = [nn.Linear(self.features, hidden_layers_sizes[0], bias=bias_embeddings)]
        if relu_embeddings:
            deep.append(nn.ReLU())

        for i in range(hidden_layers_sizes[1] - 1):
            deep.append(nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[0]))
            deep.append(nn.ReLU())
        deep.append(nn.Linear(hidden_layers_sizes[0], 1))

        self.deep = nn.Sequential(*deep)

        # wide
        self.wide = nn.Linear(self.features, 1)
        # self.wide = nn.Linear(self.features + self.cross_features, 1)

        print(self)

    def get_optimizer(self):
        deep_optim = optim.Adagrad(self.deep.parameters(),
                                   **self.adagrad_props)

        wide_optim = FTRL(self.wide.parameters(), **self.ftrl_params)

        return CompositeOptim(deep_optim, wide_optim)

    def forward(self, input_data):
        # deep forward
        # deep = self.deep(input_data[:, :self.features])
        deep = self.deep(input_data)

        # wide forward
        wide = self.wide(input_data)

        return torch.sigmoid(deep + wide)

    def decision_function(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)
        if type(X) is np.ndarray:
            X = torch.from_numpy(X).float()
        else:
            # TODO iterate
            sparse_m = X.tocoo()

            values = sparse_m.data
            indices = np.vstack((sparse_m.row, sparse_m.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)

            shape = sparse_m.shape

            X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        X = X.to(device)

        with torch.no_grad():
            outputs = self(X)
        return outputs.cpu().numpy()


class CompositeOptim(optim.Optimizer):
    def __init__(self, *optims):
        super(CompositeOptim, self).__init__([], {})
        self.optims = optims

    def zero_grad(self, set_to_none: bool = False):
        for o in self.optims:
            o.zero_grad(set_to_none)

    def step(self, closure=None):
        loss = 0
        for o in self.optims:
            loss += o.step(closure)
