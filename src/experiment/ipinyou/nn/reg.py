import torch.nn as nn
import torch as torch
import numpy as np

from experiment.ipinyou.onehot.model import get_dev


class SimpleReg(nn.Module):
    def __init__(self, features, alpha):
        super(SimpleReg, self).__init__()
        self.fc = nn.Linear(features, 1)
        self.alpha = alpha

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

    def get_loss(self):
        return CompositeLoss(nn.BCELoss(),
                             L2Loss(self.alpha, self.fc.parameters()))

    def decision_function(self, X):
        device = get_dev()

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


class DmpReg(nn.Module):
    def __init__(self, features, alpha, dmp):
        super(DmpReg, self).__init__()
        self.fc = nn.Linear(features, 1)
        self.alpha = alpha
        self.dmp = dmp

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

    def get_loss(self):
        return CompositeLoss(nn.BCELoss(), AggeL2Loss(
            self.alpha,
            self.dmp,
            list(self.fc.parameters())
        ))

    def decision_function(self, X):
        device = get_dev()

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


class L2Loss(nn.Module):
    def __init__(self, alpha, params):
        super(L2Loss, self).__init__()
        self.params = list(params)
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = None

        for param in self.params:
            _loss = self.alpha * torch.sum(param * param)
            if loss is not None:
                loss += _loss
            else:
                loss = _loss

        return loss


class AggeL2Loss(nn.Module):
    def __init__(self, alpha, dmp, params):
        super(AggeL2Loss, self).__init__()
        self.params = params
        self.alpha = alpha
        self.dmp = dmp

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = None

        bias = 0
        for param in self.params:
            if bias % 2 == 0:
                p = param * param * self.dmp()[:-1]
            else:
                p = param * param * self.dmp()[-1:]

            _loss = self.alpha * torch.sum(p)
            if loss is not None:
                loss += _loss
            else:
                loss = _loss
            bias += 1
        return torch.sum(loss)


def damping_v0(avg_counts, a):
    norm = (avg_counts - torch.min(avg_counts)) / (torch.max(avg_counts) - torch.min(avg_counts))
    return 1 + a * (1 - 2 * norm)


def damping_v1(damp):
    return damp


class CompositeLoss(nn.Module):
    def __init__(self, *losses):
        super(CompositeLoss, self).__init__()
        self.losses = losses

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = None

        for l in self.losses:
            _loss = l(input, target)
            if loss is not None:
                loss += _loss
            else:
                loss = _loss

        return loss
