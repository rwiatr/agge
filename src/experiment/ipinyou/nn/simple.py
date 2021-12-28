import torch.nn as nn


def lstm(features, num_layers=1, hidden=64, out_features=1):
    class LSTM_model(nn.Module):
        def __init__(self):
            super(LSTM_model, self).__init__()
            self.out_features = out_features
            self.lstm = nn.LSTM(input_size=features, hidden_size=hidden,
                                num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden, out_features)

        def forward(self, x):
            res, _ = self.lstm(x)
            return self.fc(res[:, -1:, ])

    return LSTM_model()


def mlp(features, num_layers=1, hidden=64, out_features=1, batch_norm=False, dropout=None, input_dropout=None):
    if num_layers == 1:
        return nn.Linear(features, out_features)

    layers = [nn.Linear(features, hidden)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(hidden))
    layers.append(nn.ReLU())
    if input_dropout is not None:
        layers.append(nn.Dropout(p=input_dropout))

    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden, hidden))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(p=dropout))

    layers.append(nn.Linear(hidden, out_features))
    layers.append(nn.Sigmoid())

    result = nn.Sequential(*layers)

    return result


def classify(model):
    return nn.Sequential(model, nn.Sigmoid())
