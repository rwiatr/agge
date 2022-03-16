import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


class Mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers_sizes, bias=False, relu_embeddings=True):
        super(Mlp, self).__init__()
        # input layer
        self.embedding = nn.ModuleList()
        self.embedding.append(nn.Linear(input_size, hidden_layers_sizes[0], bias=bias))
        if relu_embeddings:
            self.embedding.append(nn.ReLU())

        # hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_layers_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i + 1]))
            self.hidden.append(nn.ReLU())

        # connect deep n wide
        self.output_layer = nn.Linear(hidden_layers_sizes[-1], output_size)

        #print(self)

    def forward(self, x):
        for net in self.embedding:
            x = net(x)
        for net in self.hidden:
            x = net(x)
        x = self.output_layer(x)

        x = torch.sigmoid(x)

        return x

    def decision_function(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)
        if type(X) is np.ndarray:
            X = torch.from_numpy(X).float()
        else:
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


class DeepWide(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers_sizes, bias=False, relu_embeddings=True):
        super(DeepWide, self).__init__()

        # deep
        self.embedding = nn.ModuleList()
        self.embedding.append(nn.Linear(input_size, hidden_layers_sizes[0], bias=bias))
        if relu_embeddings:
            self.embedding.append(nn.ReLU())

        # hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_layers_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i + 1]))
            self.hidden.append(nn.ReLU())

        # connect deep n wide
        self.output_layer = nn.Linear(hidden_layers_sizes[-1] + input_size, output_size)

        print(self)

    def forward(self, input_data):
        x = input_data
        # deep forward
        for net in self.embedding:
            x = net(x)
        for seq in self.hidden:
            x = seq(x)
        # deep n wide
        x = torch.cat((x, input_data), dim=1)
        x = self.output_layer(x)

        return torch.sigmoid(x)

    def decision_function(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)
        if type(X) is np.ndarray:
            X = torch.from_numpy(X).float()
        else:
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


class HandleDaset(Dataset):
    def __init__(self, x, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x = self.sparse_to_tensor(x).to(device)
        # self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length

    def spy_sparse2torch_sparse(self, data):
        """

        :param data: a scipy sparse csr matrix
        :return: a sparse torch tensor
        
        """

        samples = data.shape[0]
        features = data.shape[1]
        values = data.data
        coo_data = data.tocoo()
        indices = torch.LongTensor([coo_data.row, coo_data.col])
        t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
        t = torch.sparse_coo_tensor(data)
        return t

    def sparse_to_tensor(self, sparse_m):
        if type(sparse_m) is np.ndarray:
            return torch.from_numpy(sparse_m).float()
        sparse_m = sparse_m.tocoo()

        values = sparse_m.data
        indices = np.vstack((sparse_m.row, sparse_m.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = sparse_m.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


def define_model(input_size, output_size, hidden_layer_sizes, bias):
    return Mlp(input_size, output_size, hidden_layer_sizes, bias)


def prep_data(X, y, batch_size, shuffle=False):
    return DataLoader(HandleDaset(x=X, y=y), batch_size=batch_size, shuffle=shuffle, drop_last=True)


def train_model(model, X, y, lr, epochs, batch_size, weight_decay=1e-5, patience=5, stabilization=0,
                validation_fraction=0.1, tol=0, epsilon=1e-8, beta_1=0.9, beta_2=0.999, early_stop=True,
                verbose=True, experiment_id=None):
    # define model handler and early top
    handler = ModelHandler(model, './model' if experiment_id is None else f'./model_{experiment_id}')
    stop = EarlyStop(patience=patience, max_epochs=epochs, handler=handler, tol=tol)

    # train vali split
    n = X.shape[0]
    perut = torch.randperm(X.shape[0])
    train_fraction = (1 - validation_fraction)

    X_train = X[perut][0: int(n * train_fraction)]
    y_train = y[perut][0: int(n * train_fraction)]
    X_val = X[perut][int(n * train_fraction):]
    y_val = y[perut][int(n * train_fraction):]

    # tensorize and batch data
    trainloader = prep_data(X_train, y_train, batch_size=batch_size)
    validationloader = prep_data(X_val, y_val, batch_size=batch_size) if validation_fraction > 0 else trainloader

    # determine a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The device used for training is: {device}')

    model.to(device)
    print(f'Cuda?: ', next(model.parameters()).is_cuda)
    # loss and optimizer

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=epsilon,
                                 betas=(beta_1, beta_2))
    # nn.L1Loss
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 3, factor=0.5)

    # training loop
    epoch = 1
    while not stop.is_stop():
        train_steps = 0
        train_loss = 0.0

        model.train()
        for i, (attributes, labels) in enumerate(trainloader):
            # forward
            outputs = model(attributes)
            loss = loss_fn(outputs, labels.reshape(-1, 1))

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        val_loss = 0.0
        model.eval()
        val_steps = 0
        with torch.no_grad():
            for batch_id, (X, y) in enumerate(validationloader):
                outputs = model(X)
                loss = loss_fn(outputs, y.reshape(-1, 1))
                val_loss += loss.item()
                val_steps += 1

        if stabilization > 0:
            stabilization -= 1
        else:
            stop.update_epoch_loss(validation_loss=np.abs(val_loss / val_steps),
                                   train_loss=np.abs(train_loss / train_steps))

            handler.save(mtype='last')
            if stop.is_best():
                handler.save(mtype='best')

        curr_lr = optimizer.param_groups[0]['lr']
        if verbose:
            print(f'Epoch {epoch},    \
                Training Loss: {train_loss / train_steps:.8f}\t \
                Training Steps: {train_steps}\t \
                Validation Loss:{val_loss / val_steps:.8f}\t \
                Validation Steps:{val_steps}\t \
                LR:{curr_lr}')
        scheduler.step(val_loss)
        epoch += 1

    return handler


class EarlyStop:

    def __init__(self, handler, patience=100, max_epochs=None, tol=0):
        self.patience = patience
        self.best_loss = np.inf
        self.failures = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.max_epochs = max_epochs if max_epochs is not None else np.inf
        self.is_best_loss = False
        self.handler = handler
        self.tol = tol
        self.tol_problem = False

    def update_epoch_loss(self, train_loss, validation_loss):
        self.is_best_loss = self.handler.update_epoch_loss(validation_loss)
        self.train_losses.append(train_loss)
        self.val_losses.append(validation_loss)
        self.epoch += 1
        if len(self.val_losses) > 1:
            self.tol_problem = abs(self.val_losses[-2] - self.val_losses[-1]) < self.tol

        if self.is_best_loss:
            self.best_loss = validation_loss
            if self.tol_problem:
                self.failures += 1
            else:
                self.failures = 0
        else:
            self.failures += 1
        print(f'epoch={self.epoch}/{self.max_epochs}, failures={self.failures}/{self.patience}, '
              f'best_loss={self.is_best_loss}, tol={self.tol}, tol_failure={self.tol_problem}')

    def is_best(self):
        return self.is_best_loss

    def is_stop(self):
        return (self.failures > self.patience) or (self.epoch >= self.max_epochs)


class ModelHandler:

    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf
        self.success_updates = 0

    def reset(self):
        self.best_loss = np.inf
        self.success_updates = 0

    def best(self, path=None):
        path = path if path else self.path
        if path is not None:
            self.model.load_state_dict(torch.load(path + '.best.bin'), strict=True)

    def last(self, path=None):
        path = path if path else self.path
        if path is not None:
            self.model.load_state_dict(torch.load(path + '.last.bin'), strict=True)

    def save(self, path=None, mtype='last'):
        path = path if path else self.path
        if path is not None:
            model_path = path + '.' + mtype + '.bin'
            torch.save(self.model.state_dict(), model_path)

    def update_epoch_loss(self, loss):
        if loss >= self.best_loss:
            return False
        else:
            self.best_loss = loss
            self.success_updates += 1
            return True
