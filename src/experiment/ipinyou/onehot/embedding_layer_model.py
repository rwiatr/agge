from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import torch.optim.lr_scheduler as lr_scheduler


class Mlp(nn.Module):
    def __init__(self, emb_dims, output_size, hidden_layers_sizes, no_of_cont=0):
        super(Mlp, self).__init__()
        
        # embedding layer
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x,y in emb_dims])

        no_of_embs = sum([y for x,y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # input layer
        self.input_layer = nn.Linear(self.no_of_embs + self.no_of_cont, hidden_layers_sizes[0])

        self.relu = nn.ReLU()
       
        # hidden layers
        self.linears_relus = nn.ModuleList()
        for i in range(len(hidden_layers_sizes)-1):
            self.linears_relus.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))
            self.linears_relus.append(nn.ReLU())

        #output layer
        self.output_layer = nn.Linear(hidden_layers_sizes[-1], output_size)     
    
    def forward(self, data_input):
        
        if self.no_of_embs != 0:
            x = [emb_layer(data_input[:,i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)

        x = self.input_layer(x)
        x = self.relu(x)
        for seq in self.linears_relus:
            x = seq(x)

        out = torch.sigmoid(self.output_layer(x))
        return out

    def decision_function(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X = torch.tensor(X.values,dtype=torch.int64)
        X = X.to(device)

        with torch.no_grad():
            outputs = self(X)
        return outputs.cpu().numpy()


class _Dataset(Dataset):
    def __init__(self,x,y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.x = self.sparse_to_tensor(x).to(device)
        self.x = torch.tensor(x,dtype=torch.float32).to(device)
        self.y = torch.tensor(y,dtype=torch.float32).to(device)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
    def __len__(self):
        return self.length

    def spy_sparse2torch_sparse(self, data):
        """

        :param data: a scipy sparse csr matrix
        :return: a sparse torch tensor
        
        """

        samples=data.shape[0]
        features=data.shape[1]
        values=data.data
        coo_data=data.tocoo()
        indices=torch.LongTensor([coo_data.row,coo_data.col])
        t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
        t=torch.sparse_coo_tensor(data)
        return t
    
    def sparse_to_tensor(self, sparse_m):

        sparse_m = sparse_m.tocoo()

        values = sparse_m.data
        indices = np.vstack((sparse_m.row, sparse_m.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = sparse_m.shape
        
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

from torch.utils.data import Dataset, DataLoader
class TabularDataset(Dataset):
  def __init__(self, data, cat_cols=None, output_col=None):
    """
    Characterizes a Dataset for PyTorch

    Parameters
    ----------

    data: pandas data frame
      The data frame object for the input data. It must
      contain all the continuous, categorical and the
      output columns to be used.

    cat_cols: List of strings
      The names of the categorical columns in the data.
      These columns will be passed through the embedding
      layers in the model. These columns must be
      label encoded beforehand. 

    output_col: string
      The name of the output variable column in the data
      provided.
    """

    self.n = data.shape[0]

    if output_col:
      self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
    else:
      self.y =  np.zeros((self.n, 1))

    self.cat_cols = cat_cols if cat_cols else []
    self.cont_cols = [col for col in data.columns
                      if col not in self.cat_cols + [output_col]]

    if self.cont_cols:
      self.cont_X = data[self.cont_cols].astype(np.float32).values
    else:
      self.cont_X = np.zeros((self.n, 1))

    if self.cat_cols:
      self.cat_X = data[cat_cols].astype(np.int64).values
    else:
      self.cat_X =  np.zeros((self.n, 1))

  def __len__(self):
    """
    Denotes the total number of samples.
    """
    return self.n

  def __getitem__(self, idx):
    """
    Generates one sample of data.
    """
    return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]

def define_model(embed_size, output_size, hidden_layer_sizes):
    return Mlp(embed_size, output_size, hidden_layer_sizes)

def prep_data(X, y, batch_size, shuffle=False):
    return DataLoader(_Dataset(x=X, y=y), batch_size=batch_size, shuffle=shuffle, drop_last=True)

def train_model(model, X, lr, epochs, batch_size, patience):

    patience = patience

    #define model handler and early top
    handler = ModelHandler(model, './model-emb.bin')
    stop = EarlyStop(patience=patience, max_epochs= epochs, handler=handler)

    # train vali split
    n = X.shape[0]
    train = X[:int(n*.9)]
    vali = X[int(n*.9):]

    print(X.shape, vali.shape)
    
    cols = list(X.columns)
    cols.remove('click')

    dataset = TabularDataset(data=train, cat_cols=cols, output_col='click')
    trainloader = DataLoader(dataset, batch_size, drop_last=True)

    dataset_vali = TabularDataset(data=vali, cat_cols=cols, output_col='click')
    validationloader = DataLoader(dataset_vali, batch_size, drop_last=True)
    #trainloader = prep_data(X_train, y_train, batch_size=batch_size)
    #validationloader = prep_data(X_val, y_val, batch_size=batch_size)

    # determine a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The device used for training is: {device}')
    
    model = model.to(device)
 
    # loss and optimizer
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3)

    # training loop

    n_total_steps = len(trainloader)
    epoch = 1
    while not stop.is_stop():
        loss_factor = 0
        loss_number = 0
        train_loss = 0.0

        model.train()
        for i, (labels, cat, attributes) in enumerate(trainloader):

            # forward
            outputs = model(attributes)
            loss = loss_fn(outputs, labels.reshape(-1, 1))
            loss_factor += loss.item()
            loss_number += 1

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0.0
        val_number = 0
        model.eval()
        with torch.no_grad():
            for batch_id, (y, cat, X) in enumerate(validationloader):
                outputs = model(X)
                loss = loss_fn(outputs, y.reshape(-1, 1))
                val_loss += loss.item()
                val_number += 1

        validation_loss = np.abs(val_loss/val_number)
        train_loss = np.abs((train_loss/loss_number))
        stop.update_epoch_loss(validation_loss=validation_loss, train_loss=train_loss)

        if stop.is_best():
                stop.handler.save(mtype='best')

        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch},    \
            Training Loss: {(train_loss):.8f}\t \
            Validation Loss:{(validation_loss):.8f}\t \
            LR:{curr_lr}')
        scheduler.step(validation_loss)
        epoch += 1
    if os.path.isfile('./model-emb.bin.best.bin'):
        model.load_state_dict(torch.load('./model-emb.bin.best.bin'))
    return model

class EarlyStop:

    def __init__(self, handler, patience=100, max_epochs=None):
        self.patience = patience
        self.best_loss = np.inf
        self.failures = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.max_epochs = max_epochs if max_epochs is not None else np.inf
        self.is_best_loss = False
        self.handler = handler

    def update_epoch_loss(self, train_loss, validation_loss):
        self.is_best_loss = self.handler.update_epoch_loss(validation_loss)
        self.train_losses.append(train_loss)
        self.val_losses.append(validation_loss)
        self.epoch += 1
        if self.is_best_loss:
            self.failures = 0
            self.best_loss = validation_loss
        else:
            self.failures += 1

    def is_best(self):
        return self.is_best_loss

    def is_stop(self):
        return (self.failures > self.patience) or (self.epoch >= self.max_epochs)


class ModelHandler:

    def __init__(self, model, path):
        # self.model = nn.DataParallel(model)
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
            self.model.load_state_dict(torch.load(path + '.best.bin'))

    def last(self, path=None):
        path = path if path else self.path
        if path is not None:
            self.model.load_state_dict(torch.load(path + '.last.bin'))

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