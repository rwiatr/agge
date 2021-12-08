import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


class Mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers_sizes):
        super(Mlp, self).__init__()
        # input layer
        self.input_layer = nn.Linear(input_size, hidden_layers_sizes[0])
        self.relu = nn.ReLU()
       
        # hidden layers
        self.linears_relus = nn.ModuleList()
        for i in range(len(hidden_layers_sizes)-1):
            self.linears_relus.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))
            self.linears_relus.append(nn.ReLU())

        #output layer
        self.output_layer = nn.Linear(hidden_layers_sizes[-1], output_size)     
    
    def forward(self, data_input):
        x = self.input_layer(data_input)
        x = self.relu(x)
        for seq in self.linears_relus:
            x = seq(x)

        out = torch.sigmoid(self.output_layer(x))
        return out

    def decision_function(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)
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




class dataset(Dataset):
    def __init__(self,x,y):
        self.x = self.sparse_to_tensor(x)
        #self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
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

def define_model(input_size, output_size, hidden_layer_sizes):
    return Mlp(input_size, output_size, hidden_layer_sizes)

def prep_data(X, y, batch_size, shuffle=False):
    return DataLoader(dataset(x=X, y=y), batch_size=batch_size, shuffle=shuffle, drop_last=True)

def train_model(model, X, y, lr, epochs, batch_size):
    # tensorize and batch data
    trainloader = prep_data(X, y, batch_size=batch_size)

    # determine a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The device used for training is: {device}')
    
    model.to(device)
 
    # loss and optimizer
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=.1)

    # training loop

    n_total_steps = len(trainloader)
    for epoch in range(epochs):
        loss_factor = 0
        loss_number = 0
        train_loss = 0.0

        for i, (attributes, labels) in enumerate(trainloader):
            attributes = attributes.to(device)
            labels = labels.to(device)

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
      
            #if (i+1) % 100 == 0:
                
                #print(f'epoch {epoch + 1} / {epochs}, step {i+1}/{n_total_steps}, loss = {loss:.4f}')
        #losses += [loss_factor/loss_number]
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch},    \
            Training Loss: {(train_loss/loss_number):.8f}\t \
            LR:{curr_lr}')
        scheduler.step()

if __name__ == '__main__':
    
    X = np.random.rand(986207, 561)
    y = np.random.randint(2, size=986207)

    Xt = np.random.rand(262211, 561)
    yt = np.random.randint(2, size=262211)

    mlp = define_model(input_size=X.shape[1], output_size=1, hidden_layer_sizes=(10, 15, 7))
    train_model(model = mlp, X=X, y=y, lr=1e-5, epochs=3, batch_size=500)    
    
    predictions = mlp.decision_function(Xt)
    print(predictions.shape)
    print(predictions)
