from re import M
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from experiment.display_bis import show_auc
from experiment.ipinyou.onehot.model import DeepWide, define_model, train_model
from experiment.measure import ProcessMeasure
import torch.nn as nn

from deepctr_torch.models import *
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow import keras

class AlgoRunner:
    def __init__(self, name):
        self.name = name
        self.measure = ProcessMeasure()

    def set_measure(self, measure):
        self.measure = measure
        return self

    def run(self, X, y, subject, **properties):
        self.measure.set_suffix(";{}".format(';'.join([f"{key}={properties[key]}" for key in properties.keys()])))
        self.measure.start(f"::algorithm={self.name};subject={subject}")
        aucs = self.algo(subject, X, y, **properties)

        for key in aucs.keys():
            self.measure.data_point(aucs[key], collection=f"auc_{key}_::algorithm={self.name};subject={subject}")

        self.measure.stop(f"::algorithm={self.name};subject={subject}")
        # self.measure.print()

    def algo(self, subject, X, y, **properties):
        return {"NA": 0}

    def to_auc(self, model, subject, X, y):
        return {"train": show_auc(model, X['train'], y['train'], name="Train " + subject + " " + self.name, plot=False),
                "test": show_auc(model, X['test'], y['test'], name="Test " + subject + " " + self.name, plot=False)}


class SKLearnMLPRunner(AlgoRunner):
    def __init__(self):
        super(SKLearnMLPRunner, self).__init__("SKLearn-MLP")

    def algo(self, subject, X, y, **properties):
        mlp_sk = MLPClassifier(verbose=True, **properties).fit(X['train'], y['train'])
        return self.to_auc(mlp_sk, subject, X, y)


class SKLearnLRRunner(AlgoRunner):
    def __init__(self):
        super(SKLearnLRRunner, self).__init__("SKLearn-LR")

    def algo(self, subject, X, y, **properties):
        lr = LogisticRegression(**properties).fit(X['train'], y['train'])
        return self.to_auc(lr, subject, X, y)


class MLPRunner(AlgoRunner):
    def __init__(self, experiment_id=None):
        super(MLPRunner, self).__init__("MLP-v0")
        self.experiment_id = experiment_id

    def algo(self, subject, X, y, **properties):
        mlp = define_model(X['train'].shape[1], 1, hidden_layer_sizes=properties['hidden_layer_sizes'], bias=False)
        mlp.apply(init_weights)
        handler = train_model(model=mlp, X=X['train'], y=y['train'],
                              lr=properties['learning_rate_init'],
                              epochs=properties['max_iter'],
                              batch_size=properties['batch_size'],
                              weight_decay=properties['alpha'],
                              patience=properties['n_iter_no_change'],
                              validation_fraction=properties['validation_fraction'],
                              tol=properties['tol'],
                              epsilon=properties['epsilon'],
                              early_stop=properties['early_stopping'],
                              # verbose=False,
                              experiment_id=self.experiment_id)
        handler.best()
        auc = self.to_auc(mlp, subject, X, y)
        handler.last()
        return {**self.to_auc(mlp, subject, X, y),
                "best_model_test": auc['test'],
                "best_model_train": auc['train']}


class DeepWideRunner(AlgoRunner):
    def __init__(self):
        super(DeepWideRunner, self).__init__("DeepWide")

    def algo(self, subject, X, y, **properties):
        dw = DeepWide(X['train'].shape[1], 1, hidden_layers_sizes=properties['hidden_layer_sizes'])
        dw.apply(init_weights)

        handler = train_model(model=dw, X=X['train'], y=y['train'],
                              lr=properties['learning_rate_init'],
                              epochs=properties['max_iter'],
                              batch_size=properties['batch_size'],
                              weight_decay=properties['alpha'],
                              patience=properties['n_iter_no_change'],
                              validation_fraction=properties['validation_fraction'],
                              tol=properties['tol'],
                              epsilon=properties['epsilon'],
                              early_stop=properties['early_stopping'],
                              # verbose = False
                              )

        handler.best()
        auc = self.to_auc(dw, subject, X, y)
        handler.last()
        return {**self.to_auc(dw, subject, X, y),
                "best_model_test": auc['test'],
                "best_model_train": auc['train']}

class DeepFMRunner(AlgoRunner):
    def __init__(self):
        super(DeepFMRunner, self).__init__("DeepFM")

    def algo(self, subject, X, y , linear_feature_columns, dnn_feature_columns, **properties):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepFM(
            linear_feature_columns = linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns, 
            dnn_hidden_units=properties['hidden_layer_sizes'], 
            task='binary',
            l2_reg_embedding=1e-5, 
            device=device, 
            dnn_dropout=0.9)

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=properties['learning_rate_init'],
            betas=(properties['beta_1'], properties['beta_2']),
            eps=properties['epsilon'],
            weight_decay=properties['alpha'])
        
        model.compile(
            optimizer=optimizer,
             loss='binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])

        history = model.fit(
            x=X['train'], 
            y=y['train'], 
            batch_size=properties['batch_size'], 
            epochs=properties['max_iter'], 
            verbose=0,
            validation_data=(X['test'], y['test']),
            validation_split=0.2)

        train_auc = round(roc_auc_score(y['train'], model.predict(X['train'], properties['batch_size'])), 4)
        test_auc = round(roc_auc_score(y['test'], model.predict(X['test'], properties['batch_size'])), 4)
        return {"train" : train_auc, "test": test_auc}

class WDLRunner(AlgoRunner):
    def __init__(self):
        super(WDLRunner, self).__init__("wdl")

    def algo(self, subject, X, y , linear_feature_columns, dnn_feature_columns, **properties):
        model = WDL(linear_feature_columns = linear_feature_columns, dnn_feature_columns=dnn_feature_columns, dnn_hidden_units=properties['hidden_layer_sizes'], task='binary',
                   l2_reg_embedding=1e-5, device='cpu')
        
        model.compile('adagrad', 'binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])
        history = model.fit(x=X['train'], y=y['train'], batch_size=properties['batch_size'], epochs=properties['max_iter'], verbose=0, validation_split=0)

        
        test_auc = round(roc_auc_score(y['test'], model.predict(X['test'], properties['batch_size'])), 4)
        return {"test": test_auc}

class DCNRunner(AlgoRunner):
    def __init__(self):
        super(DCNRunner, self).__init__("dcn")

    def algo(self, subject, X, y , linear_feature_columns, dnn_feature_columns, **properties):
        model = DCN(linear_feature_columns = linear_feature_columns, dnn_feature_columns=dnn_feature_columns, dnn_hidden_units=properties['hidden_layer_sizes'], task='binary',
                   l2_reg_embedding=1e-5, device='cpu')
        
        model.compile('adagrad', 'binary_crossentropy', metrics = ['binary_crossentropy', 'auc'])
        history = model.fit(x=X['train'], y=y['train'], batch_size=properties['batch_size'], epochs=properties['max_iter'], verbose=0, validation_split=0)

        
        test_auc = round(roc_auc_score(y['test'], model.predict(X['test'], properties['batch_size'])), 4)
        return {"test": test_auc}

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

