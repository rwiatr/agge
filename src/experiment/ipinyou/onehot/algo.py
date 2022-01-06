import torch
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from experiment.display_bis import show_auc
from experiment.ipinyou.onehot.model import define_model, train_model
from experiment.measure import ProcessMeasure
import torch.nn as nn


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
        self.measure.print()

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
    def __init__(self):
        super(MLPRunner, self).__init__("MLP-v0")

    def algo(self, subject, X, y, **properties):
        mlp = define_model(X['train'].shape[1], 1, hidden_layer_sizes=properties['hidden_layer_sizes'])
        mlp.apply(init_weights)
        handler = train_model(model=mlp, X=X['train'], y=y['train'],
                              lr=properties['learning_rate_init'],
                              epochs=properties['max_iter'],
                              batch_size=properties['batch_size'],
                              weight_decay=properties['alpha'],
                              patience=properties['n_iter_no_change'],
                              validation_fraction=properties['validation_fraction'],
                              tol=properties['tol'])
        handler.best()
        auc = self.to_auc(mlp, subject, X, y)
        handler.last()
        return {**self.to_auc(mlp, subject, X, y),
                "best_model_test": auc['test'],
                "best_model_train": auc['train']}


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
