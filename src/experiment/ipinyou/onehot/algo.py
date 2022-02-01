import torch
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from experiment.display_bis import show_auc
from experiment.ipinyou.nn.dnw import DeepAndWide
from experiment.ipinyou.onehot.model import DeepWide, define_model, train_model
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
        # self.measure.print()

    def algo(self, subject, X, y, **properties):
        return {"NA": 0}

    def to_auc(self, model, subject, X, y, auc_train_name='train', auc_test_name='test'):
        return {"train": show_auc(model, X[auc_train_name], y['train'],
                                  name="Train " + subject + " " + self.name, plot=False),
                "test": show_auc(model, X[auc_test_name], y['test'],
                                 name="Test " + subject + " " + self.name, plot=False)}


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
    def __init__(self, experiment_id=None):
        super(DeepWideRunner, self).__init__("DeepWide")
        self.experiment_id = experiment_id

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
                              # verbose = False,
                              experiment_id=self.experiment_id)

        handler.best()
        auc = self.to_auc(dw, subject, X, y)
        handler.last()
        return {**self.to_auc(dw, subject, X, y),
                "best_model_test": auc['test'],
                "best_model_train": auc['train']}


class DeepWideRunnerV2(AlgoRunner):
    def __init__(self, experiment_id=None):
        super(DeepWideRunnerV2, self).__init__("DeepWideV2")
        self.experiment_id = experiment_id

    def algo(self, subject, X, y, **properties):
        dw = DeepAndWide(input_sizes=(X['train'].shape[1], X['train_conj'].shape[1]),
                         hidden_layers_sizes=properties['hidden_layer_sizes'],
                         adagrad_props={
                             # lr = 1e-2, lr_decay = 0, weight_decay = 0, initial_accumulator_value = 0, eps = 1e-10
                             'lr': float(properties['learning_rate_init']),
                             'weight_decay': float(properties['alpha'])
                         },
                         ftrl_params={
                             # alpha = 1.0, beta = 1.0, l1 = 1.0, l2 = 1.0
                             "l2": 0.000001,
                             "l1": 1.0
                         })
        dw.apply(init_weights)

        handler = train_model(model=dw, X=X['train_and_conj'], y=y['train'],
                              lr=properties['learning_rate_init'],
                              epochs=properties['max_iter'],
                              batch_size=properties['batch_size'],
                              weight_decay=properties['alpha'],
                              patience=properties['n_iter_no_change'],
                              validation_fraction=properties['validation_fraction'],
                              tol=properties['tol'],
                              epsilon=properties['epsilon'],
                              early_stop=properties['early_stopping'],
                              # verbose = False,
                              experiment_id=self.experiment_id)

        handler.best()
        auc = self.to_auc(dw, subject, X, y, auc_train_name='train_and_conj', auc_test_name='test_and_conj')
        handler.last()
        return {**self.to_auc(dw, subject, X, y, auc_train_name='train_and_conj', auc_test_name='test_and_conj'),
                "best_model_test": auc['test'],
                "best_model_train": auc['train']}


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
