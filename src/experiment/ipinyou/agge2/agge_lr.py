import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def loss_function(y_hat, y):
    return np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))


def objective_function(beta, X, Y):
    error = loss_function(sigmoid(np.matmul(X, beta)), Y)
    return (error)


class LR2:

    def __init__(self, reg_type):
        self.params = None
        self.reg_type = reg_type

    def fit(self, X, y, cnt_norm=None):
        print('train')
        self.params = np.random.random(X.shape[1] + 1)
        X_tmp = np.ones((X.shape[0], X.shape[1] + 1))
        X_tmp[:, :-1] = X

        result = minimize(objective_function, self.params, args=(X_tmp, y),
                          method='BFGS', options={'maxiter': 500})

        self.params = result.x
        return self

    def decision_function(self, X):
        # if self.params is None:
        #     self.params = np.random.random(X.shape[1] + 1)
        X_tmp = np.ones((X.shape[0], X.shape[1] + 1))
        X_tmp[:, :-1] = X

        return sigmoid(np.matmul(X_tmp, self.params))


class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """

    def __init__(self, loss_function=loss_function, norm_weights=None,
                 X=None, Y=None, sample_weights=None, beta_init=None, train_norm_weights=False,
                 regularization=0.00012, new_path=None, a=None):
        self.regularization = regularization
        self.beta = None
        self.loss_function = loss_function
        self.sample_weights = sample_weights
        self.beta_init = beta_init
        self.X = X
        self.Y = Y
        # if train_norm_weights:
        #     self.X_tmp = np.ones((X.shape[0], X.shape[1] + 3))
        #     self.X_tmp[:, :-3] = X
        # else:
        self.X_tmp = np.ones((X.shape[0], X.shape[1] + 1))
        self.X_tmp[:, :-1] = X
        self.norm_weights = norm_weights
        self.train_norm_weights = train_norm_weights
        self.regularization_sqrt = np.sqrt(self.regularization)
        self.new_path = new_path
        self.a = a
        self.first_report = True

    def decision_function(self, X):
        X_tmp = np.ones((X.shape[0], X.shape[1] + 1))
        X_tmp[:, :-1] = X

        if self.train_norm_weights:
            prediction = sigmoid(np.matmul(X_tmp, self.beta[0:-2]))
        else:
            prediction = sigmoid(np.matmul(X_tmp, self.beta))
        return (prediction)

    def __decision_function(self):
        if self.train_norm_weights:
            prediction = sigmoid(np.matmul(self.X_tmp, self.beta[0:-2]))
        else:
            prediction = sigmoid(np.matmul(self.X_tmp, self.beta))
        return (prediction)

    def model_error(self):
        error = self.loss_function(self.__decision_function(), self.Y)
        return (error)

    def l2_regularized_loss(self, beta):
        self.beta = beta
        beta_loss = np.array(self.beta)

        if self.a is not None:
            reg = self.regularization + self.regularization * self.a * self.norm_weights
            if self.first_report:
                print(f'Report: a={self.a}, reg={self.regularization}, min={min(reg)}, max={max(reg)}')
                self.first_report = False
        else:
            reg = self.regularization

        reg_loss = sum(reg * (beta_loss ** 2))
        return (self.model_error() + reg_loss) / self.X.shape[0]

    def fit(self, maxiter=250):
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.beta_init) == type(None):
            if self.train_norm_weights:
                self.beta_init = np.array([1] * (self.X.shape[1] + 3))
            else:
                # set beta_init = 1 for every feature
                self.beta_init = np.array([1] * (self.X.shape[1] + 1))
        else:
            # Use provided initial values
            pass

        if self.beta != None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")

        res = minimize(self.l2_regularized_loss, self.beta_init,
                       method='L-BFGS-B', options={'maxiter': 6000}, tol=0.0001)
        self.beta = res.x
        self.beta_init = self.beta

        return self

