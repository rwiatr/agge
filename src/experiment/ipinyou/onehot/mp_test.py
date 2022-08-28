import os
import sys
from zoneinfo import available_timezones
sys.path.append(os.getcwd())
import threading

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import numpy as np

from torch.utils.data import Dataset, DataLoader
from experiment.ipinyou.onehot.data_manager import DataManager
from experiment.ipinyou.onehot.model import Mlp
from deepctr_torch.models import *
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import log_loss, roc_auc_score
import time
import pandas as pd
from optuna.samplers import TPESampler, CmaEsSampler
import copy
import torch.multiprocessing as mp


if __name__ == "__main__":
    available_gpus = torch.cuda.device_count()
    devices = []
    print(torch.device)
    for cuda in range(available_gpus):
        print(cuda)