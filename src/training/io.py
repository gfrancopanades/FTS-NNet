import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import time
from joblib import dump, load

from src.data_file_names import *

def check_and_create_folder(path):
    # Check if the path exists
    if not os.path.exists(path):
        # Create the folder if it doesn't exist
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")


# def r_squared(y_true, y_pred):
#     SS_res =  K.sum(K.square(y_true - y_pred)) 
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
#     return (1 - SS_res/(SS_tot + K.epsilon()))


###############################################################
###################### PYTORCH FUNCTIONS ######################
###############################################################


### SAVING AND LOADING 

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model


