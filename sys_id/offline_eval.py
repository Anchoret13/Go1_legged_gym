import torch 
from torch.utils.data import DataLoader
import numpy as np
import pickle
import math
import time

from sys_id.dataset import load_trajectory, WheeledTrajWindowed
from model import GPT2

def load_model(checkpoint_path, model_params):
    model = GPT2(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()
    return model

def evaluate_model(
        test_path, 
        test_params
):
    dataset = WheeledTrajWindowed(test_path, test_params['window_size'])
    test_loader = DataLoader(dataset, test_params['batch_size'], shuffle = True)