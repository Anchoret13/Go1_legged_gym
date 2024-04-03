import torch 
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np
import pickle
import math
import time

from sys_id.dataset import load_trajectory, WheeledTrajWindowed
from model import GPT2

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])

def evaluate(
        model_params,
        test_path, 
        test_params
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WheeledTrajWindowed(test_path, test_params['window_size'])
    test_loader = DataLoader(dataset, test_params['batch_size'], shuffle = True)
    input_size = (42 + 12) * test_params['window_size']
    hidden_size = (42 + 12) * test_params['window_size']
    model_params['input_size'] = input_size
    model_params["hidden_size"] = hidden_size
    model = GPT2(**model_params).to(device)

    checkpoint_path = test_params['checkpoint_path']
    load_checkpoint(model, checkpoint_path, device)

    model.eval()

    loss_fn = MSELoss()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, _ = model(inputs, None)
            loss = loss_fn(predictions, targets)
            
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)

if __name__ == "__main__":
    eval_model_params = {
        "n_layer": 2,
        "n_head": 3,
        "pdrop": 0.1,
        "max_seq_length": 1000,
        'position_encoding': 'sine'
    }
    test_eval_params = {
        'batch_size' : 1,
        'checkpoint_path': "./log"
    }