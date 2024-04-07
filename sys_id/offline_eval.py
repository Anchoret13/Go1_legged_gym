import torch 
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np
import pickle
import math
import time

from sys_id.dataset import load_trajectory, WheeledTrajWindowed, PhysProps
from model import GPT2

def evaluate(model_params, eval_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(eval_params['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    dataset = PhysProps(eval_params['dataset_folder_path'], window_size=eval_params['window_size'])
    data_loader = DataLoader(dataset, batch_size=eval_params['batch_size'], shuffle=False)

    model.eval()
    total_loss = 0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():  # Evaluation, no gradient needed
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, _ = model(inputs, None)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Average Loss: {avg_loss}")
    return avg_loss


if __name__ == "__main__":
    eval_params = {
         'checkpoint_path': 'path/to/your/checkpoint.pth', 
        'dataset_folder_path': '../dataset/wheeled_flat', 
        'window_size': 10,
        'batch_size': 1, 
    }

    model_params = {
        "n_layer": 2,
        "n_head": 3,
        "pdrop": 0.1,
        "max_seq_length": 1000,
        'position_encoding': 'sine',
        "output_size": 3,
        "input_size": (42 + 12) * eval_params['window_size'], 
        "hidden_size": (42 + 12) * eval_params['window_size'], 
    }