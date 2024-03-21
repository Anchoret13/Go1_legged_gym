import os
import time
import math
import pickle
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import transformers

# from sys_id.trajectory_gpt2 import GPT2Model
from sys_id.dataset import load_trajectory, WheeledTraj, WheeledTrajWindowed
from model import GPT2

def test_script():
    input_size = 270
    hidden_size = 270
    n_layer = 2
    n_head =3
    pdrop = 0.1
    max_seq_length = 1000
    position_encoding = "sine"
    traj_transformer = GPT2(input_size, hidden_size, n_layer, n_head, pdrop, max_seq_length, position_encoding)

    test_path = "../dataset/example_dataset"
    dataset = WheeledTrajWindowed(test_path, window_size=5)
    data_loader = DataLoader(dataset, batch_size = 5, shuffle = True)
    for obs, act in data_loader:
        observation = obs
        print(observation.shape)

    traj_transformer.eval()
    with torch.no_grad():
        dummy_output = traj_transformer(observation, None)
    
    # print(dummy_output)

def train(
        dataset_folder_path,
        batch_size,
        shuffle, 
        window_size
):
    dataset = WheeledTraj(dataset_folder_path)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    input_size = 42 * window_size
    hidden_size = 42 * window_size
    model = GPT2(
        # THIS SHITTY IDIOT IS HARDCODING EVERYTHING AGAIN
        input_size = input_size,
        hidden_size = window_size, 
        n_layer = 2,
        n_head = 3,
        pdrop = 0.1,
        max_seq_length = 1000
    )

    
if __name__ == "__main__":
    test_script()