import os
import time
import math
import pickle
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

# from sys_id.trajectory_gpt2 import GPT2Model
from sys_id.dataset import load_trajectory, WheeledTraj, WheeledTrajWindowed
from model import GPT2

def test_script():
    input_size = 270 # obs 42 act 12 
    hidden_size = 270
    n_layer = 2
    n_head = 3
    pdrop = 0.1
    max_seq_length = 1000
    position_encoding = "sine"
    traj_transformer = GPT2(input_size, hidden_size, n_layer, n_head, pdrop, max_seq_length, position_encoding)

    test_path = "../dataset/example_dataset"
    dataset = WheeledTrajWindowed(test_path, window_size=5)
    data_loader = DataLoader(dataset, batch_size = 10, shuffle = True)
    for obs, act in data_loader:
        observation = obs
        print("OBSERVATION SHAPE")
        print(observation.shape)

    traj_transformer.eval()
    with torch.no_grad():
        dummy_output = traj_transformer(observation, None)
    
    print(dummy_output)

def save_checkpoint(state, filename):
    torch.save(state, filename)

def train(
        model_params,
        train_params,
        window_size,
        dataset_folder_path = "../dataset/example_dataset"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WheeledTrajWindowed(dataset_folder_path, window_size=window_size)
    data_loader = DataLoader(dataset, batch_size = train_params['batch_size'], shuffle = False)
    input_size = (42+12) * window_size
    hidden_size = (42+12) * window_size
    model_params['input_size'] = input_size
    model_params["hidden_size"] = hidden_size
    model = GPT2(**model_params).to(device)
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=train_params['learning_rate'])
    
    writer = SummaryWriter(log_dir = "./logs")
    # Training loop
    num_epochs = train_params['epochs']
    model.train()  
    
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions, _ = model(inputs, None) 
            loss = loss_fn(predictions, targets)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
        writer.add_scalar("Training_Loss", avg_loss, epoch)

        if (epoch + 1) % 500 == 0:
            print(f"Saving periodic checkpoint at epoch {epoch+1}...")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch, filename=os.path.join(writer.log_dir, "checkpoint_epoch_{}.pth.tar"))

    writer.close()
    
if __name__ == "__main__":
    test_model_params = {
        "n_layer": 2,
        "n_head": 3,
        "pdrop": 0.1,
        "max_seq_length": 1000,
        'position_encoding': 'sine'
    }
    test_train_params = {
        'epochs': 10000,
        'batch_size': 5,
        'learning_rate': 0.001
    }

    train(test_model_params, test_train_params, window_size = 20)
    # test_script()