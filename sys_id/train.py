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
from datetime import datetime
from tqdm import tqdm

# from sys_id.trajectory_gpt2 import GPT2Model
from sys_id.dataset import load_phyprops, PhysProps, WheeledTrajWindowed
from model import GPT2

def save_checkpoint(state, epoch, filename):
    file = filename.format(epoch = epoch)
    torch.save(state, file)

def train(
        model_params,
        train_params,
        window_size,
        dataset_folder_path = "../dataset/wheeled_flat"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PhysProps(dataset_folder_path, window_size= window_size)
    data_loader = DataLoader(dataset, batch_size = train_params['batch_size'], shuffle = False)
    input_size = (42 + 12) * window_size
    hidden_size = (42 + 12) * window_size
    model_params['input_size'] = input_size
    model_params['hidden_size'] = hidden_size
    model = GPT2(**model_params).to(device)
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr = train_params['learning_rate'])
    base_log_dir = "./logs"
    date_time_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(base_log_dir, date_time_folder)
    writer = SummaryWriter(log_dir = checkpoint_dir)
    num_epochs = train_params['epochs']
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            targets = targets.squeeze()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            predictions, _ = model(inputs, None)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
        writer.add_scalar("Training_Loss", avg_loss, epoch)

        if (epoch + 1) % 20 == 0:
            print(f"Saving periodic checkpoint at epoch {epoch+1}...")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch = epoch + 1, filename=os.path.join(writer.log_dir, "checkpoint_epoch_{epoch}.pth"))
    
    writer.close()

if __name__ == "__main__":
    tmp_model_params = {
        "n_layer": 2,
        "n_head": 3,
        "pdrop": 0.1,
        "max_seq_length": 1000,
        'position_encoding': 'sine',
        "output_size": 3
    }
    test_train_params = {
        'epochs': 1000,
        'batch_size': 100,
        'learning_rate': 0.001
    }
    train(tmp_model_params, test_train_params, window_size=50)