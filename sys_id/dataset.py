from torch.utils.data import Dataset, DataLoader
import torch
import pickle as pkl
import os
import re
import numpy as np

def load_trajectory(traj_path):
    with open(traj_path, 'rb') as file:
        data = pkl.load(file)
    obs = data['obs']
    act = data['act']

    return obs, act

class WheeledTraj(Dataset):
    def __init__(self, directory):
        self.file_paths = sorted(
            [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pkl')],
            key = lambda x: int(re.search(r'traj_(\d+).pkl', x).group(1))
        )

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        obs, act = load_trajectory(self.file_paths[idx])
        obs_tensor = torch.tensor(obs, dtype=torch.float)
        act_tensor = torch.tensor(act, dtype=torch.long)
        
        return obs_tensor, act_tensor
    
class WheeledTrajWindowed(Dataset):
    def __init__(self, directory, window_size):
        self.file_paths = sorted(
            [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pkl')],
            key=lambda x: int(re.search(r'traj_(\d+).pkl', x).group(1))
        )
        self.window_size = window_size

    def __len__(self):
        # Each file is considered one sample for simplicity; adjust as needed for windowing
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        obs, act = load_trajectory(self.file_paths[idx])
        
        # Prepare windowed sequences
        input_seq = []  
        target_seq = []  
        
        for i in range(len(obs) - self.window_size):
            # Concatenate observation and action history as input
            window_obs = np.array(obs[i:i+self.window_size]).flatten()
            window_act = np.array(act[i:i+self.window_size]).flatten()
            window_input = np.concatenate([window_obs, window_act])
            
            target_obs = obs[i + self.window_size]
            
            input_seq.append(window_input)
            target_seq.append(target_obs)
        
        # Convert to tensors
        input_seq_tensor = torch.tensor(input_seq, dtype=torch.float)
        target_seq_tensor = torch.tensor(target_seq, dtype=torch.float) 
        
        return input_seq_tensor, target_seq_tensor

class WheeledTracjSAS(Dataset):
    # This returns the transitions
    def __init__(self, directory, window_size):
        self.file_paths = sorted(
            [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pkl')],
            key=lambda x: int(re.search(r'traj_(\d+).pkl', x).group(1))
        )
        self.window_size = window_size

    def __len__(self):
        # Each file is considered one sample for simplicity; adjust as needed for windowing
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        obs, act = load_trajectory(self.file_paths[idx])
        
        # Initialize lists to hold the formatted inputs and targets
        input_seq = []
        target_seq = []

        # Start from the first timestep, and ensure there is a next observation to predict
        for i in range(1, len(obs) - 1):
            # Previous observation and action, and current observation
            input_features = np.concatenate([obs[i-1], act[i-1], obs[i]])
            input_seq.append(input_features)
            
            # Next observation as target
            target_obs = obs[i + 1]
            target_seq.append(target_obs)
        
        # Convert lists to tensors
        input_seq_tensor = torch.tensor(input_seq, dtype=torch.float)
        target_seq_tensor = torch.tensor(target_seq, dtype=torch.float)
        
        return input_seq_tensor, target_seq_tensor