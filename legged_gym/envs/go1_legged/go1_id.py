from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import *

import torch
from typing import Tuple, Dict
from ..base.wheeled_robot import WheeledRobot
from ..go1_legged.go1 import Go1_Flat
from .go1_id_config import Go1FwFlatIDCfg

from sys_id.model import GPT2
from sys_id.RNN import GRU
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np

def update_history(history, new_obs):
    new_obs = new_obs.unsqueeze(1)
    updated_history = torch.cat((history[:, 1:, :], new_obs), dim=1)
    return updated_history

class Go1FlatID(Go1_Flat):
    def load_sys_id(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        id_model = GRU(**self.sys_model_params).to(device)
        checkpoint = torch.load(self.run_params['checkpoint_path'], map_location = device)
        id_model.load_state_dict(checkpoint['state_dict'])
        id_model.eval()
        return id_model
    
    def _init_buffers(self):
        super()._init_buffers()
        self.sys_id_path = '../../sys_id/logs/GRU/2024-05-12_13-41-58/checkpoint_epoch_20.pth'
        self.run_params = {
            'window_size': 50,
        }
        self.run_params['checkpoint_path'] = self.sys_id_path
        self.sys_model_params = {
            "n_layer": 2,
            "output_size": 16,
            "input_size": 27, 
            "hidden_size": 150, 
        } # NOTE: modify this
        self.window_size = self.run_params['window_size'] # NOTE: this stupid asshole hardcode again
        self.adaptive_module = self.load_sys_id()
        self.obs_history = torch.zeros(self.num_envs, self.window_size, self.sys_model_params["input_size"], dtype=torch.float, device=self.device, requires_grad=False)
    
    def compute_observations(self):
        adapt_output = self.compute_adapt_target()
        id_input = self.compute_adapt_input()
        self.history = update_history(self.obs_history, id_input)
        with torch.no_grad():
            id_output = self.adaptive_module(self.obs_history, None)
        self.obs_buf = torch.cat((
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            torch.clip(self.actions, -1, 1),
            id_output
        ), dim = -1)

        self.privileged_obs_buf = torch.cat((
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            torch.clip(self.actions, -1, 1),  
            adapt_output,
        ), dim = -1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.obs_history[env_ids] = torch.zeros(
        (self.window_size, self.sys_model_params["input_size"]),
        dtype=torch.float, 
        device=self.device, 
        requires_grad=False
        )