from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import *
from legged_gym.utils.terrain import Terrain

import torch
from typing import Tuple, Dict
from ..base.wheeled_robot import WheeledRobot
from ..go1_fw_clock.go1_fw import Go1FwClock
from .go1_fw_config import Go1FwTerrainCfg

class Go1FwTerrain(Go1FwClock):
    cfg : Go1FwTerrainCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # # self.num_passive_joints = self.cfg.env.num_passive_joints
        # self.frequencies = 3.0
        # self.current_step = 0

    # def _reward_masked_legs_energy(self):
    #     mask = torch.ones(self.torques.size(-1), device=self.torques.device, dtype=torch.bool)
    #     mask[self.dof_roller_ids] = False
    #     mask[9] = False
    #     mask[12] = False

    #     masked_torques = self.torques[:, mask]
    #     masked_dof_vel = self.dof_vel[:, mask]
    #     return torch.sum(torch.square(masked_torques * masked_dof_vel), dim=1)

    def compute_observations(self):
        super().compute_observations()
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

