from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import *

import torch
from typing import Tuple, Dict
from ..base.wheeled_robot import WheeledRobot
from ..go1_fw_clock.go1_fw import Go1FwClock
from .go1_fw_config import Go1FwTerrainCfg

def sigmoid(x, k, lower, upper):
    midpoint = (lower + upper) / 2.0
    scale = k / (upper - lower)
    return 1 / (1 + torch.exp(-scale * (x - midpoint)))

# def torch_rand_sigmoid(lower, upper, shape, device):
#     uniform_samples = torch.rand(*shape, device = device)
#     sigmoid_samples = torch.sigmoid((uniform_samples - 0.5) * 10) 
#     scaled_samples = (upper - lower) * sigmoid_samples + lower
#     return scaled_samples

def frequency_ac_vel(cmd_vel):
    stride_length = 0.6 # NOTE: adjust this
    frequency = cmd_vel / stride_length
    return frequency

def adaptive_sample_vel_cmd(min_vel, max_vel, current_step, env_ids, device, total_iterations = 60000, n_samples = 1000, steps_per_iteration = 24):
    #  NOTE: STUPID HARD CODING Method
    # compute k with total_iteration, k_range
    # num_steps_per_env = 24, so consider 24 steps as one iteration.
    if len(env_ids) == 0:
        env_ids = torch.tensor([0])
    current_iteration = current_step // steps_per_iteration
    k_min = -10
    k_max = 1
    if current_iteration < (total_iterations * 0.5):
        k = k_min + (current_iteration * (k_max - k_min) /  (total_iterations * 0.5))
    else:
        k = k_max
    values = torch.linspace(min_vel, max_vel, n_samples)
    probs = sigmoid(values, k, min_vel, max_vel)
    probs /= probs.sum()
    sampled_indices = torch.multinomial(probs, num_samples=len(env_ids), replacement=True)
    sampled_velocities = values[sampled_indices]
    commands = torch.zeros_like(env_ids, dtype = torch.float, device = device)

    for i, env_id in enumerate(env_ids):
        commands[i] = sampled_velocities[i]
    return commands

class Go1FwTerrain(Go1FwClock):
    cfg : Go1FwTerrainCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # # self.num_passive_joints = self.cfg.env.num_passive_joints
        # self.frequencies = 3.0
        # self.current_step = 0

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['rough', 'discrete', 'slope']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='rough':
            self._create_terrain_rough()
        elif mesh_type=='discrete':
            self._create_terrain_discrete()
        elif mesh_type=='slope':
            self._create_terrain_slope()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_terrain_rough(self):
        num_terains = 8
        terrain_width = 12.
        terrain_length = 12.
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)


        SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
        heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), 
                                                            min_height=-0.2, 
                                                            max_height=0.2, 
                                                            step=0.2, 
                                                            downsampled_scale=0.5
                                                            ).height_field_raw
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -1.
        tm_params.transform.p.y = -1.
        self.gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

    def _create_terrain_discrete(self):
        pass

    def _create_terrain_slope(self):
        pass
