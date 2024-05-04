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
from .go1_fw_config import Go1FwTestCfg

# def sigmoid(x, k, lower, upper):
#     midpoint = (lower + upper) / 2.0
#     scale = k / (upper - lower)
#     return 1 / (1 + torch.exp(-scale * (x - midpoint)))

# # def torch_rand_sigmoid(lower, upper, shape, device):
# #     uniform_samples = torch.rand(*shape, device = device)
# #     sigmoid_samples = torch.sigmoid((uniform_samples - 0.5) * 10) 
# #     scaled_samples = (upper - lower) * sigmoid_samples + lower
# #     return scaled_samples

# def frequency_ac_vel(cmd_vel):
#     stride_length = 0.6 # NOTE: adjust this
#     frequency = cmd_vel / stride_length
#     return frequency

# def adaptive_sample_vel_cmd(min_vel, max_vel, current_step, env_ids, device, total_iterations = 60000, n_samples = 1000, steps_per_iteration = 24):
#     #  NOTE: STUPID HARD CODING Method
#     # compute k with total_iteration, k_range
#     # num_steps_per_env = 24, so consider 24 steps as one iteration.
#     if len(env_ids) == 0:
#         env_ids = torch.tensor([0])
#     current_iteration = current_step // steps_per_iteration
#     k_min = -10
#     k_max = 1
#     if current_iteration < (total_iterations * 0.5):
#         k = k_min + (current_iteration * (k_max - k_min) /  (total_iterations * 0.5))
#     else:
#         k = k_max
#     values = torch.linspace(min_vel, max_vel, n_samples)
#     probs = sigmoid(values, k, min_vel, max_vel)
#     probs /= probs.sum()
#     sampled_indices = torch.multinomial(probs, num_samples=len(env_ids), replacement=True)
#     sampled_velocities = values[sampled_indices]
#     commands = torch.zeros_like(env_ids, dtype = torch.float, device = device)

#     for i, env_id in enumerate(env_ids):
#         commands[i] = sampled_velocities[i]
#     return commands

class Go1FwTest(Go1FwClock):
    cfg : Go1FwTestCfg
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

    def _create_envs(self):
        super()._create_envs()
        self.roller_foot_body_ids = [3, 8]  


    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.rear_foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.rear_feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        self.rear_foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.rear_feet_indices,
                               0:3]
        
        self.roller_foot_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.roller_foot_body_ids,
                               0:3]
        self.roller_foot_orn = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.roller_foot_body_ids,
                               3:7]    
        self._post_physics_step_callback()
        
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() 
        
        # self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        # self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        # self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()


    def _reward_roller_orn(self):
        pass
