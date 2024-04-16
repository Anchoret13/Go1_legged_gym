# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import *

import torch
from typing import Tuple, Dict
from ..base.wheeled_robot import WheeledRobot
from .go1_fw_config import Go1FwFlatClockCfg

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

class Go1FwClock(WheeledRobot):
    cfg : Go1FwFlatClockCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # self.num_passive_joints = self.cfg.env.num_passive_joints
        self.frequencies = 3.0
        self.current_step = 0

    def compute_transformer_input(self):
        dofs_to_keep = torch.ones(self.num_dof, dtype=torch.bool)
        dofs_to_keep[self.dof_roller_ids] = False

        self.active_dof_pos = self.dof_pos[:, dofs_to_keep]
        self.active_default_dof_pos = self.default_dof_pos[:, dofs_to_keep]

        body_lin_vel = self.base_lin_vel
        body_ang_vel = self.base_ang_vel
        
        trans_input = torch.cat((
            self.projected_gravity, #3
            (self.active_dof_pos - self.active_default_dof_pos) * self.obs_scales.dof_pos, #12
            body_lin_vel, #2
            body_ang_vel, #3
        ), dim = -1)
        return trans_input

    def compute_observations(self):
        """ Computes observations to exclude passive joint
        """
        dofs_to_keep = torch.ones(self.num_dof, dtype=torch.bool)
        dofs_to_keep[self.dof_roller_ids] = False

        # Select the columns
        self.active_dof_pos = self.dof_pos[:, dofs_to_keep]
        self.active_default_dof_pos = self.default_dof_pos[:, dofs_to_keep]
        active_dof_vel = self.dof_vel[:, dofs_to_keep]

        # self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype = torch.float, device = self.device, requires_grad = False)
        # self.desired_rear_contact_states = torch.zeros(self.num_envs, 2, dtype = torch.float, device = self.device, requires_grad = False)

        self.obs_buf = torch.cat((
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.active_dof_pos - self.active_default_dof_pos) * self.obs_scales.dof_pos,
            active_dof_vel * self.obs_scales.dof_vel,
            torch.clip(self.actions, -1, 1)
        ), dim = -1)

        # privileged observation
        roller_dofs = torch.tensor([False, False, False, True,  False, False, False, True, False, False,
        False, False, False, False])
        self.roller_obs = self.dof_pos[:, roller_dofs]
        friction_coeff = self.friction_coeffs[:,0].to(self.device)
        # TO BE ADDED: TERRAIN INFO
        self.privileged_obs_buf = torch.cat((self.obs_buf,
                                             self.roller_obs,
                                             friction_coeff), dim = -1)
        # print(self.privileged_obs_buf)

    def _init_buffers(self):
        # # add for wheel robot 
        # self.num_rollers = 2
        super()._init_buffers()
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.wheel_indices = torch.tensor([5, 10], device = self.device)
        self.rear_feet_indices = torch.tensor([14, 18], device = self.device)
        
        # gait index from WTW
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        # desired_contact_state from WTW
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        
        self.desired_rear_contact_states = torch.zeros(self.num_envs, 2, dtype = torch.float, device = self.device, 
                                                       requires_grad = False)


        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.rear_foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.rear_feet_indices,
                               7:10]
        
        self.left_idx = torch.tensor([14], device = self.device)
        self.right_idx = torch.tensor([18], device = self.device)
        self.left_rear_contact = torch.zeros(self.num_envs, len(self.left_idx), dtype=torch.bool, device=self.device, requires_grad=False)
        self.right_rear_contact = torch.zeros(self.num_envs, len(self.right_idx), dtype=torch.bool, device=self.device, requires_grad=False)
        
        self.last_wheel_contacts = torch.zeros(self.num_envs, len(self.wheel_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_rear_feet_contacts = torch.zeros(self.num_envs, len(self.rear_feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # tmp_foot_forces = torch.exp(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) **2 / 100.)
        # ***************   modified
        # self.contact_detect = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_detect = self.contact_forces[:, self.feet_indices, 2] > 0.1
        self.contact_detect = self.contact_detect.float()

        self.wheel_air_time = torch.zeros(self.num_envs, self.wheel_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.rear_feet_air_time = torch.zeros(self.num_envs, self.rear_feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_masked_legs_energy(self):
        mask = torch.ones(self.torques.size(-1), device=self.torques.device, dtype=torch.bool)
        mask[self.dof_roller_ids] = False

        masked_torques = self.torques[:, mask]
        masked_dof_vel = self.dof_vel[:, mask]

        return torch.sum(torch.square(masked_torques * masked_dof_vel), dim=1)

    def step(self, actions):
        self.current_step += 1
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # self.actions
        n = self.actions.size(0)
        modified_actions = torch.zeros(n, 14)
        modified_actions[:, :3] = self.actions[:, :3]  
        modified_actions[:, 3] = 0  
        modified_actions[:, 4:7] = self.actions[:, 3:6]  
        modified_actions[:, 7] = 0  
        modified_actions[:, 8:] = self.actions[:, 6:] 
        modified_actions = modified_actions.to(self.device)


        # Note: Following code fixed the action of front joint, trying to solve split problem
        # modified_actions[:, 0] = 0
        # modified_actions[:, 4] = 0

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(modified_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # print("-"*50)
        # print(self.desired_contact_states)
        # print("-"*50)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]
        
        self.rear_foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.rear_feet_indices,
                               0:3]
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def _create_envs(self):
        super()._create_envs()
        self.all_feet_body_ids = [4, 9, 14,18] #  the id of body of feet , which used  to  get orn respect to ground
        self.roller_body_ids = [4, 9] 
        self.feet_only_body_ids = [14,18] 
        '''
        first step;
            1. roller feet perpendular to grounds
            2. penalize Z of feet
            3. four dis(foot, calf joint) diff < a number; and 
            4. initial values: e.g back legs deflaut value, front leg defalut(foot perpd to ground)
            5. hip joint limit
            
        second step:
            1. project [base , all_feet] onto SE(2), ground.
            2. [four feet(points)] became a quadhredual, centre is point of base on 2D.
            3. design gait based on the pattern of four lines.
        '''
        # self.envs # envs handle 
        # self.actor_handles # robot handle
        self.bodies_orns = []
        for i in range(self.num_envs): 
            body_states = self.gym.get_actor_rigid_body_states(
                                self.envs[i], self.actor_handles[i], gymapi.STATE_ALL)
            body_orns = body_states["pose"]["r"]
            self.bodies_orns.append(body_orns)


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

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.gait_indices[env_ids] = 0

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # adaptive_sample_vel_cmd(min_vel, max_vel, current_step, env_ids, device, total_iterations = 10000, n_samples = 1000, steps_per_iteration = 24):
        # self.commands[env_ids, 0]  = adaptive_sample_vel_cmd(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], self.current_step, env_ids, self.device)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # self.frequencies = frequency_ac_vel(self.commands[:, 0])
        self.frequencies = 3.0 # NOTE: to use adaptive frequency, use above
        self._step_contact_targets()

    def sigmoid_contact_signal(self, x, kappa):
        return 1 / (1 + torch.exp(-kappa * (x - 0.5)))

    def _step_contact_targets(self, smoothing_option = "normal_cdf"):
        # NOTE:THIS SHITY CODE IS FUKING HARD CODING EVERYTHING
        frequencies = self.frequencies
        phase = 0.5 # NOTE: 0.5 For Trotting
        offsets = 0
        bounds = 0
        durations = 0.5 # legacy variable, keep it for memorial
        if smoothing_option == "sigmoid":
            kappa = 40. 
        elif smoothing_option == "normal_cdf":
            kappa = 0.05

            # von mises distribution for gait
            smoothing_cdf_start = torch.distributions.normal.Normal(0,kappa).cdf

        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
        
        # foot_indices = [self.gait_indices + phase + offsets + bounds,
        #                 self.gait_indices + offsets,
        #                 self.gait_indices + bounds,
        #                 self.gait_indices + phase]

        # FOR TROTTING:
        foot_indices = [self.gait_indices,
                    torch.remainder(self.gait_indices + phase, 1.0),
                    torch.remainder(self.gait_indices + phase, 1.0),
                    self.gait_indices]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        self.rear_foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(2, 4)], dim=1), 1.0)

        smoothing_multiplier_FL = 1.
        smoothing_multiplier_FR = 1.
        # smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
        #             1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
        #                                smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
        #                                        1 - smoothing_cdf_start(
        #                                    torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        # smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
        #             1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
        #                                smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
        #                                        1 - smoothing_cdf_start(
        #                                    torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))


        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        self.desired_rear_contact_states[:, 0] = smoothing_multiplier_RL
        self.desired_rear_contact_states[:, 1] = smoothing_multiplier_RR

        self.contact_detect = self.contact_forces[:, self.feet_indices, 2] > 1
        self.contact_detect = self.contact_detect.float()
        
    ## ADDITIONAL REWARD FUNCTION FOR WHEELED ROBOT
    def _reward_legs_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim = 1)
    
    def _reward_legs_energy_abs(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_lin_vel_x(self):
        return self.root_states[:, 7]

    def _reward_lin_vel_y_abs(self):
        return torch.abs(self.root_states[:, 8])
    
    def _reward_lin_vel_y_square(self):
        return torch.square(self.root_states[:, 8])
    
    def _reward_exceed_torque_limits_i(self):
        max_torques = torch.abs(self.torque_limits) 
        exceed_torque_each_dof = max_torques > self.torque_limits
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)

    def _reward_alive(self):
        return 1.
    
    def _reward_tracking_lin_vel_x(self):
        # Reward for Tracking of linear velocity commands on x-axis
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_roller_orn(self):
        # TODO: to be updated
        roller_orn = self.bodies_orns[:self.roller_body_ids]

    def _reward_roller_action_rate(self):
        return torch.sum(torch.square(self.last_actions[:,:6] - self.actions[:,:6]), dim = 1)
    
    def _reward_roller_action_diff(self):
        return torch.sum(torch.square(self.actions[:,0:3] - self.actions[:, 3:6]), dim = 1)
    
    def _reward_hip(self):
        # penalize hip action
        hips_idxs = torch.tensor([0, 4, 8, 11], device=self.torques.device)
        self.hips_default_pos = torch.index_select(self.default_dof_pos, 1, hips_idxs)
        self.hips_pos = torch.index_select(self.dof_pos, 1, hips_idxs)
        diff = torch.sum(torch.square(self.hips_default_pos - self.hips_pos), dim = 1)
        return diff

    
    def _reward_front_hip(self):
        front_hips_idxs = torch.tensor([0, 4], device=self.torques.device)
        front_hips_default_pos = torch.index_select(self.default_dof_pos, 1, front_hips_idxs)
        front_hips_pos = torch.index_select(self.dof_pos, 1, front_hips_idxs)
        diff = torch.sum(torch.square(front_hips_default_pos - front_hips_pos), dim = 1)
        return diff

    def _reward_front_leg(self):
        front_leg_idxs = torch.tensor([0, 1, 2, 4, 5, 6], device = self.torques.device)
        self.front_default_pos = torch.index_select(self.default_dof_pos, 1, front_leg_idxs)
        self.front_pos = torch.index_select(self.dof_pos, 1, front_leg_idxs)
        diff = torch.sum(torch.square(self.front_default_pos - self.front_pos), dim = 1)
        return diff

    def _reward_front_hip(self):
        front_hips_idxs = torch.tensor([0, 4], device=self.torques.device)
        self.front_hips_default_pos = torch.index_select(self.default_dof_pos, 1, front_hips_idxs)
        self.front_hips_pos = torch.index_select(self.dof_pos, 1, front_hips_idxs)
        diff = self.front_hips_default_pos - self.front_hips_pos
        return torch.sum(torch.square(diff), dim = 1)
    
    # def _reward_front_hip(self):
    #     front_hips_idxs = torch.tensor([0, 4], device=self.torques.device)
    #     self.front_hips_default_pos = torch.index_select(self.default_dof_pos, 1, front_hips_idxs)
    #     self.front_hips_pos = torch.index_select(self.dof_pos, 1, front_hips_idxs)
    #     diff = self.front_hips_default_pos - self.front_hips_pos
    #     if torch.any(torch.abs(diff) > 0.3):
    #         return 100000000
    #     else:
    #         return torch.sum(torch.square(diff), dim = 1)
    
    def _reward_penalize_roll(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :1]), dim=1)
    
    def _reward_tracking_contacts_binary(self):
        desired_contact = self.desired_contact_states
        contact_detect = self.contact_detect
        contact_alignment_loss = torch.sum((contact_detect * (1-desired_contact))**2)
        non_contact_alignment_loss = torch.sum(((1 - contact_detect) * desired_contact)**2)
        # overlap_loss = contact_alignment_loss + non_contact_alignment_loss
        total_loss = contact_alignment_loss + non_contact_alignment_loss
        total_loss = torch.abs(total_loss)
        normalized_loss = total_loss / contact_detect.numel()

        return normalized_loss
    
    # PERIODIC 1 
    
    def _reward_tracking_rear_swing_force(self):
        desired_contact = self.desired_rear_contact_states
        foot_forces = torch.norm(self.contact_forces[:, self.rear_feet_indices, :], dim = 1)
        reward = 0
        for i in range(2):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        return reward / 2
    
    def _reward_tracking_swing_force(self):
        desired_contact = self.desired_contact_states
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        return reward / 4

    # PERIODIC 2

    def _reward_tracking_rear_stance_vel(self):
        desired_contact = self.desired_rear_contact_states
        foot_velocities = torch.norm(self.rear_foot_velocities, dim = 2).view(self.num_envs, -1)
        reward = 0
        for i in range(2):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / 10.)))
        return reward / 4
    
    def _reward_tracking_stance_vel(self):
        desired_contact = self.desired_contact_states
        foot_velocities = torch.norm(self.foot_velocities, dim = 2).view(self.num_envs, -1)
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / 10.)))
        return reward / 4

    def _reward_raibert_heuristic(self):
        # pass 
        cur_footsteps_translated = self.rear_foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 2, 3, device = self.device)
        for i in range(2):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])
        
        # nomial positions: FR, FL, RR, RL
        desired_stance_width = 0.3
        desired_stance_length = 0.45

        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)
        desired_xs_nom = torch.tensor([-desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phase = torch.abs(1.0 - (self.rear_foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.frequencies
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        # desired_ys_offset = phase * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset = phase * y_vel_des * (0.5 / frequencies) # NOTE: for fixed frequency
        desired_ys_offset[:, 2:4] *= -1
        # desired_xs_offset = phase * x_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_xs_offset = phase * x_vel_des * (0.5 / frequencies)

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        return reward
    

    # NOTE: TRYING TO BE BRUTAL
    def _reward_wheel_air_time(self):
        contact = self.contact_forces[:, self.wheel_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_wheel_contacts)
        self.last_wheel_contacts = contact
        first_contact = (self.wheel_air_time > 0.) * contact_filt
        self.wheel_air_time += self.dt
        rew_airTime = torch.sum((self.wheel_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.wheel_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_rear_feet_air_time(self):
        contact =  self.contact_forces[:, self.rear_feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_rear_feet_contacts)
        self.last_rear_feet_contacts = contact
        first_contact = (self.rear_feet_air_time > 0.) * contact_filt
        self.rear_feet_air_time += self.dt
        rew_airTime = torch.sum((self.rear_feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.rear_feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_penalize_slow_x_vel(self):
        # penalize not moving forward
        if torch.any(torch.abs(self.base_lin_vel[:, 0]) < 0.5):
            return -1.
        else:
            return 0.
        
    def _reward_feet_clearance(self):
        # NOTE: based on phase, not working as expected?
        phases = 1 - torch.abs(1.0 - torch.clip((self.rear_foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.rear_foot_positions[:, :, 2]).view(self.num_envs, -1)
        target_height = 0.03 * phases + 0.03 # currently target height is 0.02
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_rear_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    # NOTE: simulate front hip joint noise
    def _apply_curriculum_noise(self):
        pass