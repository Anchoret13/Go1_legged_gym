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

import torch
from typing import Tuple, Dict
from ..base.wheeled_robot import WheeledRobot
from .go1_fw_config import Go1FwFlatCfg

class Go1Fw(WheeledRobot):
    cfg : Go1FwFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # self.num_passive_joints = self.cfg.env.num_passive_joints

    def compute_observations(self):
        """ Computes observations to exclude passive joint
        """
        dofs_to_keep = torch.ones(self.num_dof, dtype=torch.bool)
        dofs_to_keep[self.dof_roller_ids] = False

        # Select the columns
        active_dof_pos = self.dof_pos[:, dofs_to_keep]
        active_default_dof_pos = self.default_dof_pos[:, dofs_to_keep]
        active_dof_vel = self.dof_vel[:, dofs_to_keep]
        # active_actions = self.actions[:, dofs_to_keep]


        
        # self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             self.commands[:, :3] * self.commands_scale,
        #                             (active_dof_pos - active_default_dof_pos) * self.obs_scales.dof_pos,
        #                             active_dof_vel * self.obs_scales.dof_vel,
        #                             active_actions
        #                             ),dim=-1)

        self.obs_buf = torch.cat((
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (active_dof_pos - active_default_dof_pos) * self.obs_scales.dof_pos,
            active_dof_vel * self.obs_scales.dof_vel,
            torch.clip(self.actions, -1, 1)
        ), dim = -1)
        # print("+"*50)
        # print(self.obs_buf)
        # print(self.obs_buf.shape)
        # # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec



    def _init_buffers(self):
        # # add for wheel robot *************************************************************************************
        # self.num_rollers = 2
        super()._init_buffers()
        self.base_pos = self.root_states[:self.num_envs, 0:3]

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]


    def _reward_masked_legs_energy(self):
        mask = torch.ones(self.torques.size(-1), device=self.torques.device, dtype=torch.bool)
        mask[self.dof_roller_ids] = False

        masked_torques = self.torques[:, mask]
        masked_dof_vel = self.dof_vel[:, mask]

        return torch.sum(torch.square(masked_torques * masked_dof_vel), dim=1)

    # def _compute_torques(self, actions):
    #     #pd controller
    #     actions_scaled = actions * self.cfg.control.action_scale
    #     control_type = self.cfg.control.control_type
    #     gaits_type = self.cfg.control.gaits_type
    #     if gaits_type == "fix_f":
    #         actions_scaled[:8] = 0.0

    #     if control_type=="P":    
            # torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
    #     elif control_type=="V":
    #         torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
    #     elif control_type=="T":
    #         torques = actions_scaled
    #     else:
    #         raise NameError(f"Unknown controller type: {control_type}")
        
    #     return torch.clip(torques, -self.torque_limits, self.torque_limits)
    

    # def _reward_legs_energy(self):
    #     return torch.sum(torch.square(self.torques * self.dof_vel), dim = 1)
    
    # def _reward_legs_energy_abs(self):
    #     return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        

    def step(self, actions):
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

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(modified_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    




# def _create_envs(self):
#         super()._create_envs()
#         self.all_feet_body_ids = [4, 9, 14,18] #  the id of body of feet , which used  to  get orn respect to ground
#         self.roller_body_ids = [4, 9] 
#         self.feet_only_body_ids = [14,18] 
#         '''
#         first step;
#             1. roller feet perpendular to grounds
#             2. penalize Z of feet
            
#         second step:
#             1. project [base , all_feet] onto SE(2), ground.
#             2. [four feet(points)] became a quadhredual, centre is point of base on 2D.
#             3. design gait based on the pattern of four lines.
#         '''
#         # self.envs # envs handle 
#         # self.actor_handles # robot handle
#         # for i in range(self.num_envs): 
#         body_states = self.gym.get_actor_rigid_body_states(
#         self.envs[0], self.actor_handles[0], gymapi.STATE_ALL)

#         body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
#             # Print some state slices
#         print("Poses from Body State:")
#         print(body_states['pose'])          # print just the poses

#         print("\nVelocities from Body State:")
#         print(body_states['vel'])          # print just the velocities
#         print()

#         # iterate through bodies and print name and position
#         body_positions = body_states['pose']['p']
#         for i in range(len(body_names)):
#             print("Body '%s' has position" % body_names[i], body_positions[i])
#             # print(body_states['pose']) 
#         #     self.feet_pose[i] = body_states[self.feet_names]
#         #     print(self.feet_pose[i])
#         # print(self.feet_pose)
#         body_positions[0] = body_positions[1]
#         print('  ')
#         print('  ')
#         print("Body '%s' has position" % body_names[0], body_positions[0])
#         print('  ')
#         print('  ')
#         # iterate through bodies and print name and orn
#         body_orn = body_states['pose']['r']
#         for i in range(len(body_names)):
#             print("Body '%s' has orn" % body_names[i], body_orn[i])
#             # print(body_states['pose']) 
#         #     self.feet_pose[i] = body_states[self.feet_names]
#         #     print(self.feet_pose[i])
#         # print(self.feet_pose)
            
