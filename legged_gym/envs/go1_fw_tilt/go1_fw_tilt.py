from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import *

import torch
from typing import Tuple, Dict
from ..base.wheeled_robot import WheeledRobot
from .go1_fw_tilt_config import Go1FwFlatTiltCfg
from ..go1_fw_clock.go1_fw import Go1FwClock

class Go1FwTilt(Go1FwClock):
    cfg : Go1FwFlatTiltCfg
    
    def compute_observations(self):
        """ Computes observations to exclude passive joint
        """
        dofs_to_keep = torch.ones(self.num_dof, dtype=torch.bool)
        dofs_to_keep[self.dof_roller_ids] = False
        dofs_to_keep[self.dof_roller_tilt_ids] = False

        # Select the columns
        self.active_dof_pos = self.dof_pos[:, dofs_to_keep]
        self.active_default_dof_pos = self.default_dof_pos[:, dofs_to_keep]
        active_dof_vel = self.dof_vel[:, dofs_to_keep]

        self.obs_buf = torch.cat((
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.active_dof_pos - self.active_default_dof_pos) * self.obs_scales.dof_pos,
            active_dof_vel * self.obs_scales.dof_vel,
            torch.clip(self.actions, -1, 1),
            self.base_lin_vel,
            self.base_ang_vel
        ), dim = -1)

        # TODO add friction for roller dof?
        # privileged observation
        roller_dofs = torch.tensor([False, False, False, True, True,
                                    False, False, False, True, True,
                                    False, False, False,
                                    False, False, False])
        self.roller_obs = self.dof_vel[:, roller_dofs]
        friction_coeff = self.friction_coeffs[:,0].to(self.device)
        # TO BE ADDED: TERRAIN INFO
        self.privileged_obs_buf = torch.cat((self.obs_buf,
                                             self.roller_obs,
                                             friction_coeff), dim = -1)
        # print(self.privileged_obs_buf)

    def _init_buffers(self):
        # # add for wheel robot 
        # self.num_rollers = 2
        # TODO the idx will change with tilt joint added
        super()._init_buffers()
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.wheel_indices = torch.tensor([5, 10], device = self.device)
        self.rear_feet_indices = torch.tensor([14, 18], device = self.device)
        
        # gait index from WTW
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        # desired_contact_state from WTW
         # TODO is 4 and 2 change?
        
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
        modified_actions = torch.zeros(n, self.num_dof)
        modified_actions[:, :3] = self.actions[:, :3]  
        modified_actions[:, 3] = 0
        modified_actions[:, 4] = 0 
        modified_actions[:, 5:8] = self.actions[:, 3:6]  
        modified_actions[:, 8] = 0
        modified_actions[:, 9] = 0  
        modified_actions[:, 10:] = self.actions[:, 6:] 
        modified_actions = modified_actions.to(self.device)


        # NOTE: Following code fixed the action of front joint, trying to solve split problem
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


        
    ## ADDITIONAL REWARD FUNCTION FOR WHEELED ROBOT
    def _reward_roller_action_rate(self):
        return torch.sum(torch.square(self.last_actions[:,:6] - self.actions[:,:6]), dim = 1)
    
    def _reward_roller_action_diff(self):
        return torch.sum(torch.square(self.actions[:,0:3] - self.actions[:, 3:6]), dim = 1)
    
    def _reward_hip(self):
        # penalize hip action
        hips_idxs = torch.tensor([0, 5, 10, 13], device=self.torques.device)
        self.hips_default_pos = torch.index_select(self.default_dof_pos, 1, hips_idxs)
        self.hips_pos = torch.index_select(self.dof_pos, 1, hips_idxs)
        diff = torch.sum(torch.square(self.hips_default_pos - self.hips_pos), dim = 1)
        return diff

    
    def _reward_front_hip(self):
        front_hips_idxs = torch.tensor([0, 5], device=self.torques.device)
        front_hips_default_pos = torch.index_select(self.default_dof_pos, 1, front_hips_idxs)
        front_hips_pos = torch.index_select(self.dof_pos, 1, front_hips_idxs)
        diff = torch.sum(torch.square(front_hips_default_pos - front_hips_pos), dim = 1)
        return diff

    def _reward_front_leg(self):
        front_leg_idxs = torch.tensor([0, 1, 2, 5, 6, 7], device = self.torques.device)
        self.front_default_pos = torch.index_select(self.default_dof_pos, 1, front_leg_idxs)
        self.front_pos = torch.index_select(self.dof_pos, 1, front_leg_idxs)
        diff = torch.sum(torch.square(self.front_default_pos - self.front_pos), dim = 1)
        return diff

    def _reward_front_hip(self):
        front_hips_idxs = torch.tensor([0, 5], device=self.torques.device)
        self.front_hips_default_pos = torch.index_select(self.default_dof_pos, 1, front_hips_idxs)
        self.front_hips_pos = torch.index_select(self.dof_pos, 1, front_hips_idxs)
        diff = self.front_hips_default_pos - self.front_hips_pos
        return torch.sum(torch.square(diff), dim = 1)
    
    # PERIODIC 1 
    
    # PERIODIC 2

    

    # NOTE: TRYING TO BE BRUTAL