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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from moviepy.editor import ImageSequenceClip

ENV_NUM = 1

'''
cmd_type:
- 'default':
- 'f' : forward only
- 'r' : turn only
- 'fr'
'''
CMD_TYPE = None

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, ENV_NUM)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False

    env_cfg.terrain.curriculum = False
    # NOTE: [plane, rough, slope, stair, discrete, stepping]
    env_cfg.terrain.terrain_proportions = [0.4, 0.3, 0.1, 0.1, 0.1]
    env_cfg.terrain.selected = False
    terrain_type = 'stair'
    if env_cfg.terrain.selected :
        if terrain_type == 'rough':
            env_cfg.terrain.terrain_kwargs = {
                'type': 'rough',
                'min_height': -0.01,
                'max_height': 0.02,
                'step': 0.01,
                'downsampled_scale': 0.2
            }
        elif terrain_type == 'slope':
            env_cfg.terrain.terrain_kwargs = {
                'type': 'pyramid_sloped',
                'slope': 0.2,
            }    
        elif terrain_type == 'stair':
            env_cfg.terrain.terrain_kwargs = {
                'type': 'pyramid_stairs',
                'step_width': 0.35,
                'step_height': 0.08,
            }
        elif terrain_type == 'discrete':
            env_cfg.terrain.terrain_kwargs = {
                'type': 'discrete',
                'max_height': 0.05,
                'min_size': 1.0,
                'max_size': 5.0,
                'num_rects': 15
            }
        else:
            print(f'terrain type {terrain_type} is included')
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    start_state_log = 100 # ignore starting
    stop_state_log = 400 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_position = np.array([-1., -1.5, 0.8], dtype = np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    frames = []
    # for i,dof_name in enumerate(env.dof_names):
    #     print(i, " : ",dof_name)
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
                
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if TRACKING_ROBOT:
            moving_camera_position = camera_position.copy()
            init_position = env.base_init_state[:3].cpu().numpy()
            robot_idx = int(ENV_NUM/3)
            robot_position = env.root_states[robot_idx][:3].cpu().numpy()
            # moving_camera_position = camera_position + (robot_position - init_position)
            # direction = robot_position - init_position
            # camera_look_at = camera_position + direction
            # env.set_camera(moving_camera_position, camera_look_at)
            desired_camera_position = camera_position + (robot_position - init_position)
            moving_camera_position = moving_camera_position * 0.9 + desired_camera_position * 0.1
            camera_look_at = robot_position
            env.set_camera(desired_camera_position, camera_look_at)
        # print(env.commands[robot_index, 0].item())
        # print(env.commands[robot_index, 2].item())
        if i < stop_state_log and i > start_state_log:
            logger.log_states(
                {
                    # 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    # 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_pos_target': actions[robot_index,:].cpu().detach().numpy() * env.cfg.control.action_scale,
                    'dof_pos':env.dof_pos[robot_index,:].cpu().detach().numpy(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    # 'roller_angle': env.dof_pos[robot_index, env.dof_roller_ids].cpu().numpy(),  # add for debug by xiaoyu
                    'desired_contact': env.desired_contact_states[robot_index, :].cpu().numpy(),
                    'actual_contact': env.contact_detect[robot_idx,:].cpu().numpy(),
                    # 'tilt angle': env.dof_pos[robot_index, env.dof_roller_tilt_ids].cpu().numpy(),  # add for debug by xiaoyu
                }
            )

        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
    
    # if RECORD_FRAMES:
        # print(frames)
        # clip = ImageSequenceClip(frames, fps = 30)
        # print(frames)
        # clip.write_videofile("./tmp.mp4", codec = "libx264")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    TRACKING_ROBOT = True
    args = get_args()
    play(args)
