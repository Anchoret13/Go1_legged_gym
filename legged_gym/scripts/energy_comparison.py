from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

ENV_NUM = 1
CMD_TYPE = None

def record_cot(args, task_name):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, ENV_NUM)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    start_state_log = 0 # ignore starting
    stop_state_log = 500 # number of steps before plotting states
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    joint_names = [ "FL_hip_joint",
                    "FL_thigh_joint",
                    "FL_calf_joint",
                    "FR_hip_joint",
                    "FR_thigh_joint",
                    "FR_calf_joint",
                    "RL_hip_joint",
                    "RL_thigh_joint",
                    "RL_calf_joint",
                    "RR_hip_joint",
                    "RR_thigh_joint",
                    "RR_calf_joint"]
    joint_indices = []

    for i,dof_name in enumerate(env.dof_names):
        # print(i, " : ",dof_name)
        if dof_name in joint_names:
            joint_indices.append(i)

    
    for i in range(int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        cot = 0
        if i < stop_state_log and i > start_state_log:
            for joint in joint_indices:
                dof_vel = env.dof_vel[robot_index, joint].item()
                dof_torque = env.torques[robot_index, joint].item()
                cot = dof_vel * dof_torque
            logger.log_states(
                {
                    "cot": cot
                }
            )

    logger.save_log(task_name)

if __name__ == "__main__":
    args = get_args()
    record_cot(args, "pure")