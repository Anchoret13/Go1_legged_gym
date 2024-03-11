from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import pickle as pkl

ENV_NUM = 1

def collect_trajectory(args, traj_num):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, ENV_NUM)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    args.load_run = "/home/well/Desktop/Skating/legged_gym/logs/roller_skating_gait_cond_xyz/Mar10_14-52-43_"
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    dataset_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'dataset', 'wheeled_flat')
    os.makedirs(dataset_dir, exist_ok=True)
    for i in range(traj_num):
        single_trajectory = {}
        traj_obs = []
        traj_act = []
        for j in range(int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            traj_obs.append(obs[0].cpu().detach().numpy().tolist())
            traj_act.append(actions[0].cpu().detach().numpy().tolist())
        single_trajectory["obs"] = traj_obs
        single_trajectory["act"] = traj_act
        file_path = os.path.join(dataset_dir, f'traj_{i:04d}.pkl') 
        with open(file_path, "wb") as f:
            pkl.dump(single_trajectory, f)



if __name__ == "__main__":
    args = get_args()
    collect_trajectory(args, 10)