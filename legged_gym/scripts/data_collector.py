from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import pickle as pkl

ENV_NUM = 500

def collect_trajectory(args, traj_num):
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
    args.load_run = "/home/well/Desktop/Skating/legged_gym/logs/roller_skating_asac/May02_00-04-12_"
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    dataset_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'dataset', 'short', 'wheeled_flat')
    os.makedirs(dataset_dir, exist_ok=True)
    # for i in range(traj_num):
    #     single_trajectory = {}
    #     traj_obs = []
    #     traj_act = []
    #     physical_props = []
    #     for j in range(int(env.max_episode_length)):
    #         actions = policy(obs.detach())
    #         obs, _, rews, dones, infos = env.step(actions.detach())
    #         priv_obs = env.get_privileged_observations()
    #         # phys_props = priv_obs[:, 42:]
    #         # NOTE: only use projected gravity and joint pos
    #         simp_obs = env.compute_adapt_input()
    #         targets = env.compute_adapt_target()
    #         # targets = torch.cat((phys_props, body_vel), dim = -1)
    #         traj_obs.append(simp_obs[i].cpu().detach().numpy().tolist())
    #         traj_act.append(actions[i].cpu().detach().numpy().tolist())
    #         physical_props.append(targets[i].cpu().detach().numpy().tolist())
    #         # print(simp_obs.shape)
    #         # print(targets.shape)
    #     single_trajectory["obs"] = traj_obs
    #     single_trajectory["act"] = traj_act
    #     single_trajectory['physprops'] = physical_props
    #     file_path = os.path.join(dataset_dir, f'traj_{i:04d}.pkl') 
    #     print(f'{i} trajectories collected')
    #     with open(file_path, "wb") as f:
    #         pkl.dump(single_trajectory, f)
    multi_trajectories = [{} for _ in range(traj_num)]
    multi_obs = [[] for _ in range(traj_num)]
    multi_physprops = [[] for _ in range(traj_num)]
    multi_actions = [[] for _ in range(traj_num)]
    for episode in range(int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        adapt_obs = env.compute_adapt_input()
        adapt_targets = env.compute_adapt_target()
        for env_idx in range(traj_num):            
            multi_obs[env_idx].append(adapt_obs[env_idx].cpu().detach().numpy().tolist())
            multi_physprops[env_idx].append(adapt_targets[env_idx].cpu().detach().numpy().tolist())
            multi_actions[env_idx].append(actions[env_idx].cpu().detach().numpy().tolist())

    for idx in range(len(multi_trajectories)):
        multi_trajectories[idx]["obs"] = multi_obs[idx]
        multi_trajectories[idx]["physprops"] = multi_physprops[idx]
        multi_trajectories[idx]["act"] = multi_actions[idx]
        file_path = os.path.join(dataset_dir, f'traj_{idx:04d}.pkl')
        print(f'{idx} trajectories collected')
        with open(file_path, "wb") as f:
            pkl.dump(multi_trajectories[idx], f)

if __name__ == "__main__":
    args = get_args()
    collect_trajectory(args, ENV_NUM)