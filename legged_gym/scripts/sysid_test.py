from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from sys_id.dataset import PhysProps
from sys_id.model import GPT2

ENV_NUM = 1

CMD_TYPE = None

def update_history(current_obs, history):
    pass

def transformer_test(args, eval_params, model_params):
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
    
    device=env.device
    window_size = eval_params['window_size']

    oa_input_shape = 54

    # init history buffer
    history = torch.zeros(ENV_NUM, oa_input_shape* window_size)

    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device = device)
    model = GPT2(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(eval_params['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    
    logger = Logger(env.dt)
    robot_index = 0
    start_state_log = 0
    stop_state_log = 1000

    for i in range(1050):
        pass

if __name__ == "__main__":
    eval_params = {
        'checkpoint_path': './logs/2024-04-10_22-02-09/checkpoint_epoch_940.pth', 
        'dataset_folder_path': '../dataset/eval_model/wheeled_flat', 
        'window_size': 10,
        'batch_size': 1, 
    }

    model_params = {
        "n_layer": 2,
        "n_head": 3,
        "pdrop": 0.1,
        "max_seq_length": 1000,
        'position_encoding': 'sine',
        "output_size": 3,
        "input_size": (42 + 12) * eval_params['window_size'], 
        "hidden_size": (42 + 12) * eval_params['window_size'], 
    }
    