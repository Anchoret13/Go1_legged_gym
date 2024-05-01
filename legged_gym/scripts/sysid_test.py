from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from sys_id.dataset import PhysProps
from sys_id.model import GPT2
from sys_id.RNN import GRU
from sys_id.MLP import MLP

ENV_NUM = 1            

CMD_TYPE = None

def update_history(history, new_obs):
    new_obs = new_obs.unsqueeze(1)
    updated_history = torch.cat((history[:, 1:, :], new_obs), dim=1)
    return updated_history

def udpate_mlp_history(history, new_obs):
    new_obs_flattened = new_obs.flatten()  # This should have a shape of [15]
    updated_history = torch.cat((history[:, 15:], new_obs_flattened.unsqueeze(0)), dim=1)
    
    return updated_history



# def transformer_test(args, eval_params, model_params):
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     # override some parameters for testing
#     env_cfg.env.num_envs = min(env_cfg.env.num_envs, ENV_NUM)
#     env_cfg.terrain.num_rows = 1
#     env_cfg.terrain.num_cols = 1
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = True
#     env_cfg.domain_rand.push_robots = False
    
#     # prepare environment
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
#     device=env.device
#     window_size = eval_params['window_size']

#     oa_input_shape = 54

#     # init history buffer
#     # history = torch.zeros(ENV_NUM, oa_input_shape* window_size)

#     obs = env.get_observations()
#     train_cfg.runner.resume = True
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#     policy = ppo_runner.get_inference_policy(device = device)
#     # model = GPT2(**model_params).to(device)
#     # model = GRU(**model_params).to(device)
#     # optimizer = torch.optim.Adam(model.parameters())
#     # checkpoint = torch.load(eval_params['checkpoint_path'], map_location=device)
#     # model.load_state_dict(checkpoint['state_dict'])

    
#     logger = Logger(env.dt)
#     robot_index = 0
#     start_state_log = 0
#     stop_state_log = 1000

#     for i in range(1050):
#         pass


def GRU_test(args, eval_params, model_params):
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

    input_shape = 15
    history = torch.zeros(ENV_NUM, window_size, input_shape).cuda()
    # tensor([[ 0.0463,  0.0487,  0.0087, -0.0476,  0.0367,  0.0201, -0.0883, -0.1690,
    #       0.5352,  0.0176, -0.2382,  0.5574]], device='cuda:0')
    history[:, :, 0] = 0.0   
    history[:, :, 1] = 0.0   
    history[:, :, 2] = -1    
    history[:, :, 3] = 0.05  
    history[:, :, 4] = 0.05  
    history[:, :, 5] = 0.01  
    history[:, :, 6] = -0.05 
    history[:, :, 7] = 0.05  
    history[:, :, 8] = 0.01  
    history[:, :, 9] = -0.08 
    history[:, :, 10] = -0.02
    history[:, :, 11] = 0.54 
    history[:, :, 12] = 0.02 
    history[:, :, 13] = -0.24
    history[:, :, 14] = 0.56 
    print(history)


    obs = env.get_observations()

    # load model
    model = GRU(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(eval_params['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    start_state_log = 0 # do not ignore starting
    stop_state_log = 500 # number of steps before plotting states
    for i in range(600):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        adapt_input = env.compute_adapt_input().cuda()
        history = update_history(history, adapt_input)
        prediction = model(history, None)
        target = env.compute_adapt_target().cuda()
        if i < stop_state_log and i > start_state_log:
            logger.log_states({
                "prediction": prediction,
                "target": target
            })

    logger.plot_pred_true()
    # logger.save_log("Comparison")

def MLP_test(args, eval_params, model_params):
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

    input_shape = 15
    history = torch.zeros(ENV_NUM, window_size * input_shape).cuda()

    obs = env.get_observations()

    # load model
    model = MLP(**model_params).to(device)
    checkpoint = torch.load(eval_params['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    logger = Logger(env.dt)
    start_state_log = 0 # do not ignore starting
    stop_state_log = 500 # number of steps before plotting states
    for i in range(5000):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        adapt_input = env.compute_adapt_input().cuda()
        history = udpate_mlp_history(history, adapt_input)
        prediction = model(history)
        target = env.compute_adapt_target().cuda()
        if i < stop_state_log and i > start_state_log:
            logger.log_states({
                "prediction": prediction,
                "target": target
            })
    logger.plot_pred_true()

if __name__ == "__main__":
    GRU_eval_params = {
        'checkpoint_path': '../../sys_id/logs/GRU/2024-04-30_15-49-58/checkpoint_epoch_120.pth', 
        'dataset_folder_path': '../dataset/eval/wheeled_flat', 
        'window_size': 50,
        'batch_size': 1, 
    }

    GRU_model_params = {
        "input_size": 15,
        "hidden_size": 150,
        "n_layer": 2,
        "output_size": 9
    }
    args = get_args()
    GRU_test(args, GRU_eval_params, GRU_model_params)

    MLP_eval_params = {
        'checkpoint_path': '../../sys_id/logs/MLP/2024-04-22_16-19-45/mlp_model_epoch_1000.pth', 
        'window_size': 50,
        'batch_size': 1, 
    }

    MLP_model_params = {
        "input_size": 750,
        "hidden_size": 150,
        "output_size": 9
    }
    # MLP_test(args, MLP_eval_params, MLP_model_params)