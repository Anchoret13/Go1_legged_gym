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

ENV_NUM = 1            

CMD_TYPE = None

def update_history(history, new_obs):
    # print("+"*50)
    # print(history.shape)
    # print(new_obs.shape)
    # print("+"*50)
    new_obs = new_obs.unsqueeze(1)
    print("++++++++++++++++OBS SHAPE++++++++++++++++")
    print(new_obs.shape)
    updated_history = torch.cat((history[:, 1:, :], new_obs), dim=1)
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
    print(history.shape)

    obs = env.get_observations()

    # load model
    model = GRU(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(GRU_eval_params['checkpoint_path'], map_location=device)
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
    stop_state_log = 1000 # number of steps before plotting states
    for i in range(1100):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        adapt_input = env.compute_adapt_input().cuda()
        history = update_history(history, adapt_input)
        print(history.shape)
        prediction = model(history, None)
        target = env.compute_adapt_target().cuda()
        print(prediction)
        print(target)
        if i < stop_state_log and i > start_state_log:
            logger.log_states({
                "prediction": prediction,
                "target": target
            })

    logger.plot_pred_true()
    # logger.save_log("Comparison")


if __name__ == "__main__":
    GRU_eval_params = {
        'checkpoint_path': '../../sys_id/logs/GRU/2024-04-20_18-14-17/checkpoint_epoch_440.pth', 
        'dataset_folder_path': '../dataset/eval/wheeled_flat', 
        'window_size': 50,
        'batch_size': 1, 
    }

    model_params = {
        "input_size": 15,
        "hidden_size": 150,
        "n_layer": 2,
        "output_size": 9
    }
    args = get_args()
    GRU_test(args, GRU_eval_params, model_params)