from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

ENV_NUM = 10

def test_sim_model(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, ENV_NUM)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    for i in range( 10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

def load_deploy_model(model_dir, device = "cuda:0"):
    from rsl_rl.modules import ActorCritic
    from rsl_rl.algorithms import PPO
    train_cfg_dict = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 'init_member_classes': {}, 'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'roller_skating', 'load_run': -1, 'max_iterations': 600, 'num_steps_per_env': 24, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 'runner_class_name': 'OnPolicyRunner', 'seed': 1}
    num_critic_obs = 42
    actor_critic = ActorCritic(42, num_critic_obs, 14, **train_cfg_dict['policy']).to(device)
    alg = PPO(actor_critic, device, train_cfg_dict["algorithm"])
    cfg = train_cfg_dict["runner"]
    alg.init_storage(1, cfg["num_steps_per_env"], [42], [0], [14])

    # load prvious model
    loaded_dict = torch.load(model_dir)
    alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    # load optimzier
    alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

    # get inference policy
    alg.actor_critic.eval()
    alg.actor_critic.to(device)
    policy = alg.actor_critic.act_inference
    return policy

def test_deploy_model(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    model_dir = "../../logs/roller_skating/Feb16_19-26-18_/model_500.pt"
    policy = load_deploy_model(model_dir)
    print(policy)
    for i in range(10 * int(env.max_episode_length)):
        action = policy(obs)
        print(action)
        obs, _, rews, dones, infos = env.step(action)

if __name__ == "__main__":
    args = get_args()
    test_deploy_model(args)
    # test_sim_model(args)