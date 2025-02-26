# train_rarl.py

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
import pickle
import random
import copy
import yaml
import time
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from sam_ppo_continuous import PPO_continuous  # Protagonist PPO
from sam_ppo_continuous import PPO_continuous as PPO_Adversary  # Adversary PPO
from pprint import pprint

def evaluate_policy(args, env, protagonist, adversary, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done and episode_steps < args.max_episode_steps:
            episode_steps += 1
            with torch.no_grad():
                a_pro = protagonist.evaluate(s)
                a_adv = adversary.evaluate(s)
            action = a_pro + args.adv_fraction * a_adv  
            if args.policy_dist == "Beta":
                action = 2 * (action - 0.5) * args.max_action
            action = np.clip(action, -args.max_action, args.max_action)
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    return evaluate_reward / times

def save_agents(protagonist, adversary, save_path_pro, save_path_adv, state_norm, reward_scaling):
    protagonist.actor.save(f'{save_path_pro}_actor')
    protagonist.critic.save(f'{save_path_pro}_critic')
    adversary.actor.save(f'{save_path_adv}_actor')
    adversary.critic.save(f'{save_path_adv}_critic')
    with open(f'{save_path_pro}_state_norm', 'wb') as file1:
        pickle.dump(state_norm, file1)
    if reward_scaling is not None:
        with open(f'{save_path_pro}_reward_scaling', 'wb') as file2:
            pickle.dump(reward_scaling, file2)

def main(args, number):

    total_steps = 0  
    iteration = 0  
    max_value = -np.inf
    convergence_threshold = args.convergence_threshold  
    convergence_iteration = None  

    seed, GAMMA = args.seed, args.GAMMA
    env = gym.make(args.env)
    env_evaluate = gym.make(args.env)  

    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed + 100)
    env_evaluate.action_space.seed(seed + 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  
    evaluate_rewards = [] 
    update_time_p = []
    update_time_a = []
    total_steps = 0  
    max_value = -np.inf

    save_path_pro = f"./models/RARL_Protagonist_{args.env}_{GAMMA}"
    save_path_adv = f"./models/RARL_Adversary_{args.env}_{GAMMA}"

    replay_buffer_pro = ReplayBuffer(args)
    replay_buffer_adv = ReplayBuffer(args)

    protagonist = PPO_continuous(args)
    adversary_args = copy.deepcopy(args)
    adversary_args.lr_a = args.lr_a_adv 
    adversary = PPO_Adversary(adversary_args)

    if args.load_pro:
        protagonist.actor.load(f'{args.load_pro}_actor')
        protagonist.critic.load(f'{args.load_pro}_critic')
    if args.load_adv:
        adversary.actor.load(f'{args.load_adv}_actor')
        adversary.critic.load(f'{args.load_adv}_critic')

    writer = SummaryWriter(log_dir='runs/RARL/env_{}_{}_number_{}_seed_{}_GAMMA_{}'.format(args.env, args.policy_dist, number, seed, GAMMA))

    state_norm = Normalization(shape=args.state_dim) 
    if args.use_reward_norm:  
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling: 
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    else:
        reward_scaling = None

    start_time = time.time() 

    converge = False

    while total_steps < args.max_train_steps:
        iteration += 1  

        for _ in range(args.n_pro_itr):
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done and episode_steps < args.max_episode_steps:
                episode_steps += 1
                total_steps += 1
                a_pro, a_logprob_pro = protagonist.choose_action(s)
                with torch.no_grad():
                    a_adv = adversary.evaluate(s)

                action = a_pro + args.adv_fraction * a_adv
                if args.policy_dist == "Beta":
                    action = 2 * (action - 0.5) * args.max_action
                action = np.clip(action, -args.max_action, args.max_action)

                s_, r, done, info = env.step(action)
                if args.use_state_norm:
                    s_ = state_norm(s_)
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

                if done and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False

                replay_buffer_pro.store(s, a_pro, a_logprob_pro, r, s_, dw, done)
                s = s_

                if replay_buffer_pro.count == args.batch_size:
                    update_time_start = time.time()
                    protagonist.update(replay_buffer_pro, total_steps)
                    replay_buffer_pro.count = 0
                    update_time_finish = time.time()
                    update_time_1 = update_time_finish - update_time_start
                    update_time_p.append(update_time_1)
                    print(f'time per 1p update : {update_time_1}')

                if total_steps % args.evaluate_freq == 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time 
                    evaluate_num += 1
                    evaluate_reward = evaluate_policy(args, env_evaluate, protagonist, adversary, state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    print(f"evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward} \t elapsed_time:{elapsed_time} \t episode_step:{episode_steps} \t total_steps:{total_steps}" )
                    writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
                    if evaluate_num % args.save_freq == 0:
                        np.save('./data_train/RARL_PPO_{}_env_{}_number_{}_seed_{}_GAMMA_{}.npy'.format(args.policy_dist, args.env, number, seed, GAMMA), np.array(evaluate_rewards))
                    if evaluate_reward > max_value:
                        save_agents(protagonist, adversary, save_path_pro, save_path_adv, state_norm, reward_scaling)
                        max_value = evaluate_reward

                    if evaluate_reward >= convergence_threshold and convergence_iteration is None:
                        convergence_iteration = iteration
                        convergence_steps = total_steps
                        print(f"Convergence reached at iteration {convergence_iteration}, total steps: {convergence_steps}")
                        converge = True
        if converge : break    

        for _ in range(args.n_adv_itr):
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s)
            episode_steps = 0
            done = False
            while not done and episode_steps < args.max_episode_steps:
                episode_steps += 1
                total_steps += 1
                with torch.no_grad():
                    a_pro = protagonist.evaluate(s)
                a_adv, a_logprob_adv = adversary.choose_action(s)

                action = a_pro + args.adv_fraction * a_adv
                if args.policy_dist == "Beta":
                    action = 2 * (action - 0.5) * args.max_action
                action = np.clip(action, -args.max_action, args.max_action)

                s_, r, done, info = env.step(action)
                if args.use_state_norm:
                    s_ = state_norm(s_)

                adv_r = -r

                if done and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False

                replay_buffer_adv.store(s, a_adv, a_logprob_adv, adv_r, s_, dw, done)
                s = s_

                if replay_buffer_adv.count == args.batch_size:
                    update_time_start = time.time()
                    adversary.update(replay_buffer_adv, total_steps)
                    replay_buffer_adv.count = 0
                    update_time_finish = time.time()
                    update_time_1 = update_time_finish - update_time_start
                    update_time_a.append(update_time_1)
                    print(f'time per 1a update : {update_time_1}')

    end_time = time.time() 
    total_training_time = end_time - start_time 
    average_time_per_iteration = total_training_time / iteration  
    update_time = np.array(update_time_a)[:len(update_time_p)] + np.array(update_time_p)
    
    print(f"Total training time: {total_training_time / 3600:.2f} hours")
    print(f"Iterations to convergence: {convergence_iteration}")
    print(f"Total iterations: {iteration}")
    print(f"Average time per iteration: {average_time_per_iteration:.2f} seconds")
    print(f"Average update time: {sum(update_time) / len(update_time)}")

    writer.close()

if __name__ == '__main__':
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str, default="configs/rarl_ppo_hopper.yaml", help="Path to the config file")
    initial_args, remaining_argv = initial_parser.parse_known_args()

    with open(initial_args.config, 'r') as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(
        parents=[initial_parser],
        description="Hyperparameters Setting for RARL-PPO"
    )

    parser.add_argument("--use_adv_norm", action='store_true', help="Trick 1: advantage normalization")
    parser.add_argument("--use_state_norm", action='store_true', help="Trick 2: state normalization")
    parser.add_argument("--use_reward_norm", action='store_true', help="Trick 3: reward normalization")
    parser.add_argument("--use_reward_scaling", action='store_true', help="Trick 4: reward scaling")
    parser.add_argument("--use_lr_decay", action='store_true', help="Trick 6: learning rate decay")
    parser.add_argument("--use_grad_clip", action='store_true', help="Trick 7: gradient clip")
    parser.add_argument("--use_orthogonal_init", action='store_true', help="Trick 8: orthogonal initialization")
    parser.add_argument("--use_tanh", action='store_true', help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", action='store_true', help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--use_sam", action='store_true', help="Use SAM optimizer")

    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument("--max_train_steps", type=int, default=int(6e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency (in terms of evaluation number)")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=4000, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="Number of neurons in hidden layers")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of protagonist actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of protagonist critic")
    parser.add_argument("--lr_a_adv", type=float, default=3e-4, help="Learning rate of adversary actor")
    parser.add_argument("--lr_c_adv", type=float, default=3e-4, help="Learning rate of adversary critic")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.97, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=2, help="Seed")
    parser.add_argument("--GAMMA", type=str, default='0', help="File name")
    parser.add_argument("--rho", type=float, default=0.0, help="Rho value for SAM optimizer")
    parser.add_argument("--load_pro", type=str, default=None, help="Path to load Protagonist model")
    parser.add_argument("--load_adv", type=str, default=None, help="Path to load Adversary model")
    parser.add_argument("--adv_fraction", type=float, default=0.25, help="Fraction of adversarial action")
    parser.add_argument("--n_pro_itr", type=int, default=1, help="Number of iterations for the protagonist")
    parser.add_argument("--n_adv_itr", type=int, default=1, help="Number of iterations for the adversary")
    parser.add_argument("--convergence_threshold", type=float, default=8000,
                        help="Desired average reward for convergence")

    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)

    args.config = initial_args.config

    print("\n===== Final Training Configuration =====")
    config_dict = vars(args)
    pprint(config_dict)
    print("========================================\n")

    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    main(args, number=1)
