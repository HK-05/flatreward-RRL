import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
import pickle
import math
import random
import copy
import mujoco_py
import yaml
import time
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from sam_ppo_continuous_efficient import PPO_continuous
from torch.distributions import Uniform
from pprint import pprint

class NoisyRewardEnv(gym.Wrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyRewardEnv, self).__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        noise = np.random.normal(0, self.noise_std)
        reward += noise
        return state, reward, done, info



def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset(state=None, x_pos=None)
        if args.use_state_norm:
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    return evaluate_reward / times

def save_agent(agent, save_path, state_norm, reward_scaling):
    agent.actor.save(f'{save_path}_actor')
    agent.critic.save(f'{save_path}_critic')
    with open(f'{save_path}_state_norm', 'wb') as file1:
        pickle.dump(state_norm, file1)
    with open(f'{save_path}_reward_scaling', 'wb') as file2:
        pickle.dump(reward_scaling, file2)
    print(f'model saved : {save_path}')

def main(args, number):

    total_steps = 0
    iteration = 0  
    max_value = -np.inf
    convergence_threshold = args.convergence_threshold 
    convergence_iteration = None

    seed, GAMMA = args.seed, args.GAMMA
    env = gym.make(args.env)
    env_evaluate = gym.make(args.env)
    env_reset = gym.make(args.env)
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    env_reset.seed(seed)
    env_reset.action_space.seed(seed)
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


    if args.reward_noise:
        noise_std_train = args.reward_noise_std  
        env = NoisyRewardEnv(env, noise_std=noise_std_train)
        print(f"Training with reward noise: std={noise_std_train}")

    evaluate_num = 0
    evaluate_rewards = []
    update_time = []
    total_steps = 0
    max_value = -np.inf
    if args.use_sam:
        save_path = f"./models/SAM_PPO_reward_{args.reward_noise_std}_{args.env}_rho_{args.rho}_{GAMMA}" if args.reward_noise else f"./models/SAM_PPO_{args.env}_rho_{args.rho}_{GAMMA}"
    else:
        save_path = f"./models/PPO_reward_{args.reward_noise_std}_{args.env}_{GAMMA}" if args.reward_noise else f"./models/PPO_{args.env}_{GAMMA}"

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    if args.use_sam:
        writer = SummaryWriter(log_dir=f'runs/SAM_PPO_continuous/reward_env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}' if args.reward_noise else f'runs/SAM_PPO_continuous/env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}')
    else:
        writer = SummaryWriter(log_dir=f'runs/PPO_continuous/reward_env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}' if args.reward_noise else f'runs/PPO_continuous/env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}')

    state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_norm:
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    start_time = time.time()  
    converge = False
    
    while total_steps < args.max_train_steps:
        iteration += 1  

        s = env.reset(state=None, x_pos=None)
        s_org, x_pos = copy.deepcopy(s), np.array([env.sim.data.qpos[0]])
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            s_, r, done, info = env.step(action)
            x_pos = np.array([info['x_position']])
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

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = copy.deepcopy(s_)
            s_org = copy.deepcopy(state_norm.denormal(s_, update=False))
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                update_time_start = time.time()
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0
                update_time_finish = time.time()
                update_time_per_1 = update_time_finish - update_time_start
                update_time.append(update_time_per_1)
                print(f'time per 1 update : {update_time_per_1}')

            if total_steps % args.evaluate_freq == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time 
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print(f"evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward} \t elapsed_time:{elapsed_time} \t episode_step:{episode_steps} \t total_steps:{total_steps}" )
                writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
                if evaluate_num % args.save_freq == 0:
                    if args.use_sam:
                        np.save('./data_train/SAM_PPO_continuous_{}_env_{}_number_{}_seed_{}_GAMMA_{}.npy'.format(args.policy_dist, args.env, number, seed, GAMMA), np.array(evaluate_rewards))                   
                    else:
                        np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}_GAMMA_{}.npy'.format(args.policy_dist, args.env, number, seed, GAMMA), np.array(evaluate_rewards))                         
                if evaluate_reward > max_value:
                    save_agent(agent, save_path, state_norm, reward_scaling)
                    max_value = evaluate_reward

                if evaluate_reward >= convergence_threshold and convergence_iteration is None:
                    convergence_iteration = iteration
                    convergence_steps = total_steps
                    print(f"Convergence reached at iteration {convergence_iteration}, total steps: {convergence_steps}")
                    converge = True
        if converge : break
    
    end_time = time.time()  
    total_training_time = end_time - start_time
    average_time_per_iteration = total_training_time / iteration

    print(f"Total training time: {total_training_time / 3600:.2f} hours")
    print(f"Iterations to convergence: {convergence_iteration}")
    print(f"Total iterations: {iteration}")
    print(f"Average time per iteration: {average_time_per_iteration:.2f} seconds")
    print(f"Average update time: {sum(update_time)/len(update_time)}")

if __name__ == '__main__':
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str, default="configs/Hopper.yaml", help="Path to the config file")
    initial_args, remaining_argv = initial_parser.parse_known_args()

    with open(initial_args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    
    parser = argparse.ArgumentParser(
        parents=[initial_parser],
        description="Evaluation for Action Robustness in SAM-PPO-continuous"
    )

    parser.add_argument("--use_adv_norm", action='store_true', help="Trick 1: advantage normalization")
    parser.add_argument("--use_state_norm", action='store_true', help="Trick 2: state normalization")
    parser.add_argument("--use_reward_norm", action='store_true', help="Trick 3: reward normalization")
    parser.add_argument("--use_reward_scaling", action='store_true', help="Trick 4: reward scaling")
    parser.add_argument("--use_lr_decay", action='store_true', help="Trick 6: learning rate decay")
    parser.add_argument("--use_grad_clip", action='store_true', help="Trick 7: gradient clip")
    parser.add_argument("--use_orthogonal_init", action='store_true', help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", action='store_true', help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", action='store_true', help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", action='store_true', help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--use_sam", action='store_true', help="Use SAM optimizer")
    parser.add_argument("--reward_noise", action='store_true', help="Train with reward noise")
 
    parser.add_argument("--env", type=str, help="HalfCheetah-v3/Hopper-v3/Walker2d-v3")
    parser.add_argument("--next_steps", type=int, help="Number of next states")
    parser.add_argument("--random_steps", type=int, help="Uniformly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, help="Number of neurons in hidden layers")
    parser.add_argument("--lr_a", type=float, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--lamda", type=float, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, help="PPO parameter")
    parser.add_argument("--entropy_coef", type=float, help="Trick 5: policy entropy")
    parser.add_argument("--weight_reg", type=float, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--GAMMA", type=str, help="File name")
    parser.add_argument("--rho", type=float, help="Rho value for SAM optimizer") 
    parser.add_argument("--convergence_threshold", type=float, default=8000,
                        help="Desired average reward for convergence")
    parser.add_argument("--reward_noise_std", type=float, default=0.1,
                        help="Standard deviation of reward noise during training")


    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)

    print("\n===== Evaluation Configuration =====")
    config_dict = vars(args)
    pprint(config_dict)
    print("====================================\n")


    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    main(args, number=1)
