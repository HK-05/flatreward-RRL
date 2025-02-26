# eval_reward_sam_ppo.py

import os
import numpy as np
import gym
import argparse
import pickle
import yaml
from normalization import Normalization
from sam_ppo_continuous import PPO_continuous
from pprint import pprint

class NoisyRewardEnv(gym.Wrapper):
    def __init__(self, env, noise_std=0.0):
        super(NoisyRewardEnv, self).__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        noise = np.random.normal(0, self.noise_std)
        reward += noise
        return state, reward, done, info

def evaluate_policy(args, env, agent, state_norm, num_evals=5):
    evaluate_rewards = []
    for eval_idx in range(num_evals):
        s = env.reset()
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
        evaluate_rewards.append(episode_reward)
        # print(f'Episode {eval_idx + 1}: Reward = {episode_reward:.3f}')
    return np.mean(evaluate_rewards), np.std(evaluate_rewards)

def load_agent(agent, load_path, device):
    agent.actor.load(f'{load_path}_actor', device=device)
    agent.critic.load(f'{load_path}_critic', device=device)
    with open(f'{load_path}_state_norm', 'rb') as file1:
        state_norm = pickle.load(file1)
    return agent, state_norm

def main(args):
    seed, GAMMA, rho = args.seed, args.GAMMA, args.rho

    # Determine the load path based on whether reward noise was used during training
    if args.use_sam:
        load_path = f"./models/SAM_PPO_reward_{args.reward_noise_std}_{args.env}_rho_{args.rho}_{GAMMA}"
    else:
        load_path = f"./models/PPO_reward_{args.reward_noise_std}_{args.env}_{GAMMA}"

    print(f'load model from : {load_path}')
    # Create evaluation environment
    env = gym.make(args.env)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)



    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps
    print("Environment Information:")
    print(f"  env = {args.env}")
    print(f"  state_dim = {args.state_dim}")
    print(f"  action_dim = {args.action_dim}")
    print(f"  max_action = {args.max_action}")
    print(f"  max_episode_steps = {args.max_episode_steps}")

    # Optionally add reward noise during evaluation (set to 0.0 by default)
    if args.eval_reward_noise_std >= 0.0:
        env = NoisyRewardEnv(env, noise_std=args.eval_reward_noise_std)
        print(f"Evaluating with reward noise: std={args.eval_reward_noise_std}")
    else:
        print("Evaluating without reward noise.")


    agent = PPO_continuous(args)
    agent, state_norm = load_agent(agent, load_path, args.device)
    agent.gamma = args.gamma

    avg_reward, std_reward = evaluate_policy(args, env, agent, state_norm, num_evals=args.eval_episodes)
    print("---------------------------------------")
    print(f'Average Reward over {args.eval_episodes} episodes: {avg_reward:.3f} ± {std_reward:.3f}')
    print("---------------------------------------")

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
   # Boolean arguments
    parser.add_argument("--use_state_norm", action='store_true', help="Use state normalization")
    parser.add_argument("--use_sam", action='store_true', help="Use SAM optimizer")
    parser.add_argument("--reward_noise", action='store_true', help="Model was trained with reward noise")

    # Other arguments
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--policy_dist", type=str, help="Policy distribution: Beta or Gaussian")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--GAMMA", type=str, help="Gamma value for naming")
    parser.add_argument("--rho", type=float, help="Rho value for SAM optimizer")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run on")
    parser.add_argument("--reward_noise_std", type=float, default=0.1, help="Reward noise std during training")
    parser.add_argument("--eval_reward_noise_std", type=float, default=0, help="Reward noise std during evaluation")

    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)

    print("\n===== Evaluation Configuration =====")
    config_dict = vars(args)
    pprint(config_dict)
    print("====================================\n")

    main(args)

