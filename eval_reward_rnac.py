# eval_reward_sam_ppo.py

import os
import numpy as np
import gym
import argparse
import pickle
import yaml
from normalization import Normalization
from ppo_continuous_original import PPO_continuous
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
    seed, GAMMA = args.seed, args.GAMMA

    # Determine the load path based on whether reward noise was used during training
    load_path = f"./models/RNAC_reward_{args.reward_noise_std}_{args.env}_{GAMMA}"
    load_path_nom = f"./models/RNAC_{args.env}_0"

    print(f'load model from : {load_path}')
    print(f'load nominal model from : {load_path_nom}')
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

    agent_nom = PPO_continuous(args)
    agent_nom, state_norm_nom = load_agent(agent_nom, load_path_nom, args.device)
    agent_nom.gamma = args.gamma
 
    avg_reward_nom, std_reward_nom = evaluate_policy(args, env, agent_nom, state_norm_nom, num_evals=args.eval_episodes)
    print("---------------------------------------")
    print(f'Average Reward(nominal) over {args.eval_episodes} episodes: {avg_reward_nom:.3f} ± {std_reward_nom:.3f}')
    print("---------------------------------------")
 

    # Optionally add reward noise during evaluation (set to 0.0 by default)
    if args.eval_reward_noise_std >= 0.0:
        env_noise = NoisyRewardEnv(env, noise_std=args.eval_reward_noise_std)
        print(f"Evaluating with reward noise: std={args.eval_reward_noise_std}")
    else:
        print("Evaluating without reward noise.")


    agent = PPO_continuous(args)
    agent, state_norm = load_agent(agent, load_path, args.device)
    agent.gamma = args.gamma

    avg_reward, std_reward = evaluate_policy(args, env_noise, agent, state_norm, num_evals=args.eval_episodes)
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

    parser.add_argument('--hard', default='False', type=str)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument("--env", type=str, default='Hopper-v3', help="HalfCheetah-v3/Hopper-v3/Walker2d-v3")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--weight_reg", type=float, default=1e-5, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=2, help="Seed")
    parser.add_argument("--GAMMA", type=str, default='10', help="Name")
    parser.add_argument("--reward_noise_std", type=float, default=0.1,
                        help="Standard deviation of reward noise during training")
    parser.add_argument("--eval_reward_noise_std", type=float, default=0.0, help="Reward noise std during evaluation")

    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)

    print("\n===== Evaluation Configuration =====")
    config_dict = vars(args)
    pprint(config_dict)
    print("====================================\n")

    main(args)

# python eval_rnac.py --env Hopper-v3 