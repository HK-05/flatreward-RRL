# eval_action_sam_ppo.py

import os
import numpy as np
import gym
import argparse
import pickle
import yaml
from normalization import Normalization
from sam_ppo_continuous import PPO_continuous
from pprint import pprint

def evaluate_policy(args, env, agent, state_norm, action_noise_std, num_evals=5):
    """
    Evaluates the policy with added action noise.

    Args:
        args: Arguments.
        env: Gym environment.
        agent: Trained agent.
        state_norm: State normalization object.
        action_noise_std: Standard deviation of the action noise.
        num_evals: Number of evaluation episodes.

    Returns:
        Average reward, standard deviation.
    """
    evaluate_rewards = []

    for eval_idx in range(num_evals):
        try:
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

                noise = np.random.normal(0, action_noise_std, size=action.shape)
                noisy_action = action + noise
                noisy_action = np.clip(noisy_action, -args.max_action, args.max_action)

                s_, r, done, _ = env.step(noisy_action)
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
            evaluate_rewards.append(episode_reward)
            # print(f'Episode Reward: {episode_reward:.3f}')

        except Exception as e:
            print(f'Error during evaluation {eval_idx+1}: {e}')
            evaluate_rewards.append(float('-inf'))  

    return np.mean(evaluate_rewards), np.std(evaluate_rewards)

def load_agent(agent, load_path, device):
    agent.actor.load(f'{load_path}_actor', device=device)
    agent.critic.load(f'{load_path}_critic', device=device)
    with open(f'{load_path}_state_norm', 'rb') as file1:
        state_norm = pickle.load(file1)
    return agent, state_norm

def save_evals(save_path, action_noise_std_list, avgs, stds, GAMMA, rho):
    if args.use_sam:
        np.save(f'{save_path}_action_noise_avgs_{GAMMA}_rho{rho}', avgs)
        np.save(f'{save_path}_action_noise_stds_{GAMMA}_rho{rho}', stds)
    else:
        np.save(f'{save_path}_action_noise_avgs_{GAMMA}', avgs)
        np.save(f'{save_path}_action_noise_stds_{GAMMA}', stds)
    np.save(f'{save_path}_action_noise_levels', action_noise_std_list)

def main(args):
    seed, GAMMA, rho = args.seed, args.GAMMA, args.rho
    if args.use_sam:
        load_path = f"./models/SAM_PPO_{args.env}_rho_{args.rho}_{GAMMA}"
        save_path = f"./perturbed_results/SAM_PPO_{args.env}_{GAMMA}"
    elif args.compare:
        compare_algo = args.compare_algo
        load_path = f"./models/{compare_algo}_PPO_{args.env}_{GAMMA}"
        save_path = f"./perturbed_results/{compare_algo}_PPO_{args.env}_{GAMMA}"
    else:
        load_path = f"./models/PPO_{args.env}_{GAMMA}"
        save_path = f"./perturbed_results/PPO_{args.env}_{GAMMA}"

    # Create environment
    env = gym.make(args.env)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    print(f'load model from : {load_path}')

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    agent = PPO_continuous(args)
    agent, state_norm = load_agent(agent, load_path, args.device)
    agent.gamma = args.gamma

    eval_episodes = args.eval_episodes

    # Action noise levels to evaluate
    action_noise_std_list = np.linspace(0.0, 0.5, 11)  # From 0.0 to 0.5 with 11 steps
    avgs_action_noise = []
    stds_action_noise = []

    print("\n=== Evaluating Action Robustness ===")
    for action_noise_std in action_noise_std_list:
        rewards = []
        for _ in range(eval_episodes):
            env.seed(seed=np.random.randint(1000))
            avg_reward, std_reward = evaluate_policy(
                args,
                env,
                agent,
                state_norm,
                action_noise_std=action_noise_std,
                num_evals=1
            )
            rewards.append(avg_reward)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        # print("---------------------------------------")
        # print(f'Action Noise STD: {action_noise_std}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
        # print("---------------------------------------")
        avgs_action_noise.append(avg_reward)
        stds_action_noise.append(std_reward)

    save_evals(save_path, action_noise_std_list, avgs_action_noise, stds_action_noise, GAMMA, rho)
    print(f'results saved to : {save_path}')
    print(*map(int, avgs_action_noise))

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

    parser.add_argument("--use_state_norm", action='store_true', help="Use state normalization")
    parser.add_argument("--use_sam", action='store_true', help="Use SAM optimizer")

    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument("--policy_dist", type=str, help="Policy distribution: Beta or Gaussian")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--GAMMA", type=str, help="Gamma value for naming")
    parser.add_argument("--rho", type=float, help="Rho value for SAM optimizer")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--compare", action='store_true', help="Comparison with other robust algoritms")
    parser.add_argument("--compare_algo", type=str, help="other robust algorithms")

    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)

    print("\n===== Evaluation Configuration =====")
    config_dict = vars(args)
    pprint(config_dict)
    print("====================================\n")

    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")

    main(args)
