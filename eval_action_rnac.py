import os
import numpy as np
import gym
import argparse
import pickle
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous_original import PPO_continuous


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
                a = agent.evaluate(s)  # Deterministic policy for evaluation
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1] -> [-max, max]
                else:
                    action = a

                # Add Gaussian noise to the action
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
            evaluate_rewards.append(float('-inf'))  # Add a very low reward in case of failure

    return np.mean(evaluate_rewards), np.std(evaluate_rewards)

def load_agent(agent, load_path, device):
    agent.actor.load(f'{load_path}_actor', device=device)
    agent.critic.load(f'{load_path}_critic', device=device)
    with open(f'{load_path}_state_norm', 'rb') as file1:
        state_norm = pickle.load(file1)
    return agent, state_norm

def save_evals(save_path, action_noise_std_list, avgs, stds, GAMMA):
    np.save(f'{save_path}_action_noise_avgs_{GAMMA}', avgs)
    np.save(f'{save_path}_action_noise_stds_{GAMMA}', stds)
    np.save(f'{save_path}_action_noise_levels', action_noise_std_list)
    print(f'save model to : {save_path}_action_noise_avgs_{GAMMA}')



def main(args):
    seed, GAMMA = args.seed, args.GAMMA
    # evaluate PPO on perturbed environments
    load_path = f"./models/RNAC_{args.env}_{GAMMA}"
    save_path = f"./perturbed_results/RNAC_{args.env}_{GAMMA}"

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

    save_evals(save_path, action_noise_std_list, avgs_action_noise, stds_action_noise, GAMMA)
    print(*map(int, avgs_action_noise))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
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
    parser.add_argument("--GAMMA", type=str, default='0', help="Name")

    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")

    main(args)
