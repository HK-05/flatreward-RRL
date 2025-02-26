# eval_fricmass_rarl.py

import os
import numpy as np
import gym
import argparse
import pickle
import yaml
import copy
from normalization import Normalization
from sam_ppo_continuous import PPO_continuous  
from pprint import pprint


def inspect_env(env):
    print("=== Environment Structure ===")
    print("env:", env)
    print("env.unwrapped:", env.unwrapped)
    print("dir(env):", dir(env))
    print("dir(env.unwrapped):", dir(env.unwrapped))
    print("=============================")


def set_friction(env, geom_ids, friction_factors):
    """
    Modify friction coefficients of specific geometries.
    Args:
        env: Gym environment
        geom_ids: List of geometry IDs to modify friction
        friction_factors: List of friction values [mu1, mu2, mu3]
    """
    model = env.unwrapped.model
    for geom_id in geom_ids:
        if len(friction_factors) != 3:
            raise ValueError("friction_factors must be a list of three elements [mu1, mu2, mu3].")
        model.geom_friction[geom_id] = friction_factors


def set_mass(env, body_id, mass_factor, original_masses):
    """
    Modify mass of a specific body.
    Args:
        env: Gym environment
        body_id: ID of the body to modify mass
        mass_factor: Scaling factor for the mass
        original_masses: Dictionary of original mass values
    """
    model = env.unwrapped.model
    model.body_mass[body_id] = original_masses[body_id] * mass_factor


def evaluate_policy(
    args,
    env,
    protagonist,
    adversary,
    state_norm,
    friction_perturbation=None,
    mass_perturbation=None,
    num_evals=20):

    """
    Evaluates the policy.

    Args:
        args: Arguments
        env: Gym environment
        protagonist: Protagonist agent
        adversary: Adversary agent
        state_norm: State normalization object
        friction_perturbation: {'geom_ids': [...], 'friction_factors': [mu1, mu2, mu3]}
        mass_perturbation: {'body_id': int, 'mass_factor': float}
        num_evals: Number of evaluation episodes

    Returns:
        Average reward, standard deviation
    """
    evaluate_rewards = []

    for eval_idx in range(num_evals):
        try:
            # Modify friction
            if friction_perturbation is not None:
                set_friction(env, friction_perturbation['geom_ids'], friction_perturbation['friction_factors'])

            # Modify mass
            if mass_perturbation is not None:
                set_mass(env, mass_perturbation['body_id'], mass_perturbation['mass_factor'], mass_perturbation['original_masses'])

            # Reset environment
            s = env.reset()

            if args.use_state_norm:
                s = state_norm(s, update=False)

            done = False
            episode_reward = 0
            episode_steps = 0
            while not done and episode_steps < args.max_episode_steps:
                episode_steps += 1

                a_pro = protagonist.evaluate(s)
                a_adv = adversary.evaluate(s)
                # Combine actions: protagonist action + scaled adversary action
                action = a_pro + args.adv_fraction * a_adv

                if args.policy_dist == "Beta":
                    action = 2 * (action - 0.5) * args.max_action  # [0,1] -> [-max, max]
                action = np.clip(action, -args.max_action, args.max_action)

                s_, r, done, _ = env.step(action)
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
            evaluate_rewards.append(episode_reward)
            print(f'Episode Reward: {episode_reward:.3f}')

        except Exception as e:
            print(f'Error during evaluation {eval_idx+1}: {e}')
            evaluate_rewards.append(float('-inf'))  # Add a very low reward in case of failure

    return np.mean(evaluate_rewards), np.std(evaluate_rewards)


def load_agents(protagonist, adversary, load_path_pro, load_path_adv, device):
    protagonist.actor.load(f'{load_path_pro}_actor', device=device)
    protagonist.critic.load(f'{load_path_pro}_critic', device=device)
    adversary.actor.load(f'{load_path_adv}_actor', device=device)
    adversary.critic.load(f'{load_path_adv}_critic', device=device)
    with open(f'{load_path_pro}_state_norm', 'rb') as file1:
        state_norm = pickle.load(file1)
    print(f'protagonist loaded from : {load_path_pro}')
    print(f'adversary loaded from : {load_path_adv}')
    return protagonist, adversary, state_norm


def save_evals(save_path, setting, avgs, stds, GAMMA):
    np.save(f'{save_path}_{setting}_avgs_{GAMMA}', avgs)
    np.save(f'{save_path}_{setting}_stds_{GAMMA}', stds)
    print(f'model saved to : {save_path}_{setting}_avgs_{GAMMA}')



def main(args):
    seed = args.seed
    GAMMA = args.GAMMA

    # Load model paths
    load_path_pro = f"./models/RARL_Protagonist_{args.env}_{GAMMA}"
    load_path_adv = f"./models/RARL_Adversary_{args.env}_{GAMMA}"
    save_path = f"./perturbed_results/RARL_{args.env}_{GAMMA}"

    # Create environment
    env = gym.make(args.env)
    env_evaluate = gym.make(args.env)  # Separate environment for evaluation

    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    # Initialize Protagonist and Adversary agents
    protagonist = PPO_continuous(args)
    adversary_args = copy.deepcopy(args)
    adversary_args.lr_a = args.lr_a_adv  # Adversary may have a different learning rate
    adversary = PPO_continuous(adversary_args)

    # Load agents
    protagonist, adversary, state_norm = load_agents(protagonist, adversary, load_path_pro, load_path_adv, args.device)
    protagonist.gamma = args.gamma
    adversary.gamma = args.gamma

    eval_episodes = args.eval_episodes

    # Inspect environment structure
    inspect_env(env_evaluate)

    # Depending on the environment, set up evaluation
    if args.env in ['HalfCheetah-v3', 'Walker2d-v3', 'Hopper-v3']:
        if args.env == 'HalfCheetah-v3':
            # Define friction and mass parameters for HalfCheetah
            fric_fractions = np.linspace(0.4, 1.6, 11)  # Example friction factors
            fric_bodies = ['fthigh']  # Bodies to apply friction changes
            mass_fractions = np.linspace(0.5, 1.5, 11)  # Mass scaling factors
            mass_bodies = ['torso']  # Bodies to apply mass changes

        elif args.env == 'Walker2d-v3':
            # Define friction and mass parameters for Walker2d
            fric_fractions = np.linspace(0.4, 1.6, 11)
            fric_bodies = ['foot', 'foot_left']  # Bodies to apply friction changes
            mass_fractions = np.linspace(0.5, 1.5, 11)
            mass_bodies = ['torso']

        elif args.env == 'Hopper-v3':
            # Define friction and mass parameters for Hopper
            fric_fractions = np.linspace(0.4, 1.6, 11)
            fric_bodies = ['foot']
            mass_fractions = np.linspace(0.5, 1.5, 11)
            mass_bodies = ['torso']

        # Retrieve original mass and friction values
        original_masses = {}
        for body in mass_bodies:
            body_id = env_evaluate.unwrapped.model.body_names.index(body)
            original_masses[body_id] = env_evaluate.unwrapped.model.body_mass[body_id]

        original_frictions = {}
        for body in fric_bodies:
            body_id = env_evaluate.unwrapped.model.body_names.index(body)
            geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == body_id]
            original_frictions[body] = {geom_id: env_evaluate.unwrapped.model.geom_friction[geom_id].copy() for geom_id in geom_ids}

        # Evaluate with only friction changes
        setting_friction = 'friction'
        avgs_friction = []
        stds_friction = []
        print("\n=== Evaluating Friction Robustness ===")
        for f in fric_fractions:
            rewards = []
            for _ in range(eval_episodes):
                env_evaluate.seed(seed=np.random.randint(1000))
                try:
                    # Modify friction
                    for body in fric_bodies:
                        geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body)]
                        for geom_id in geom_ids:
                            base_frictions = original_frictions[body][geom_id]
                            scaled_frictions = base_frictions.copy()
                            scaled_frictions[:2] *= f  # Scale Mu1 and Mu2
                            scaled_frictions[2] = 0.1  # Keep Mu3 fixed
                            set_friction(env_evaluate, [geom_id], scaled_frictions)

                    # Reset environment to apply changes
                    s = env_evaluate.reset()

                    if args.use_state_norm:
                        s = state_norm(s, update=False)

                    done = False
                    episode_reward = 0
                    episode_steps = 0
                    while not done and episode_steps < args.max_episode_steps:
                        episode_steps += 1
                        a_pro = protagonist.evaluate(s)
                        a_adv = adversary.evaluate(s)
                        action = a_pro + args.adv_fraction * a_adv
                        if args.policy_dist == "Beta":
                            action = 2 * (action - 0.5) * args.max_action
                        action = np.clip(action, -args.max_action, args.max_action)
                        s_, r, done, _ = env_evaluate.step(action)
                        if args.use_state_norm:
                            s_ = state_norm(s_, update=False)
                        episode_reward += r
                        s = s_
                    rewards.append(episode_reward)
                except Exception as e:
                    print(f'Error during friction evaluation with f={f}: {e}')
                    rewards.append(float('-inf'))  # Failure case
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # print("---------------------------------------")
            # print(f'Friction factor: {f}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
            # print("---------------------------------------")
            avgs_friction.append(avg_reward)
            stds_friction.append(std_reward)

        save_evals(save_path, setting_friction, avgs_friction, stds_friction, GAMMA)
        print(*map(int, avgs_friction))


        # Restore original friction
        print("\n=== Restoring Friction to Original Values ===")
        for body in fric_bodies:
            geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body)]
            for geom_id in geom_ids:
                original_friction = original_frictions[body][geom_id]
                set_friction(env_evaluate, [geom_id], original_friction)
        print("Friction restored to original values.")

        # Evaluate with only mass changes
        setting_mass = 'mass'
        avgs_mass = []
        stds_mass = []
        print("\n=== Evaluating Mass Robustness ===")
        for m in mass_fractions:
            rewards = []
            for _ in range(eval_episodes):
                env_evaluate.seed(seed=np.random.randint(1000))
                try:
                    # Modify mass
                    for body in mass_bodies:
                        body_id = env_evaluate.unwrapped.model.body_names.index(body)
                        set_mass(env_evaluate, body_id, m, original_masses)

                    # Reset environment to apply changes
                    s = env_evaluate.reset()

                    if args.use_state_norm:
                        s = state_norm(s, update=False)

                    done = False
                    episode_reward = 0
                    episode_steps = 0
                    while not done and episode_steps < args.max_episode_steps:
                        episode_steps += 1
                        a_pro = protagonist.evaluate(s)
                        a_adv = adversary.evaluate(s)
                        action = a_pro + args.adv_fraction * a_adv
                        if args.policy_dist == "Beta":
                            action = 2 * (action - 0.5) * args.max_action
                        action = np.clip(action, -args.max_action, args.max_action)
                        s_, r, done, _ = env_evaluate.step(action)
                        if args.use_state_norm:
                            s_ = state_norm(s_, update=False)
                        episode_reward += r
                        s = s_
                    rewards.append(episode_reward)
                except Exception as e:
                    print(f'Error during mass evaluation with m={m}: {e}')
                    rewards.append(float('-inf'))  # Failure case
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # print("---------------------------------------")
            # print(f'Mass factor: {m}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
            # print("---------------------------------------")
            avgs_mass.append(avg_reward)
            stds_mass.append(std_reward)
            

        save_evals(save_path, setting_mass, avgs_mass, stds_mass, GAMMA)
        print(*map(int, avgs_mass))


        # Restore original mass
        print("\n=== Restoring Mass to Original Values ===")
        for body_id, original_mass in original_masses.items():
            env_evaluate.unwrapped.model.body_mass[body_id] = original_mass
        print("Mass restored to original values.")


        avgs_both = []
        stds_both = []
        setting_both = 'mass_and_friction'

        for m in mass_fractions:
            for f in fric_fractions:
                rewards = []
                for _ in range(eval_episodes):
                    env_evaluate.seed(seed=np.random.randint(1000))
                    try:
                        # Modify friction
                        for body in fric_bodies:
                            geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body)]
                            for geom_id in geom_ids:
                                base_frictions = original_frictions[body][geom_id]
                                scaled_frictions = base_frictions.copy()
                                scaled_frictions[:2] *= f  # Scale Mu1 and Mu2
                                scaled_frictions[2] = 0.1  # Keep Mu3 fixed
                                set_friction(env_evaluate, [geom_id], scaled_frictions)

                        # Modify mass
                        for body in mass_bodies:
                            body_id = env_evaluate.unwrapped.model.body_names.index(body)
                            set_mass(env_evaluate, body_id, m, original_masses)

                        # Reset environment to apply changes
                        s = env_evaluate.reset()

                        if args.use_state_norm:
                            s = state_norm(s, update=False)

                        done = False
                        episode_reward = 0
                        episode_steps = 0
                        while not done and episode_steps < args.max_episode_steps:
                            episode_steps += 1
                            a_pro = protagonist.evaluate(s)
                            a_adv = adversary.evaluate(s)
                            action = a_pro + args.adv_fraction * a_adv
                            if args.policy_dist == "Beta":
                                action = 2 * (action - 0.5) * args.max_action
                            action = np.clip(action, -args.max_action, args.max_action)
                            s_, r, done, _ = env_evaluate.step(action)
                            if args.use_state_norm:
                                s_ = state_norm(s_, update=False)
                            episode_reward += r
                            s = s_
                        rewards.append(episode_reward)
                    except Exception as e:
                        print(f'Error during mass and friction evaluation with m={m}, f={f}: {e}')
                        rewards.append(float('-inf'))  # Failure case
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                # print("---------------------------------------")
                # print(f'Mass factor: {m}, Friction factor: {f}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
                # print("---------------------------------------")
                avgs_both.append(avg_reward)
                stds_both.append(std_reward)

        save_evals(save_path, setting_both, avgs_both, stds_both, GAMMA)
        print(*map(int, avgs_both))


        # Restore original mass and friction
        print("\n=== Restoring All Parameters to Original Values ===")
        # Restore friction
        for body in fric_bodies:
            geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body)]
            for geom_id in geom_ids:
                original_friction = original_frictions[body][geom_id]
                set_friction(env_evaluate, [geom_id], original_friction)
        # Restore mass
        for body_id, original_mass in original_masses.items():
            env_evaluate.unwrapped.model.body_mass[body_id] = original_mass
        print("All parameters restored to original values.")

    else:
        print(f"Environment {args.env} not recognized for perturbation evaluation.")


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

    # Other arguments
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--policy_dist", type=str, help="Policy distribution: Beta or Gaussian")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--GAMMA", type=str, help="Gamma value for naming")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--adv_fraction", type=float, default=0.25, help="Fraction of adversary's action")
    parser.add_argument("--lr_a_adv", type=float, default=3e-4, help="Learning rate of adversary actor")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run on")

    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)
    

    print("\n===== Evaluation Configuration =====")
    config_dict = vars(args)
    pprint(config_dict)
    print("====================================\n")


    # Create results directory
    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")

    main(args)
