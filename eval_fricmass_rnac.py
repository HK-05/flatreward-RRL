import os
import numpy as np
import gym
import argparse
import pickle
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous_original import PPO_continuous


def inspect_env(env):
    print("=== Environment Structure ===")
    print("env:", env)
    print("env.unwrapped:", env.unwrapped)
    print("dir(env):", dir(env))
    print("dir(env.unwrapped):", dir(env.unwrapped))
    print("=============================")


def set_friction(env, geom_ids, friction_factors):

    model = env.unwrapped.model
    for geom_id in geom_ids:
        if len(friction_factors) != 3:
            raise ValueError("friction_factors must be a list of three elements [mu1, mu2, mu3].")
        model.geom_friction[geom_id] = friction_factors
        # print(f'Set friction of geom_id {geom_id} to {friction_factors}')


def set_mass(env, body_id, mass_factor, original_masses):

    model = env.unwrapped.model
    model.body_mass[body_id] = original_masses[body_id] * mass_factor
    # print(f'Set mass of body_id {body_id} to {model.body_mass[body_id]} (Original: {original_masses[body_id]}, Factor: {mass_factor})')


def evaluate_policy(
    args, 
    env, 
    agent, 
    state_norm, 
    friction_perturbation=None, 
    mass_perturbation=None, 
    num_evals=20):

    evaluate_rewards = []
    
    for eval_idx in range(num_evals):
        try:
            # Friction 
            if friction_perturbation is not None:
                set_friction(env, friction_perturbation['geom_ids'], friction_perturbation['friction_factors'])
            
            # Mass 
            if mass_perturbation is not None:
                set_mass(env, mass_perturbation['body_id'], mass_perturbation['mass_factor'], mass_perturbation['original_masses'])
            
            s = env.reset()
            
            if args.use_state_norm:
                s = state_norm(s, update=False)
            
            done = False
            episode_reward = 0
            while not done:
                a = agent.evaluate(s)
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1] -> [-max, max]
                else:
                    action = a
                s_, r, done, _ = env.step(action)
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
            evaluate_rewards.append(episode_reward)
            print(f'Episode Reward: {episode_reward:.3f}')
        
        except Exception as e:
            print(f'Error during evaluation {eval_idx+1}: {e}')
            evaluate_rewards.append(float('-inf'))  
    
    return np.mean(evaluate_rewards), np.std(evaluate_rewards)


def load_agent(agent, load_path, device):
    agent.actor.load(f'{load_path}_actor', device=device)
    agent.critic.load(f'{load_path}_critic', device=device)
    with open(f'{load_path}_state_norm', 'rb') as file1:
        state_norm = pickle.load(file1)
    with open(f'{load_path}_reward_scaling', 'rb') as file2:
        reward_scaling = pickle.load(file2)
    return agent, state_norm, reward_scaling


def save_evals(save_path, setting, avgs, stds, GAMMA):
    np.save(f'{save_path}_{setting}_avgs_{GAMMA}', avgs)
    np.save(f'{save_path}_{setting}_stds_{GAMMA}', stds)


def main(args):
    seed, GAMMA = args.seed, args.GAMMA
    # Evaluate PPO on perturbed environments
    
    load_path = f"./models/RNAC_{args.env}_{GAMMA}"  
    save_path = f"./perturbed_results/RNAC_{args.env}_{GAMMA}"


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


    agent = PPO_continuous(args)
    agent, state_norm, reward_scaling = load_agent(agent, load_path, args.device)
    agent.gamma = args.gamma


    eval_episodes = args.eval_episodes
    if args.hard == 'False':
        hard = False
    else:
        hard = True


    inspect_env(env_evaluate)


    if args.env in ['HalfCheetah-v3', 'Walker2d-v3', 'Hopper-v3']:
        if args.env == 'HalfCheetah-v3':
            # Define friction and mass parameters for HalfCheetah
            # fric_fractions = np.linspace(0.2, 0.8, 11)  
            fric_fractions = np.linspace(0.4, 1.6, 11)  
            # fric_bodies = ['fthigh']  
            fric_bodies = ['ffoot'] 
            mass_fractions = np.linspace(0.5, 1.5, 11)  
            mass_bodies = ['torso']  


        elif args.env == 'Walker2d-v3':
            # Define friction and mass parameters for Walker2d
            fric_fractions = np.linspace(0.4, 1.6, 11)  
            fric_bodies = ['foot_left']  
            # fric_bodies = ['foot', 'foot_left']  
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
        
        setting_friction = 'friction'
        avgs_friction = []
        stds_friction = []
        print("\n=== Evaluating Friction Robustness ===")
        for f in fric_fractions:
            for body in fric_bodies:
                geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body)]
                rewards = []
                for _ in range(eval_episodes):
                    env_evaluate.seed(seed=np.random.randint(1000))
                    try:
 
                        for geom_id in geom_ids:
                            base_frictions = original_frictions[body][geom_id]
                            scaled_frictions = base_frictions.copy()
                            scaled_frictions[:2] *= f  
                            scaled_frictions[2] = 0.1  
                            set_friction(env_evaluate, [geom_id], scaled_frictions)
                        
         
                        s = env_evaluate.reset()
                        
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
                            s_, r, done, _ = env_evaluate.step(action)
                            if args.use_state_norm:
                                s_ = state_norm(s_, update=False)
                            episode_reward += r
                            s = s_
                        rewards.append(episode_reward)
                    except Exception as e:
                        print(f'Error during friction evaluation with f={f}: {e}')
                        rewards.append(float('-inf'))  
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                # print("---------------------------------------")
                # print(f'Friction factor: {f} on body: {body}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
                # print("---------------------------------------")
                avgs_friction.append(avg_reward)
                stds_friction.append(std_reward)
        
        save_evals(save_path, setting_friction, avgs_friction, stds_friction, GAMMA)
        print(*map(int, avgs_friction))


        print("\n=== Restoring Friction to Original Values ===")
        for body in fric_bodies:
            geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body)]
            for geom_id in geom_ids:
                original_friction = original_frictions[body][geom_id]
                set_friction(env_evaluate, [geom_id], original_friction)
        print("Friction restored to original values.")


        setting_mass = 'mass'
        avgs_mass = []
        stds_mass = []
        print("\n=== Evaluating Mass Robustness ===")
        for m in mass_fractions:
            for body in mass_bodies:
                body_id = env_evaluate.unwrapped.model.body_names.index(body)
                rewards = []
                for _ in range(eval_episodes):
                    env_evaluate.seed(seed=np.random.randint(1000))
                    try:
                        
                        set_mass(env_evaluate, body_id, m, original_masses)
                        
                        
                        s = env_evaluate.reset()
                        
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
                            s_, r, done, _ = env_evaluate.step(action)
                            if args.use_state_norm:
                                s_ = state_norm(s_, update=False)
                            episode_reward += r
                            s = s_
                        rewards.append(episode_reward)
                    except Exception as e:
                        print(f'Error during mass evaluation with m={m}: {e}')
                        rewards.append(float('-inf'))  
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                # print("---------------------------------------")
                # print(f'Mass factor: {m} on body: {body}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
                # print("---------------------------------------")
                avgs_mass.append(avg_reward)
                stds_mass.append(std_reward)
        
        save_evals(save_path, setting_mass, avgs_mass, stds_mass, GAMMA)
        print(*map(int, avgs_mass))


        setting_both = 'mass_and_friction'
        avgs_both = []
        stds_both = []
        print("\n=== Evaluating Mass and Friction Robustness ===")
        for f in fric_fractions:
            for m in mass_fractions:
                for body_fric, body_mass in zip(fric_bodies, mass_bodies):
                    geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body_fric)]
                    body_id = env_evaluate.unwrapped.model.body_names.index(body_mass)
                    rewards = []
                    for _ in range(eval_episodes):
                        env_evaluate.seed(seed=np.random.randint(1000))
                        try:
                            for geom_id in geom_ids:
                                base_frictions = original_frictions[body_fric][geom_id]
                                scaled_frictions = base_frictions.copy()
                                scaled_frictions[:2] *= f  
                                scaled_frictions[2] = 0.1  
                                set_friction(env_evaluate, [geom_id], scaled_frictions)
                            
                            set_mass(env_evaluate, body_id, m, original_masses)
                            
                            s = env_evaluate.reset()
                            
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
                                s_, r, done, _ = env_evaluate.step(action)
                                if args.use_state_norm:
                                    s_ = state_norm(s_, update=False)
                                episode_reward += r
                                s = s_
                            rewards.append(episode_reward)
                        except Exception as e:
                            print(f'Error during mass and friction evaluation with f={f}, m={m}: {e}')
                            rewards.append(float('-inf'))  
                    avg_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    # print("---------------------------------------")
                    # print(f'Friction factor: {f} on body: {body_fric}, Mass factor: {m} on body: {body_mass}: Avg Reward: {avg_reward:.3f}, Std: {std_reward:.3f}')
                    # print("---------------------------------------")
                    avgs_both.append(avg_reward)
                    stds_both.append(std_reward)
        
        save_evals(save_path, setting_both, avgs_both, stds_both, GAMMA)
        print(*map(int, avgs_both))

        
        print("\n=== Restoring All Parameters to Original Values ===")
        for body in fric_bodies:
            geom_ids = [i for i, geom_body in enumerate(env_evaluate.unwrapped.model.geom_bodyid) if geom_body == env_evaluate.unwrapped.model.body_names.index(body)]
            for geom_id in geom_ids:
                original_friction = original_frictions[body][geom_id]
                set_friction(env_evaluate, [geom_id], original_friction)
        for body_id, original_mass in original_masses.items():
            env_evaluate.unwrapped.model.body_mass[body_id] = original_mass
            print(f'Restored mass of body_id {body_id} to {original_mass}')
        print("All parameters restored to original values.")


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
