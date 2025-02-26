# plot_results_reward.py

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


def load_results(base_path, gamma, rho=None):
    """Load average rewards, standard deviations, and reward noise levels."""
    if rho is not None:
        avg_rewards_path = f"{base_path}_reward_noise_avgs_{gamma}_rho{rho}.npy"
        std_devs_path = f"{base_path}_reward_noise_stds_{gamma}_rho{rho}.npy"
        noise_levels_path = f"{base_path}_reward_noise_levels.npy"
    else:
        avg_rewards_path = f"{base_path}_reward_noise_avgs_{gamma}.npy"
        std_devs_path = f"{base_path}_reward_noise_stds_{gamma}.npy"
        noise_levels_path = f"{base_path}_reward_noise_levels.npy"

    print(f"Loading results from {avg_rewards_path}, {std_devs_path}, and {noise_levels_path}")

    if os.path.exists(avg_rewards_path) and os.path.exists(std_devs_path) and os.path.exists(noise_levels_path):
        avg_rewards = np.load(avg_rewards_path)
        std_devs = np.load(std_devs_path)
        noise_levels = np.load(noise_levels_path)
        return avg_rewards, std_devs, noise_levels
    else:
        print(f"Files not found: {avg_rewards_path}, {std_devs_path}, or {noise_levels_path}")
        return None, None, None


def plot_reward_noise_results(env, ppo_results, sam_ppo_results, output_path):
    """Plot Reward Function Robustness Evaluation results."""
    plt.figure(figsize=(12, 8))

    # Plot PPO results
    if ppo_results[0] is not None:
        ppo_avgs, ppo_stds, noise_levels = ppo_results
        plt.plot(
            noise_levels,
            ppo_avgs,
            label='PPO',
            marker='o',
            linestyle='-',
            color='blue'
        )
        if ppo_stds is not None:
            plt.fill_between(
                noise_levels,
                ppo_avgs - ppo_stds,
                ppo_avgs + ppo_stds,
                color='blue',
                alpha=0.2
            )

    # Plot SAM+PPO results
    for rho, results in sam_ppo_results.items():
        if results[0] is not None:
            sam_ppo_avgs, sam_ppo_stds, noise_levels = results
            plt.plot(
                noise_levels,
                sam_ppo_avgs,
                label=f'SAM+PPO (rho={rho})',
                marker='s',
                linestyle='-',
                color='orange'
            )
            if sam_ppo_stds is not None:
                plt.fill_between(
                    noise_levels,
                    sam_ppo_avgs - sam_ppo_stds,
                    sam_ppo_avgs + sam_ppo_stds,
                    color='orange',
                    alpha=0.2
                )

    plt.xlabel('Reward Noise Standard Deviation')
    plt.ylabel('Average Reward')
    plt.title(f'Reward Function Robustness Evaluation on {env}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Reward function robustness plot saved as {output_path}")


def main(args):
    # environments = ['HalfCheetah-v3', 'Walker2d-v3', 'Hopper-v3']
    environments = args.env

    gamma = args.GAMMA
    rho_values = [0.008]  # Add other rho values if needed

    for env in environments:
        base_path = f"./perturbed_results/PPO_{env}_{gamma}"
        ppo_avgs, ppo_stds, noise_levels = load_results(base_path, gamma)

        sam_ppo_results = {}
        for rho in rho_values:
            sam_ppo_base_path = f"./perturbed_results/SAM_PPO_{env}_{gamma}"
            sam_ppo_avgs, sam_ppo_stds, sam_noise_levels = load_results(sam_ppo_base_path, gamma, rho)
            sam_ppo_results[rho] = (sam_ppo_avgs, sam_ppo_stds, sam_noise_levels)

        output_path = f'./plot_results/{env}_reward_noise_{gamma}.png'
        plot_reward_noise_results(
            env,
            (ppo_avgs, ppo_stds, noise_levels),
            sam_ppo_results,
            output_path
        )

    print("All reward noise plots have been generated and saved in './plot_results/'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot results for Action Robustness in SAM-PPO-continuous")

    # Other arguments
    parser.add_argument("--env", nargs='+', default=[], help="Environment names")
    parser.add_argument("--GAMMA", type=str, help="Gamma value for naming")

    
    args = parser.parse_args()
    
    if not os.path.exists("./plot_results"):
        os.makedirs("./plot_results")
    main(args)