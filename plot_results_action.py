# plot_results_action.py

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


def load_results(base_path, gamma, rho=None):
    """Load average rewards, standard deviations, and action noise levels."""
    if rho is not None:
        avg_rewards_path = f"{base_path}_action_noise_avgs_{gamma}_rho{rho}.npy"
        std_devs_path = f"{base_path}_action_noise_stds_{gamma}_rho{rho}.npy"
        noise_levels_path = f"{base_path}_action_noise_levels.npy"
    else:
        avg_rewards_path = f"{base_path}_action_noise_avgs_{gamma}.npy"
        std_devs_path = f"{base_path}_action_noise_stds_{gamma}.npy"
        noise_levels_path = f"{base_path}_action_noise_levels.npy"

    print(f"Loading results from {avg_rewards_path}, {std_devs_path}, and {noise_levels_path}")

    if os.path.exists(avg_rewards_path) and os.path.exists(std_devs_path) and os.path.exists(noise_levels_path):
        avg_rewards = np.load(avg_rewards_path)
        std_devs = np.load(std_devs_path)
        noise_levels = np.load(noise_levels_path)
        return avg_rewards, std_devs, noise_levels
    else:
        print(f"Files not found: {avg_rewards_path}, {std_devs_path}, or {noise_levels_path}")
        return None, None, None


def plot_action_noise_results(env, ppo_results, rnac_ppo_results, rarl_ppo_results,sam_ppo_results,  output_path):
    """Plot Action Robustness Evaluation results."""
    plt.figure(figsize=(13, 9))

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 37
    })

    # Plot PPO results
    if ppo_results[0] is not None:
        ppo_avgs, ppo_stds, noise_levels = ppo_results
        plt.plot(
            noise_levels,
            ppo_avgs,
            label='PPO',
            marker='o',
            markersize=10,
            linestyle='-',
            color='blue',
            linewidth=4
        )
        if ppo_stds is not None:
            plt.fill_between(
                noise_levels,
                ppo_avgs - ppo_stds,
                ppo_avgs + ppo_stds,
                color='blue',
                alpha=0.2
            )

    # Plot RNAC_PPO results
    if rnac_ppo_results[0] is not None:
        rnac_ppo_avgs, rnac_ppo_stds, rnac_noise_levels = rnac_ppo_results
        plt.plot(
            rnac_noise_levels,
            rnac_ppo_avgs,
            label='RNAC',
            marker='o',
            markersize=10,
            linestyle='-',
            color='green',
            linewidth=4
        )
        if rnac_ppo_stds is not None:
            plt.fill_between(
                rnac_noise_levels,
                rnac_ppo_avgs - rnac_ppo_stds,
                rnac_ppo_avgs + rnac_ppo_stds,
                color='green',
                alpha=0.2
            )

    if rarl_ppo_results[0] is not None:
        rarl_ppo_avgs, rarl_ppo_stds, rarl_noise_levels = rarl_ppo_results
        plt.plot(
            rarl_noise_levels,
            rarl_ppo_avgs,
            label='RARL',
            marker='o',
            markersize=10,
            linestyle='-',
            color='brown',
            linewidth=4
        )
        if rarl_ppo_stds is not None:
            plt.fill_between(
                rarl_noise_levels,
                rarl_ppo_avgs - rarl_ppo_stds,
                rarl_ppo_avgs + rarl_ppo_stds,
                color='brown',
                alpha=0.2
            )

    # Plot SAM+PPO results
    for rho, results in sam_ppo_results.items():
        if results[0] is not None:
            sam_ppo_avgs, sam_ppo_stds, noise_levels = results
            plt.plot(
                noise_levels,
                sam_ppo_avgs,
                label=f'SAM+PPO',
                marker='s',
                markersize=10,
                linestyle='-',
                color='orange',
                linewidth=4
            )
            if sam_ppo_stds is not None:
                plt.fill_between(
                    noise_levels,
                    sam_ppo_avgs - sam_ppo_stds,
                    sam_ppo_avgs + sam_ppo_stds,
                    color='orange',
                    alpha=0.2
                )

    plt.xlabel('Action Noise Standard Deviation', fontsize=40)
    plt.ylabel('Average Reward', fontsize=40)
    plt.title(f'Action Robustness Evaluation', fontsize=40)
    plt.legend(fontsize=35)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Action robustness plot saved as {output_path}")


def main(args):
    # environments = ['HalfCheetah-v3', 'Walker2d-v3', 'Hopper-v3']
    environments = args.env
    gamma = args.GAMMA
    rnac = args.rnac
    rarl = args.rarl

    rho_values = args.rho  # Add other rho values if needed

    for env in environments:
        base_path = f"./perturbed_results/PPO_{env}_{gamma}"
        ppo_avgs, ppo_stds, noise_levels = load_results(base_path, gamma)

        sam_ppo_results = {}
        for rho in rho_values:
            sam_ppo_base_path = f"./perturbed_results/SAM_PPO_{env}_{gamma}"
            sam_ppo_avgs, sam_ppo_stds, sam_noise_levels = load_results(sam_ppo_base_path, gamma, rho)
            sam_ppo_results[rho] = (sam_ppo_avgs, sam_ppo_stds, sam_noise_levels)
        
        if rnac:
            rnac_base_path = f"./perturbed_results/RNAC_{env}_0"
            rnac_ppo_avgs, rnac_ppo_stds, rnac_noise_levels = load_results(rnac_base_path, 0)
        else:
            rnac_ppo_avgs, rnac_ppo_stds, rnac_noise_levels = None, None, None

        if rarl:
            rarl_base_path = f"./perturbed_results/RARL_{env}_21"
            rarl_ppo_avgs, rarl_ppo_stds, rarl_noise_levels = load_results(rarl_base_path, 21)
        else:
            rarl_ppo_avgs, rarl_ppo_stds, rarl_noise_levels = None, None, None


        output_path = f'./plot_results/{env}_action_noise_{gamma}.pdf'
        plot_action_noise_results(
            env,
            (ppo_avgs, ppo_stds, noise_levels),
            (rnac_ppo_avgs, rnac_ppo_stds, rnac_noise_levels),
            (rarl_ppo_avgs, rarl_ppo_stds, rarl_noise_levels),
            sam_ppo_results,
            output_path
        )

    print("All action noise plots have been generated and saved in './plot_results/'.")


if __name__ == "__main__":

    # HalfCheetah : 3, Hopper 1, Walker : 2

    parser = argparse.ArgumentParser("Plot results for Action Robustness in SAM-PPO-continuous")

    # Other arguments
    parser.add_argument("--env", nargs='+', default=[], help="Environment names")
    parser.add_argument("--GAMMA", type=str, help="Gamma value for naming")
    parser.add_argument("--rho", nargs='+', default=[], help="Rho value for SAM optimizer")
    parser.add_argument("--rnac", action='store_true', help="involve rnac results")
    parser.add_argument("--rarl", action='store_true', help="involve rarl results")


    args = parser.parse_args()
    
    if not os.path.exists("./plot_results"):
        os.makedirs("./plot_results")
    main(args)
