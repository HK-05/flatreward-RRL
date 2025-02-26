import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_results(base_path, setting, gamma, rho=None):
    """Load average rewards and standard deviations from numpy files for a specific rho value."""
    if rho is not None:
        avg_rewards_path = f"{base_path}_{setting}_avgs_{gamma}_rho{rho}.npy"
        std_devs_path = f"{base_path}_{setting}_stds_{gamma}_rho{rho}.npy"
    else:
        avg_rewards_path = f"{base_path}_{setting}_avgs_{gamma}.npy"
        std_devs_path = f"{base_path}_{setting}_stds_{gamma}.npy"
    
    print(f"Loading results from {avg_rewards_path} and {std_devs_path}")
    
    if os.path.exists(avg_rewards_path) and os.path.exists(std_devs_path):
        avg_rewards = np.load(avg_rewards_path)
        std_devs = np.load(std_devs_path)
        return avg_rewards, std_devs
    else:
        print(f"Files not found: {avg_rewards_path} or {std_devs_path}")
        return None, None

def plot_friction_results(env, setting, ppo_avgs, ppo_stds, rnac_avgs, rnac_stds, rarl_avgs, rarl_stds, sam_ppo_results, friction_fractions, output_dir, geom_per_friction=1):
    """Plot Friction Robustness Evaluation results with separate plots for each friction body."""
    # Define labels for geom_ids
    

  
    
    geom_labels = [f'Geom {i+1}' for i in range(geom_per_friction)]
    
    # Iterate over each geom_id and create separate plots
    for geom_idx in range(geom_per_friction):
        plt.figure(figsize=(13, 9))
        
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 37
        })

        # PPO results for current geom_id
        if ppo_avgs is not None:
            try:
                # Extract avg and std for the current geom_id
                ppo_mean = ppo_avgs.reshape(-1, geom_per_friction)[:, geom_idx]
                ppo_std = ppo_stds.reshape(-1, geom_per_friction)[:, geom_idx] if ppo_stds is not None else None
                plt.plot(
                    friction_fractions,
                    ppo_mean,
                    label='PPO',
                    marker='o',
                    markersize=10,
                    linestyle='-',
                    color='blue',
                    linewidth=4
                )
                if ppo_std is not None:
                    plt.fill_between(
                        friction_fractions,
                        ppo_mean - ppo_std,
                        ppo_mean + ppo_std,
                        color='blue',
                        alpha=0.2
                    )
            except ValueError:
                print("Error reshaping PPO averages and standard deviations for friction setting.")
        # PPO results for current geom_id
        if rnac_avgs is not None:
            try:
                # Extract avg and std for the current geom_id
                rnac_mean = rnac_avgs.reshape(-1, geom_per_friction)[:, geom_idx]
                rnac_std = rnac_stds.reshape(-1, geom_per_friction)[:, geom_idx] if rnac_stds is not None else None
                plt.plot(
                    friction_fractions,
                    rnac_mean,
                    label='RNAC',
                    marker='o',
                    markersize=10,
                    linestyle='-',
                    color='green',
                    linewidth=4
                )
                if rnac_std is not None:
                    plt.fill_between(
                        friction_fractions,
                        rnac_mean - rnac_std,
                        rnac_mean + rnac_std,
                        color='green',
                        alpha=0.2
                    )
            except ValueError:
                print("Error reshaping RNAC averages and standard deviations for friction setting.")
        if rarl_avgs is not None:
            try:
                # Extract avg and std for the current geom_id
                rarl_mean = rarl_avgs.reshape(-1, geom_per_friction)[:, geom_idx]
                rarl_std = rarl_stds.reshape(-1, geom_per_friction)[:, geom_idx] if rnac_stds is not None else None
                plt.plot(
                    friction_fractions,
                    rarl_mean,
                    label='RARL',
                    marker='o',
                    markersize=10,
                    linestyle='-',
                    color='brown',
                    linewidth=4
                )
                if rarl_std is not None:
                    plt.fill_between(
                        friction_fractions,
                        rarl_mean - rarl_std,
                        rarl_mean + rarl_std,
                        color='brown',
                        alpha=0.2
                    )
            except ValueError:
                print("Error reshaping RARL averages and standard deviations for friction setting.")

        # SAM+PPO results for current geom_id
        for rho, (sam_ppo_avgs, sam_ppo_stds) in sam_ppo_results.items():
            if sam_ppo_avgs is not None:
                try:
                    sam_ppo_mean = sam_ppo_avgs.reshape(-1, geom_per_friction)[:, geom_idx]
                    sam_ppo_std = sam_ppo_stds.reshape(-1, geom_per_friction)[:, geom_idx] if sam_ppo_stds is not None else None
                    plt.plot(
                        friction_fractions,
                        sam_ppo_mean,
                        label=f'SAM+PPO',
                        marker='s',
                        markersize=10,
                        linestyle='-',
                        color='orange',
                        linewidth=4
                    )
                    if sam_ppo_std is not None:
                        plt.fill_between(
                            friction_fractions,
                            sam_ppo_mean - sam_ppo_std,
                            sam_ppo_mean + sam_ppo_std,
                            color='orange',
                            alpha=0.2
                        )
                except ValueError:
                    print(f"Error reshaping SAM+PPO averages and standard deviations for rho={rho} in friction setting.")
        
        plt.xlabel('Friction Factor', fontsize=45)
        plt.ylabel('Average Reward', fontsize=45)
        plt.title(f'Friction Robustness Evaluation', fontsize=45)
        plt.legend(fontsize=40)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the individual plot
        individual_output_path = os.path.join(output_dir, f"{env}_{setting}_geom{geom_idx+1}.pdf")
        plt.savefig(individual_output_path, format='pdf')
        plt.close()
        print(f"Friction robustness plot saved as {individual_output_path}")

def plot_mass_results(env, setting, ppo_avgs, ppo_stds, rnac_avgs, rnac_stds, sam_ppo_results, rarl_avgs, rarl_stds, mass_fractions, output_path):
    """Plot Mass Robustness Evaluation results."""
    plt.figure(figsize=(13, 9))

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 37
    })

    
    if ppo_avgs is not None:
        plt.plot(
            mass_fractions,
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
                mass_fractions,
                ppo_avgs - ppo_stds,
                ppo_avgs + ppo_stds,
                color='blue',
                alpha=0.2
            )
    
    if rnac_avgs is not None:
        plt.plot(
            mass_fractions,
            rnac_avgs,
            label='RNAC',
            marker='o',
            markersize=10,
            linestyle='-',
            color='green',
            linewidth=4
        )
        if rnac_stds is not None:
            plt.fill_between(
                mass_fractions,
                rnac_avgs - rnac_stds,
                rnac_avgs + rnac_stds,
                color='green',
                alpha=0.2
            )
    
    if rarl_avgs is not None:
        plt.plot(
            mass_fractions,
            rarl_avgs,
            label='RARL',
            marker='o',
            markersize=10,
            linestyle='-',
            color='brown',
            linewidth=4
        )
        if rarl_stds is not None:
            plt.fill_between(
                mass_fractions,
                rarl_avgs - rarl_stds,
                rarl_avgs + rarl_stds,
                color='brown',
                alpha=0.2
            )
    
    for rho, (sam_ppo_avgs, sam_ppo_stds) in sam_ppo_results.items():
        if sam_ppo_avgs is not None:
            plt.plot(
                mass_fractions,
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
                    mass_fractions,
                    sam_ppo_avgs - sam_ppo_stds,
                    sam_ppo_avgs + sam_ppo_stds,
                    color='orange',
                    alpha=0.2
                )
    
    plt.xlabel('Mass Factor', fontsize=45)
    plt.ylabel('Average Reward', fontsize=45)
    plt.title(f'Mass Robustness Evaluation', fontsize=45)
    plt.legend(fontsize=40)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Mass robustness plot saved as {output_path}")

def plot_heatmap_results(env, setting, ppo_avgs, ppo_stds, rnac_avgs, rnac_stds, rarl_avgs, rarl_stds, sam_ppo_results, friction_fractions, mass_fractions, output_path):
    """Plot Mass and Friction Robustness Evaluation results as a heatmap."""
    # Assuming ppo_avgs is a flat array of size len(friction_fractions) * len(mass_fractions)
    fric_len = len(friction_fractions)
    mass_len = len(mass_fractions)
    
    try:
        ppo_avgs_matrix = ppo_avgs.reshape(fric_len, mass_len)
        ppo_stds_matrix = ppo_stds.reshape(fric_len, mass_len) if ppo_stds is not None else None
    except ValueError:
        print("Error reshaping PPO averages and standard deviations for heatmap.")
        ppo_avgs_matrix = None
        ppo_stds_matrix = None
    
    try:
        # print(rnac_avgs)
        rnac_avgs_matrix = rnac_avgs.reshape(fric_len, mass_len)
        rnac_stds_matrix = rnac_stds.reshape(fric_len, mass_len) if rnac_stds is not None else None
    except ValueError:
        print("Error reshaping RNAC averages and standard deviations for heatmap.")
        rnac_avgs_matrix = None
        rnac_stds_matrix = None
    
    try:
        print(rarl_avgs)
        rarl_avgs_matrix = rarl_avgs.reshape(fric_len, mass_len)
        rarl_stds_matrix = rarl_stds.reshape(fric_len, mass_len) if rarl_stds is not None else None
    except ValueError:
        print("Error reshaping RARL averages and standard deviations for heatmap.")
        rarl_avgs_matrix = None
        rarl_stds_matrix = None

    # Similarly reshape SAM+PPO results
    sam_ppo_avgs_matrices = {}
    sam_ppo_stds_matrices = {}
    for rho, (sam_ppo_avgs, sam_ppo_stds) in sam_ppo_results.items():
        if sam_ppo_avgs is not None:
            try:
                sam_ppo_avgs_matrices[rho] = sam_ppo_avgs.reshape(fric_len, mass_len)
                if sam_ppo_stds is not None:
                    sam_ppo_stds_matrices[rho] = sam_ppo_stds.reshape(fric_len, mass_len)
            except ValueError:
                print(f"Error reshaping SAM+PPO averages and standard deviations for rho={rho} in heatmap.")
    
    # plt.figure(figsize=(14, 10))
    
    if rnac_avgs_matrix is not None:
        for rho, sam_ppo_avgs_matrix in sam_ppo_avgs_matrices.items():
            vmin = min(ppo_avgs_matrix.min(), rnac_avgs_matrix.min(), sam_ppo_avgs_matrices[rho].min())
            vmax = max(ppo_avgs_matrix.max(), rnac_avgs_matrix.max(), sam_ppo_avgs_matrices[rho].max())

            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.09], wspace=0.3)

            # Plot PPO heatmap
            ax0 = plt.subplot(gs[0])
            if ppo_avgs_matrix is not None:
                c = ax0.pcolormesh(mass_fractions, friction_fractions, ppo_avgs_matrix, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                ax0.set_ylabel('Friction Factor', fontsize=45)
                ax0.set_title('PPO', fontsize=45)

            # Plot RNAC heatmap
            ax1 = plt.subplot(gs[1])
            if rnac_avgs_matrix is not None:
                c = ax1.pcolormesh(mass_fractions, friction_fractions, rnac_avgs_matrix, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                ax1.set_title('RNAC', fontsize=45)
                ax1.set_xlabel('        Mass', fontsize=45)
                ax1.set_yticks([])  # Remove y-tick labels for RNAC

            ax1 = plt.subplot(gs[2])
            if rarl_avgs_matrix is not None:
                c = ax1.pcolormesh(mass_fractions, friction_fractions, rarl_avgs_matrix, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                ax1.set_title('RARL', fontsize=45)
                ax1.set_xlabel('Factor        ', fontsize=45)
                ax1.set_yticks([])  # Remove y-tick labels for RNAC

            # Plot SAM+PPO heatmap
            ax2 = plt.subplot(gs[3])
            if sam_ppo_avgs_matrices:
                for rho, sam_ppo_avgs_matrix in sam_ppo_avgs_matrices.items():
                    c = ax2.pcolormesh(mass_fractions, friction_fractions, sam_ppo_avgs_matrix, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                ax2.set_title('SAM+PPO', fontsize=45)
                ax2.set_yticks([])  # Remove y-tick labels for SAM+PPO

            # Create a separate axis for the color bar
            cbar_ax = plt.subplot(gs[4])
            fig.colorbar(c, cax=cbar_ax)

            # Adjust the layout to prevent colorbar overlap without using plt.tight_layout()
            plt.subplots_adjust(wspace=0.3, right=0.90)

    else:
        # Plot PPO Heatmap
        if ppo_avgs_matrix is not None:
            plt.subplot(1, 2, 1)
            c = plt.pcolormesh(mass_fractions, friction_fractions, ppo_avgs_matrix, shading='auto', cmap='viridis')
            colorbar = plt.colorbar(c)
            plt.xlabel('Mass Factor', fontsize=45)
            plt.ylabel('Friction Factor', fontsize=45)
            plt.title(f'PPO', fontsize=45)
        
        # Plot SAM+PPO Heatmap for each rho
        if sam_ppo_avgs_matrices:
            plt.subplot(1, 2, 2)
            for rho, sam_ppo_avgs_matrix in sam_ppo_avgs_matrices.items():
                c = plt.pcolormesh(mass_fractions, friction_fractions, sam_ppo_avgs_matrix, shading='auto', cmap='viridis')
            colorbar = plt.colorbar(c)
            plt.xlabel('Mass Factor', fontsize=45)
            # plt.ylabel('Friction Factor', fontsize=45)
            plt.title(f'SAM+PPO', fontsize=45)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Mass and Friction robustness heatmap saved as {output_path}")

def main(args):
    # environments = ['HalfCheetah-v3', 'Walker2d-v3', 'Hopper-v3']
    environments = args.env
    settings = ['friction', 'mass', 'mass_and_friction']
    gamma = args.GAMMA  
    rnac = args.rnac
    rarl = args.rarl
    rho_values = args.rho  # Add other rho values if needed
    
    for env in environments:
        for setting in settings:
            base_path = f"./perturbed_results/PPO_{env}_{gamma}"
            avgs, stds = load_results(base_path, setting, gamma)
            
            if rnac:
                rnac_base_path = f"./perturbed_results/RNAC_{env}_0"
                rnac_avgs, rnac_stds = load_results(rnac_base_path, setting, 0)

            if rarl:
                rarl_base_path = f"./perturbed_results/RARL_{env}_21"
                rarl_avgs, rarl_stds = load_results(rarl_base_path, setting, 21)
                print(rarl_avgs)
            sam_ppo_results = {}
            for rho in rho_values:
                sam_ppo_base_path = f"./perturbed_results/SAM_PPO_{env}_{gamma}"
                sam_ppo_avgs, sam_ppo_stds = load_results(sam_ppo_base_path, setting, gamma, rho)
                sam_ppo_results[rho] = (sam_ppo_avgs, sam_ppo_stds)
            
            if setting == 'friction':
                # Define friction_fractions based on evaluation script
                friction_fractions = np.linspace(0.4, 1.6, 11)
                
                # Determine number of geom_ids per friction_factor based on environment
                if env == 'HalfCheetah-v3':
                    geom_per_friction = 1  # 'bfoot', 'ffoot'
                elif env == 'Walker2d-v3':
                    geom_per_friction = 1  # 'foot', 'foot_left'
                elif env == 'Hopper-v3':
                    geom_per_friction = 1  # 'foot'
                else:
                    geom_per_friction = 1  # Default
                
                # Define output directory for friction plots within plot_results
                # All plots are saved in ./plot_results/
                output_dir = "./plot_results"
                
                # Plot friction results with separate graphs for each geom_id
                plot_friction_results(
                    env,
                    setting,
                    avgs,
                    stds,
                    rnac_avgs,
                    rnac_stds,
                    rarl_avgs,
                    rarl_stds,
                    sam_ppo_results,
                    friction_fractions,
                    output_dir,
                    geom_per_friction
                )
            
            elif setting == 'mass':
                # Define mass_fractions based on evaluation script
                mass_fractions = np.linspace(0.5, 1.5, 11)
                output_path = f'./plot_results/{env}_{setting}_{gamma}.pdf'
                plot_mass_results(
                    env,
                    setting,
                    avgs,
                    stds,
                    rnac_avgs,
                    rnac_stds,
                    sam_ppo_results,
                    rarl_avgs,
                    rarl_stds,
                    mass_fractions,
                    output_path
                )
            
            elif setting == 'mass_and_friction':
                # Define friction_fractions and mass_fractions based on evaluation script
                friction_fractions = np.linspace(0.4, 1.6, 11)
                mass_fractions = np.linspace(0.5, 1.5, 11)
                output_path = f'./plot_results/{env}_{setting}_{gamma}_heatmap.pdf'
                plot_heatmap_results(
                    env,
                    setting,
                    avgs,
                    stds,
                    rnac_avgs,
                    rnac_stds,
                    rarl_avgs,
                    rarl_stds,
                    sam_ppo_results,
                    friction_fractions,
                    mass_fractions,
                    output_path
                )
    
    print("All plots have been generated and saved in './plot_results/'.")

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
