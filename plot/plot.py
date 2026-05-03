import json
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def plot_training_logs(file_path_str):
    # Convert string path to a Path object for easy manipulation
    log_path = Path(file_path_str)
    
    if not log_path.exists():
        print(f"Error: The file '{log_path}' was not found.")
        sys.exit(1)

    # Initialize lists to store metrics
    batches = []
    policy_loss = []
    value_loss = []
    entropy = []
    kl = []

    # Evaluation specific lists
    eval_batches = []
    win_rate = []
    draw_rate = []
    champions_replaced = []

    # Read and parse the JSON Lines from the file
    try:
        with open(log_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line}")
                    continue
                
                # Skip meta and summary rows
                if "meta" in data or "summary" in data:
                    continue
                    
                batch = data.get("batch")
                if batch is None:
                    continue

                batches.append(batch)
                policy_loss.append(data.get("policy_loss"))
                value_loss.append(data.get("value_loss"))
                entropy.append(data.get("entropy"))
                kl.append(data.get("kl"))
                
                # Track evaluations (which happen periodically)
                if data.get("win_rate") is not None:
                    eval_batches.append(batch)
                    win_rate.append(data["win_rate"])
                    draw_rate.append(data["draw_rate"])
                    
                # Track champion replacements
                if data.get("champion_replaced"):
                    champions_replaced.append(batch)
                    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # If no data was found, exit early
    if not batches:
        print("No valid batch data found in the provided log file.")
        sys.exit(1)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reinforcement Learning Training Metrics', fontsize=16)

    # Plot 1: Policy Loss and Value Loss (Twin Axes)
    axs[0, 0].plot(batches, policy_loss, label='Policy Loss', color='tab:blue')
    axs[0, 0].set_xlabel('Batch')
    axs[0, 0].set_ylabel('Policy Loss', color='tab:blue')
    axs[0, 0].tick_params(axis='y', labelcolor='tab:blue')
    axs[0, 0].grid(True, alpha=0.3)

    ax_loss_twin = axs[0, 0].twinx()
    ax_loss_twin.plot(batches, value_loss, label='Value Loss', color='tab:red')
    ax_loss_twin.set_ylabel('Value Loss', color='tab:red')
    ax_loss_twin.tick_params(axis='y', labelcolor='tab:red')
    axs[0, 0].set_title('Losses over Batches')

    # Plot 2: Entropy
    axs[0, 1].plot(batches, entropy, label='Entropy', color='tab:purple')
    axs[0, 1].set_xlabel('Batch')
    axs[0, 1].set_ylabel('Entropy')
    axs[0, 1].set_title('Policy Entropy')
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: KL Divergence
    axs[1, 0].plot(batches, kl, label='KL Divergence', color='tab:orange')
    axs[1, 0].set_xlabel('Batch')
    axs[1, 0].set_ylabel('KL Divergence')
    axs[1, 0].set_yscale('log') # Log scale is generally better for KL
    axs[1, 0].set_title('KL Divergence (Log Scale)')
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Win/Draw Rates and Champion Replacements
    if eval_batches:
        axs[1, 1].plot(eval_batches, win_rate, marker='o', label='Win Rate', color='tab:green')
        axs[1, 1].plot(eval_batches, draw_rate, marker='s', label='Draw Rate', color='tab:gray')

        # Mark where the champion was replaced
        for champ_batch in champions_replaced:
            # Only add the label to the legend once
            label = 'Champion Replaced' if champ_batch == champions_replaced[0] else ""
            axs[1, 1].axvline(x=champ_batch, color='gold', linestyle='--', label=label)

        axs[1, 1].set_xlabel('Batch')
        axs[1, 1].set_ylabel('Rate (0.0 to 1.0)')
        axs[1, 1].set_title('Evaluation Win/Draw Rates')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].set_ylim(0, 1)
    else:
        axs[1, 1].text(0.5, 0.5, 'No Evaluation Data Found', 
                       horizontalalignment='center', verticalalignment='center')
        axs[1, 1].set_title('Evaluation Win/Draw Rates')

    # Adjust layout
    plt.tight_layout()
    
    # --- Saving Logic ---
    # Create the output directory: parent_folder / file_name_without_extension
    out_dir = log_path.parent / log_path.stem
    
    # Make the directory (exist_ok=True prevents errors if you run it twice)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the save path
    save_path = out_dir / "training_metrics_dashboard.png"
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to: {save_path.resolve()}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Plot RL training metrics from a JSONL log file.")
    parser.add_argument("log_path", help="Path to the JSONL log file")
    
    args = parser.parse_args()
    
    # Run the plotting function
    plot_training_logs(args.log_path)