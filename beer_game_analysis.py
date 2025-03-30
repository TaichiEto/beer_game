import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from collections import defaultdict

def parse_log_file(log_file_path):
    """Parse the training log file to extract episode, reward, and loss data."""
    data = defaultdict(list)
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Extract episode number
            episode_match = re.search(r'Episode (\d+)/\d+', line)
            if episode_match:
                episode = int(episode_match.group(1))
                data['episodes'].append(episode)
                
                # Extract total reward
                reward_match = re.search(r'Total Reward: ([-\d.]+)', line)
                if reward_match:
                    reward = float(reward_match.group(1))
                    data['rewards'].append(reward)
                
                # Extract average reward
                avg_reward_match = re.search(r'Avg Reward\(100ep\): ([-\d.]+)', line)
                if avg_reward_match:
                    avg_reward = float(avg_reward_match.group(1))
                    data['avg_rewards'].append(avg_reward)
                
                # Extract loss
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                if loss_match:
                    loss = float(loss_match.group(1))
                    data['losses'].append(loss)
                    
    return data

def analyze_training_stability(rewards):
    """Analyze the stability of training by looking at reward variance."""
    if len(rewards) < 100:
        return "Not enough data for stability analysis"
        
    # Calculate moving variance with window of 20 episodes
    variances = []
    for i in range(len(rewards) - 20):
        window = rewards[i:i+20]
        variances.append(np.var(window))
    
    # High variance indicates instability
    avg_variance = np.mean(variances)
    max_variance = np.max(variances)
    
    return {
        "average_variance": avg_variance,
        "max_variance": max_variance,
        "stable_training": avg_variance < 1000  # Threshold can be adjusted
    }

def find_most_recent_output():
    """Find the most recent output directory."""
    output_dirs = glob.glob("output/*/")
    if not output_dirs:
        print("No output directories found.")
        return None
        
    # Sort by modification time
    newest_dir = max(output_dirs, key=os.path.getmtime)
    return newest_dir

def main():
    # Find most recent output directory
    output_dir = find_most_recent_output()
    if not output_dir:
        return
    
    print(f"Analyzing data from: {output_dir}")
    
    # Parse log file
    log_file = os.path.join(output_dir, "log.txt")
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
        
    data = parse_log_file(log_file)
    
    # Analyze training stability
    stability = analyze_training_stability(data['rewards'])
    
    # Create enhanced visualizations
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Reward progression with moving average
    plt.subplot(2, 2, 1)
    plt.plot(data['episodes'], data['rewards'], 'b-', alpha=0.3, label='Episode Reward')
    plt.plot(data['episodes'], data['avg_rewards'], 'r-', label='Moving Average (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Progression')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Loss progression
    plt.subplot(2, 2, 2)
    plt.plot(data['episodes'], data['losses'], 'g-')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Progression')
    plt.grid(True)
    
    # Plot 3: Reward distribution histogram
    plt.subplot(2, 2, 3)
    plt.hist(data['rewards'], bins=30, alpha=0.7, color='blue')
    plt.axvline(np.mean(data['rewards']), color='r', linestyle='dashed', 
                linewidth=1, label=f'Mean: {np.mean(data["rewards"]):.2f}')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Performance stability (reward variance)
    if isinstance(stability, dict):
        plt.subplot(2, 2, 4)
        window_size = 20
        variances = []
        for i in range(len(data['rewards']) - window_size):
            window = data['rewards'][i:i+window_size]
            variances.append(np.var(window))
        
        plt.plot(range(window_size, len(data['rewards'])), variances)
        plt.xlabel('Episode')
        plt.ylabel('Reward Variance (20 ep window)')
        plt.title('Training Stability')
        plt.yscale('log')  # Log scale for better visualization
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "enhanced_analysis.png"))
    plt.close()
    
    # Print analysis summary
    print("\nAnalysis Summary:")
    print(f"Total episodes: {max(data['episodes'])}")
    print(f"Final average reward (last 100 ep): {data['avg_rewards'][-1]:.2f}")
    print(f"Best episode reward: {max(data['rewards']):.2f}")
    print(f"Worst episode reward: {min(data['rewards']):.2f}")
    
    if isinstance(stability, dict):
        print(f"Training stability: {'Stable' if stability['stable_training'] else 'Unstable'}")
        print(f"Average reward variance: {stability['average_variance']:.2f}")
    else:
        print(f"Stability analysis: {stability}")
    
    # Save summary to file
    with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
        f.write("Analysis Summary:\n")
        f.write(f"Total episodes: {max(data['episodes'])}\n")
        f.write(f"Final average reward (last 100 ep): {data['avg_rewards'][-1]:.2f}\n")
        f.write(f"Best episode reward: {max(data['rewards']):.2f}\n")
        f.write(f"Worst episode reward: {min(data['rewards']):.2f}\n")
        
        if isinstance(stability, dict):
            f.write(f"Training stability: {'Stable' if stability['stable_training'] else 'Unstable'}\n")
            f.write(f"Average reward variance: {stability['average_variance']:.2f}\n")

if __name__ == "__main__":
    main()
