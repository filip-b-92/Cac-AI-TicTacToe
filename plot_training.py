"""
Training History Visualization

Reads training history CSV files and creates plots to visualize training progress.

Usage:
    python plot_training.py training_history_20231213_143022.csv
    python plot_training.py --latest
"""

import argparse
import csv
import os
import glob


def load_training_history(csv_file):
    """Load training history from CSV file"""
    episodes = []
    win_rates = []
    loss_rates = []
    draw_rates = []
    avg_losses = []
    epsilons = []
    learning_rates = []
    eval_win_rates = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            win_rates.append(float(row['win_rate']))
            loss_rates.append(float(row['loss_rate']))
            draw_rates.append(float(row['draw_rate']))
            avg_losses.append(float(row['avg_loss']))
            epsilons.append(float(row['epsilon']))
            learning_rates.append(float(row['learning_rate']))

            eval_wr = float(row['eval_win_rate'])
            eval_win_rates.append(eval_wr if eval_wr >= 0 else None)

    return {
        'episodes': episodes,
        'win_rates': win_rates,
        'loss_rates': loss_rates,
        'draw_rates': draw_rates,
        'avg_losses': avg_losses,
        'epsilons': epsilons,
        'learning_rates': learning_rates,
        'eval_win_rates': eval_win_rates
    }


def print_training_summary(data, csv_file):
    """Print a text-based summary of training progress"""
    print("=" * 70)
    print(f"Training History Summary: {csv_file}")
    print("=" * 70)

    episodes = data['episodes']
    win_rates = data['win_rates']
    avg_losses = data['avg_losses']
    eval_win_rates = [x for x in data['eval_win_rates'] if x is not None]

    print(f"\nTotal Episodes: {episodes[-1]}")
    print(f"Data Points: {len(episodes)}")

    print("\n--- Performance Metrics ---")
    print(f"Initial Win Rate (vs opponent): {win_rates[0]:.1f}%")
    print(f"Final Win Rate (vs opponent): {win_rates[-1]:.1f}%")
    print(f"Best Win Rate (vs opponent): {max(win_rates):.1f}%")
    print(f"Improvement: +{win_rates[-1] - win_rates[0]:.1f}%")

    if eval_win_rates:
        print(f"\nFinal Win Rate (vs random): {eval_win_rates[-1]:.1f}%")
        print(f"Best Win Rate (vs random): {max(eval_win_rates):.1f}%")

    print("\n--- Training Loss ---")
    print(f"Initial Avg Loss: {avg_losses[0]:.4f}")
    print(f"Final Avg Loss: {avg_losses[-1]:.4f}")
    print(f"Best (Minimum) Loss: {min(avg_losses):.4f}")

    print("\n--- Exploration ---")
    print(f"Initial Epsilon: {data['epsilons'][0]:.3f}")
    print(f"Final Epsilon: {data['epsilons'][-1]:.3f}")

    print("\n--- Learning Rate ---")
    print(f"Initial LR: {data['learning_rates'][0]:.6f}")
    print(f"Final LR: {data['learning_rates'][-1]:.6f}")

    # Find when model started showing good performance
    good_performance_threshold = 60.0
    good_episodes = [ep for ep, wr in zip(episodes, win_rates) if wr >= good_performance_threshold]
    if good_episodes:
        print(f"\nReached {good_performance_threshold}% win rate at episode: {good_episodes[0]}")

    print()


def create_ascii_plot(data, metric_name, metric_data, height=15):
    """Create a simple ASCII plot"""
    if not metric_data:
        return

    print(f"\n{metric_name} over Training")
    print("-" * 70)

    # Filter out None values
    episodes = data['episodes']
    valid_data = [(e, v) for e, v in zip(episodes, metric_data) if v is not None]
    if not valid_data:
        print("No data to plot")
        return

    valid_episodes, valid_values = zip(*valid_data)

    min_val = min(valid_values)
    max_val = max(valid_values)
    val_range = max_val - min_val if max_val != min_val else 1

    # Create plot
    for i in range(height, -1, -1):
        threshold = min_val + (val_range * i / height)
        line = f"{threshold:6.2f} |"

        for val in valid_values:
            if val >= threshold:
                line += "█"
            else:
                line += " "

        print(line)

    # X-axis
    print("       " + "-" * len(valid_values))
    print(f"       Episodes: {valid_episodes[0]} to {valid_episodes[-1]}")
    print()


def plot_with_matplotlib(data, csv_file):
    """Create plots using matplotlib if available"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return False

    episodes = data['episodes']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training History: {os.path.basename(csv_file)}', fontsize=14)

    # Win/Loss/Draw rates
    ax = axes[0, 0]
    ax.plot(episodes, data['win_rates'], label='Win Rate', color='green', linewidth=2)
    ax.plot(episodes, data['loss_rates'], label='Loss Rate', color='red', linewidth=2)
    ax.plot(episodes, data['draw_rates'], label='Draw Rate', color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Win/Loss/Draw Rates (vs Opponent)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Evaluation win rate
    ax = axes[0, 1]
    eval_episodes = [e for e, v in zip(episodes, data['eval_win_rates']) if v is not None]
    eval_values = [v for v in data['eval_win_rates'] if v is not None]
    if eval_values:
        ax.plot(eval_episodes, eval_values, label='Win Rate vs Random', color='purple',
                linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Evaluation Performance (vs Random Player)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center')
        ax.set_title('Evaluation Performance')

    # Average loss
    ax = axes[1, 0]
    ax.plot(episodes, data['avg_losses'], label='Avg Loss', color='orange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Epsilon
    ax = axes[1, 1]
    ax.plot(episodes, data['epsilons'], label='Epsilon', color='brown', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_filename = csv_file.replace('.csv', '_plot.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")

    # Show plot
    plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('csv_file', nargs='?', help='Training history CSV file')
    parser.add_argument('--latest', action='store_true',
                       help='Use the most recent training history file')
    parser.add_argument('--text-only', action='store_true',
                       help='Show text summary only (no plots)')
    parser.add_argument('--ascii-plot', action='store_true',
                       help='Show ASCII plots in terminal')

    args = parser.parse_args()

    # Find CSV file
    if args.latest or args.csv_file is None:
        csv_files = glob.glob('training_history_*.csv')
        if not csv_files:
            print("❌ No training history files found")
            print("Train a model first to generate history files")
            return
        csv_file = max(csv_files, key=os.path.getctime)
        print(f"Using latest file: {csv_file}\n")
    else:
        csv_file = args.csv_file

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return

    # Load data
    try:
        data = load_training_history(csv_file)
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return

    # Print summary
    print_training_summary(data, csv_file)

    # Create ASCII plots if requested
    if args.ascii_plot:
        create_ascii_plot(data, "Win Rate", data['win_rates'], height=12)
        create_ascii_plot(data, "Average Loss", data['avg_losses'], height=12)

    # Create matplotlib plots if not text-only
    if not args.text_only:
        if not plot_with_matplotlib(data, csv_file):
            print("\n💡 Tip: Install matplotlib for visual plots:")
            print("   pip install matplotlib")
            print("\nOr use --ascii-plot for terminal-based plots")


if __name__ == "__main__":
    main()
