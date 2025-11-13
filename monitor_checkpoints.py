"""
Checkpoint Progress Monitor

This script monitors training progress by analyzing saved checkpoints and comparing
different checkpoint stages against perfect play.

Usage:
    python monitor_checkpoints.py                    # Monitor all checkpoints
    python monitor_checkpoints.py --live             # Watch for new checkpoints
    python monitor_checkpoints.py --compare 5000 10000  # Compare specific episodes
"""

import os
import argparse
import time
import glob
from datetime import datetime


def list_checkpoints(checkpoint_dir='checkpoints'):
    """List all available checkpoints"""
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_episode_*.pth'))

    # Extract episode numbers and sort
    checkpoint_info = []
    for cp in checkpoints:
        basename = os.path.basename(cp)
        try:
            episode = int(basename.split('_')[-1].replace('.pth', ''))
            size = os.path.getsize(cp)
            mtime = os.path.getmtime(cp)
            checkpoint_info.append({
                'path': cp,
                'episode': episode,
                'size': size,
                'modified': datetime.fromtimestamp(mtime)
            })
        except ValueError:
            continue

    checkpoint_info.sort(key=lambda x: x['episode'])
    return checkpoint_info


def display_checkpoint_list(checkpoints):
    """Display checkpoint information"""
    if not checkpoints:
        print("No checkpoints found in 'checkpoints/' directory")
        print("\nCheckpoints are saved every 5000 episodes during training.")
        return

    print("=" * 80)
    print("AVAILABLE CHECKPOINTS")
    print("=" * 80)
    print(f"{'Episode':<12} {'Size':<12} {'Modified':<25} {'Path':<30}")
    print("-" * 80)

    for cp in checkpoints:
        size_kb = cp['size'] / 1024
        modified_str = cp['modified'].strftime('%Y-%m-%d %H:%M:%S')
        basename = os.path.basename(cp['path'])
        print(f"{cp['episode']:<12} {size_kb:>8.1f} KB {modified_str:<25} {basename:<30}")

    print("-" * 80)
    print(f"Total checkpoints: {len(checkpoints)}")
    print()


def evaluate_checkpoint(checkpoint_path, num_games=100):
    """Evaluate a checkpoint against minimax"""
    try:
        import torch
        from evaluate_model import ModelEvaluator

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {os.path.basename(checkpoint_path)}")
        print(f"{'=' * 80}")

        evaluator = ModelEvaluator(checkpoint_path)

        # Quick evaluation
        print("\n🎮 Testing against Minimax (Perfect Play)...")
        wins, losses, draws = evaluator.evaluate_vs_minimax(num_games, model_plays_first=True)

        win_rate = wins / num_games * 100
        loss_rate = losses / num_games * 100
        draw_rate = draws / num_games * 100

        print(f"\nResults ({num_games} games):")
        print(f"  Wins:   {wins:3d} ({win_rate:5.1f}%)")
        print(f"  Losses: {losses:3d} ({loss_rate:5.1f}%)")
        print(f"  Draws:  {draws:3d} ({draw_rate:5.1f}%)")

        # Verdict
        if losses == 0:
            if draw_rate >= 90:
                print(f"\n  ✅ EXCELLENT - Near-perfect play!")
            elif draw_rate >= 70:
                print(f"\n  ✅ GOOD - Strong play")
            else:
                print(f"\n  ⚠️  LEARNING - Needs more training")
        else:
            print(f"\n  ❌ WEAK - Lost to perfect opponent")

        return {'wins': wins, 'losses': losses, 'draws': draws}

    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ Error evaluating checkpoint: {e}")
        return None


def compare_checkpoints(checkpoint_paths, num_games=100):
    """Compare multiple checkpoints"""
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)

    results = []
    for cp_path in checkpoint_paths:
        result = evaluate_checkpoint(cp_path, num_games)
        if result:
            episode = int(os.path.basename(cp_path).split('_')[-1].replace('.pth', ''))
            results.append({
                'episode': episode,
                'path': cp_path,
                **result
            })

    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)
        print(f"{'Episode':<12} {'Wins':<8} {'Losses':<8} {'Draws':<8} {'Draw Rate':<12}")
        print("-" * 80)

        for r in results:
            total = r['wins'] + r['losses'] + r['draws']
            draw_rate = r['draws'] / total * 100 if total > 0 else 0
            print(f"{r['episode']:<12} {r['wins']:<8} {r['losses']:<8} {r['draws']:<8} {draw_rate:>8.1f}%")

        print("\n📈 Training Progress:")
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            first_draw_rate = first['draws'] / num_games * 100
            last_draw_rate = last['draws'] / num_games * 100
            improvement = last_draw_rate - first_draw_rate

            print(f"  Episode {first['episode']}: {first_draw_rate:.1f}% draws")
            print(f"  Episode {last['episode']}: {last_draw_rate:.1f}% draws")

            if improvement > 0:
                print(f"  ✅ Improved by {improvement:.1f}%")
            elif improvement < 0:
                print(f"  ⚠️  Declined by {abs(improvement):.1f}%")
            else:
                print(f"  → No change")


def watch_checkpoints(checkpoint_dir='checkpoints', interval=60):
    """Watch for new checkpoints and evaluate them"""
    print("=" * 80)
    print("CHECKPOINT MONITORING - LIVE MODE")
    print("=" * 80)
    print(f"Watching: {checkpoint_dir}/")
    print(f"Check interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")

    seen_checkpoints = set()

    try:
        while True:
            current_checkpoints = list_checkpoints(checkpoint_dir)
            current_paths = {cp['path'] for cp in current_checkpoints}

            # Find new checkpoints
            new_checkpoints = current_paths - seen_checkpoints

            if new_checkpoints:
                print(f"\n🔔 New checkpoint(s) detected: {len(new_checkpoints)}")
                for cp_path in sorted(new_checkpoints):
                    evaluate_checkpoint(cp_path, num_games=50)

                seen_checkpoints.update(new_checkpoints)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Monitor training checkpoint progress')
    parser.add_argument('--live', action='store_true',
                       help='Watch for new checkpoints in real-time')
    parser.add_argument('--compare', nargs='+', type=int,
                       help='Compare specific checkpoint episodes (e.g., --compare 5000 10000)')
    parser.add_argument('--evaluate', type=int,
                       help='Evaluate specific checkpoint episode')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games for evaluation (default: 100)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds for --live mode (default: 60)')

    args = parser.parse_args()

    if args.live:
        watch_checkpoints(interval=args.interval)

    elif args.compare:
        checkpoints = list_checkpoints()
        if not checkpoints:
            print("No checkpoints found. Train a model first.")
            return

        # Find requested checkpoints
        cp_dict = {cp['episode']: cp['path'] for cp in checkpoints}
        requested_paths = []

        for episode in args.compare:
            if episode in cp_dict:
                requested_paths.append(cp_dict[episode])
            else:
                print(f"⚠️  Checkpoint for episode {episode} not found")

        if requested_paths:
            compare_checkpoints(requested_paths, args.games)
        else:
            print("No valid checkpoints to compare")

    elif args.evaluate:
        checkpoints = list_checkpoints()
        cp_dict = {cp['episode']: cp['path'] for cp in checkpoints}

        if args.evaluate in cp_dict:
            evaluate_checkpoint(cp_dict[args.evaluate], args.games)
        else:
            print(f"Checkpoint for episode {args.evaluate} not found")
            display_checkpoint_list(checkpoints)

    else:
        # Default: list all checkpoints
        checkpoints = list_checkpoints()
        display_checkpoint_list(checkpoints)

        if checkpoints:
            print("\n💡 Usage examples:")
            print(f"  python monitor_checkpoints.py --evaluate {checkpoints[0]['episode']}")
            if len(checkpoints) >= 2:
                print(f"  python monitor_checkpoints.py --compare {checkpoints[0]['episode']} {checkpoints[-1]['episode']}")
            print(f"  python monitor_checkpoints.py --live")


if __name__ == "__main__":
    main()
