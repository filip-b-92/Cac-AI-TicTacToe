# Visualization and Inspection Guide

This guide explains all the visualization and monitoring tools available for understanding your Tic-Tac-Toe AI training.

## 🎯 Overview

The training system now includes comprehensive monitoring and visualization tools:

1. **TensorBoard** - Real-time training metrics
2. **Training History CSV** - Exportable training data
3. **Model Inspection** - Understand model decisions
4. **Training Plots** - Visualize progress after training
5. **Checkpoints** - Resume training from any point
6. **Evaluation** - Test against random player

---

## 📊 1. TensorBoard (Real-time Monitoring)

### What is TensorBoard?
TensorBoard is a professional ML visualization tool that shows training metrics in real-time as your model trains.

### How to Use:

1. **Start training** (TensorBoard logging is automatic):
   ```bash
   python main.py
   ```

2. **In another terminal**, start TensorBoard:
   ```bash
   tensorboard --logdir=runs
   ```

3. **Open your browser** to http://localhost:6006

### What You'll See:

- **Training/WinRate** - Win rate against self-play opponent
- **Training/LossRate** - Loss rate over time
- **Training/DrawRate** - Draw rate over time
- **Training/AvgLoss** - Average training loss (should decrease)
- **Training/Epsilon** - Exploration rate (should decay)
- **Training/LearningRate** - Learning rate schedule
- **Evaluation/WinRate_vs_Random** - Performance vs random player

### Disable TensorBoard:
```bash
export DISABLE_TENSORBOARD=1
python main.py
```

---

## 📈 2. Training History & Plots

### CSV Export
Every training run automatically saves a CSV file:
- **Filename**: `training_history_YYYYMMDD_HHMMSS.csv`
- **Contains**: Episode-by-episode metrics
- **Can be**: Opened in Excel, analyzed in Python, shared

### Visualize Training Progress

#### Quick Summary:
```bash
python plot_training.py --latest
```

#### Visualize Specific Run:
```bash
python plot_training.py training_history_20231213_143022.csv
```

#### Text-only Summary:
```bash
python plot_training.py --latest --text-only
```

#### ASCII Plots (in terminal):
```bash
python plot_training.py --latest --ascii-plot
```

### Output:
- **Console**: Training summary statistics
- **PNG File**: Beautiful matplotlib plots saved automatically
- **Charts**: Win/Loss/Draw rates, Evaluation results, Loss curve, Epsilon decay

---

## 🔍 3. Model Inspection Tool

Understand what your trained model is thinking!

### Interactive Mode (Recommended):
```bash
python inspect_model.py --interactive
```

Then enter board states like:
- `X..O.....` - X in top-left, O in position 3
- `XX.OO....` - Two X's and two O's
- `.........` - Empty board

The tool shows:
- Q-values for each position
- Model's action ranking
- Best move highlighted

### Analyze Specific Position:
```bash
python inspect_model.py --board "X..O....."
```

### See Example Scenarios:
```bash
python inspect_model.py --examples
```

Shows interesting scenarios like:
- Opening moves
- Winning in one move
- Blocking opponent's win
- Fork opportunities

### Use Different Model:
```bash
python inspect_model.py --model checkpoint_episode_10000.pth --interactive
```

---

## 💾 4. Checkpoints

### Automatic Checkpoints
Checkpoints are saved automatically:
- **Every 5000 episodes** during training
- **At the end** of training
- **Location**: `checkpoints/` directory

### What's in a Checkpoint?
- Model weights (policy, target, opponent networks)
- Optimizer state
- Scheduler state
- Episode number
- Best win rate achieved

### Resume Training from Checkpoint:
```python
import torch
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/checkpoint_episode_10000.pth')

# Create trainer with loaded weights
policy_net = DQN(device)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

trainer = Trainer(device, policy_net)
# Restore optimizer, scheduler, etc.
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
trainer.steps_done = checkpoint['steps_done']

# Continue training
trainer.train()
```

---

## 🎮 5. Evaluation Against Random Player

### Automatic Evaluation
During training, the model is evaluated against a random player:
- **Every 500 episodes**
- **100 games per evaluation**
- **Results logged** to console and TensorBoard

### Why This Matters:
- Self-play (training) win rate can be misleading
- Random player provides absolute performance measure
- Should reach 90%+ win rate against random player

### Manual Evaluation:
```python
from trainer import Trainer
import torch

device = torch.device('cpu')
trainer = Trainer(device)
# Load your model first
trainer.policy_net.load_state_dict(torch.load('tic_tac_toe_best_model.pth'))

wins, losses, draws = trainer.evaluate_against_random(num_games=1000)
win_rate = wins / 1000 * 100
print(f"Win rate: {win_rate:.1f}%")
```

---

## 📋 6. Training Output Example

```
Starting training for 50000 episodes...
Epsilon decay: 10000.0, Min memory: 1000
Learning rate: 0.001, Batch size: 64
TensorBoard logging enabled. Run: tensorboard --logdir=runs

Episode 100/50000 | Win: 45.2% | Loss: 34.1% | Draw: 20.7% | Avg Loss: 0.1234 | Epsilon: 0.905 | LR: 0.001000
  → New best model saved! Win rate: 45.2%

Episode 200/50000 | Win: 52.3% | Loss: 27.8% | Draw: 19.9% | Avg Loss: 0.0987 | Epsilon: 0.819 | LR: 0.001000
  → New best model saved! Win rate: 52.3%

...

Episode 500/50000 | Win: 68.5% | Loss: 15.2% | Draw: 16.3% | Avg Loss: 0.0543 | Epsilon: 0.606 | LR: 0.001000
  → New best model saved! Win rate: 68.5%

=== Evaluation vs Random Player (100 games) ===
Wins: 87 | Losses: 2 | Draws: 11 | Win Rate: 87.0%
===============================================

...

Episode 5000/50000 | Win: 85.7% | Loss: 5.1% | Draw: 9.2% | Avg Loss: 0.0123 | Epsilon: 0.105 | LR: 0.000500
  → Checkpoint saved: checkpoints/checkpoint_episode_5000.pth
```

---

## 🔧 Configuration Options

### Environment Variables:

```bash
# Training parameters
export NUM_EPISODES=50000
export EPSILON_DECAY=10000
export MIN_MEMORY_SIZE=1000
export BATCH_SIZE=64
export LEARNING_RATE=0.001

# Visualization
export DISABLE_TENSORBOARD=1  # Disable TensorBoard logging

# Run training
python main.py
```

---

## 📚 What to Look For

### Signs of Good Training:
✅ Win rate increases over time
✅ Loss rate decreases over time
✅ Average loss decreases and stabilizes
✅ Epsilon decreases smoothly
✅ Evaluation win rate > 85% vs random
✅ Win rate vs opponent > 70%

### Signs of Problems:
❌ Win rate oscillates wildly
❌ Loss increases or doesn't decrease
❌ Win rate stays below 40%
❌ Evaluation win rate < 70% after 10k episodes
❌ Model loses to random player frequently

---

## 💡 Pro Tips

1. **Always check TensorBoard** during long training runs to catch issues early
2. **Use `--examples` mode** in inspect_model.py to see if your model learned basic strategies
3. **Compare checkpoints** from different episodes to see if later training helps
4. **Look at evaluation metrics** - they're more reliable than self-play metrics
5. **Save your CSV files** - you can analyze them later even if TensorBoard logs are deleted

---

## 🆘 Troubleshooting

### TensorBoard won't start
```bash
pip install tensorboard
```

### Plots won't display
```bash
pip install matplotlib
```

### Model inspection fails
- Make sure the model file exists
- Check you're using the correct model path
- Ensure the model was trained (not just initialized)

### Can't find training history
```bash
ls -la training_history_*.csv
```

---

## 📖 Next Steps

1. Train your model
2. Watch TensorBoard while it trains
3. When done, run `python plot_training.py --latest`
4. Test model understanding with `python inspect_model.py --interactive`
5. Play against it with `python main.py` (choose not to retrain)

Happy training! 🎉
