# Setup and Run Guide

Complete guide to install dependencies, train the model, and track progress with checkpoints.

---

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `torch` - PyTorch for neural networks
- `tensorboard` - Real-time training visualization
- `matplotlib` - Plotting training history
- `numpy` - Numerical operations
- `python-dotenv` - Environment configuration

### 2. Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed ✅')"
```

---

## 🎮 Training the Model

### Basic Training (50,000 episodes)

```bash
python main.py
```

**What happens:**
- Trains for 50,000 episodes (default)
- Saves checkpoints every 5,000 episodes to `checkpoints/`
- Saves best model to `tic_tac_toe_best_model.pth`
- Saves final model to `tic_tac_toe_model.pth`
- Exports training history to CSV
- Logs to TensorBoard (if installed)

**Expected time:**
- CPU: ~30-60 minutes
- GPU: ~10-20 minutes

### Custom Training Configuration

```bash
# Train for more episodes
NUM_EPISODES=100000 python main.py

# Faster epsilon decay
EPSILON_DECAY=5000 python main.py

# Larger batch size (faster on GPU)
BATCH_SIZE=128 python main.py

# Combine multiple settings
NUM_EPISODES=100000 BATCH_SIZE=128 LEARNING_RATE=0.0005 python main.py
```

**Available environment variables:**
- `NUM_EPISODES` - Total training episodes (default: 50000)
- `EPSILON_DECAY` - Epsilon decay rate (default: 10000)
- `MIN_MEMORY_SIZE` - Min experiences before training (default: 1000)
- `BATCH_SIZE` - Batch size for training (default: 64)
- `LEARNING_RATE` - Learning rate (default: 0.001)
- `DISABLE_TENSORBOARD` - Set to 1 to disable TensorBoard

---

## 📊 Monitoring Training Progress

### Option 1: Watch TensorBoard (Real-time)

**In one terminal**, start training:
```bash
python main.py
```

**In another terminal**, start TensorBoard:
```bash
tensorboard --logdir=runs
```

**Open browser:**
- Go to http://localhost:6006
- See real-time graphs of win rates, loss, epsilon, etc.

### Option 2: Monitor Console Output

Training prints progress every 100 episodes:

```
Episode 100/50000 | Win: 45.2% | Loss: 34.1% | Draw: 20.7% | Avg Loss: 0.1234 | Epsilon: 0.905 | LR: 0.001000
  → New best model saved! Win rate: 45.2%
```

Every 500 episodes, evaluates against random player:

```
=== Evaluation vs Random Player (100 games) ===
Wins: 87 | Losses: 2 | Draws: 11 | Win Rate: 87.0%
===============================================
```

Every 5000 episodes, saves checkpoint:

```
Episode 5000/50000 | Win: 85.7% | Loss: 5.1% | Draw: 9.2% | Avg Loss: 0.0123 | Epsilon: 0.105 | LR: 0.000500
  → Checkpoint saved: checkpoints/checkpoint_episode_5000.pth
```

### Option 3: Monitor Checkpoints

**List all saved checkpoints:**
```bash
python monitor_checkpoints.py
```

**Evaluate a specific checkpoint:**
```bash
python monitor_checkpoints.py --evaluate 5000
```

**Compare two checkpoints:**
```bash
python monitor_checkpoints.py --compare 5000 10000
```

**Watch for new checkpoints (live monitoring):**
```bash
python monitor_checkpoints.py --live
```

This will automatically evaluate new checkpoints as they're created during training!

---

## 🔍 Checking Training Results

### After Training: Comprehensive Evaluation

```bash
python evaluate_model.py
```

**This tests:**
1. Model vs Minimax (perfect play) - both as first and second player
2. Model vs Random player
3. Provides verdict on training success

**Expected output:**
```
================================================================================
FINAL VERDICT: Was RL Training Worth It?
================================================================================

🏆 VERDICT: RL TRAINING WAS HIGHLY SUCCESSFUL!
  ✅ Model learned near-optimal or optimal play
  ✅ Never loses to perfect opponent
  ✅ Dominates random players
  ✅ Achieves ~95% draws vs perfect play
```

### Inspect Model Decisions

```bash
python inspect_model.py --interactive
```

Enter board positions to see:
- Q-values for each position
- Model's action ranking
- Best move choice

### Visualize Training History

```bash
python plot_training.py --latest
```

Shows:
- Win/Loss/Draw rate progression
- Training loss curve
- Epsilon decay
- Learning rate schedule
- Evaluation results

---

## 📁 Checkpoint System

### What Are Checkpoints?

Checkpoints save **complete training state** including:
- Model weights (policy, target, opponent networks)
- Optimizer state
- Learning rate scheduler state
- Episode number
- Best win rate achieved

### When Are Checkpoints Saved?

1. **Every 5000 episodes** - Automatic checkpoints
2. **End of training** - Final checkpoint
3. **Best model** - Saved when win rate improves

### Checkpoint File Structure

```
checkpoints/
├── checkpoint_episode_5000.pth
├── checkpoint_episode_10000.pth
├── checkpoint_episode_15000.pth
├── ...
└── checkpoint_episode_50000.pth

tic_tac_toe_best_model.pth  # Best performing model
tic_tac_toe_model.pth        # Final model
```

### Using Checkpoints

**Evaluate checkpoint:**
```bash
python evaluate_model.py --model checkpoints/checkpoint_episode_10000.pth
```

**Inspect checkpoint:**
```bash
python inspect_model.py --model checkpoints/checkpoint_episode_10000.pth --interactive
```

**Resume training from checkpoint:**
```python
import torch
from model import DQN
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_episode_10000.pth')

# Restore model
policy_net = DQN(device)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

# Create trainer with restored model
trainer = Trainer(device, policy_net)
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
trainer.steps_done = checkpoint['steps_done']

# Continue training
trainer.train()
```

---

## 🎯 Complete Workflow Example

### Terminal 1: Training
```bash
# Start training
python main.py
```

### Terminal 2: TensorBoard
```bash
# Watch training in real-time
tensorboard --logdir=runs
```

### Terminal 3: Checkpoint Monitoring
```bash
# Auto-evaluate new checkpoints
python monitor_checkpoints.py --live
```

### After Training

```bash
# 1. Evaluate against perfect play
python evaluate_model.py

# 2. Visualize training progress
python plot_training.py --latest

# 3. Inspect what model learned
python inspect_model.py --examples

# 4. Compare early vs late checkpoints
python monitor_checkpoints.py --compare 5000 50000
```

---

## 🔧 Troubleshooting

### Training is slow

**On CPU:**
- Training 50k episodes takes ~30-60 minutes
- This is normal for CPU training
- Consider reducing episodes: `NUM_EPISODES=10000 python main.py`

**To use GPU:**
- Ensure CUDA is installed
- PyTorch will automatically use GPU if available
- Check with: `python -c "import torch; print(torch.cuda.is_available())"`

### Checkpoints not saving

- Checkpoints save every 5000 episodes
- If training < 5000 episodes, only final checkpoint is saved
- Check `checkpoints/` directory exists

### TensorBoard not working

```bash
pip install tensorboard
tensorboard --logdir=runs
```

Or disable it:
```bash
DISABLE_TENSORBOARD=1 python main.py
```

### Model not learning

1. Check evaluation: `python evaluate_model.py`
2. Inspect training history: `python plot_training.py --latest`
3. Look at checkpoints: `python monitor_checkpoints.py --compare 5000 10000`

If model isn't improving after 10k+ episodes, there may be training issues.

---

## 📊 What to Expect

### Training Progress (Typical)

| Episode | Win Rate vs Self | Win Rate vs Random | Draw Rate vs Perfect |
|---------|------------------|-------------------|---------------------|
| 1,000   | ~40%            | ~60%              | ~50%                |
| 5,000   | ~60%            | ~80%              | ~70%                |
| 10,000  | ~75%            | ~90%              | ~85%                |
| 25,000  | ~85%            | ~95%              | ~90%                |
| 50,000  | ~90%            | ~98%              | ~95%                |

### Signs of Good Training

✅ Win rate increasing over time
✅ Loss decreasing and stabilizing
✅ Epsilon decaying smoothly (1.0 → 0.1)
✅ Evaluation win rate > 85% vs random
✅ Never loses to perfect minimax player
✅ Draws 90%+ against perfect play

### Signs of Problems

❌ Win rate oscillating wildly
❌ Loss increasing or not decreasing
❌ Win rate stuck below 40%
❌ Loses to minimax player
❌ Low win rate vs random (< 70%)

---

## 💡 Tips

1. **Start with default settings** - They're tuned for good performance
2. **Watch TensorBoard** - Catch issues early
3. **Use checkpoints** - Compare different training stages
4. **Evaluate early** - Test checkpoint at 5000 episodes to verify training works
5. **Be patient** - 50k episodes takes time but ensures good learning

---

## 🎓 Next Steps

After successful training:

1. **Compete against your AI**: `python main.py` (choose not to retrain)
2. **Challenge it with tricky positions**: `python inspect_model.py --interactive`
3. **Understand what it learned**: `python inspect_model.py --examples`
4. **Compare to perfection**: `python evaluate_model.py`

Enjoy your trained AI! 🎉
