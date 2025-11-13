# AI TicTacToe

Deep Reinforcement Learning (DQN) implementation for Tic-Tac-Toe with comprehensive evaluation against optimal algorithmic solutions.

## 🎯 Key Features

- **Deep Q-Network (DQN)** implementation for Tic-Tac-Toe
- **Minimax algorithm** for perfect play comparison
- **Comprehensive evaluation** system to verify training success
- **Real-time monitoring** with TensorBoard
- **Model inspection tools** to understand AI decisions
- **Training visualizations** and performance tracking

## 🤔 Why Both RL and Minimax?

Tic-Tac-Toe is a **solved game** with only ~5,478 legal positions. This makes it perfect for:
1. **Learning RL concepts** in a simple environment
2. **Verifying RL success** against provably optimal play (minimax)
3. **Comparing approaches**: RL vs algorithmic solutions

**Spoiler**: For Tic-Tac-Toe, minimax is simpler and better. But RL scales to complex games where minimax fails!

## 🚀 Quick Start

### Train the Model
```bash
python main.py
```

### Evaluate Against Perfect Play
```bash
python evaluate_model.py
```

### Inspect Model Decisions
```bash
python inspect_model.py --interactive
```

### Visualize Training
```bash
python plot_training.py --latest
tensorboard --logdir=runs
```

## 📊 Game Complexity

- **Total possible states**: 19,683 (3^9)
- **Legal positions**: ~5,478
- **Unique positions (symmetry)**: ~765
- **Result with perfect play**: DRAW (always!)

## 📈 Evaluation System

The evaluation system tests your RL model against:

1. **Minimax (Perfect Play)** - The gold standard
   - A perfect player NEVER loses
   - Can only draw or win
   - Tests if RL learned optimal strategy

2. **Random Player** - Baseline performance
   - Tests if model learned anything useful
   - Should achieve >90% win rate

3. **Performance Metrics**
   - Win/Loss/Draw rates
   - Comparison: RL vs Minimax
   - Verdict on training success

## 📖 Documentation

- **VISUALIZATION_GUIDE.md** - Complete guide to monitoring and visualization tools
- **evaluate_model.py** - Comprehensive evaluation suite
- **inspect_model.py** - Interactive Q-value inspection
- **plot_training.py** - Training history visualization

## 🔧 Tools Included

| Tool | Purpose |
|------|---------|
| `main.py` | Train and play against AI |
| `evaluate_model.py` | Test RL vs minimax (perfect play) |
| `minimax_solver.py` | Optimal Tic-Tac-Toe algorithm |
| `inspect_model.py` | Visualize model Q-values |
| `plot_training.py` | Generate training plots |

## 💡 Expected Results

A well-trained RL model should:
- ✅ **Never lose** to perfect minimax player
- ✅ **Draw 90%+** against perfect play
- ✅ **Win 90%+** against random player
- ✅ Show understanding of winning strategies

If your model doesn't achieve this, the evaluation will tell you why!
