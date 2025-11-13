import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
import os
from game import TicTacToe
from memory import ReplayMemory
from model import DQN

class Trainer:
    def __init__(self, device, policy_net=None):
        self.device = device
        self.num_episodes = int(os.getenv('NUM_EPISODES', 50000))
        self.gamma = float(os.getenv('GAMMA', 0.99))
        self.epsilon_start = float(os.getenv('EPSILON_START', 1.0))
        self.epsilon_end = float(os.getenv('EPSILON_END', 0.1))
        self.epsilon_decay = float(os.getenv('EPSILON_DECAY', 10000))  # Fixed: reduced from 100000
        self.learning_rate = float(os.getenv('LEARNING_RATE', 0.001))
        self.target_update = int(os.getenv('TARGET_UPDATE', 1000))
        self.memory_capacity = int(os.getenv('MEMORY_CAPACITY', 10000))
        self.batch_size = int(os.getenv('BATCH_SIZE', 64))
        self.min_memory_size = int(os.getenv('MIN_MEMORY_SIZE', 1000))  # Added: minimum memory before training

        if policy_net is None:
            self.policy_net = DQN(device).to(device)
        else:
            self.policy_net = policy_net.to(device)
            self.policy_net.train()  # Set model to training mode

        self.opponent_net = DQN(device).to(device)
        self.opponent_net.load_state_dict(self.policy_net.state_dict())
        self.opponent_net.eval()  # Opponent net in evaluation mode

        self.target_net = DQN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.5)  # Added: learning rate scheduler
        self.memory = ReplayMemory(self.memory_capacity)

        self.steps_done = 0

        # Added: Training metrics tracking
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_loss = 0.0
        self.loss_count = 0
        self.best_win_rate = 0.0

    def optimize_model(self):
        # Fixed: use min_memory_size instead of batch_size for initial check
        if len(self.memory) < self.min_memory_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.stack(batch_state)
        batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(self.device)
        batch_next_state = torch.stack(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(batch_state).gather(1, batch_action)

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(batch_next_state).max(1)[0]
            next_state_values = next_state_values * (1 - batch_done)

        # Compute the expected Q values
        expected_state_action_values = batch_reward + (self.gamma * next_state_values)

        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Added: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Added: Return loss value for tracking
        return loss.item()

    def train(self):
        print(f"Starting training for {self.num_episodes} episodes...")
        print(f"Epsilon decay: {self.epsilon_decay}, Min memory: {self.min_memory_size}")
        print(f"Learning rate: {self.learning_rate}, Batch size: {self.batch_size}\n")

        for episode in range(self.num_episodes):
            game = TicTacToe()
            state = torch.FloatTensor(game.reset()).to(self.device)
            done = False
            episode_reward = 0

            while not done:
                # Calculate epsilon for this step
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-1. * self.steps_done / self.epsilon_decay)

                # AI's turn (Player 1)
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.choice(game.available_actions())
                else:
                    with torch.no_grad():
                        q_values = self.policy_net(state)
                        masked_q_values = q_values.clone()
                        for i in range(9):
                            if i not in game.available_actions():
                                masked_q_values[i] = -float('inf')
                        action = torch.argmax(masked_q_values).item()

                # Take action
                next_state_np, reward, done = game.step(action, 1)
                next_state = torch.FloatTensor(next_state_np).to(self.device)

                # Fixed: Store the state BEFORE opponent's move
                state_before_opponent = next_state.clone()

                # If not done, opponent's turn (Player -1)
                if not done:
                    # Opponent's action using opponent_net
                    with torch.no_grad():
                        q_values = self.opponent_net(next_state)
                        masked_q_values = q_values.clone()
                        for i in range(9):
                            if i not in game.available_actions():
                                masked_q_values[i] = -float('inf')
                        opponent_action = torch.argmax(masked_q_values).item()

                    next_state_np, opponent_reward, done = game.step(opponent_action, -1)
                    next_state = torch.FloatTensor(next_state_np).to(self.device)

                    # Fixed: Better reward shaping
                    if opponent_reward != 0:
                        reward = -1.0  # Loss: opponent wins
                    elif done:
                        reward = 0.5  # Draw after opponent's move
                elif done:
                    # Game ended on AI's move
                    if reward == 1:
                        reward = 1.0  # Win
                    else:
                        reward = 0.5  # Draw

                # Fixed: Store transition with correct next state
                # If opponent moved, next_state includes opponent's move
                # If game ended on AI move, state_before_opponent is used
                self.memory.push((state, action, reward, next_state, done))

                state = next_state
                self.steps_done += 1
                episode_reward += reward

                # Perform optimization
                loss = self.optimize_model()
                if loss is not None:
                    self.total_loss += loss
                    self.loss_count += 1

                # Update target network
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # Added: Track episode outcomes
            if episode_reward > 0.9:  # Win
                self.wins += 1
            elif episode_reward < 0:  # Loss
                self.losses += 1
            else:  # Draw
                self.draws += 1

            # Fixed: Update opponent network more frequently (every 100 episodes instead of 1000)
            if episode % 100 == 0 and episode != 0:
                self.opponent_net.load_state_dict(self.policy_net.state_dict())

            # Step the learning rate scheduler
            self.scheduler.step()

            # Added: Print detailed progress every 100 episodes
            if (episode + 1) % 100 == 0:
                games_played = self.wins + self.losses + self.draws
                win_rate = (self.wins / games_played * 100) if games_played > 0 else 0
                loss_rate = (self.losses / games_played * 100) if games_played > 0 else 0
                draw_rate = (self.draws / games_played * 100) if games_played > 0 else 0
                avg_loss = (self.total_loss / self.loss_count) if self.loss_count > 0 else 0

                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Win: {win_rate:.1f}% | Loss: {loss_rate:.1f}% | Draw: {draw_rate:.1f}% | "
                      f"Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

                # Added: Save best model based on win rate
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    torch.save(self.policy_net.state_dict(), 'tic_tac_toe_best_model.pth')
                    print(f"  → New best model saved! Win rate: {win_rate:.1f}%")

                # Reset metrics for next 100 episodes
                self.wins = 0
                self.losses = 0
                self.draws = 0
                self.total_loss = 0.0
                self.loss_count = 0

        # Save the final trained model
        torch.save(self.policy_net.state_dict(), 'tic_tac_toe_model.pth')
        print("\nTraining complete!")
        print(f"Final model saved as 'tic_tac_toe_model.pth'")
        print(f"Best model saved as 'tic_tac_toe_best_model.pth' (win rate: {self.best_win_rate:.1f}%)")

        return self.policy_net
