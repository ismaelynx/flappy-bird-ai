import pygame
import random
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pickle
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Game constants
WIDTH, HEIGHT = 400, 600
PIPE_WIDTH = 60
PIPE_GAP = 150
GRAVITY = 0.8
JUMP_STRENGTH = -12
FPS = 60
BIRD_SIZE = 15

# Set up matplotlib - FIX: Only one figure
plt.ion()
matplotlib.use('TkAgg')  # Use consistent backend
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class FlappyBirdEnv:
    def __init__(self, render_mode=True):
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Flappy Bird AI")
            self.clock = pygame.time.Clock()
        
        # Action space setup
        self.action_space = type('', (), {})()
        self.action_space.n = 2
        self.action_space.sample = lambda: random.randint(0, 1)
        
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.frame = 0
        self.done = False
        
        # Start with first pipe at reasonable distance
        self.spawn_pipe()
        return self.get_state()

    def spawn_pipe(self):
        # Ensure gap is in reasonable position
        gap_y = random.randint(PIPE_GAP, HEIGHT - PIPE_GAP - 100)
        self.pipes.append([WIDTH, gap_y])

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
            
        self.frame += 1
        reward = 0.1  # Small reward for staying alive

        # Handle bird physics
        if action == 1:  # Flap
            self.bird_vel = JUMP_STRENGTH
        
        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        # Move pipes
        for pipe in self.pipes:
            pipe[0] -= 3

        # Spawn new pipes
        if self.frame % 120 == 0:  # Spawn every 2 seconds at 60 FPS
            self.spawn_pipe()

        # Remove off-screen pipes and award points
        if self.pipes and self.pipes[0][0] < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.score += 1
            reward = 10  # Big reward for passing pipe

        # Check collisions
        if self.check_collision():
            self.done = True
            reward = -100  # Large penalty for dying

        # Additional reward for staying in good position
        if self.pipes:
            pipe_x, pipe_gap_y = self.pipes[0]
            distance_to_gap_center = abs(self.bird_y - (pipe_gap_y + PIPE_GAP // 2))
            if distance_to_gap_center < PIPE_GAP // 4:
                reward += 0.5  # Bonus for staying centered

        return self.get_state(), reward, self.done, {'score': self.score}

    def check_collision(self):
        # Check bounds
        if self.bird_y <= BIRD_SIZE or self.bird_y >= HEIGHT - BIRD_SIZE:
            return True
        
        # Check pipe collision
        bird_rect = pygame.Rect(50 - BIRD_SIZE, self.bird_y - BIRD_SIZE, 
                               BIRD_SIZE * 2, BIRD_SIZE * 2)
        
        for pipe_x, pipe_gap_y in self.pipes:
            # Top pipe
            top_pipe = pygame.Rect(pipe_x, 0, PIPE_WIDTH, pipe_gap_y)
            # Bottom pipe
            bottom_pipe = pygame.Rect(pipe_x, pipe_gap_y + PIPE_GAP, 
                                    PIPE_WIDTH, HEIGHT - pipe_gap_y - PIPE_GAP)
            
            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                return True
        
        return False

    def get_state(self):
        if not self.pipes:
            return np.array([0.5, 0, 1, 0], dtype=np.float32)
        
        # Get the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe[0] > 50 - PIPE_WIDTH:  # Bird is at x=50
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = self.pipes[0]
        
        pipe_x, pipe_gap_y = next_pipe
        
        # Normalize state values
        state = np.array([
            self.bird_y / HEIGHT,                    # Bird height (0-1)
            (self.bird_vel + 15) / 30,              # Bird velocity (-1 to 1)
            pipe_x / WIDTH,                          # Pipe distance (0-1)
            (pipe_gap_y + PIPE_GAP//2 - self.bird_y) / HEIGHT  # Distance to gap center
        ], dtype=np.float32)
        
        return state

    def render(self):
        if not self.render_mode:
            return True
            
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen
        self.screen.fill((135, 206, 235))  # Sky blue
        
        # Draw pipes
        for pipe_x, pipe_gap_y in self.pipes:
            # Top pipe
            pygame.draw.rect(self.screen, (0, 128, 0), 
                           (pipe_x, 0, PIPE_WIDTH, pipe_gap_y))
            # Bottom pipe
            pygame.draw.rect(self.screen, (0, 128, 0), 
                           (pipe_x, pipe_gap_y + PIPE_GAP, PIPE_WIDTH, 
                            HEIGHT - pipe_gap_y - PIPE_GAP))
            # Pipe caps
            pygame.draw.rect(self.screen, (0, 100, 0), 
                           (pipe_x - 5, pipe_gap_y - 20, PIPE_WIDTH + 10, 20))
            pygame.draw.rect(self.screen, (0, 100, 0), 
                           (pipe_x - 5, pipe_gap_y + PIPE_GAP, PIPE_WIDTH + 10, 20))
        
        # Draw bird
        bird_color = (255, 255, 0) if not self.done else (255, 0, 0)
        pygame.draw.circle(self.screen, bird_color, (50, int(self.bird_y)), BIRD_SIZE)
        pygame.draw.circle(self.screen, (0, 0, 0), (50, int(self.bird_y)), BIRD_SIZE, 2)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def close(self):
        if self.render_mode:
            pygame.quit()

# Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Experience Replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.005
LR = 0.0001

# Initialize environment and networks
env = FlappyBirdEnv(render_mode=True)
n_actions = env.action_space.n
n_observations = 4

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(50000)

# Training variables
steps_done = 0
episode_durations = []
episode_rewards = []
recent_scores = deque(maxlen=100)

# FIX: Single figure for all plots
fig = None
plot_initialized = False

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                 device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def plot_progress():
    global fig, plot_initialized
    
    # FIX: Use single figure
    if not plot_initialized:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Flappy Bird AI Training Progress')
        plot_initialized = True
    else:
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        fig.suptitle('Flappy Bird AI Training Progress')
    
    # Plot episode durations
    ax1.set_title('Episode Durations')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration (Steps)')
    if episode_durations:
        ax1.plot(episode_durations, 'b-', alpha=0.7)
        if len(episode_durations) >= 100:
            durations_t = torch.tensor(episode_durations, dtype=torch.float)
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax1.plot(means.numpy(), 'r-', alpha=0.8, linewidth=2, label='100-episode average')
            ax1.legend()
    
    # Plot episode rewards
    ax2.set_title('Episode Rewards')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    if episode_rewards:
        ax2.plot(episode_rewards, 'g-', alpha=0.7)
        if len(episode_rewards) >= 100:
            rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax2.plot(means.numpy(), 'r-', alpha=0.8, linewidth=2, label='100-episode average')
            ax2.legend()
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

def save_model(episode, score, suffix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flappy_model_{timestamp}{suffix}.pth"
    
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'steps_done': steps_done,
        'episode_durations': episode_durations,
        'episode_rewards': episode_rewards,
        'recent_scores': list(recent_scores),
        'hyperparameters': {
            'BATCH_SIZE': BATCH_SIZE,
            'GAMMA': GAMMA,
            'EPS_START': EPS_START,
            'EPS_END': EPS_END,
            'EPS_DECAY': EPS_DECAY,
            'TAU': TAU,
            'LR': LR,
        }
    }, filename)
    
    # Also save as latest
        # Also save as latest
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'steps_done': steps_done,
    }, "flappy_model_latest.pth")
    
    print(f"Model saved: {filename} (Episode {episode}, Score {score})")

def load_model():
    global steps_done, episode_durations, episode_rewards, recent_scores
    
    if os.path.exists("flappy_model_latest.pth"):
        checkpoint = torch.load("flappy_model_latest.pth", map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        steps_done = checkpoint.get('steps_done', 0)
        
        # Load training progress if available
        episode_durations = checkpoint.get('episode_durations', [])
        episode_rewards = checkpoint.get('episode_rewards', [])
        recent_scores = deque(checkpoint.get('recent_scores', []), maxlen=100)
        
        start_episode = checkpoint.get('episode', 0)
        print(f"✅ Model loaded from flappy_model_latest.pth (Episode {start_episode})")
        return start_episode
    else:
        print("🆕 No saved model found, starting from scratch")
        return 0

# Training function
def train_agent(num_episodes=2000, render_every=10, save_every=100):
    global steps_done, fig, plot_initialized
    
    start_episode = load_model()
    best_score = max(recent_scores) if recent_scores else 0
    
    print(f"🚀 Starting training from episode {start_episode}")
    print(f"📊 Best score so far: {best_score}")
    
    try:
        for i_episode in range(start_episode, start_episode + num_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            total_reward = 0
            episode_steps = 0
            
            for t in count():
                # Select and perform action
                action = select_action(state)
                observation, reward, done, info = env.step(action.item())
                
                total_reward += reward
                episode_steps += 1
                
                # Render occasionally
                if i_episode % render_every == 0:
                    if not env.render():  # Returns False if window closed
                        print("Window closed, stopping training...")
                        return
                
                reward_tensor = torch.tensor([reward], device=device)
                
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, 
                                            device=device).unsqueeze(0)
                
                # Store transition in memory
                memory.push(state, action, next_state, reward_tensor)
                state = next_state
                
                # Optimize model
                optimize_model()
                
                # Soft update target network
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (policy_net_state_dict[key] * TAU + 
                                                target_net_state_dict[key] * (1 - TAU))
                target_net.load_state_dict(target_net_state_dict)
                
                if done:
                    episode_durations.append(episode_steps)
                    episode_rewards.append(total_reward)
                    recent_scores.append(info['score'])
                    break
            
            # Print progress
            current_score = info['score']
            avg_score = np.mean(recent_scores) if recent_scores else 0
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            
            if i_episode % 10 == 0:
                print(f"Episode {i_episode:4d} | Score: {current_score:2d} | "
                      f"Reward: {total_reward:6.1f} | Steps: {episode_steps:3d} | "
                      f"Avg Score: {avg_score:.1f} | Epsilon: {eps_threshold:.3f}")
            
            # Plot progress - FIX: Only update every 50 episodes to reduce overhead
            if i_episode % 50 == 0 and i_episode > 0:
                plot_progress()
            
            # Save model
            if i_episode % save_every == 0 and i_episode > 0:
                save_model(i_episode, current_score)
            
            # Save best model
            if current_score > best_score:
                best_score = current_score
                save_model(i_episode, current_score, "_best")
                print(f"🎉 New best score: {best_score}!")
            
            # Early stopping if performing well
            if len(recent_scores) >= 100 and np.mean(recent_scores) > 50:
                print(f"🏆 Training completed! Average score over last 100 episodes: {np.mean(recent_scores):.1f}")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
    finally:
        # Save final model
        save_model(i_episode, info.get('score', 0), "_final")
        # FIX: Close plot properly
        if plot_initialized and fig:
            plt.close(fig)
        env.close()

# Evaluation function
def evaluate_agent(num_episodes=10, render=True):
    """Evaluate the trained agent"""
    load_model()
    
    eval_env = FlappyBirdEnv(render_mode=render)
    policy_net.eval()
    
    scores = []
    
    print("🎮 Evaluating agent...")
    
    try:
        for episode in range(num_episodes):
            state = eval_env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            total_reward = 0
            steps = 0
            
            while True:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
                
                observation, reward, done, info = eval_env.step(action.item())
                total_reward += reward
                steps += 1
                
                if render:
                    if not eval_env.render():
                        break
                
                if done:
                    scores.append(info['score'])
                    print(f"Episode {episode + 1}: Score = {info['score']}, "
                          f"Reward = {total_reward:.1f}, Steps = {steps}")
                    break
                
                state = torch.tensor(observation, dtype=torch.float32, 
                                   device=device).unsqueeze(0)
    
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
    finally:
        eval_env.close()
    
    if scores:
        print(f"\n📊 Evaluation Results:")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Best Score: {max(scores)}")
        print(f"Worst Score: {min(scores)}")
        print(f"Standard Deviation: {np.std(scores):.2f}")
    
    return scores

# Play manually function
def play_manually():
    """Let human play the game"""
    manual_env = FlappyBirdEnv(render_mode=True)
    
    print("🎮 Manual Play Mode")
    print("Press SPACE to flap, ESC to quit")
    
    state = manual_env.reset()
    
    try:
        while True:
            action = 0  # Default: don't flap
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1  # Flap
                    elif event.key == pygame.K_ESCAPE:
                        return
            
            observation, reward, done, info = manual_env.step(action)
            
            if not manual_env.render():
                break
            
            if done:
                print(f"Game Over! Final Score: {info['score']}")
                
                # Wait for restart or quit
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                state = manual_env.reset()
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                return
                    
                    # Show game over screen
                    manual_env.screen.fill((0, 0, 0))
                    font = pygame.font.Font(None, 48)
                    game_over_text = font.render("Game Over!", True, (255, 255, 255))
                    score_text = font.render(f"Score: {info['score']}", True, (255, 255, 255))
                    restart_text = font.render("Press SPACE to restart, ESC to quit", True, (255, 255, 255))
                    
                    manual_env.screen.blit(game_over_text, (WIDTH//2 - 100, HEIGHT//2 - 100))
                    manual_env.screen.blit(score_text, (WIDTH//2 - 80, HEIGHT//2 - 50))
                    manual_env.screen.blit(restart_text, (WIDTH//2 - 200, HEIGHT//2))
                    
                    pygame.display.flip()
                    manual_env.clock.tick(30)
    
    except KeyboardInterrupt:
        print("\n⏹️ Manual play interrupted")
    finally:
        manual_env.close()

# Main execution
if __name__ == "__main__":
    print("🐦 Flappy Bird AI")
    print("=" * 40)
    print("1. Train AI")
    print("2. Evaluate AI") 
    print("3. Play Manually")
    print("4. Train then Evaluate")
    
    try:
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == "1":
            print("\n🏋️ Starting Training...")
            train_agent(num_episodes=2000, render_every=20, save_every=200)
        
        elif choice == "2":
            print("\n🎯 Starting Evaluation...")
            evaluate_agent(num_episodes=5, render=True)
        
        elif choice == "3":
            print("\n🎮 Manual Play Mode...")
            play_manually()
        
        elif choice == "4":
            print("\n🏋️ Starting Training...")
            train_agent(num_episodes=1000, render_every=50, save_every=200)
            print("\n🎯 Starting Evaluation...")
            evaluate_agent(num_episodes=3, render=True)
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n⏹️ Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # FIX: Ensure proper cleanup
        plt.close('all')  # Close all matplotlib figures
        pygame.quit()
        print("\n✅ Program ended cleanly")
