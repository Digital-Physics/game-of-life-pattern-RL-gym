#!/usr/bin/env python3
"""
main.py

Unified interface for the GoL-2x2-v0 environment.

Usage:
    # Test the environment
    python main.py test --mode random --episodes 3 --render
    python main.py test --mode greedy --episodes 5 --render
    python main.py test --mode manual
    
    # Train an agent
    python main.py train --episodes 100 --save-path my_agent.pth --render
    
    # Evaluate a trained agent
    python main.py eval --load-path my_agent.pth --episodes 10 --render
    
    # Quick benchmark
    python main.py benchmark
"""

import argparse
import numpy as np
import gymnasium as gym
import rl_GoL_env_gym
import torch
import os
import sys
from tqdm import tqdm

# Configuration
DEFAULT_PATTERN_FILE = 'target_patterns.json'
DEFAULT_AGENT_PATH = 'agent.pth'
GRID_SIZE = 12
MAX_STEPS = 10
OBS_SIZE = GRID_SIZE * GRID_SIZE * 3 + 1  # grid + target + head_mask + remaining_steps
ACTION_SIZE = 21
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# AGENT DEFINITION
# ============================================================================

class SimpleAgent:
    """Base agent class - replace with your own implementation."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.policy = torch.nn.Linear(OBS_SIZE, action_space.n).to(DEVICE)
    
    def select_action(self, obs: dict) -> int:
        """Select action given observation dictionary."""
        # Simple random policy - replace with your trained policy
        return self.action_space.sample()
    
    def save(self, path: str):
        """Save agent state."""
        torch.save(self.policy.state_dict(), path)
        print(f"✓ Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent file not found: {path}")
        state_dict = torch.load(path, map_location=DEVICE)
        self.policy.load_state_dict(state_dict)
        print(f"✓ Agent loaded from {path}")


# ============================================================================
# TEST POLICIES
# ============================================================================

def run_random_policy(env, episodes=5, render=False):
    """Test with random actions."""
    print("\n=== Random Policy ===")
    all_rewards = []
    all_accuracies = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        target = obs['target']
        print(f"\nEpisode {ep + 1}/{episodes} | Target density: {target.mean():.3f}")
        
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            if terminated:
                print(f"  Accuracy: {info['accuracy']:.3f} | Reward: {reward:.3f} | "
                      f"Matches: {info['matches']}/{GRID_SIZE*GRID_SIZE}")
                all_accuracies.append(info['accuracy'])
        
        all_rewards.append(episode_reward)
    
    print(f"\n=== Results ===")
    print(f"Avg reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Avg accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")


def run_greedy_policy(env, episodes=5, render=False):
    """Test with greedy pattern-matching policy."""
    print("\n=== Greedy Policy ===")
    all_rewards = []
    all_accuracies = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        target = obs['target']
        print(f"\nEpisode {ep + 1}/{episodes} | Target density: {target.mean():.3f}")
        
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            current_grid = obs['grid']
            head_mask = obs['head_mask']
            
            # Find head position
            head_pos = np.where(head_mask == 1)
            head_r, head_c = head_pos[0][0], head_pos[1][0]
            
            # Extract 2×2 regions
            target_2x2 = []
            current_2x2 = []
            for dr in range(2):
                for dc in range(2):
                    r = (head_r + dr) % env.unwrapped.grid_size
                    c = (head_c + dc) % env.unwrapped.grid_size
                    target_2x2.append(target[r, c])
                    current_2x2.append(current_grid[r, c])
            
            # Decide action
            target_pattern_id = sum(bit << i for i, bit in enumerate(target_2x2))
            current_matches = sum(c == t for c, t in zip(current_2x2, target_2x2))
            
            if 4 > current_matches:  # Write if it improves
                action = 5 + target_pattern_id
            else:
                action = np.random.choice([0, 1, 2, 3, 4])  # Move/pass
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            if terminated:
                print(f"  Accuracy: {info['accuracy']:.3f} | Reward: {reward:.3f} | "
                      f"Matches: {info['matches']}/{GRID_SIZE*GRID_SIZE}")
                all_accuracies.append(info['accuracy'])
        
        all_rewards.append(episode_reward)
    
    print(f"\n=== Results ===")
    print(f"Avg reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Avg accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")


def run_manual_control(env):
    """Interactive manual control."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    # Disable matplotlib key bindings
    for key in ['fullscreen', 'home', 'back', 'forward', 'pan', 'zoom', 
                'save', 'quit', 'grid', 'yscale', 'xscale']:
        mpl.rcParams[f'keymap.{key}'] = []
    
    print("\n=== Manual Control ===")
    print("Controls:")
    print("  Arrow keys: Move write head")
    print("  Space: Pass")
    print("  0-9, a-f: Write pattern (hex)")
    print("  R: Reset")
    print("  Q/Esc: Quit")
    
    obs, info = env.reset()
    env.render()
    
    state = {'obs': obs, 'info': info, 'target': obs['target'], 'finished': False}
    
    def on_key(event):
        if event.key is None:
            return
        
        key = event.key.lower()
        
        if key in ['q', 'escape']:
            plt.close()
            return
        
        if key == 'r':
            state['obs'], state['info'] = env.reset()
            state['target'] = state['obs']['target']
            state['finished'] = False
            env.render()
            print("\nEpisode reset")
            return
        
        if state['finished']:
            return
        
        # Map keys to actions
        action_map = {
            'up': (0, '↑'), 'down': (1, '↓'), 
            'left': (2, '←'), 'right': (3, '→'), 
            ' ': (4, '○')
        }
        
        if key in action_map:
            action, name = action_map[key]
        elif key in '0123456789abcdef':
            pattern_id = int(key, 16)
            action = 5 + pattern_id
            name = f'{pattern_id:X}'
        else:
            return
        
        obs, reward, terminated, truncated, info = env.step(action)
        state['obs'] = obs
        state['info'] = info
        env.render()
        
        if terminated:
            state['finished'] = True
            print(f"\nComplete! Accuracy: {info['accuracy']:.3f} | "
                  f"Reward: {reward:.3f}")
            print("Press R to reset or Q to quit")
        else:
            print(f"Step {info['step']}: {name} | Matches: {info['matches']}/144 | "
                  f"Remaining: {obs['remaining_steps'][0]}")
    
    env.unwrapped._fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_agent(env, agent, episodes=100, render=False):
    """Train agent (placeholder - implement your training loop)."""
    print(f"\n=== Training Agent ({episodes} episodes) ===")
    
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        terminated = truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
        
        if ep % 10 == 0 or ep == episodes:
            print(f"Episode {ep}/{episodes} | Reward: {episode_reward:.3f} | "
                  f"Accuracy: {info['accuracy']:.3f}")
    
    print("✓ Training complete")


def evaluate_agent(env, agent, episodes=10, render=False):
    """Evaluate trained agent."""
    print(f"\n=== Evaluating Agent ({episodes} episodes) ===")
    all_rewards = []
    all_accuracies = []
    
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        terminated = truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
        
        all_rewards.append(episode_reward)
        all_accuracies.append(info['accuracy'])
        print(f"Episode {ep}/{episodes} | Reward: {episode_reward:.3f} | "
              f"Accuracy: {info['accuracy']:.3f}")
    
    print(f"\n=== Results ===")
    print(f"Avg reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Avg accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark():
    """Quick environment verification."""
    print("\n=== Environment Benchmark ===")
    
    env = gym.make("GoL-2x2-v0", grid_size=GRID_SIZE, max_steps=MAX_STEPS,
                   pattern_file=DEFAULT_PATTERN_FILE)
    
    # Test reset
    obs, info = env.reset(seed=42)
    print("✓ Reset successful")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Grid shape: {obs['grid'].shape}")
    print(f"  Target shape: {obs['target'].shape}")
    print(f"  Head mask shape: {obs['head_mask'].shape}")
    print(f"  Remaining steps: {obs['remaining_steps']}")
    
    # Test all actions
    print("\n✓ Testing all 21 actions...")
    for action in tqdm(range(21), desc="Actions"):
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs['grid'].shape == (GRID_SIZE, GRID_SIZE)
        assert 0.0 <= reward <= 1.0
    
    # Test full episode
    print("\n✓ Testing full episode...")
    obs, info = env.reset(seed=42)
    for step in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs['remaining_steps'][0] == MAX_STEPS - (step + 1)
        
        if step < MAX_STEPS - 1:
            assert reward == 0.0
            assert not terminated
        else:
            assert terminated
    
    env.close()
    print("\n✓ All tests passed!")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified interface for GoL-2x2-v0 environment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # TEST command
    test_parser = subparsers.add_parser('test', help='Test environment with policies')
    test_parser.add_argument('--mode', choices=['random', 'greedy', 'manual'],
                            default='random', help='Test policy')
    test_parser.add_argument('--episodes', type=int, default=5,
                            help='Number of episodes')
    test_parser.add_argument('--render', action='store_true',
                            help='Enable rendering')
    
    # TRAIN command
    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--episodes', type=int, default=100,
                             help='Number of training episodes')
    train_parser.add_argument('--save-path', type=str, default=DEFAULT_AGENT_PATH,
                             help='Path to save trained agent')
    train_parser.add_argument('--render', action='store_true',
                             help='Enable rendering')
    
    # EVAL command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained agent')
    eval_parser.add_argument('--load-path', type=str, default=DEFAULT_AGENT_PATH,
                            help='Path to load agent')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--render', action='store_true',
                            help='Enable rendering')
    
    # BENCHMARK command
    subparsers.add_parser('benchmark', help='Run environment benchmark')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'benchmark':
        run_benchmark()
    
    elif args.command == 'test':
        render_mode = "human" if args.render else None
        env = gym.make("GoL-2x2-v0", grid_size=GRID_SIZE, max_steps=MAX_STEPS,
                      pattern_file=DEFAULT_PATTERN_FILE, render_mode=render_mode)
        
        if args.mode == 'random':
            run_random_policy(env, args.episodes, args.render)
        elif args.mode == 'greedy':
            run_greedy_policy(env, args.episodes, args.render)
        elif args.mode == 'manual':
            run_manual_control(env)
        
        env.close()
    
    elif args.command == 'train':
        render_mode = "human" if args.render else None
        env = gym.make("GoL-2x2-v0", grid_size=GRID_SIZE, max_steps=MAX_STEPS,
                      pattern_file=DEFAULT_PATTERN_FILE, render_mode=render_mode)
        
        agent = SimpleAgent(env.action_space)
        train_agent(env, agent, args.episodes, args.render)
        agent.save(args.save_path)
        env.close()
    
    elif args.command == 'eval':
        render_mode = "human" if args.render else None
        env = gym.make("GoL-2x2-v0", grid_size=GRID_SIZE, max_steps=MAX_STEPS,
                      pattern_file=DEFAULT_PATTERN_FILE, render_mode=render_mode)
        
        agent = SimpleAgent(env.action_space)
        try:
            agent.load(args.load_path)
            evaluate_agent(env, agent, args.episodes, args.render)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run 'python main.py train' first to create an agent")
        
        env.close()


if __name__ == "__main__":
    main()