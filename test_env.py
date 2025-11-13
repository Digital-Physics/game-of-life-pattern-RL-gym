#!/usr/bin/env python3
"""
test_env.py

Test the GoL-2x2-v0 environment with different algorithms:
1. Random policy
2. Greedy pattern matching policy
3. Interactive manual control

Usage:
    python test_env.py --mode random --episodes 3 --render
    python test_env.py --mode greedy --episodes 5 --render
    python test_env.py --mode manual
"""

import argparse
import numpy as np
import gymnasium as gym
import rl_GoL_env_gym  # Register the environment
from tqdm import tqdm

DEFAULT_PATTERN_FILE = 'target_patterns.json' 

def random_policy_test(episodes=5, render=False):
    """Test with random actions, loading the default pattern file."""
    print("\n=== Testing Random Policy ===")
    
    render_mode = "human" if render else None
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, 
                   pattern_file=DEFAULT_PATTERN_FILE, render_mode=render_mode)
    
    all_rewards = []
    all_accuracies = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        # --- MODIFIED: Get target from obs ---
        target = obs['target']
        # ------------------------------------
        print(f"\nEpisode {ep + 1}/{episodes}")
        print(f"Target density: {target.mean():.3f}")
        print(f"Remaining steps: {obs['remaining_steps'][0]}")
        
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            if terminated:
                print(f"  Final accuracy: {info['accuracy']:.3f}")
                print(f"  Final fitness: {reward:.3f}")
                print(f"  Matches: {info['matches']}/{12*12}")
                all_accuracies.append(info['accuracy'])
        
        all_rewards.append(episode_reward)
        
        if render and ep < episodes - 1:
            import time
            time.sleep(1.0)  # Pause between episodes
    
    env.close()
    
    print(f"\n=== Random Policy Results ===")
    print(f"Average reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Average accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")


def greedy_policy_test(episodes=5, render=False):
    """
    Greedy policy: Look at target, write patterns that maximize matches.
    This is a simple heuristic, not RL.
    """
    print("\n=== Testing Greedy Policy ===")
    
    render_mode = "human" if render else None
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, 
                   pattern_file=DEFAULT_PATTERN_FILE, render_mode=render_mode)
    
    all_rewards = []
    all_accuracies = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        # --- MODIFIED: Get target from obs ---
        target = obs['target']
        # ------------------------------------
        print(f"\nEpisode {ep + 1}/{episodes}")
        print(f"Target density: {target.mean():.3f}")
        
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Extract current state from dictionary observation
            current_grid = obs['grid']
            head_mask = obs['head_mask']
            remaining_steps = obs['remaining_steps'][0]
            
            # Find head position
            head_pos = np.where(head_mask == 1)
            head_r, head_c = head_pos[0][0], head_pos[1][0]
            
            # Greedy strategy: look at 2×2 target area under write head
            target_2x2 = []
            for dr in range(2):
                for dc in range(2):
                    r = (head_r + dr) % env.unwrapped.grid_size
                    c = (head_c + dc) % env.unwrapped.grid_size
                    target_2x2.append(target[r, c])
            
            # Convert to pattern ID
            target_pattern_id = sum(bit << i for i, bit in enumerate(target_2x2))
            
            # Check if we should write or move
            current_2x2 = []
            for dr in range(2):
                for dc in range(2):
                    r = (head_r + dr) % env.unwrapped.grid_size
                    c = (head_c + dc) % env.unwrapped.grid_size
                    current_2x2.append(current_grid[r, c])
            
            current_matches = sum(c == t for c, t in zip(current_2x2, target_2x2))
            target_matches = 4  # Perfect match if we write
            
            # Write if it improves matches, otherwise move randomly
            if target_matches > current_matches:
                action = 5 + target_pattern_id  # Write pattern
            else:
                action = np.random.choice([0, 1, 2, 3, 4])  # Move or pass
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            if terminated:
                print(f"  Final accuracy: {info['accuracy']:.3f}")
                print(f"  Final fitness: {reward:.3f}")
                print(f"  Matches: {info['matches']}/{12*12}")
                all_accuracies.append(info['accuracy'])
        
        all_rewards.append(episode_reward)
        
        if render and ep < episodes - 1:
            import time
            time.sleep(1.0)  # Pause between episodes
    
    env.close()
    
    print(f"\n=== Greedy Policy Results ===")
    print(f"Average reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Average accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")


def manual_control():
    """
    Interactive manual control with integrated rendering.
    
    Controls:
        Arrow keys: Move write head
        Space: Pass (do nothing)
        0-9, a-f: Write hex pattern (0x0 to 0xF)
        R: Reset
        Q: Quit
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    # CRITICAL: Disable ALL default matplotlib key bindings 
    mpl.rcParams['keymap.fullscreen'] = []
    mpl.rcParams['keymap.home'] = []
    mpl.rcParams['keymap.back'] = []
    mpl.rcParams['keymap.forward'] = []
    mpl.rcParams['keymap.pan'] = []
    mpl.rcParams['keymap.zoom'] = []
    mpl.rcParams['keymap.save'] = []
    mpl.rcParams['keymap.quit'] = []
    mpl.rcParams['keymap.grid'] = []
    mpl.rcParams['keymap.yscale'] = []
    mpl.rcParams['keymap.xscale'] = []
    
    print("\n=== Manual Control Mode ===")
    print("Controls:")
    print("  Arrow keys: Move write head")
    print("  Space: Pass")
    print("  0-9, a-f: Write pattern (hex)")
    print("  R: Reset episode")
    print("  Q/Esc: Quit")
    
    # Create environment with human rendering
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, render_mode="human")
    obs, info = env.reset()
    # --- MODIFIED: Get target from obs ---
    target = obs['target']
    # ------------------------------------
    
    # Initial render
    env.render()
    
    # State tracking
    state = {'obs': obs, 'info': info, 'target': target, 'finished': False}
    
    def on_key(event):
        if event.key is None:
            return
        
        key = event.key.lower()
        
        # Handle quit
        if key == 'q' or key == 'escape':
            plt.close()
            return
        
        # Handle reset
        if key == 'r':
            state['obs'], state['info'] = env.reset()
            # --- MODIFIED: Get target from obs after reset ---
            state['target'] = state['obs']['target']
            # ------------------------------------------------
            state['finished'] = False
            env.render()
            print("\nEpisode reset")
            return
        
        # Don't accept actions if episode is finished
        if state['finished']:
            return
        
        # Map keys to actions
        action = None
        action_name = None
        
        if key == 'up':
            action = 0
            action_name = '↑'
        elif key == 'down':
            action = 1
            action_name = '↓'
        elif key == 'left':
            action = 2
            action_name = '←'
        elif key == 'right':
            action = 3
            action_name = '→'
        elif key == ' ':
            action = 4
            action_name = '○'
        elif key in '0123456789abcdef':
            pattern_id = int(key, 16)
            action = 5 + pattern_id
            action_name = f'{pattern_id:X}'
        
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            state['obs'] = obs
            state['info'] = info
            
            env.render()
            
            if terminated:
                state['finished'] = True
                print(f"\nEpisode complete!")
                print(f"Final accuracy: {info['accuracy']:.3f}")
                print(f"Final fitness: {reward:.3f}")
                print(f"Press R to reset or Q to quit")
            else:
                print(f"Step {info['step']}: Action {action_name}, Matches {info['matches']}/144, Remaining: {obs['remaining_steps'][0]}")
    
    # Connect event handler
    env.unwrapped._fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show plot
    plt.show(block=True)
    
    env.close()


def benchmark_environment():
    """Quick benchmark to verify environment is working."""
    print("\n=== Environment Benchmark ===")
    
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, 
                   pattern_file=DEFAULT_PATTERN_FILE) 
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation type: {type(obs)}")
    print(f"  Observation keys: {obs.keys()}")
    print(f"  Grid shape: {obs['grid'].shape}")
    print(f"  Head mask shape: {obs['head_mask'].shape}")
    print(f"  Remaining steps: {obs['remaining_steps']}")
    # --- MODIFIED: Check target shape in obs ---
    print(f"  Target shape: {obs['target'].shape}")
    # -------------------------------------------
    
    # Test all action types
    print(f"\n✓ Testing all 21 actions...")
    for action in tqdm(range(21), desc="Actions"):
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs['grid'].shape == (12, 12)
        assert obs['head_mask'].shape == (12, 12)
        assert obs['remaining_steps'].shape == (1,)
        # Optional: check target is still there
        assert obs['target'].shape == (12, 12)
        assert 0.0 <= reward <= 1.0
    
    # Test full episode
    print(f"\n✓ Testing full episode...")
    obs, info = env.reset(seed=42)
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify remaining_steps is correct
        assert obs['remaining_steps'][0] == 10 - (step + 1), \
            f"Expected {10 - (step + 1)} remaining steps, got {obs['remaining_steps'][0]}"
        
        if step < 9:
            assert reward == 0.0, f"Expected 0 reward at step {step}, got {reward}"
            assert not terminated
        else:
            assert terminated
            assert 0.0 <= reward <= 1.0
    
    env.close()
    print(f"\n✓ All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GoL-2x2-v0 environment")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["random", "greedy", "manual", "benchmark"],
        default="benchmark",
        help="Test mode"
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    
    args = parser.parse_args()
    
    if args.mode == "random":
        random_policy_test(episodes=args.episodes, render=args.render)
    elif args.mode == "greedy":
        greedy_policy_test(episodes=args.episodes, render=args.render)
    elif args.mode == "manual":
        manual_control()
    elif args.mode == "benchmark":
        benchmark_environment()