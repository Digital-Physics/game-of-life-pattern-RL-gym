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
    
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, pattern_file=DEFAULT_PATTERN_FILE)
    
    all_rewards = []
    all_accuracies = []
    
    if render:
        import matplotlib.pyplot as plt
        # Disable default key bindings
        for key in ['keymap.fullscreen', 'keymap.home', 'keymap.back', 'keymap.forward',
                    'keymap.pan', 'keymap.zoom', 'keymap.save', 'keymap.quit',
                    'keymap.grid', 'keymap.yscale', 'keymap.xscale']:
            plt.rcParams[key] = []
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax_current, ax_target, ax_info = axes
    
    for ep in range(episodes):
        obs, info = env.reset()
        target = info['target']
        print(f"\nEpisode {ep + 1}/{episodes}")
        print(f"Target density: {target.mean():.3f}")
        
        if render:
            # Setup visualization
            ax_current.clear()
            ax_target.clear()
            ax_info.clear()
            
            im_current = ax_current.imshow(obs[:, :, 0], cmap='binary', vmin=0, vmax=1)
            ax_current.set_title('Current Grid (Step 0/10)')
            ax_current.set_xticks([])
            ax_current.set_yticks([])
            
            ax_target.imshow(target, cmap='binary', vmin=0, vmax=1)
            ax_target.set_title('Target Pattern')
            ax_target.set_xticks([])
            ax_target.set_yticks([])
            
            ax_info.axis('off')
            info_text = ax_info.text(0.1, 0.5, "", fontsize=12, verticalalignment='center', fontfamily='monospace')
            
            head_patches = []
            plt.ion()
            plt.show()
        
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                # Update display
                im_current.set_data(obs[:, :, 0])
                
                # Remove old head patches
                for patch in head_patches:
                    patch.remove()
                head_patches.clear()
                
                # Draw write head
                head_mask = obs[:, :, 1]
                head_pos = np.where(head_mask == 1)
                if len(head_pos[0]) > 0:
                    r, c = head_pos[0][0], head_pos[1][0]
                    for dr in range(2):
                        for dc in range(2):
                            rect = plt.Rectangle(
                                (c + dc - 0.5, r + dr - 0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=2
                            )
                            ax_current.add_patch(rect)
                            head_patches.append(rect)
                
                # Update info panel
                action_names = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
                info_str = (
                    f"Episode: {ep + 1}/{episodes}\n"
                    f"Step: {info['step']}/10\n"
                    f"Action: {action_names[action]}\n"
                    f"Matches: {info['matches']}/144\n"
                    f"Accuracy: {info['accuracy']:.1%}\n"
                    f"Reward: {reward:.3f}\n"
                )
                if terminated:
                    info_str += f"\n{'='*20}\n"
                    info_str += f"FINAL FITNESS: {reward:.3f}\n"
                    info_str += f"{'='*20}"
                
                info_text.set_text(info_str)
                ax_current.set_title(f"Current Grid (Step {info['step']}/10)")
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)
            
            if terminated:
                print(f"  Final accuracy: {info['accuracy']:.3f}")
                print(f"  Final fitness: {reward:.3f}")
                print(f"  Matches: {info['matches']}/{12*12}")
                all_accuracies.append(info['accuracy'])
        
        all_rewards.append(episode_reward)
        
        if render and ep < episodes - 1:
            plt.pause(1.0)  # Pause between episodes
    
    if render:
        plt.ioff()
        plt.show(block=True)
    
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
    
    # Use the default pattern file defined at the top
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, pattern_file=DEFAULT_PATTERN_FILE)
    
    all_rewards = []
    all_accuracies = []
    
    if render:
        import matplotlib.pyplot as plt
        # Disable default key bindings
        for key in ['keymap.fullscreen', 'keymap.home', 'keymap.back', 'keymap.forward',
                    'keymap.pan', 'keymap.zoom', 'keymap.save', 'keymap.quit',
                    'keymap.grid', 'keymap.yscale', 'keymap.xscale']:
        
            plt.rcParams[key] = []
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax_current, ax_target, ax_info = axes
        plt.ion()
        plt.show()
    
    for ep in range(episodes):
        obs, info = env.reset()
        target = info['target']
        print(f"\nEpisode {ep + 1}/{episodes}")
        print(f"Target density: {target.mean():.3f}")
        
        if render:
            # Setup visualization
            ax_current.clear()
            ax_target.clear()
            ax_info.clear()
            
            im_current = ax_current.imshow(obs[:, :, 0], cmap='binary', vmin=0, vmax=1)
            ax_current.set_title('Current Grid (Step 0/10)')
            ax_current.set_xticks([])
            ax_current.set_yticks([])
            
            ax_target.imshow(target, cmap='binary', vmin=0, vmax=1)
            ax_target.set_title('Target Pattern')
            ax_target.set_xticks([])
            ax_target.set_yticks([])
            
            ax_info.axis('off')
            info_text = ax_info.text(0.1, 0.5, "", fontsize=12, verticalalignment='center', fontfamily='monospace')
            
            head_patches = []
            plt.ion()
            plt.show()
        
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Extract current state
            current_grid = obs[:, :, 0]
            head_mask = obs[:, :, 1]
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
                # Update display
                im_current.set_data(obs[:, :, 0])
                
                # Remove old head patches
                for patch in head_patches:
                    patch.remove()
                head_patches.clear()
                
                # Draw write head
                head_mask = obs[:, :, 1]
                head_pos = np.where(head_mask == 1)
                if len(head_pos[0]) > 0:
                    r, c = head_pos[0][0], head_pos[1][0]
                    for dr in range(2):
                        for dc in range(2):
                            rect = plt.Rectangle(
                                (c + dc - 0.5, r + dr - 0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=2
                            )
                            ax_current.add_patch(rect)
                            head_patches.append(rect)
                
                # Update info panel
                action_names = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
                info_str = (
                    f"Episode: {ep + 1}/{episodes}\n"
                    f"Step: {info['step']}/10\n"
                    f"Action: {action_names[action]}\n"
                    f"Matches: {info['matches']}/144\n"
                    f"Accuracy: {info['accuracy']:.1%}\n"
                    f"Reward: {reward:.3f}\n"
                )
                if terminated:
                    info_str += f"\n{'='*20}\n"
                    info_str += f"FINAL FITNESS: {reward:.3f}\n"
                    info_str += f"{'='*20}"
                
                info_text.set_text(info_str)
                ax_current.set_title(f"Current Grid (Step {info['step']}/10)")
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)
            
            if terminated:
                print(f"  Final accuracy: {info['accuracy']:.3f}")
                print(f"  Final fitness: {reward:.3f}")
                print(f"  Matches: {info['matches']}/{12*12}")
                all_accuracies.append(info['accuracy'])
        
        all_rewards.append(episode_reward)
        
        if render and ep < episodes - 1:
            plt.pause(1.0)  # Pause between episodes
    
    if render:
        plt.ioff()
        plt.show(block=True)
    
    env.close()
    
    print(f"\n=== Greedy Policy Results ===")
    print(f"Average reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Average accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")


def manual_control():
    """
    Interactive manual control.
    
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
    
    # Manual control does NOT use the pattern file; it uses a blank target by default
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10) 
    obs, info = env.reset()
    target = info['target']
    
    # Create figure with same layout as agent modes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_current, ax_target, ax_info = axes
    
    # Current grid
    im_current = ax_current.imshow(obs[:, :, 0], cmap='binary', vmin=0, vmax=1)
    ax_current.set_title('Current Grid (Step 0/10)')
    ax_current.set_xticks([])
    ax_current.set_yticks([])
    
    # Target pattern
    ax_target.imshow(target, cmap='binary', vmin=0, vmax=1)
    ax_target.set_title('Target Pattern')
    ax_target.set_xticks([])
    ax_target.set_yticks([])
    
    # Info panel
    ax_info.axis('off')
    info_text = ax_info.text(0.1, 0.5, "", fontsize=12, verticalalignment='center', fontfamily='monospace')
    
    head_patches = []
    
    # State tracking
    state = {'obs': obs, 'info': info, 'target': target, 'finished': False}
    
    def update_display():
        im_current.set_data(state['obs'][:, :, 0])
        
        # Remove old patches
        for patch in head_patches:
            patch.remove()
        head_patches.clear()
        
        # Draw write head
        head_mask = state['obs'][:, :, 1]
        head_pos = np.where(head_mask == 1)
        if len(head_pos[0]) > 0:
            r, c = head_pos[0][0], head_pos[1][0]
            for dr in range(2):
                for dc in range(2):
                    rect = plt.Rectangle(
                        (c + dc - 0.5, r + dr - 0.5), 1, 1,
                        fill=False, edgecolor='red', linewidth=2
                    )
                    ax_current.add_patch(rect)
                    head_patches.append(rect)
        
        # Update info
        matches = int((state['obs'][:, :, 0] == state['target']).sum())
        accuracy = matches / (12 * 12)
        step = env.unwrapped.step_count
        
        info_str = (
            f"Step: {step}/10\n"
            f"Matches: {matches}/144\n"
            f"Accuracy: {accuracy:.1%}\n"
            f"\n"
            f"Controls:\n"
            f"  ↑↓←→: Move\n"
            f"  Space: Pass\n"
            f"  0-F: Write pattern\n"
            f"  R: Reset\n"
            f"  Q: Quit\n"
        )
        
        if state['finished']:
            final_reward = matches / 144.0
            info_str += f"\n{'='*20}\n"
            info_str += f"FINAL FITNESS: {final_reward:.3f}\n"
            info_str += f"{'='*20}\n"
            info_str += f"\nPress R to restart"
        
        info_text.set_text(info_str)
        ax_current.set_title(f'Current Grid (Step {step}/10)')
        fig.canvas.draw_idle()
    
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
            state['target'] = state['info']['target']
            state['finished'] = False
            
            # Update target display
            ax_target.clear()
            ax_target.imshow(state['target'], cmap='binary', vmin=0, vmax=1)
            ax_target.set_title('Target Pattern')
            ax_target.set_xticks([])
            ax_target.set_yticks([])
            
            update_display()
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
            
            if terminated:
                state['finished'] = True
                print(f"\nEpisode complete!")
                print(f"Final accuracy: {info['accuracy']:.3f}")
                print(f"Final fitness: {reward:.3f}")
                print(f"Press R to reset or Q to quit")
            else:
                print(f"Step {info['step']}: Action {action_name}, Matches {info['matches']}/144")
            
            update_display()
    
    # Connect event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display()
    
    # Show plot
    plt.show(block=True)
    
    env.close()


def benchmark_environment():
    """Quick benchmark to verify environment is working."""
    print("\n=== Environment Benchmark ===")
    
    # Use the default pattern file defined at the top
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, pattern_file=DEFAULT_PATTERN_FILE) 
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Target shape: {info['target'].shape}")
    
    # Test all action types
    print(f"\n✓ Testing all 21 actions...")
    for action in tqdm(range(21), desc="Actions"):
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (12, 12, 2)
        assert 0.0 <= reward <= 1.0
    
    # Test full episode
    print(f"\n✓ Testing full episode...")
    obs, info = env.reset(seed=42)
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
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
    
    # The pattern file is now hardcoded as a constant and passed inside the functions
    if args.mode == "random":
        random_policy_test(
            episodes=args.episodes, 
            render=args.render
        )
    elif args.mode == "greedy":
        greedy_policy_test(
            episodes=args.episodes, 
            render=args.render
        )
    elif args.mode == "manual":
        manual_control()
    elif args.mode == "benchmark":
        benchmark_environment()