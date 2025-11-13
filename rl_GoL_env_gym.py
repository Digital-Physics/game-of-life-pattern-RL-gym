#!/usr/bin/env python3
"""
rl_GoL_env_gym.py

Gymnasium-compatible Game of Life 2×2-write-head environment.

Usage:
    import gymnasium as gym
    import rl_ca_env_gym  # Registers "GoL-2x2-v0"

    # Create environment
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, render_mode="human")

    # Training loop
    obs, info = env.reset()
    target = obs['target']  # Target pattern is now in the observation!

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Your agent selects action (0-20)
            # obs is now a dict with 'grid', 'target', 'head_mask', and 'remaining_steps'
            action = agent.select_action(obs)
            
            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Optional rendering
            env.render()
            
            # reward is 0.0 until step 10, then it's the match percentage
            if terminated:
                print(f"Match accuracy: {info['accuracy']:.2%}")

    env.close()
"""

import numpy as np
from scipy.signal import convolve2d
import gymnasium as gym
from gymnasium import spaces
import json
import os


class GoL2x2Env(gym.Env):
    """
    Game of Life pattern matching environment.
    
    Observation Space:
        Dict with:
        - 'grid': Box(0, 1, (grid_size, grid_size), int8) - Current Game of Life grid
        - 'target': Box(0, 1, (grid_size, grid_size), int8) - NEW: The pattern to match
        - 'head_mask': Box(0, 1, (grid_size, grid_size), int8) - Write head position (2×2 area)
        - 'remaining_steps': Box(0, max_steps, (1,), int32) - Steps remaining in episode
    
    Action Space:
        Discrete(21):
        - 0: Move head up
        - 1: Move head down
        - 2: Move head left
        - 3: Move head right
        - 4: Do nothing (pass)
        - 5-20: Write pattern 0-15 (4-bit patterns for 2×2 grid)
    
    Reward:
        - Steps 0-9: reward = 0.0
        - Step 10 (terminal): reward = fraction of matching cells (0.0 to 1.0)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=12, max_steps=10, render_mode=None, 
                 pattern_file="target_patterns.json"):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Conway's Game of Life rules
        self.rules = {"birth": [3], "survive": [2, 3]}
        self.ca_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
        
        # Action space: 4 moves + 1 pass + 16 patterns
        self.action_space = spaces.Discrete(21)
        
        # Observation: Dictionary with grid, target, head_mask, and remaining_steps
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=1, 
                shape=(grid_size, grid_size), 
                dtype=np.int8
            ),
            'target': spaces.Box(
                low=0, high=1, 
                shape=(grid_size, grid_size), 
                dtype=np.int8
            ),
            'head_mask': spaces.Box(
                low=0, high=1, 
                shape=(grid_size, grid_size), 
                dtype=np.int8
            ),
            'remaining_steps': spaces.Box(
                low=0, high=max_steps, 
                shape=(1,), 
                dtype=np.int32
            )
        })
        
        # Load target patterns if provided
        self.target_patterns = []
        self.pattern_file = pattern_file
        if pattern_file:
            self._load_patterns(pattern_file)
        
        # Rendering
        self._fig = None
        self._ax_current = None
        self._ax_target = None
        self._ax_info = None
        self._im_current = None
        self._info_text = None
        self._head_patches = []
        
        # Set a dummy target for initialization if no patterns loaded
        self.target = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
    
    def _load_patterns(self, filename):
        """Load pre-generated target patterns from JSON file."""
        if not os.path.exists(filename):
            print(f"Warning: Pattern file {filename} not found. Using a zero-grid as target.")
            return
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Check for grid size consistency
            if data['patterns'] and len(data['patterns'][0]['grid']) != self.grid_size:
                print(f"Error: Pattern file grid size ({len(data['patterns'][0]['grid'])}) does not match env size ({self.grid_size}). Using a zero-grid as target.")
                return

            self.target_patterns = [np.array(p['grid'], dtype=np.int8) 
                                   for p in data['patterns']]
            print(f"Loaded {len(self.target_patterns)} target patterns from {filename}")
        except Exception as e:
            print(f"Error loading patterns: {e}. Using a zero-grid as target.")
            self.target_patterns = []

    def _apply_ca(self, grid):
        """Apply Conway's Game of Life rules."""
        neighbor_counts = convolve2d(grid, self.ca_kernel, mode="same", boundary="wrap")
        birth_mask = np.isin(neighbor_counts, self.rules["birth"]) & (grid == 0)
        survive_mask = np.isin(neighbor_counts, self.rules["survive"]) & (grid == 1)
        new_grid = np.zeros_like(grid)
        new_grid[birth_mask | survive_mask] = 1
        return new_grid.astype(np.int8)

    def _get_head_mask(self):
        """Create head mask showing 2×2 write head position."""
        head_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        r, c = self.head_pos
        for dr in [0, 1]:
            for dc in [0, 1]:
                head_mask[(r + dr) % self.grid_size, (c + dc) % self.grid_size] = 1
        return head_mask

    def _obs(self):
        """Construct observation dictionary."""
        return {
            'grid': self.ca_grid.copy(),
            'target': self.target.copy(),
            'head_mask': self._get_head_mask(),
            'remaining_steps': np.array([self.max_steps - self.step_count], dtype=np.int32)
        }

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.ca_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.head_pos = (self.grid_size // 2 - 1, self.grid_size // 2 - 1)
        
        # Select target pattern using the environment's RNG
        if self.target_patterns:
            idx = self.np_random.integers(len(self.target_patterns))
            self.target = self.target_patterns[idx].copy()
        else:
            self.target = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            
        obs = self._obs()
        # The target is now in obs, so we remove it from info
        info = {} 
        return obs, info

    def _write_pattern_at_head(self, pattern_id: int):
        """Write a 4-bit pattern as 2×2 grid at write head position."""
        bits = [(pattern_id >> i) & 1 for i in range(4)]
        r, c = self.head_pos
        coords = [
            (r % self.grid_size, c % self.grid_size),
            (r % self.grid_size, (c + 1) % self.grid_size),
            ((r + 1) % self.grid_size, c % self.grid_size),
            ((r + 1) % self.grid_size, (c + 1) % self.grid_size),
        ]
        
        for (rr, cc), b in zip(coords, bits):
            self.ca_grid[rr, cc] = b

    def step(self, action: int):
        """Execute one step: apply CA rules, then execute action."""
        if not (0 <= action < self.action_space.n):
            raise ValueError(f"Action must be in [0, {self.action_space.n - 1}]")
        
        # First, update the Game of Life
        self.ca_grid = self._apply_ca(self.ca_grid)
        
        # Then, execute the action
        if action == 0:  # up
            self.head_pos = ((self.head_pos[0] - 1) % self.grid_size, self.head_pos[1])
        elif action == 1:  # down
            self.head_pos = ((self.head_pos[0] + 1) % self.grid_size, self.head_pos[1])
        elif action == 2:  # left
            self.head_pos = (self.head_pos[0], (self.head_pos[1] - 1) % self.grid_size)
        elif action == 3:  # right
            self.head_pos = (self.head_pos[0], (self.head_pos[1] + 1) % self.grid_size)
        elif action == 4:  # pass
            pass
        else:  # write pattern (actions 5-20)
            self._write_pattern_at_head(action - 5)

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Reward only at the end
        reward = 0.0
        if terminated:
            matches = (self.ca_grid == self.target).sum()
            reward = float(matches) / (self.grid_size * self.grid_size)
        
        obs = self._obs()
        info = {
            "step": self.step_count,
            "matches": int((self.ca_grid == self.target).sum()),
            "accuracy": float((self.ca_grid == self.target).sum()) / (self.grid_size * self.grid_size)
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
            
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            
            if self._fig is None:
                # Initialize figure
                plt.ion()
                self._fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                self._ax_current, self._ax_target, self._ax_info = axes
                
                # Current grid
                self._im_current = self._ax_current.imshow(
                    self.ca_grid, cmap="binary", vmin=0, vmax=1
                )
                self._ax_current.set_title(f"Current Grid (Step {self.step_count}/{self.max_steps})")
                self._ax_current.set_xticks([])
                self._ax_current.set_yticks([])
                
                # Target pattern
                self._target_current = self._ax_target.imshow(self.target, cmap="binary", vmin=0, vmax=1)
                self._ax_target.set_title("Target Pattern")
                self._ax_target.set_xticks([])
                self._ax_target.set_yticks([])
                
                # Info panel
                self._ax_info.axis('off')
                self._info_text = self._ax_info.text(
                    0.1, 0.5, "", fontsize=12, 
                    verticalalignment='center', fontfamily='monospace'
                )
                
                plt.show(block=False)
            
            # Update current grid
            self._im_current.set_data(self.ca_grid)

            # Update Target Pattern (target patterns changes between episodes, not every step, but we do it here every step)
            self._target_current.set_data(self.target)
            
            # Clear old write head patches
            for patch in self._head_patches:
                patch.remove()
            self._head_patches.clear()
            
            # Draw write head (2×2 red outline)
            head_mask = self._get_head_mask()
            head_pos = np.where(head_mask == 1)
            if len(head_pos[0]) > 0:
                r, c = head_pos[0][0], head_pos[1][0]
                for dr in range(2):
                    for dc in range(2):
                        rect = plt.Rectangle(
                            (c + dc - 0.5, r + dr - 0.5), 1, 1,
                            fill=False, edgecolor="red", linewidth=2
                        )
                        self._ax_current.add_patch(rect)
                        self._head_patches.append(rect)
            
            # Update info panel
            matches = int((self.ca_grid == self.target).sum())
            accuracy = matches / (self.grid_size * self.grid_size)
            
            info_str = (
                f"Step: {self.step_count}/{self.max_steps}\n"
                f"Remaining: {self.max_steps - self.step_count}\n"
                f"Matches: {matches}/{self.grid_size * self.grid_size}\n"
                f"Accuracy: {accuracy:.1%}\n"
            )
            
            if self.step_count >= self.max_steps:
                final_reward = matches / (self.grid_size * self.grid_size)
                info_str += f"\n{'='*20}\n"
                info_str += f"FINAL FITNESS: {final_reward:.3f}\n"
                info_str += f"{'='*20}"
            
            self._info_text.set_text(info_str)
            self._ax_current.set_title(f"Current Grid (Step {self.step_count}/{self.max_steps})")
            
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.01)
            
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            # Create a composite image with current grid, target, and head visualization
            grid_with_head = self.ca_grid.copy().astype(float)
            
            # Overlay head position with a different color (0.5 = gray)
            head_mask = self._get_head_mask()
            grid_with_head[head_mask == 1] = 0.5
            
            # Convert to RGB
            grid_rgb = np.repeat(grid_with_head[:, :, np.newaxis], 3, axis=2) * 255
            return grid_rgb.astype(np.uint8)

    def close(self):
        """Clean up rendering resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None


# Register environment
gym.register(
    id="GoL-2x2-v0",
    entry_point="rl_GoL_env_gym:GoL2x2Env",
)