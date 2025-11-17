#!/usr/bin/env python3
"""
main.py

1) Reinforcement Learning Interface for the GoL-2x2-v0 environment.
2) Evolutionary Search Agent 

"""

import argparse
import numpy as np
import gymnasium as gym
import rl_GoL_env_gym
import torch
import os
import sys
import time
from tqdm import tqdm
from gymnasium import Wrapper

# Configuration
DEFAULT_PATTERN_FILE = 'target_patterns.json'
DEFAULT_AGENT_PATH = 'agent.pth'
GRID_SIZE = 12
MAX_STEPS = 10
OBS_SIZE = GRID_SIZE * GRID_SIZE * 3 + 1  # grid + target + head_mask + remaining_steps
ACTION_SIZE = 21
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_TIME_LIMIT = 30.0  # seconds per episode


# ============================================================================
# TIME LIMIT WRAPPER
# ============================================================================

class TimeLimitWrapper(Wrapper):
    """
    Wrapper that enforces a time limit per episode.
    If the agent takes too long, the episode is truncated.
    """
    def __init__(self, env, time_limit: float):
        """
        Args:
            env: The environment to wrap
            time_limit: Maximum time in seconds per episode
        """
        super().__init__(env)
        self.time_limit = time_limit
        self.episode_start_time = None
        self.total_episode_time = 0.0
        self.timed_out = False
    
    def reset(self, **kwargs):
        """Reset and start timing the episode."""
        self.episode_start_time = time.time()
        self.total_episode_time = 0.0
        self.timed_out = False
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step and check if time limit exceeded."""
        step_start = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        step_time = time.time() - step_start
        
        # Track total episode time
        self.total_episode_time = time.time() - self.episode_start_time
        
        # Check if time limit exceeded
        if self.total_episode_time > self.time_limit:
            truncated = True
            self.timed_out = True
            info['timeout'] = True
            info['episode_time'] = self.total_episode_time
            info['time_limit'] = self.time_limit
        
        # Always track timing info
        info['step_time'] = step_time
        info['total_episode_time'] = self.total_episode_time
        
        return obs, reward, terminated, truncated, info


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


class EvolutionarySearchAgent:
    """
    Agent that uses evolutionary algorithms to find action sequences.
    
    On first call, runs evolution to find the best action sequence for the target.
    Then executes actions from that sequence one by one.
    """
    def __init__(self, action_space, generations=50, population_size=100, 
                 elite_fraction=0.2, mutation_rate=0.1, mutation_mode='sexual', verbose=True):
        self.action_space = action_space
        self.generations = generations
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.mutation_mode = mutation_mode  # 'sexual', 'asexual', or 'both'
        self.verbose = verbose
        
        # Episode state
        self.current_sequence = None
        self.action_index = 0
        self.current_episode = 0  # Track current episode within this session
        
        # Evolution tracking
        self.best_fitness_ever = -float('inf')
        self.best_sequence_ever = None
        
    def select_action(self, obs: dict) -> int:
        """Select next action from evolved sequence."""
        # On first step of episode, run evolution
        if obs['remaining_steps'][0] == MAX_STEPS:
            self.current_episode += 1
            if self.verbose:
                print(f"\n Episode {self.current_episode}: Running evolution...")
            
            target = obs['target']
            current_grid = obs['grid']
            self.current_sequence = self._evolve_sequence(target, current_grid)
            self.action_index = 0
        
        # Return next action from sequence
        if self.current_sequence is not None and self.action_index < len(self.current_sequence):
            action = self.current_sequence[self.action_index]
            self.action_index += 1
            return int(action)
        
        # Fallback (shouldn't happen)
        return self.action_space.sample()
    
    def _evolve_sequence(self, target: np.ndarray, initial_grid: np.ndarray) -> np.ndarray:
        """Run evolutionary algorithm to find best action sequence."""
        from scipy.signal import convolve2d
        
        # Initialize population
        population = [np.random.randint(0, ACTION_SIZE, size=MAX_STEPS) 
                      for _ in range(self.population_size)]
        fitness_scores = np.zeros(self.population_size)
        
        ca_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
        
        # Helper functions
        def apply_ca_rules(grid):
            """Conway's Game of Life rules using convolution."""
            neighbor_counts = convolve2d(grid, ca_kernel, mode='same', boundary='wrap')
            birth_mask = (neighbor_counts == 3) & (grid == 0)
            survive_mask = np.isin(neighbor_counts, [2, 3]) & (grid == 1)
            new_grid = np.zeros_like(grid)
            new_grid[birth_mask | survive_mask] = 1
            return new_grid
        
        def evaluate_sequence(sequence):
            """
            Evaluate fitness of an action sequence starting from current grid state.
            
            CRITICAL: The gym environment applies CA rules BEFORE the first action,
            so we must do the same here to match the actual execution.
            """
            # Start from the current grid state
            grid = initial_grid.copy()
            agent_x, agent_y = 5, 5
            
            for step_idx, action in enumerate(sequence):
                # IMPORTANT: Apply CA rules FIRST (this happens before every action in gym)
                grid = apply_ca_rules(grid)
                
                # Then apply action
                if action == 0:  # Up
                    agent_y = (agent_y - 1) % GRID_SIZE
                elif action == 1:  # Down
                    agent_y = (agent_y + 1) % GRID_SIZE
                elif action == 2:  # Left
                    agent_x = (agent_x - 1) % GRID_SIZE
                elif action == 3:  # Right
                    agent_x = (agent_x + 1) % GRID_SIZE
                elif action >= 5:  # Write pattern
                    pattern_id = action - 5
                    # CRITICAL: Match the gym environment's bit ordering
                    # bit i corresponds to position (i//2, i%2) relative to agent
                    # So bits [0,1,2,3] map to positions [(0,0), (0,1), (1,0), (1,1)]
                    bits = [(pattern_id >> i) & 1 for i in range(4)]
                    for i in range(2):
                        for j in range(2):
                            y = (agent_y + i) % GRID_SIZE
                            x = (agent_x + j) % GRID_SIZE
                            bit_index = i * 2 + j  # 0,1,2,3
                            grid[y, x] = bits[bit_index]
            
            # Calculate fitness as percentage match
            match_fraction = np.mean(grid == target)
            return match_fraction * 100
        
        # Evolution loop
        best_fitness = -float('inf')
        best_sequence = None
        
        for gen in range(self.generations):
            # Evaluate population
            for i in range(self.population_size):
                fitness_scores[i] = evaluate_sequence(population[i])
                
                if fitness_scores[i] > best_fitness:
                    best_fitness = fitness_scores[i]
                    best_sequence = population[i].copy()
            
            # Print progress
            if self.verbose and (gen % 10 == 0 or gen == self.generations - 1):
                avg_fitness = np.mean(fitness_scores)
                print(f"  Gen {gen+1}/{self.generations}: Best={best_fitness:.1f}%, Avg={avg_fitness:.1f}%")
            
            # Check for perfect solution
            if best_fitness >= 100:
                if self.verbose:
                    print(f"  ✓ Perfect solution found at generation {gen+1}!")
                break
            
            # Selection and breeding
            if gen < self.generations - 1:
                # Select elite
                elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
                elites = [population[i].copy() for i in elite_indices]
                
                # Create new population starting with elites
                new_population = elites.copy()
                
                # Apply mutation strategy
                if self.mutation_mode == 'asexual':
                    # Asexual: mutate elites directly
                    while len(new_population) < self.population_size:
                        # Select random elite
                        parent = elites[np.random.randint(len(elites))].copy()
                        
                        # Mutate
                        for i in range(MAX_STEPS):
                            if np.random.random() < self.mutation_rate:
                                parent[i] = np.random.randint(0, ACTION_SIZE)
                        
                        new_population.append(parent)
                
                elif self.mutation_mode == 'sexual':
                    # Sexual: crossover and mutation
                    while len(new_population) < self.population_size:
                        parent_indices = np.random.choice(len(elites), size=2, replace=False)
                        parent1 = elites[parent_indices[0]]
                        parent2 = elites[parent_indices[1]]
                        
                        # Crossover
                        crossover_point = np.random.randint(1, MAX_STEPS)
                        child = np.concatenate([parent1[:crossover_point], 
                                              parent2[crossover_point:]])
                        
                        # Mutation
                        for i in range(MAX_STEPS):
                            if np.random.random() < self.mutation_rate:
                                child[i] = np.random.randint(0, ACTION_SIZE)
                        
                        new_population.append(child)
                
                elif self.mutation_mode == 'both':
                    # Both: mix of asexual and sexual reproduction
                    target_size = self.population_size - len(new_population)
                    asexual_count = target_size // 2
                    
                    # Asexual reproduction
                    for _ in range(asexual_count):
                        parent = elites[np.random.randint(len(elites))].copy()
                        for i in range(MAX_STEPS):
                            if np.random.random() < self.mutation_rate:
                                parent[i] = np.random.randint(0, ACTION_SIZE)
                        new_population.append(parent)
                    
                    # Sexual reproduction
                    while len(new_population) < self.population_size:
                        parent_indices = np.random.choice(len(elites), size=2, replace=False)
                        parent1 = elites[parent_indices[0]]
                        parent2 = elites[parent_indices[1]]
                        
                        crossover_point = np.random.randint(1, MAX_STEPS)
                        child = np.concatenate([parent1[:crossover_point], 
                                              parent2[crossover_point:]])
                        
                        for i in range(MAX_STEPS):
                            if np.random.random() < self.mutation_rate:
                                child[i] = np.random.randint(0, ACTION_SIZE)
                        
                        new_population.append(child)
                
                population = new_population[:self.population_size]
        
        # Track all-time best
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_sequence_ever = best_sequence.copy()
        
        if self.verbose:
            print(f"  Final best fitness: {best_fitness:.1f}%")
        
        return best_sequence
    
    def save(self, path: str):
        """
        Save agent hyperparameters (not learned - just configuration).
        
        Note: Evolutionary agents learn from scratch each episode, so there's
        nothing meaningful to save except the hyperparameters for convenience.
        """
        data = {
            'agent_type': 'evolutionary',
            'generations': self.generations,
            'population_size': self.population_size,
            'elite_fraction': self.elite_size / self.population_size,
            'mutation_rate': self.mutation_rate,
            'mutation_mode': self.mutation_mode,
            'sessions_run': self.current_episode
        }
        with open(path, 'w') as f:
            import json
            json.dump(data, f, indent=2)
        print(f"✓ Evolutionary agent config saved to {path}")
        print(f"  (Note: Agent learns from scratch each episode - only hyperparameters saved)")
    
    def load(self, path: str):
        """Load agent hyperparameters from config file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent file not found: {path}")
        
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        if data.get('agent_type') != 'evolutionary':
            raise ValueError(f"Config file is not for evolutionary agent")
        
        self.generations = data['generations']
        self.population_size = data['population_size']
        self.elite_size = int(data['population_size'] * data['elite_fraction'])
        self.mutation_rate = data['mutation_rate']
        self.mutation_mode = data.get('mutation_mode', 'sexual')  # Default to sexual for backwards compatibility
        self.current_episode = 0  # Always start fresh
        
        print(f"✓ Evolutionary agent config loaded from {path}")
        print(f"  Generations: {self.generations}, Population: {self.population_size}")
        print(f"  Elite: {data['elite_fraction']}, Mutation: {self.mutation_rate}")
        print(f"  Mutation mode: {self.mutation_mode}")


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
    all_times = []
    timeouts = 0
    
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
        
        # Check for timeout
        episode_time = info.get('total_episode_time', 0.0)
        all_times.append(episode_time)
        timed_out = info.get('timeout', False)
        if timed_out:
            timeouts += 1
        
        status = " [TIMEOUT]" if timed_out else ""
        print(f"Episode {ep}/{episodes} | Reward: {episode_reward:.3f} | "
              f"Accuracy: {info['accuracy']:.3f} | Time: {episode_time:.2f}s{status}")
    
    print(f"\n=== Results ===")
    print(f"Avg reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Avg accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")
    print(f"Avg time: {np.mean(all_times):.2f}s ± {np.std(all_times):.2f}s")
    print(f"Timeouts: {timeouts}/{episodes}")
    if timeouts > 0:
        time_limit = env.time_limit if hasattr(env, 'time_limit') else 'N/A'
        print(f"⚠️  Warning: {timeouts} episode(s) exceeded time limit of {time_limit}s")


def run_benchmark_suite(args):
    """
    Run evolutionary agent on ALL patterns in target_patterns.json.
    Creates a benchmark log for comparison.
    """
    import json
    from datetime import datetime
    
    print("\n" + "="*70)
    print("BENCHMARK SUITE: Evolutionary Agent on All Target Patterns")
    print("="*70)
    
    # Load all target patterns
    if not os.path.exists(DEFAULT_PATTERN_FILE):
        print(f"Error: {DEFAULT_PATTERN_FILE} not found!")
        print("Run 'python generate_target_patterns.py' first to create patterns.")
        return
    
    with open(DEFAULT_PATTERN_FILE, 'r') as f:
        data = json.load(f)
    
    patterns_data = data["patterns"]
    
    num_patterns = len(patterns_data)
    print(f"\nLoaded {num_patterns} target patterns from {DEFAULT_PATTERN_FILE}")
    print(f"Time limit: {args.time_limit}s per pattern")
    print(f"Evolutionary parameters:")
    print(f"  Generations: {args.evo_generations}")
    print(f"  Population: {args.evo_population}")
    print(f"  Elite fraction: {args.evo_elite}")
    print(f"  Mutation rate: {args.evo_mutation}")
    print(f"  Mutation mode: {args.evo_mutation_mode}")
    print("\nStarting benchmark...\n")
    
    # Create agent
    render_mode = "human" if args.render else None
    print(render_mode)
    env = gym.make("GoL-2x2-v0", grid_size=GRID_SIZE, max_steps=MAX_STEPS,
                   pattern_file=DEFAULT_PATTERN_FILE, render_mode=render_mode)

    env = TimeLimitWrapper(env, args.time_limit)
    
    agent = EvolutionarySearchAgent(
        env.action_space,
        generations=args.evo_generations,
        population_size=args.evo_population,
        elite_fraction=args.evo_elite,
        mutation_rate=args.evo_mutation,
        mutation_mode=args.evo_mutation_mode,
        verbose=False  # Disable per-episode printing for cleaner output
    )
    
    # Results tracking
    results = {
        'benchmark_info': {
            'date': datetime.now().isoformat(),
            'num_patterns': num_patterns,
            'time_limit_seconds': args.time_limit,
            'evo_generations': args.evo_generations,
            'evo_population': args.evo_population,
            'evo_elite': args.evo_elite,
            'evo_mutation': args.evo_mutation,
            'evo_mutation_mode': args.evo_mutation_mode,
        },
        'pattern_results': []
    }
    
    rewards = []
    times = []
    timeouts = 0
    perfect_matches = 0
    
    # Run benchmark on each pattern
    for pattern_idx in tqdm(range(num_patterns), desc="Testing patterns"):
        # This ensures the agent's internal state (current_episode, best_sequence_ever) is reset, forcing a new evolutionary search for the current target pattern.
        # agent = EvolutionarySearchAgent(
        #     env.action_space,
        #     generations=args.evo_generations,
        #     population_size=args.evo_population,
        #     elite_fraction=args.evo_elite,
        #     mutation_rate=args.evo_mutation,
        #     verbose=False  # Disable per-episode printing for cleaner output
        # )

        # Reset with this specific pattern using the proper API
        obs, info = env.reset(options={'pattern_index': pattern_idx})
        
        terminated = truncated = False
        episode_reward = 0
        step_count = 0
        
        while not (terminated or truncated):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if args.render:
                env.render()
        
        # Record results
        episode_time = info.get('total_episode_time', 0.0)
        timed_out = info.get('timeout', False)
        accuracy = info.get('accuracy', 0.0)
        
        rewards.append(episode_reward)
        times.append(episode_time)
        
        if timed_out:
            timeouts += 1
        if accuracy >= 0.999:  # Account for floating point
            perfect_matches += 1
        
        pattern_result = {
            'pattern_index': pattern_idx,
            'reward': float(episode_reward),
            'accuracy': float(accuracy),
            'time_seconds': float(episode_time),
            'timeout': bool(timed_out),
            'steps_taken': step_count,
            'best_action_sequence': agent.current_sequence.tolist() if agent.current_sequence is not None else []  
        }
        results['pattern_results'].append(pattern_result)
    
    env.close()
    
    # Calculate summary statistics
    rewards_array = np.array(rewards)
    times_array = np.array(times)
    
    summary = {
        'total_patterns': num_patterns,
        'perfect_matches': perfect_matches,
        'perfect_match_rate': perfect_matches / num_patterns,
        'timeouts': timeouts,
        'timeout_rate': timeouts / num_patterns,
        'avg_reward': float(np.mean(rewards_array)),
        'std_reward': float(np.std(rewards_array)),
        'min_reward': float(np.min(rewards_array)),
        'max_reward': float(np.max(rewards_array)),
        'median_reward': float(np.median(rewards_array)),
        'avg_time': float(np.mean(times_array)),
        'std_time': float(np.std(times_array)),
        'total_time': float(np.sum(times_array))
    }
    
    results['summary'] = summary
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_file = f"benchmark_evolutionary_{timestamp}.json"
    
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"Total Patterns:        {num_patterns}")
    print(f"Perfect Matches:       {perfect_matches} ({summary['perfect_match_rate']:.1%})")
    print(f"Timeouts:              {timeouts} ({summary['timeout_rate']:.1%})")
    print(f"\nReward Statistics:")
    print(f"  Average:             {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}")
    print(f"  Median:              {summary['median_reward']:.4f}")
    print(f"  Range:               [{summary['min_reward']:.4f}, {summary['max_reward']:.4f}]")
    print(f"\nTime Statistics:")
    print(f"  Average per pattern: {summary['avg_time']:.2f}s ± {summary['std_time']:.2f}s")
    print(f"  Total time:          {summary['total_time']:.1f}s ({summary['total_time']/60:.1f} min)")
    print(f"\nResults saved to: {benchmark_file}")
    print("="*70 + "\n")
    
    # Create human-readable summary file
    summary_file = f"benchmark_evolutionary_{timestamp}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("EVOLUTIONARY AGENT BENCHMARK SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {results['benchmark_info']['date']}\n")
        f.write(f"Patterns Tested: {num_patterns}\n")
        f.write(f"Time Limit: {args.time_limit}s per pattern\n\n")
        
        f.write("Evolutionary Algorithm Parameters:\n")
        f.write(f"  Generations: {args.evo_generations}\n")
        f.write(f"  Population: {args.evo_population}\n")
        f.write(f"  Elite Fraction: {args.evo_elite}\n")
        f.write(f"  Mutation Rate: {args.evo_mutation}\n")
        f.write(f"  Mutation Mode: {args.evo_mutation_mode}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Perfect Matches: {perfect_matches}/{num_patterns} ({summary['perfect_match_rate']:.1%})\n")
        f.write(f"Timeouts:        {timeouts}/{num_patterns} ({summary['timeout_rate']:.1%})\n\n")
        
        f.write("Reward Statistics:\n")
        f.write(f"  Average: {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}\n")
        f.write(f"  Median:  {summary['median_reward']:.4f}\n")
        f.write(f"  Min:     {summary['min_reward']:.4f}\n")
        f.write(f"  Max:     {summary['max_reward']:.4f}\n\n")
        
        f.write("Time Statistics:\n")
        f.write(f"  Average per pattern: {summary['avg_time']:.2f}s ± {summary['std_time']:.2f}s\n")
        f.write(f"  Total time:          {summary['total_time']:.1f}s ({summary['total_time']/60:.1f} minutes)\n\n")
        
        f.write("="*70 + "\n")
        f.write("To reproduce this benchmark:\n")
        f.write(f"python main.py benchmark-suite --time-limit {args.time_limit} \\\n")
        f.write(f"  --evo-generations {args.evo_generations} --evo-population {args.evo_population} \\\n")
        f.write(f"  --evo-elite {args.evo_elite} --evo-mutation {args.evo_mutation} \\\n")
        f.write(f"  --evo-mutation-mode {args.evo_mutation_mode}\n")
    
    print(f"Human-readable summary saved to: {summary_file}\n")


# ============================================================================
# Verification
# ============================================================================

def run_verification():
    """Quick environment verification."""
    print("\n=== Environment Verification ===")
    
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
    train_parser.add_argument('--agent-type', choices=['simple', 'evolutionary'],
                             default='simple', help='Type of agent to train')
    train_parser.add_argument('--episodes', type=int, default=100,
                             help='Number of training episodes')
    train_parser.add_argument('--save-path', type=str, default=DEFAULT_AGENT_PATH,
                             help='Path to save trained agent')
    train_parser.add_argument('--render', action='store_true',
                             help='Enable rendering')
    # Evolutionary agent parameters
    train_parser.add_argument('--evo-generations', type=int, default=50,
                             help='Generations per evolution run (evolutionary only)')
    train_parser.add_argument('--evo-population', type=int, default=100,
                             help='Population size (evolutionary only)')
    train_parser.add_argument('--evo-elite', type=float, default=0.2,
                             help='Elite fraction (evolutionary only)')
    train_parser.add_argument('--evo-mutation', type=float, default=0.1,
                             help='Mutation rate (evolutionary only)')
    train_parser.add_argument('--evo-mutation-mode', choices=['sexual', 'asexual', 'both'],
                             default='sexual', help='Mutation mode (evolutionary only)')
    
    # EVAL command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained agent')
    eval_parser.add_argument('--agent-type', choices=['simple', 'evolutionary'],
                            default='simple', help='Type of agent to evaluate')
    eval_parser.add_argument('--load-path', type=str, default=DEFAULT_AGENT_PATH,
                            help='Path to load agent')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--render', action='store_true',
                            help='Enable rendering')
    eval_parser.add_argument('--time-limit', type=float, default=DEFAULT_TIME_LIMIT,
                            help=f'Time limit per episode in seconds (default: {DEFAULT_TIME_LIMIT}s)')
    eval_parser.add_argument('--evo-generations', type=int, default=50,
                             help='Generations per evolution run. (evolutionary only)')
    eval_parser.add_argument('--evo-population', type=int, default=100,
                             help='Population size (evolutionary only)')
    eval_parser.add_argument('--evo-elite', type=float, default=0.2,
                             help='Elite fraction (evolutionary only)')
    eval_parser.add_argument('--evo-mutation', type=float, default=0.1,
                             help='Mutation rate (evolutionary only)')
    eval_parser.add_argument('--evo-mutation-mode', choices=['sexual', 'asexual', 'both'],
                             default='sexual', help='Mutation mode (evolutionary only)')

    #  Evolutionary Search Agent Benchmark suite
    benchmark_suite_parser = subparsers.add_parser('benchmark-suite', help='Run Evolutionary Algorithm Agent Benchmark Suite')
    benchmark_suite_parser.add_argument('--time-limit', type=float, default=7.0, help='Time limit per episode in seconds')
    benchmark_suite_parser.add_argument('--render', action='store_true',
                             help='Enable rendering')
    benchmark_suite_parser.add_argument('--evo-generations', type=int, default=50,
                             help='Generations per evolution run.')
    benchmark_suite_parser.add_argument('--evo-population', type=int, default=100,
                             help='Population size (evolutionary only)')
    benchmark_suite_parser.add_argument('--evo-elite', type=float, default=0.2,
                             help='Elite fraction (evolutionary only)')
    benchmark_suite_parser.add_argument('--evo-mutation', type=float, default=0.1,
                             help='Mutation rate (evolutionary only)')
    benchmark_suite_parser.add_argument('--evo-mutation-mode', choices=['sexual', 'asexual', 'both'],
                             default='sexual', help='Mutation mode')

    # Verification command
    subparsers.add_parser('verification', help='Run environment verification check')

    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'verification':
        run_verification()

    elif args.command == 'benchmark-suite':
        run_benchmark_suite(args)
    
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
        # Check if evolutionary agent was requested
        if args.agent_type == 'evolutionary':
            print("\n⚠️  Evolutionary agents don't support training mode for developing memory at the moment.")
            print("Evolutionary agents evolve solutions from scratch each episode.")
            print("\nPlease use one of these commands instead:")
            print("  - 'eval' to evaluate the evolutionary agent on random patterns")
            print("  - 'benchmark-suite' to test on all patterns systematically")
            print("\nExample:")
            print(f"  python main.py eval --agent-type evolutionary --episodes {args.episodes}")
            print(f"  python main.py benchmark-suite --evo-generations {args.evo_generations}")
            sys.exit(1)
        
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
        
        # Wrap with time limit
        env = TimeLimitWrapper(env, args.time_limit)
        print(f"⏱️  Time limit: {args.time_limit}s per episode")
        
        # Create and load agent based on type
        if args.agent_type == 'evolutionary':
            agent = EvolutionarySearchAgent(env.action_space, verbose=True)
            # Adjust load path for evolutionary agent
            load_path = args.load_path
            if not load_path.endswith('.json'):
                load_path = load_path.replace('.pth', '.json')
            
            # Try to load config, but it's optional for evolutionary agents
            if os.path.exists(load_path):
                try:
                    agent.load(load_path)
                except ValueError:
                    print(f"Warning: {load_path} is not an evolutionary agent config")
                    print("Using default evolutionary parameters")
            else:
                print(f"No config found at {load_path}, using default evolutionary parameters")
        else:
            agent = SimpleAgent(env.action_space)
            agent.load(args.load_path)
        
        try:
            evaluate_agent(env, agent, args.episodes, args.render)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Run 'python main.py train --agent-type {args.agent_type}' first to create an agent")
        
        env.close()


if __name__ == "__main__":
    main()