# Reinforcement Learning Gym Environment for Pattern Matching in Conway's Game of Life

A Gymnasium-compatible environment for training RL agents to construct target patterns in a cellular automaton that evolves according to Conway's Game of Life rules.

![Screenshot of the pattern matching game](./screenshot_GoL_pattern_matching_game.png)

## Game Overview

**Objective**: Match a target pattern after exactly 10 timesteps of Game of Life evolution and agent actions.

**Core Mechanics**:
- Each turn: Game of Life update → Agent action → Repeat
- Agent controls a 2×2 "write head" that can move around the grid and write patterns
- Final reward based on how well the resulting grid matches the target pattern

**Actions** (Manual play keyboard shortcuts):
- **Arrow Keys**: Move 2×2 write head (Up/Down/Left/Right)
- **Space Bar**: Pass (do nothing)
- **0-9, A-F**: Write one of 16 possible 2×2 patterns (hex encoding)

## Installation

```bash
uv pip install .
```

## Quick Start

### 1. Verify Installation
```bash
python main.py benchmark
```

### 2. Test Built-in Policies

```bash
# Random policy (baseline)
python main.py test --mode random --episodes 3 --render

# Greedy policy (smarter baseline)
python main.py test --mode greedy --episodes 5 --render

# Manual control (play yourself!)
python main.py test --mode manual
```

### 3. Train and Evaluate Your Own Agent

#### Simple RL Agent (Placeholder)
```bash
# Train simple agent (replace SimpleAgent class with your RL implementation)
python main.py train --agent-type simple --episodes 100 --render

# Evaluate your trained agent
python main.py eval --agent-type simple --load-path agent.pth --episodes 10 --render

# Evaluate with strict time limit
python main.py eval --agent-type simple --load-path agent.pth --episodes 10 --time-limit 5 --render
```

#### Evolutionary Search Agent (Built-in)
```bash
# Run evolutionary agent (searches for solution each episode)
python main.py eval --agent-type evolutionary --episodes 5 --render

# Customize evolutionary parameters
python main.py eval --agent-type evolutionary --episodes 5 --render \
  --evo-generations 100 --evo-population 200 --evo-elite 0.3 --evo-mutation 0.15

# Test with time limit (will timeout if evolution takes too long)
python main.py eval --agent-type evolutionary --episodes 5 --time-limit 60 --render
```

**Note on Evolutionary Agents**: Unlike RL agents that learn across episodes, evolutionary agents solve each episode from scratch by running evolution. The agent runs a genetic algorithm at the start of each episode to find a good action sequence for the current target pattern. There's no training phase - it's pure search.

## Implementing Your Own Agent

To use your own RL algorithm, replace the `SimpleAgent` class in `main.py` with your implementation. Required interface:

```python
class YourAgent:
    def __init__(self, action_space):
        # Initialize your model
        pass
    
    def select_action(self, obs: dict) -> int:
        # Return action index (0-20)
        # Called once per timestep
        pass
    
    def save(self, path: str):
        # Save model weights/parameters
        pass
    
    def load(self, path: str):
        # Load model weights/parameters
        pass
```

The training and evaluation infrastructure handles everything else (environment setup, episode loops, rendering, metrics, time limits).

### Example: Adding an RL Algorithm

```python
class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_network = self._build_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        # ... your implementation
    
    def select_action(self, obs: dict) -> int:
        # Convert obs dict to tensor
        state_tensor = self._obs_to_tensor(obs)
        # Get Q-values and select action
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def save(self, path: str):
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path: str):
        self.q_network.load_state_dict(torch.load(path))
```

## Environment Specification

### Observation Space

**Type**: `Dict` with the following keys:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `grid` | `(12, 12)` | `int8` | Current Game of Life grid (0=dead, 1=alive) |
| `target` | `(12, 12)` | `int8` | Target pattern to match (0=dead, 1=alive) |
| `head_mask` | `(12, 12)` | `int8` | Write head position (1 marks the 2×2 area) |
| `remaining_steps` | `(1,)` | `int32` | Steps remaining in episode |

### Action Space

**Type**: `Discrete(21)`

| Action | Description |
|--------|-------------|
| 0 | Move head up |
| 1 | Move head down |
| 2 | Move head left |
| 3 | Move head right |
| 4 | Pass (do nothing) |
| 5-20 | Write pattern 0x0 to 0xF |

**Pattern Encoding**: 4-bit patterns map to 2×2 grids:
```
bit 0 = top-left    bit 1 = top-right
bit 2 = bottom-left bit 3 = bottom-right

Example: 0x9 = 0b1001 = ■ □
                        □ ■
```

### Reward Structure

- **Steps 0-9**: `reward = 0.0` (sparse reward)
- **Step 10** (terminal): `reward = matching_cells / total_cells`
  - Range: `[0.0, 1.0]`
  - `1.0` = perfect match

### Episode Dynamics

1. **Reset**: Grid initialized to all dead cells; random target pattern loaded from `target_patterns.json`
2. **Each step**:
   - Game of Life update applied to grid
   - Agent action executed
3. **Termination**: After 10 steps, final reward computed based on pattern match

## Generating Achievable Target Patterns

By default, the environment samples from patterns in `target_patterns.json`. To generate new patterns that are provably achievable within N steps:

```bash
python generate_target_patterns.py --num-steps 6
```

This creates/updates `target_patterns.json`, which the environment uses for sampling targets.

## Command Line Reference

### Test Mode
```bash
python main.py test --mode {random|greedy|manual} [options]

Options:
  --episodes N     Number of episodes (default: 5)
  --render         Enable visualization
```

### Train Mode
```bash
python main.py train --agent-type {simple|evolutionary} [options]

Options:
  --episodes N            Number of episodes (default: 100)
  --save-path PATH        Save location (default: agent.pth or agent.json)
  --render                Enable visualization
  
Evolutionary agent options:
  --evo-generations N     Generations per episode (default: 50)
  --evo-population N      Population size (default: 100)
  --evo-elite FRAC        Elite fraction (default: 0.2)
  --evo-mutation RATE     Mutation rate (default: 0.1)
```

### Eval Mode
```bash
python main.py eval --agent-type {simple|evolutionary} [options]

Options:
  --load-path PATH        Agent to load (default: agent.pth or agent.json)
  --episodes N            Number of episodes (default: 10)
  --render                Enable visualization
  --time-limit SECONDS    Per-episode time limit (default: 30.0)
  
Evolutionary agent options (same as train mode)
```

### Benchmark Mode
```bash
python main.py benchmark
```
Quick sanity check that verifies the environment is working correctly.

### Benchmark Suite Mode
```bash
python main.py benchmark-suite [options]

Options:
  --time-limit SECONDS    Per-episode time limit (default: 5.0)
  --render                Enable visualization (slow)
  
Evolutionary agent options:
  --evo-generations N     Generations per episode (default: 50)
  --evo-population N      Population size (default: 100)
  --evo-elite FRAC        Elite fraction (default: 0.2)
  --evo-mutation RATE     Mutation rate (default: 0.1)
```

### Benchmark Suite Mode

Runs the evolutionary agent on **every pattern** in `target_patterns.json` with a 5-second time limit per pattern. See [video](https://vimeo.com/1136735348).

**Basic usage:**
```bash
python main.py benchmark-suite --time-limit 5.0
```

**With custom evolutionary parameters:**
```bash
python main.py benchmark-suite --time-limit 5.0 \
  --evo-generations 100 --evo-population 200 \
  --evo-elite 0.3 --evo-mutation 0.15
```

**With rendering (warning: slow):**
```bash
python main.py benchmark-suite --time-limit 5.0 --render
```

**Output Files**

Creates two files with timestamps:

1. **JSON file** (`benchmark_evolutionary_TIMESTAMP.json`):
   - Complete results for each pattern
   - Summary statistics
   - All parameters used
   - Machine-readable for analysis

2. **Summary file** (`benchmark_evolutionary_TIMESTAMP_summary.txt`):
   - Human-readable overview
   - Key statistics (perfect matches, timeouts, avg reward, etc.)
   - Command to reproduce the benchmark

**Statistics Tracked**
- Perfect match rate (accuracy ≥ 99.9%)
- Timeout rate 
- Reward statistics (mean, std, median, min, max)
- Time statistics (mean, std, total)
- Per-pattern details

**Progress Display**

Uses `tqdm` for a progress bar while testing all patterns:
```
Testing patterns: 100%|██████████████| 100/100 [05:42<00:00,  3.42s/it]
```

## Evolutionary Algorithm Website

For browser-based experimentation with evolutionary approaches to this problem, see: https://evolutionary-ca-webgpu.onrender.com/


## Project Structure

```
├── rl_GoL_env_gym.py           # Gymnasium environment implementation
├── main.py                      # Unified CLI for testing, training, evaluation
├── generate_target_patterns.py # Utility to create achievable targets
├── evo_ca.py                    # Standalone evolutionary algorithm with visualization
├── target_patterns.json         # Generated target patterns (sampled during episodes)
├── agent.pth                    # Saved agent weights (created after training)
└── README.md                    # This file
```

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{gol_pattern_matching_env,
  author = {Khanlian, Jon},
  title = {Conway's Game of Life Pattern Matching RL Environment},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Digital-Physics/game-of-life-pattern-RL-gym}
}
```

## 📜 License

This project is licensed under the MIT License.