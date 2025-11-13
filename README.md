# Reinforcement Learning Gym Environment for Pattern Matching in Conway's Game of Life

A Gymnasium-compatible environment for training RL agents to construct target patterns in a cellular automaton that evolves according to Conway's Game of Life rules.

![Screenshot of the pattern matching game](./screenshot_GoL_pattern_matching_game.png)

## Game Overview

**Objective**: Match a target pattern after exactly 10 timesteps of Game of Life evolution and agent actions.

**Core Mechanics**:
- Each turn: Game of Life update → Agent action 
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

### Verify Installation
```bash
uv run main.py benchmark
```

### Test Built-in Policies

```bash
# Random policy (baseline)
python main.py test --mode random --episodes 3 --render

# Greedy policy (smarter baseline)
python main.py test --mode greedy --episodes 5 --render

# Manual control (play yourself!)
python main.py test --mode manual
```

### Train and Evaluate an Agent

```bash
# Train with default settings
python main.py train --episodes 100

# Train your agent with custom save path and visualization
python main.py train --episodes 100 --save-path my_agent.pth --render

# Evaluate your agent
python main.py eval --load-path my_agent.pth --episodes 10 --render

# Evaluate sample agent
python main.py eval --load-path agent.pth --episodes 10 --render
```

## Implementing Your Own Agent

To use your own RL algorithm, replace the `SimpleAgent` class in `main.py` with your implementation. Required interface:

```python
class YourAgent:
    def __init__(self, action_space):
        # Initialize your model
        pass
    
    def select_action(self, obs: dict) -> int:
        # Return action index (0-20)
        pass
    
    def save(self, path: str):
        # Save model weights
        pass
    
    def load(self, path: str):
        # Load model weights
        pass
```

The training and evaluation infrastructure handles everything else (environment setup, episode loops, rendering, metrics).

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

1. **Reset**: Grid initialized to all dead cells; random target pattern loaded
2. **Each step**:
   - Game of Life update applied to grid
   - Agent action executed
3. **Termination**: After 10 steps, final reward computed

## Generating Achievable Target Patterns

By default, the environment samples from random patterns. To generate patterns that are provably achievable within N steps:

```bash
python generate_target_patterns.py --num-steps 6
```

This creates `target_patterns.json`, which the environment uses for sampling targets.

## Alternative: Evolutionary Algorithms

For browser-based experimentation with evolutionary approaches to this problem, see: https://evolutionary-ca-webgpu.onrender.com/

## Project Structure

```
├── rl_GoL_env_gym.py           # Gymnasium environment implementation
├── main.py                      # Unified CLI for testing, training, evaluation
├── generate_target_patterns.py # Utility to create achievable targets
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