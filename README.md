# Reinforcement Learning Gym Environment for a Pattern Matching Game in Conway's "Game of Life"

A Gymnasium-compatible environment for training RL agents to create action sequences that lead to constructing a target pattern in a cellular environment that updates according to Conway's "Game of Life".

## Game Overview - Pattern Matching in Conway's "Game of Life"
• Goal: Match Target Pattern at step 10  
• Actions (Keyboard Shortcuts for Manual Play):  

- Arrow Keys: MOVE 2x2 write window Up, Down, Left, or Right  
- Space Bar: PASS  
- Hex Keys(0-9 & A-F): WRITE a 2×2 pattern 

• Update Rule: An action (i.e. move, pass, or write) is always preceded by a "Game of Life" update step.   

![](./screenshot_GoL_pattern_matching_game.png)

## Python UV Environment Installation

```bash
uv pip install .
```

## Quick Start

### 1. Test with random policy
```bash
python test_env.py --mode random --render
```

### 2. Test with greedy heuristic
```bash
uv run test_env.py --mode greedy --render
```

### 3. Manual game play!
```bash
uv run test_env.py --mode manual
```

### 4. Test a saved agent
First, run the RL integration example to save the model:
```bash
uv run integrated_RL_example.py
```
```
uv run test_saved_agent.py --episodes 5
```

## Using the Environment in Your RL Code

```python
import gymnasium as gym
import rl_GoL_env_gym  # Registers "GoL-2x2-v0"

# Create environment
env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10)

# Training loop
obs, info = env.reset()
target = info['target']  # The pattern to match

for episode in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Your agent selects action (0-20)
        # obs is now a dict with 'grid', 'head_mask', and 'remaining_steps'
        action = agent.select_action(obs)
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # reward is 0.0 until step 10, then it's the match percentage
        if terminated:
            print(f"Match accuracy: {info['accuracy']:.2%}")

env.close()
```

## Evolutionary Algorithms (alternative exploration)
Alternatively, to play the game in the browser and experiment with Evolutionary Algorithms for mutating action sequences to find a good solution, check out https://evolutionary-ca-webgpu.onrender.com/

## Environment Details

### Observation Space
- **Type**: `Dict`
- **'grid' Shape**: `(12, 12)` with dtype `int8`. Current Game of Life grid (0=dead, 1=alive).
- **'head_mask' Shape**: `(12, 12)` with dtype `int8`. Write head position mask (1 for the 2×2 write area).
- **'remaining_steps' Shape**: `(1,)` with dtype `int32`. Steps remaining in the episode.

---

### Action Space
- **Type**: `Discrete(21)`
- Actions:
  - `0`: Move head up
  - `1`: Move head down
  - `2`: Move head left
  - `3`: Move head right
  - `4`: Pass (do nothing)
  - `5-20`: Write patterns 0x0 to 0xF as 2×2 grids

Pattern encoding (4 bits → 2×2 grid):
```
bit 0 = top-left
bit 1 = top-right
bit 2 = bottom-left
bit 3 = bottom-right
```

### Reward Structure
- **Steps 0-9**: `reward = 0.0`
- **Step 10 (terminal)**: `reward = matching_cells / total_cells`
  - Range: [0.0, 1.0]
  - 1.0 = perfect match

### Episode Flow
1. Environment resets with empty grid and random target pattern
2. Each step:
   - Game of Life update is applied **first**
   - Then the agent's action is executed
3. Episode terminates after 10 steps
4. Final reward based on match percentage

## Example RL Training Integration

### RL Agent Integration (Example + Command Line Tool) 

The `integrated_RL_example.py` file demonstrates the complete workflow for an RL agent: saving a policy (after "training") and then loading it back for testing.

### To train, save, load, and test a sample RL model

### A. Save the default agent
```

python integrated_RL_example.py train_save --episodes 3
```
### B. Load the default agent and test
```
python integrated_RL_example.py test_load --episodes 2 --render
```

### To save the simulated training to a file named new_experiment.pth:
```
uv run integrated_RL_example.py train_save --episodes 10 --load-file new_experiment.pth
```

### To load an agent from a file named my_custom_agent.pth and test it:
```
uv run integrated_RL_example.py test_load --load-file my_custom_agent.pth --render
```

## Utility: Generate Achievable Target Patterns to Sample

Instead of using random target patterns, we can generate patterns that are actually achievable within N steps:

```bash
uv run python generate_target_patterns.py --num-steps 6
```

## Files

- `rl_GoL_env_gym.py` - Main Gymnasium environment
- `test_env.py` - Testing scripts
- `generate_target_patterns.py` - Utility to create achievable target patterns
- `target_patterns.json` - Output file from generate_target_patterns.py that is sampled for training and testing
- `integrated_RL_example.py` - Shows you how to integrate an RL agent (and save and load a trained agent)
- `simple_agent.pth` - Saved demo agent
