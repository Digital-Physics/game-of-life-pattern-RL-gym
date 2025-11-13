# test_saved_agent.py
import gymnasium as gym
import rl_GoL_env_gym  # Ensures the "GoL-2x2-v0" environment is registered
import torch
import os
import argparse
import sys

# Assume the SimpleAgent class is defined or imported from integrated_RL_example
# For a standalone script, we need the Agent class definition here or imported.
# For simplicity, we'll redefine the necessary parts or assume a successful import:

# --- Necessary Agent Definitions (Adapted from integrated_RL_example.py) ---
AGENT_FILE_PATH = 'simple_agent.pth' 
NUM_TEST_EPISODES = 10 

class SimpleAgent:
    """A dummy agent demonstrating the select_action interface."""
    def __init__(self, action_space):
        self.action_space = action_space
        # Model architecture must match the saved file
        self.dummy_policy = torch.nn.Linear(1, action_space.n) 
    
    def select_action(self, obs: dict) -> int:
        """
        In a real scenario, this uses the loaded policy network for inference.
        For this simple agent, it's still random, but the loading process is correct.
        """
        # In a real agent, this would be: action = self.policy(obs).argmax().item()
        action = self.action_space.sample() 
        return action

def load_agent(action_space, path):
    """Loads an agent's policy state dictionary and returns the agent."""
    if not os.path.exists(path):
        print(f"!! Error: Saved model not found at '{path}'.")
        print("!! Please run 'uv run integrated_RL_example.py' first to generate the saved agent.")
        sys.exit(1)

    loaded_agent = SimpleAgent(action_space)
    state_dict = torch.load(path)
    loaded_agent.dummy_policy.load_state_dict(state_dict)
    
    print(f"<- Agent loaded successfully from {path}.")
    return loaded_agent
# --------------------------------------------------------------------------

def run_test_episode(env, agent, episode_num):
    """Runs a single episode with rendering."""
    print(f"\n--- Testing Episode {episode_num} ---")
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    
    while not (terminated or truncated):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        env.render() # Always render in this test script
        
        if terminated:
            print(f"Episode terminated after {step_count} steps. Final Match Accuracy: {info['accuracy']:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Test a saved RL agent in the GoL-2x2-v0 environment.")
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=NUM_TEST_EPISODES, 
        help=f"Number of episodes to run (default: {NUM_TEST_EPISODES})."
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=AGENT_FILE_PATH, 
        help=f"Path to the saved PyTorch agent file (default: {AGENT_FILE_PATH})."
    )
    
    args = parser.parse_args()
    
    # 1. Setup Environment with rendering
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, render_mode="human")
    
    # 2. Load the Agent
    loaded_agent = load_agent(env.action_space, args.model_path)
    
    # 3. Run Tests
    print(f"\n[TESTING SAVED AGENT] Running {args.episodes} episodes with visualization...")
    for episode in range(1, args.episodes + 1):
        run_test_episode(env, loaded_agent, episode)

    env.close()
    print("\nTesting complete.")

if __name__ == "__main__":
    main()