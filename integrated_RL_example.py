import gymnasium as gym
# IMPORTANT: Import the module that registers the environment
import rl_GoL_env_gym
import torch
import numpy as np
import os 
import argparse

# --- Configuration Constants ---
AGENT_FILE_PATH = 'simple_agent.pth' # Default file path
GRID_SIZE = 12
# OBS_SIZE: grid (144) + target (144) + head_mask (144) + remaining_steps (1) = 433
OBS_SIZE = GRID_SIZE * GRID_SIZE * 3 + 1
ACTION_SIZE = 21 

# Define device based on availability (or default to CPU)
DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# A minimal "Agent" class to demonstrate the interface.
class SimpleAgent:
    """A dummy agent demonstrating the select_action interface."""
    def __init__(self, action_space):
        self.action_space = action_space
        # FIX: Set the input size to the full observation space (OBS_SIZE=433)
        self.dummy_policy = torch.nn.Linear(OBS_SIZE, action_space.n).to(DEVICE)
    
    # A simplified select_action, as requested in the base file
    def select_action(self, obs: dict) -> int:
        """
        Takes the dictionary observation and returns a single action index (int).
        """
        # Simplest policy: Choose a random action
        action = self.action_space.sample()
        return action

def save_agent(agent, path):
    """Saves the agent's policy state dictionary."""
    print(f"-> Saving agent state to {path}...")
    torch.save(agent.dummy_policy.state_dict(), path)
    print("-> Agent saved.")

def load_agent(action_space, path):
    """Loads an agent's policy state dictionary and returns the agent."""
    print(f"<- Loading agent state from {path}...")
    
    if not os.path.exists(path):
        print(f"!! Error: Model file not found at {path}. Returning new (untrained) agent.")
        return SimpleAgent(action_space)

    # 1. Initialize the model architecture (must match the saved model)
    loaded_agent = SimpleAgent(action_space)
    
    # 2. Load the saved state dictionary to the correct device
    state_dict = torch.load(path, map_location=DEVICE)
    
    # 3. Apply the state dictionary to the model
    loaded_agent.dummy_policy.load_state_dict(state_dict)
    
    print("<- Agent loaded successfully.")
    return loaded_agent

def run_episode(env, agent, episode_num, render):
    """Runs a single episode and prints results."""
    print(f"\n--- Running Episode {episode_num} ---")
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    
    while not (terminated or truncated):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        # FIX: Explicitly call env.render() when rendering is requested
        if render:
            env.render()
        
        if terminated:
            print(f"Episode terminated after {step_count} steps. Final Match Accuracy: {info['accuracy']:.2%}")

def train_and_save(episodes, render, file_path):
    """Simulates training and saves the agent."""
    print("\n[PHASE: SIMULATED TRAINING AND SAVING]")
    
    render_mode = "human" if render else None
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, render_mode=render_mode)
    
    trained_agent = SimpleAgent(env.action_space)
    
    print(f"Simulating a training run ({episodes} episodes)...")
    for episode in range(1, episodes + 1):
        run_episode(env, trained_agent, episode, render)
        
    # USE THE PASSED file_path FOR SAVING
    save_agent(trained_agent, file_path)
    env.close()

def test_and_load(episodes, render, file_path):
    """Loads an agent and tests it."""
    print("\n[PHASE: TESTING LOADED AGENT]")
    
    render_mode = "human" if render else None
    env = gym.make("GoL-2x2-v0", grid_size=12, max_steps=10, render_mode=render_mode)

    # USE THE PASSED file_path FOR LOADING
    loaded_agent = load_agent(env.action_space, file_path)
    
    if loaded_agent:
        print(f"Testing loaded agent ({episodes} episodes)...")
        for episode in range(1, episodes + 1):
            run_episode(env, loaded_agent, episode, render)
    else:
        print("Cannot test agent: Model could not be loaded.")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate RL Agent integration for GoL-2x2-v0.")
    
    # Define POSITIONAL argument for mode
    parser.add_argument(
        "mode", 
        type=str, 
        choices=["train_save", "test_load"],
        help="Mode of operation: 'train_save' to simulate training and save, or 'test_load' to load and test."
    )
    
    # Define OPTIONAL arguments
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=3, 
        help="Number of episodes to run in the selected mode (default: 3)."
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Enable human rendering during episode run."
    )
    # ADDED ARGUMENT for custom file path
    parser.add_argument(
        "--load-file", 
        type=str, 
        default=AGENT_FILE_PATH, 
        help=f"File path to load/save the agent policy from (default: {AGENT_FILE_PATH})."
    )
    
    args = parser.parse_args()

    if args.mode == "train_save":
        train_and_save(args.episodes, args.render, args.load_file) 
    elif args.mode == "test_load":
        test_and_load(args.episodes, args.render, args.load_file)
    
    print("\nDemonstration complete.")