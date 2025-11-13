#!/usr/bin/env python3
"""
generate_target_patterns.py

Generate ALL achievable target patterns by exhaustively exploring all action 
sequences up to a given number of steps. This is suitable for small step counts.

Usage (runs exhaustive enumeration for 6 steps by default):
    python generate_target_patterns.py 
    python generate_target_patterns.py --num-steps 5
"""

import argparse
import json
import numpy as np
from collections import deque
import gymnasium as gym
import rl_GoL_env_gym
from tqdm import tqdm
import hashlib
from itertools import product
import os


# --- Helper Functions ---

def grid_to_hash(grid):
    """Convert grid to a hash string for deduplication."""
    return hashlib.md5(grid.tobytes()).hexdigest()


def grid_to_list(grid):
    """Convert numpy grid to list for JSON serialization."""
    # We only care about the first channel (the grid state)
    return grid.astype(int).tolist()


def list_to_grid(lst):
    """Convert list back to numpy grid."""
    return np.array(lst, dtype=np.int8)


# --- Core Enumeration Function ---

def exhaustive_enumeration(num_steps=6, grid_size=12):
    """
    Exhaustively enumerate all possible action sequences of length num_steps 
    and collect all unique final patterns.
    """
    print(f"\n=== Exhaustive Enumeration ===")
    print(f"Number of steps: {num_steps}")
    print(f"Grid size: {grid_size}x{grid_size}")
    
    # Total sequences: 21 is the default action space size (20 positions + 1 NOP)
    total_sequences = 21 ** num_steps
    print(f"Total sequences to evaluate: {total_sequences:,}")
    
    # Estimate time (assuming ~2000 sequences/second)
    if total_sequences > 0:
        print(f"Estimated time: ~{total_sequences / 2000:.0f} seconds (~{total_sequences / 2000 / 60:.1f} minutes)")
    
    # Check feasibility for num_steps > 6
    if num_steps > 6:
        print("WARNING: Exhaustive search for more than 6 steps will take a very long time.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return {}, {}

    env = gym.make("GoL-2x2-v0", grid_size=grid_size, max_steps=num_steps)
    unique_patterns = {}
    action_sequences = {}
    
    # Generate all action sequences
    all_action_sequences = product(range(21), repeat=num_steps)
    
    # Use tqdm to show progress over the total number of sequences
    with tqdm(total=total_sequences, desc="Evaluating sequences") as pbar:
        for action_seq in all_action_sequences:
            
            # Execute this sequence
            # Use a fixed seed (e.g., 0) as the environment should be deterministic
            # for a fixed initial state (empty grid in this case).
            obs, info = env.reset(seed=0) 
            
            terminated_early = False
            for action in action_seq:
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated:
                    terminated_early = True
                    break
            
            # Get final grid from observation
            if terminated_early:
                # If terminated, the final_grid should be the last step before termination,
                # but typically in GoL this means max_steps reached, so we take the observation.
                # Assuming the last observation is the final state.
                final_grid = obs['grid']  # obs[:, :, 0]
            else:
                final_grid = obs['grid']  # obs[:, :, 0]
            
            grid_hash = grid_to_hash(final_grid)
            
            # Store if unique
            if grid_hash not in unique_patterns:
                unique_patterns[grid_hash] = grid_to_list(final_grid)
                action_sequences[grid_hash] = list(action_seq)
            
            pbar.update(1)
            # Update description periodically
            if pbar.n % 1000 == 0:
                pbar.set_postfix({'unique_patterns': len(unique_patterns)})
    
    env.close()
    
    print(f"\nEvaluated all {total_sequences:,} sequences")
    print(f"Found {len(unique_patterns)} unique patterns")
    return unique_patterns, action_sequences


# --- Analysis and Saving Functions (Kept for completeness) ---

def analyze_patterns(unique_patterns):
    """Analyze the diversity of generated patterns."""
    print("\n=== Pattern Analysis ===")
    
    if not unique_patterns:
        print("No patterns found to analyze.")
        return

    densities = []
    for pattern in unique_patterns.values():
        grid = list_to_grid(pattern)
        density = np.mean(grid)
        densities.append(density)
    
    densities = np.array(densities)
    
    print(f"Total unique patterns: {len(unique_patterns)}")
    print(f"Density statistics:")
    print(f"  Mean: {np.mean(densities):.3f}")
    print(f"  Std:  {np.std(densities):.3f}")
    print(f"  Min:  {np.min(densities):.3f}")
    print(f"  Max:  {np.max(densities):.3f}")
    print(f"  Median: {np.median(densities):.3f}")
    
    # Count empty and full grids
    empty_count = sum(1 for d in densities if d == 0.0)
    full_count = sum(1 for d in densities if d == 1.0)
    print(f"\nSpecial patterns:")
    print(f"  Empty grids (all 0): {empty_count}")
    print(f"  Full grids (all 1):  {full_count}")


def save_patterns(unique_patterns, action_sequences, filename="target_patterns.json", max_save=None):
    """Save patterns to JSON file."""
    # Convert to list format for saving
    patterns_list = []
    for grid_hash, pattern in unique_patterns.items():
        # Convert all numpy types to Python native types
        actions = action_sequences.get(grid_hash, [])
        actions = [int(a) for a in actions]  # Convert numpy int64 to Python int
        
        patterns_list.append({
            'grid': pattern,
            'actions': actions,
            'hash': grid_hash
        })
    
    # Optionally limit number saved
    if max_save and len(patterns_list) > max_save:
        print(f"\nLimiting saved patterns to {max_save} (randomly selected)")
        indices = np.random.choice(len(patterns_list), max_save, replace=False)
        patterns_list = [patterns_list[i] for i in indices]
    
    # Check grid size (assuming all grids are the same size)
    grid_size = len(patterns_list[0]['grid']) if patterns_list else 0
    
    data = {
        'num_patterns': int(len(patterns_list)),  # Convert to Python int
        'grid_size': grid_size,
        'patterns': patterns_list
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"\n✓ Saved {len(patterns_list)} patterns to {filename}")
    
    # Print file size
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Exhaustively generate ALL achievable target patterns for GoL environment in N steps."
    )
    parser.add_argument('--num-steps', type=int, default=6,
                        help='Exact action sequence length for exhaustive search (default: 6)')
    parser.add_argument('--grid-size', type=int, default=12,
                        help='Grid size (default: 12)')
    parser.add_argument('--max-save', type=int, default=None,
                        help='Maximum patterns to save (default: save all)')
    parser.add_argument('--output', type=str, default='target_patterns.json',
                        help='Output JSON file (default: target_patterns.json)')
    parser.add_argument('--analyze-only', type=str, default=None,
                        help='Only analyze existing pattern file')
    
    args = parser.parse_args()
    
    # If analyzing existing file
    if args.analyze_only:
        print(f"Loading patterns from {args.analyze_only}...")
        with open(args.analyze_only, 'r') as f:
            data = json.load(f)
        # Extract grid data from the loaded JSON structure
        unique_patterns = {p['hash']: p['grid'] for p in data['patterns']}
        analyze_patterns(unique_patterns)
        return
    
    # Run the exhaustive generation
    unique_patterns, action_sequences = exhaustive_enumeration(
        num_steps=args.num_steps,
        grid_size=args.grid_size
    )
    
    # Analyze
    analyze_patterns(unique_patterns)
    
    # Save
    save_patterns(unique_patterns, action_sequences, 
                  filename=args.output, max_save=args.max_save)


# if __name__ == "__main__":
main()