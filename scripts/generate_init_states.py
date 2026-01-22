import argparse
import os
import glob
import torch
import numpy as np
import robosuite
import init_path
from libero.libero.envs import BDDLBaseDomain
from libero.libero import get_libero_path

def generate_init_states(bddl_folder, save_folder, num_states=20):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    bddl_files = glob.glob(os.path.join(bddl_folder, "*.bddl"))
    print(f"Found {len(bddl_files)} BDDL files in {bddl_folder}")

    for bddl_file in bddl_files:
        task_name = os.path.splitext(os.path.basename(bddl_file))[0]
        # output file path
        output_file = os.path.join(save_folder, f"{task_name}.init")
        output_file_pruned = os.path.join(save_folder, f"{task_name}.pruned_init")
        
        if os.path.exists(output_file_pruned):
            print(f"Already exists: {output_file_pruned}, skipping...")
            continue
            
        print(f"Generating init states for {task_name}...")

        try:
            # Initialize environment
            # Note: We use the Panda robot as default. 
            # Make sure your LIBERO installation supports it (it usually does as it is the default).
            env = BDDLBaseDomain(
                bddl_file_name=bddl_file,
                robots=["Panda"],
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                use_object_obs=True,
                control_freq=20,
            )
        except Exception as e:
            print(f"FAILED to initialize env for {task_name}: {e}")
            continue

        init_states = []
        success_count = 0
        
        # We try to collect `num_states` valid states.
        attempts = 0
        max_attempts = num_states * 3
        
        while success_count < num_states and attempts < max_attempts:
            attempts += 1
            try:
                env.reset()
                # Get the simulation state and flatten it to 1D array
                sim_state = env.sim.get_state().flatten()
                init_states.append(sim_state)
                success_count += 1
            except Exception as e:
                # Reset might fail if placement sampling fails repeatedly
                print(f"Reset failed: {e}")
        
        env.close()

        if success_count > 0:
            torch.save(init_states, output_file)
            torch.save(init_states, output_file_pruned)
            print(f"Saved {success_count} states to {output_file_pruned}")
        else:
            print(f"Could not generate any valid states for {task_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-folder", type=str, required=True, help="Path to the folder containing .bddl files")
    parser.add_argument("--save-folder", type=str, required=True, help="Path to the output folder for .init files")
    parser.add_argument("--num-states", type=int, default=20, help="Number of init states to generate per task")
    args = parser.parse_args()

    generate_init_states(args.bddl_folder, args.save_folder, args.num_states)
