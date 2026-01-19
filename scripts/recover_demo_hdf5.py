
import argparse
import datetime
import h5py
import json
import numpy as np
import os
import robosuite as suite
from glob import glob
from robosuite import load_controller_config
import init_path
import libero.libero.envs.bddl_utils as BDDLUtils

def gather_demonstrations_as_hdf5(directory, out_dir, env_info, args):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print(f"Creating hdf5 at: {hdf5_path}")
    
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None 

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return

    # Iterate over subdirectories in the raw data folder
    # Raw data structure is usually: directory/ep_timestamp/state_*.npz
    # Or sometimes the directory IS the parent of ep folders.
    
    # We assume 'directory' contains subfolders like 'ep_123...'
    # Let's verify if we need to filter for specific task names or just take all folders.
    # collect_demonstration.py iterates os.listdir(directory)
    
    ep_directories = sorted(os.listdir(directory))
    
    for ep_directory in ep_directories:
        full_ep_path = os.path.join(directory, ep_directory)
        if not os.path.isdir(full_ep_path):
            continue
            
        state_paths = os.path.join(full_ep_path, "state_*.npz")
        state_files = sorted(glob(state_paths))
        
        if not state_files:
            print(f"Skipping {ep_directory}: No state files found.")
            continue
            
        states = []
        actions = []

        for state_file in state_files:
            try:
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])

                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
            except Exception as e:
                print(f"Error loading {state_file}: {e}")
                continue

        if len(states) == 0:
            print(f"Skipping {ep_directory}: Empty states.")
            continue

        # Data collection wrapper logic: Delete the first actions and the last state.
        if len(states) > 0:
            del states[-1]
            
        if len(states) != len(actions):
            print(f"Warning: {ep_directory} length mismatch. States: {len(states)}, Actions: {len(actions)}")
            # Truncate to min length to be safe, or skip
            min_len = min(len(states), len(actions))
            states = states[:min_len]
            actions = actions[:min_len]

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml
        xml_path = os.path.join(full_ep_path, "model.xml")
        xml_str = ""
        if os.path.exists(xml_path):
            with open(xml_path, "r") as xml_f:
                xml_str = xml_f.read()
        else:
             print(f"Warning: model.xml not found for {ep_directory}")

        ep_data_grp.attrs["model_file"] = xml_str

        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        print(f"Processed demo_{num_eps} from {ep_directory}")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name if env_name else "Unknown"
    grp.attrs["env_info"] = env_info
    
    # Needs problem info
    if args.bddl_file and os.path.exists(args.bddl_file):
        problem_info = BDDLUtils.get_problem_info(args.bddl_file)
        grp.attrs["problem_info"] = json.dumps(problem_info)
        grp.attrs["bddl_file_name"] = args.bddl_file
        with open(args.bddl_file, "r", encoding="utf-8") as bddl_f:
            grp.attrs["bddl_file_content"] = str(bddl_f.read())
    else:
        print("Warning: BDDL file not found or not provided. Metadata might be incomplete.")
        grp.attrs["problem_info"] = json.dumps({})

    f.close()
    print(f"Finished. Saved {num_eps} demonstrations to {hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the directory containing raw demonstrations (e.g. demonstration_data/tmp/...)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to where to store the output demo.hdf5 file",
    )
    parser.add_argument(
        "--bddl-file",
        type=str,
        required=True,
        help="Path to the BDDL file used for the task",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use",
    )
    
    args = parser.parse_args()
    
    # Mock env_info
    controller_config = load_controller_config(default_controller="OSC_POSE")
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    env_info = json.dumps(config)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    gather_demonstrations_as_hdf5(args.directory, args.out_dir, env_info, args)
