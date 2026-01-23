import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros
import glob
import traceback
import sys
import re

import init_path
import libero.libero.utils.utils as libero_utils
import cv2
from PIL import Image
from robosuite.utils import camera_utils

from libero.libero.envs import *
from libero.libero import get_libero_path

BASE_DATA_DIR = "/inspire/qb-ilm/project/wuliqifa/czxs25210147/data"

def create_dataset(
    demo_file,
    dataset_path,
    dataset_name,
    camera_height,
    camera_width,
    use_depth=False,
    no_proprio=False,
    use_camera_obs=True,
):
    hdf5_path = demo_file
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]

    env_args = f["data"].attrs["env_info"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])

    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    bddl_file_name = f["data"].attrs["bddl_file_name"]
    
    # Fix absolute path for BDDL file
    path_pattern = r'/[^"\']*?/libero/libero/bddl_files'
    current_bddl_root = os.path.join(os.getcwd(), "libero/libero/bddl_files")
    if os.path.exists(current_bddl_root):
        bddl_file_name = re.sub(path_pattern, current_bddl_root, bddl_file_name)
    
    if not os.path.exists(bddl_file_name):
        print(f"[Warning] BDDL file does not exist at processed path: {bddl_file_name}")

    target_hdf5_path = os.path.join(dataset_path, f"{dataset_name}.hdf5")

    if os.path.exists(target_hdf5_path):
        file_size = os.path.getsize(target_hdf5_path)
        # 1GB = 1 * 1024 * 1024 * 1024 bytes
        if file_size > 1 * 1024 * 1024 * 1024:
            print(f"Skipping {dataset_name}, valid file exists ({file_size/1024/1024:.2f} MB) at {target_hdf5_path}")
            f.close()
            return
        else:
            print(f"Overwriting {dataset_name}, file exists but size is abnormal ({file_size/1024/1024:.2f} MB < 1GB) at {target_hdf5_path}")

    output_parent_dir = Path(target_hdf5_path).parent
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    h5py_f = h5py.File(target_hdf5_path, "w")

    grp = h5py_f.create_group("data")

    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not use_camera_obs,
        has_offscreen_renderer=use_camera_obs,
        ignore_done=True,
        use_camera_obs=use_camera_obs,
        camera_depths=use_depth,
        camera_names=[
            "robot0_eye_in_hand",
            "agentview",
        ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_segmentations=None,
    )

    grp.attrs["bddl_file_name"] = bddl_file_name
    try:
        if os.path.exists(bddl_file_name):
            grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
        else:
            print(f"[Warning] BDDL file not found at: {bddl_file_name}")
    except Exception as e:
        print(f"[Warning] Failed to read BDDL content: {e}")

    # Initialize environment
    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )

    env_args_dict = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": bddl_file_name,
        "env_kwargs": env_kwargs,
    }

    grp.attrs["env_args"] = json.dumps(env_args_dict)
    
    total_len = 0
    cap_index = 5

    try:
        for (i, ep) in enumerate(demos):
            print(f"Processing episode {i+1}/{len(demos)} for {dataset_name}...")

            model_xml = f["data/{}".format(ep)].attrs["model_file"]
            
            # Replace absolute paths that may be from a different machine
            # Matches any absolute path ending in /libero/libero/assets and replaces with local path
            path_pattern = r'/[^"\']*?/libero/libero/assets'
            current_assets_path = os.path.join(os.getcwd(), "libero/libero/assets")
            if os.path.exists(current_assets_path):
                model_xml = re.sub(path_pattern, current_assets_path, model_xml)
            else:
                print(f"[Warning] Local assets path not found: {current_assets_path}")

            reset_success = False
            
            # Simple retry mechanism for reset
            for _ in range(5):
                try:
                    env.reset()
                    reset_success = True
                    break
                except:
                    continue
            
            if not reset_success:
                print(f"Failed to reset environment for episode {ep}")
                continue

            model_xml = libero_utils.postprocess_model_xml(model_xml, {})

            if not use_camera_obs:
                env.viewer.set_camera(0)

            states = f["data/{}/states".format(ep)][()]
            actions = np.array(f["data/{}/actions".format(ep)][()])

            num_actions = actions.shape[0]

            init_idx = 0
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[init_idx])
            env.sim.forward()
            model_xml = env.sim.model.get_xml()

            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []

            agentview_images = []
            eye_in_hand_images = []

            agentview_depths = []
            eye_in_hand_depths = []

            rewards = []
            dones = []

            valid_index = []

            for j, action in enumerate(actions):

                obs, reward, done, info = env.step(action)

                if j < num_actions - 1:
                    state_playback = env.sim.get_state().flatten()
                    err = np.linalg.norm(states[j + 1] - state_playback)

                    if err > 0.01:
                        # Warning but continue
                        pass

                if j < cap_index:
                    continue

                valid_index.append(j)

                if not no_proprio:
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])

                    joint_states.append(obs["robot0_joint_pos"])

                    ee_states.append(
                        np.hstack(
                            (
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"]),
                            )
                        )
                    )

                robot_states.append(env.get_robot_state_vector(obs))

                if use_camera_obs:
                    if use_depth:
                        agentview_depths.append(obs["agentview_depth"])
                        eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                    agentview_images.append(obs["agentview_image"])
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                else:
                    env.render()

            states_final = states[valid_index]
            actions_final = actions[valid_index]
            dones = np.zeros(len(actions_final)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions_final)).astype(np.uint8)
            rewards[-1] = 1
            
            ep_data_grp = grp.create_group(f"demo_{i}")

            obs_grp = ep_data_grp.create_group("obs")
            if not no_proprio:
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

            obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
            obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
            
            if use_depth:
                obs_grp.create_dataset("agentview_depth", data=np.stack(agentview_depths, axis=0))
                obs_grp.create_dataset("eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0))

            ep_data_grp.create_dataset("actions", data=actions_final)
            ep_data_grp.create_dataset("states", data=states_final)
            ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep_data_grp.create_dataset("rewards", data=rewards)
            ep_data_grp.create_dataset("dones", data=dones)
            ep_data_grp.attrs["num_samples"] = len(agentview_images)
            ep_data_grp.attrs["model_file"] = model_xml
            ep_data_grp.attrs["init_state"] = states[init_idx]
            total_len += len(agentview_images)

    finally:
        grp.attrs["num_demos"] = len(demos)
        grp.attrs["total"] = total_len
        env.close()
        h5py_f.close()
        f.close()
        
    print(f"Finished processing {dataset_name}. Saved to {target_hdf5_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["gr00t", "openpi"],
        help="Configuration: gr00t (224x224) or openpi (256x256)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=f"{BASE_DATA_DIR}/collected_data/",
        help="Root directory for collected data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{BASE_DATA_DIR}/converted_data/",
        help="Root directory for converted data"
    )
    
    parser.add_argument("--use-depth", action="store_true")

    args = parser.parse_args()

    if args.config == "gr00t":
        camera_height = 224
        camera_width = 224
    else:
        # openpi
        camera_height = 256
        camera_width = 256
        
    print(f"Configuration: {args.config} (Res: {camera_width}x{camera_height})")
    print(f"Searching in: {args.data_dir}")

    # Search for all demo.hdf5 files
    # Pattern: DATA_DIR / * / demo / demo.hdf5
    search_pattern = os.path.join(args.data_dir, "*", "demo", "demo.hdf5")
    demo_files = glob.glob(search_pattern)
    
    if not demo_files:
        print("No demo files found matching pattern */demo/demo.hdf5")
        return
        
    print(f"Found {len(demo_files)} datasets.")
    
    # Destination directory
    output_base_dir = os.path.join(args.output_dir, args.config)
    os.makedirs(output_base_dir, exist_ok=True)
    
    for demo_file in demo_files:
        # demo_file: .../task_name/demo/demo.hdf5
        # task_name is grandparent dir name
        task_name = os.path.basename(os.path.dirname(os.path.dirname(demo_file)))
        
        print(f"Processing task: {task_name}")
        
        try:
            create_dataset(
                demo_file=demo_file,
                dataset_path=output_base_dir,
                dataset_name=task_name,
                camera_height=camera_height,
                camera_width=camera_width,
                use_depth=args.use_depth,
                use_camera_obs=True
            )
        except Exception as e:
            print(f"FAILED processing {task_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
