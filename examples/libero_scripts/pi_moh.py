import sys
sys.path.append("/home/txs/Code/Policy_Eval_Done_Right/MixtureOfHorizons/src")  # Change to your path!
sys.path.append("/home/txs/Code/Policy_Eval_Done_Right/LIBERO")  # Use standalone LIBERO
import os
import torch
import collections
import dataclasses
import math
import pathlib
import imageio
import numpy as np
import pickle
import json
from datetime import datetime
from openpi_client import image_tools
from openpi.policies import policy_config
from openpi.training import config as _config

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224  # resolution used to render training data


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    resize_size: int = 224
    replan_steps: int = 5
    rank: int = 0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    results_dir: str = ""  # Path to save results (config.json and result.json)
    video_out_path: str = ""  # Path to save videos (defaults to <results_dir>/<timestamp>/videos/ if not provided)

    seed: int = 7  # Random Seed (for reproducibility)
    default_prompt: str | None = None
    config: str = "pi05_libero"
    checkpoint_dir: str = ""
    save_gate_weights: bool = False
    horizons: list[int] = dataclasses.field(default_factory=lambda: [3, 6, 9, 12, 15, 18, 21, 24, 27, 30])

def eval_libero(args: Args) -> None:
    device = f"cuda:{args.rank}"
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    results_subdir = None
    if args.results_dir:
        results_subdir = pathlib.Path(args.results_dir) / f"{args.task_suite_name}_{timestamp}"
        results_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save config to config.json
        config_dict = dataclasses.asdict(args)
        with open(results_subdir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    # Set video_out_path to default if not provided
    if args.video_out_path:
        current_video_out_path = args.video_out_path + args.task_suite_name
    elif results_subdir:
        current_video_out_path = str(results_subdir / "videos")
    else:
        current_video_out_path = f"./videos/{args.task_suite_name}_{timestamp}"
    
    if not os.path.exists(current_video_out_path):
        os.makedirs(current_video_out_path, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)

    # Init Policy
    config = _config.get_config(args.config)
    config.horizons = args.horizons
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir, pytorch_device=device,
                                                 sample_kwargs={"ret_weights": True})

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {args.task_suite_name}")

    # pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = {}  # Store per-task results
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        num_failure_cases = 0
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        init_states_path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        print(f"Using init states: {init_states_path}")

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.rank)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_gate_weights = []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            cur_episode_gate_weights = []

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            print(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # print(f"original image shape: {img.shape, wrist_img}")
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }

                    # Query model to get action
                    with torch.inference_mode():
                        outputs = policy.infer(element)
                        action_chunk, gate_weights = outputs["actions"], outputs["gate_weights"]
                        cur_episode_gate_weights.append(np.asarray(gate_weights))
                    assert (
                            len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1
            task_gate_weights.append(cur_episode_gate_weights)

            # Save a replay video of the episode
            if done:
                suffix = "success"
            else:
                suffix = f"failure_{num_failure_cases}"
                num_failure_cases += 1

            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(current_video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        if args.save_gate_weights:
            with open(pathlib.Path(current_video_out_path) / f"rollout_{task_segment}_weights.pkl", 'wb') as f:
                pickle.dump(task_gate_weights, f)

        # Store per-task results
        task_name = task_description.replace(" ", "_")
        task_results[task_name] = {
            "rollouts": task_episodes,
            "successes": task_successes,
            "failures": task_episodes - task_successes,
            "success_rate": float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0,
            "checkpoint_path": args.checkpoint_dir,
            "config_name": args.config
        }

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # Close the environment to release resources properly
        env.close()

    print(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    print(f"Total episodes: {total_episodes}")
    
    # Save results to result.json
    if results_subdir:
        results = {
            "tasks": task_results,
            "suite": {
                "rollouts": total_episodes,
                "successes": total_successes,
                "failures": total_episodes - total_successes,
                "success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
                "task_suite_name": args.task_suite_name,
                "checkpoint_path": args.checkpoint_dir
            }
        }
        with open(results_subdir / "result.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_subdir / 'result.json'}")


def _get_libero_env(task, resolution, seed, rank=0):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution,
                "render_gpu_device_id": rank}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    eval_libero(tyro.cli(Args))
