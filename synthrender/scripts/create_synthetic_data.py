import blenderproc

import argparse
import numpy as np
import random

from synthrender.src.simulation import KeyframeGenerator
from synthrender.src.simulation import KeyframesRenderer
from synthrender.utils.misc_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-keyframes', '-n', type=int, help="Number of keyframes to generate", default=-1)
    parser.add_argument("--start-frame", '-s', type=int, help="Frame at which the render should start (inclusive). If set, stop-frame must be set too.", default=-1)
    parser.add_argument("--stop-frame", '-e', type=int, help="Frame at which the render should stop (inclusive). If set, start-frame will be set to 0 unless it has already being set.", default=-1)
    parser.add_argument("--gpu-id", '-g', type=int, help="GPU id to use, if not specified it will use them all", default=-1)
    parser.add_argument("--config", '-c', type=str, help="Path to config file", default="config.yaml")

    # Optional argument for the new UI
    parser.add_argument("--verbose", action='store_true', help="Enable verbose mode (Used for parallelized UI)")
    
    args = parser.parse_args()

    # If start-frame is set but stop-frame not, raise issue.
    if args.start_frame == -1 and args.stop_frame == -1 and args.num_keyframes == -1:
        parser.error("You need to define at least one parameter [--num-keyframes, --stop-frame]")

    # If no gpu selected, set None.
    args.gpu_id = args.gpu_id if args.gpu_id > -1 else None

    # If start-frame is set but stop-frame not, raise issue.
    if args.start_frame != -1 and args.stop_frame == -1:
        parser.error("--stop-frame is required when --start-frame is specified.")

    # If stop-frame is set but start-frame not, set start-frame to 0
    if args.start_frame == -1 and args.stop_frame != -1:
        args.start_frame = 0

    # If not defined, adapt num-keyframes to end-frame range.
    if args.num_keyframes == -1:
        args.num_keyframes = args.stop_frame+1 

    # If start-frame is set but stop-frame not, raise issue.
    if args.start_frame != -1 and args.stop_frame == -1:
        parser.error("--stop-frame is required when --start-frame is specified.")


    return args.num_keyframes, args.start_frame, args.stop_frame, args.gpu_id, args.verbose, args.config


if __name__ == "__main__":
    num_keyframes, start_frame, stop_frame, gpu_id, verbose, config_path = parse_args()

    # Loading keyframe generator class
    config = load_config(config_path)
    keyframer = KeyframeGenerator(config_path)
    renderer  = KeyframesRenderer(config_path)
    
    # Seed:
    seed = config["seed"] if config["seed"] != -1 else random.random()
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed used: {seed}")
    
    # Setting up the scene and keyframes.
    keyframer.set_up_keyframes(num_keyframes=num_keyframes, start_frame=start_frame, stop_frame=stop_frame, verbose=verbose)
    print("Scene set-up!")

    # Render keyframes within an interval.
    print(f"start_frame: {start_frame}, stop_frame: {stop_frame}, n_keyframes: {num_keyframes}")
    if stop_frame != -1 and stop_frame >= start_frame:
        renderer.render_scene(num_frames=num_keyframes, seed=seed, start_frame=start_frame, stop_frame=stop_frame, desired_gpu=gpu_id, verbose=verbose)
        renderer.post_render()
        print("Scene rendered!")
    else:
        print("Warning! stop_frame < start_frame, skipping rendering.")