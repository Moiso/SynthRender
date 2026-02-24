import blenderproc

import os
import sys
import argparse
import numpy as np
import random

from synthrender.utils.misc_utils import load_config
from synthrender.src.simulation import KeyframeGenerator
from synthrender.src.ext_camera import ExternalCameraSetup, ExternalCameraRender
from synthrender.src.png2mp4 import mp4Merger


def parse_args():
    debugmode = sys.argv[0] == 'debug'

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-layouts', '-n', type=int, help="Number of layouts to generate", default=-1)
    parser.add_argument('--interp-time', '-i', type=float, help="Time for the interpolation animation.", default=1)
    parser.add_argument('--paused-time', '-p', type=float, help="Time for the paused animation.", default=2)
    parser.add_argument('--camera-rps', '-r', type=float, help="Revolutions per second for the camera.", default=0.05)
    parser.add_argument('--mp4', action='store_true', help="Convert the generated frames into an mp4 video.")
    parser.add_argument("--config", '-c', type=str, help="Path to config file", default="config.yaml")
    parser.add_argument("--render_engine", type=str, choices=["cycles", "eeve"], help="Render engine to use.", default="cycles")
    parser.add_argument("--used-cameras", '-u', type=int, choices=[0, 1, 2], help="Cameras used for rendering. 0: the original one; 1: the external one; 2: both (two renders and slower)", default=1)

    # Optional argument for the new UI
    parser.add_argument("--verbose", action='store_true', help="Enable verbose mode (Used for parallelized UI)")
    
    args = parser.parse_args()

    # If start-frame is set but stop-frame not, raise issue.
    if args.num_layouts == -1:
        parser.error("You need to define at least one parameter [--num-keyframes]")

    use_cycles = args.render_engine == "cycles"

    return args.num_layouts, args.interp_time, args.paused_time, args.camera_rps, args.verbose, args.config, use_cycles, debugmode, args.mp4, args.used_cameras


if __name__ == "__main__":
    num_layouts, interp_time, paused_time, camera_rps, verbose, config_path, use_cycles, debugmode, mp4, used_cameras = parse_args()

    # Loading keyframe generator class
    config = load_config(config_path)
    keyframer = KeyframeGenerator(config_path)
    extcam_setup = ExternalCameraSetup(config_path)
    extcam_render = ExternalCameraRender(config_path)
    
    # Seed:
    seed = config["seed"] if config["seed"] != -1 else random.random()
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed used: {seed}")
    
    # Setting up the scene and keyframes.
    keyframer.set_up_keyframes(num_keyframes=num_layouts, start_frame=0, stop_frame=-1, verbose=verbose)
    print("Scene set up!")

    extcam_setup.set_up_keyframes(num_layouts=num_layouts, interp_time=interp_time, 
                            pause_time=paused_time, camera_rps=camera_rps, cam_dist=7, cam_height=4) # 8, 3
    
    if not debugmode:
        extcam_render.render_scene(use_cycles=use_cycles, used_cameras=used_cameras)
        print("Scene rendered")

        if mp4:
            cameras = [used_cameras]
            cameras = cameras if used_cameras != 2 else [0,1]
            
            for camera_id in cameras:

                print("Creating mp4...")
                # output_dir = extcam_render.output_dir_rgb
                output_dir = os.path.join(config["output_dir"], "animations", f"animation_{camera_id}_frames")
                sequence = os.path.join(output_dir, 'rgb_*.png') # pattern of files: RGB_0001.png, RGB_0002.png, …
                output = os.path.join(config["output_dir"], "animations", f'animation_{camera_id}.mp4')
                framerate = extcam_setup.sec2frames(1)      # “-framerate 24”

                mp4Merger.create_mp4_default(sequence, output, framerate)






