import os
import argparse

from synthrender.utils.misc_utils import load_config
from synthrender.src.png2mp4 import mp4Merger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", '-c', type=str, help="Path to config file", default="config.yaml")
    
    args = parser.parse_args()

    return args.config


if __name__ == "__main__":
    config_path = parse_args()

    config = load_config(config_path)
    cfg_output_dir = os.path.abspath(config["output_dir"] or "./output")
    output_dir = os.path.join(cfg_output_dir, "animation")
    sequence = os.path.join(output_dir, 'rgb_*.png')
    output = os.path.join(output_dir, 'animation.mp4')
    framerate = 24

    if not os.path.isdir(output_dir):
        raise FileExistsError(f"Animation folder not found!: {output_dir}")
    
    print("Creating mp4...")
    mp4Merger.create_mp4_default(sequence, output, framerate)