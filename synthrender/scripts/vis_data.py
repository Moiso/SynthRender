import os
import argparse

from synthrender.utils import misc_utils
from synthrender.utils.bproc_utils import run_cli

def setup_argparser() -> str:
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--config_path", '-c', type=str, help="Path to config.json file", default="config.yaml")
    parser.add_argument('vis_option', type=str, help="Data to visualize", choices=["coco", "hdf5"])
    parser.add_argument('frame_id', type=int, help="Frame id to visualize")

    args = parser.parse_args()

    return args.config_path, args.vis_option, args.frame_id

if __name__ == "__main__":
    config_path, vis_option, frame_id = setup_argparser()
    config = misc_utils.load_config(config_path)

    cfg_output_dir = os.path.abspath(config["output_dir"] or "./output")

    if vis_option == "coco":
            command = ["blenderproc", "vis", vis_option, "-b", os.path.join(cfg_output_dir, "coco"), "-i", str(frame_id)]


    elif vis_option == "hdf5":
            command = ["blenderproc", "vis", vis_option, os.path.join(cfg_output_dir, "hdf5", f"{frame_id}.hdf5")]

    run_cli.run_bproc_cli(command)

