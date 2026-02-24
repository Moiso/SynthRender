import sys
import time
import subprocess
import argparse
import os
import time 

from synthrender.scripts.parallel_create_synthetic_data import run_parallel_create_syn_data
from synthrender.scripts import __path__ as scripts_path 


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="Process arguments.")

    parser.add_argument('--num-frames', '-n', type=int, help="Number of frames to render")
    parser.add_argument("--num-gpus", '-g', type=int, help="Number of GPUs to use, if not specified it will use them all", default=-1)
    parser.add_argument("--config", '-c', type=str, help="Path to config file", default="config.yaml")

    parser.add_argument('--do_coco', '-dc', default=False, action=argparse.BooleanOptionalAction, type=str, help="Get COCO annotations")
    parser.add_argument('--do_bop', '-db', default=False, action=argparse.BooleanOptionalAction, help="Get BOP annotatios")
    parser.add_argument('--to_yolo', '-ty', default=False, action=argparse.BooleanOptionalAction, help="Transform coco annotations into yolo")
    parser.add_argument('--start', '-s', type=int, help="start frame to render", default=0)
    parser.add_argument('--end', '-e', type=int, help="end frame to render",default=-1)

    args = parser.parse_args()
    
    if args.to_yolo:
        args.do_coco = True

    return args.config, args.num_frames, args.num_gpus, args.do_coco, args.do_bop, args.to_yolo, args.start, args.end

def run_annotator(script_path, config_path:str, do_coco:bool, do_bop:bool, to_yolo:bool):
    ret = False
    try:
        command = ["blenderproc", "run", script_path, "-c", config_path]
        if do_coco: command.append('-dc')
        if do_bop: command.append('-db')
        if to_yolo: command.append('-ty')

        print(f"Command: [{' '.join(command)}]")
        process = subprocess.Popen(command)
        process.wait() # Wait for the process to complete or be interrupted

        ret = True

    except subprocess.CalledProcessError as e:
        print(f"Error while running BlenderProc command: {e}")
    except KeyboardInterrupt:
        print("Process interrupted. Terminating annotator...")
        process.terminate()
        process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()

        return ret

def create_dataset(config_file_path, n_frames, n_gpus, do_coco, do_bop, do_yolo, start_frame=0, stop_frame=-1):
    blenderproc_script = os.path.join(scripts_path[0], 'create_synthetic_data.py')
    annotator_script   = os.path.join(scripts_path[0], 'annotate_synthetic_data.py')
    
    try:
        # Call the function to create synthetic data...
        start = time.time()
        ret = run_parallel_create_syn_data("run", blenderproc_script, config_file_path, keyframes=n_frames, n_gpus=n_gpus, start_frame = start_frame, stop_frame=stop_frame)
        render_elapsed = time.time()-start

        print()

        # Call the function to annotate the synthetic data...
        start = time.time()
        if ret and any([do_coco, do_bop, do_yolo]):
            run_annotator(annotator_script, config_file_path, do_coco=do_coco, do_bop=do_bop, to_yolo=do_yolo)
        annotator_elapsed = time.time()-start

        print(f"\n> Total rendering time: {render_elapsed:.3f} s")
        print(f"> Total annotating time: {annotator_elapsed:.3f} s")

    except KeyboardInterrupt:
        print("Script interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    config_file_path, n_frames, n_gpus, do_coco, do_bop, do_yolo, start, stop = parse_args()

    assert os.path.exists(config_file_path), f"Error: Could not find config file in: {config_file_path}"

    create_dataset(config_file_path, n_frames, n_gpus, do_coco, do_bop, do_yolo, start_frame=start, stop_frame=stop)