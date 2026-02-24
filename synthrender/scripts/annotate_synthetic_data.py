import blenderproc

import os
import random
import numpy as np
import argparse

from tqdm import tqdm 

import time
from datetime import datetime, timedelta

from synthrender.utils import hdf5_utils


from synthrender.utils import misc_utils
from synthrender.src.annotation import Coco_Annotator
from synthrender.src.annotation import Bop_Annotator
from synthrender.src.yolo import coco2yolo

def parse_args() -> tuple[int, int, str]:
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--config", '-c', type=str, help="Path to config file", default="config.yaml")
    
    parser.add_argument('--do_coco', '-dc', default=False, action=argparse.BooleanOptionalAction, type=str, help="Get COCO annotations")
    parser.add_argument('--do_bop', '-db', default=False, action=argparse.BooleanOptionalAction, help="Get BOP annotatios")
    parser.add_argument('--to_yolo', '-ty', default=False, action=argparse.BooleanOptionalAction, help="Transform coco annotations into yolo")
    parser.add_argument('--start-index', '-s', default=0, type=int, help="Makese the annotator start at a specific index")
    parser.add_argument('--stop-index', '-e', default=-1, type=int, help="Makese the annotator stop at a specific index (inclusive)")
    parser.add_argument('--output-dir', '-o', default=None, type=str, help="Path for the output directory (if not set it will default to the one in the config file)")

    args = parser.parse_args()
    
    return args.config, args.do_coco, args.do_bop, args.to_yolo, args.start_index, args.stop_index, args.output_dir

def generate_annotations(config:dict, hdf5_paths:list[str], do_coco:bool, do_bop:bool, keyframes:int=None, batch_size:int=200, output_dir=None):
    output_dir = output_dir or config["output_dir"]
    
    output_dir_coco = os.path.join(output_dir, "coco")
    output_dir_bop = os.path.join(output_dir, "bop")

    # Adapting batch_size to number of keyframes if its bigger.
    keyframes = keyframes or len(hdf5_paths)

    # Loading keyframes of the scene for bop dataset.
    if do_bop:
        bop_annotator = Bop_Annotator(output_dir_bop)
        bop_annotator.set_up_scene(config, keyframes)
        target_elements = bop_annotator.get_target_elements()
    
    if do_coco:
        coco_annotator = Coco_Annotator(output_dir_coco)
        
    format = 'Annotating dataset {percentage:3.0f}%[{bar:25}] {n_fmt:>3}/{total_fmt} [{elapsed}<{remaining:>5} - {rate_fmt:>12}] {desc}'
    base_batch_size, remainder = divmod(keyframes, batch_size)
    total = base_batch_size+bool(remainder)

    print()
    with tqdm(total = total*(1 + int(do_coco) + int(do_bop)), unit="step", disable=False, bar_format=format, ascii=" =") as pbar:
        for i in range(total):
            batch = (i==total-1) * remainder or batch_size
            start = i*batch_size
            stop = start+batch

            # Extract the current batch of file paths
            batch_paths = hdf5_paths[start:stop]


            # Loading batch of data
            pbar.set_description_str(f"Loading batch {i+1}/{total} with {len(batch_paths)} files...")
            hdf5 = hdf5_utils.open_multiple_hdf5_as_dict(batch_paths)
            fixed_hdf5 = {key: {id: hdf5[key][id-start] for id in range(start,stop)} for key in hdf5.keys()}
            pbar.update() # Manually update the progress after loading files


            # Saving coco annotations:
            if do_coco:
                pbar.set_description_str(f"Writting COCO for batch {i+1}/{total}...")
                coco_annotator.annotate_data(batch, hdf5)
                pbar.update()
            
            # Saving bop annotations:
            if do_bop:
                pbar.set_description_str(f"Writting BOP  for batch {i+1}/{total}...")
                bop_annotator.annotate_data(target_elements, start, batch, fixed_hdf5)
                pbar.update()


        pbar.colour='green'





if __name__ == "__main__":
    config_path, do_coco, do_bop, to_yolo, start, stop, output_dir = parse_args()

    _t0 = time.perf_counter()
    _start_time = datetime.now()

    config = misc_utils.load_config(config_path)

    # Seed:
    seed = config["seed"] if config["seed"] != -1 else random.random()
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed used: {seed}")

    # Generate data:
    if do_coco or do_bop:
        hdf5_path  = os.path.join(config["output_dir"],"hdf5")
        hdf5_paths = [os.path.join(hdf5_path, file) for file in os.listdir(hdf5_path) if file.endswith(".hdf5")]
        hdf5_paths = sorted(hdf5_paths, key = lambda x: int(os.path.basename(x).split(".hdf5")[0]))

        # Taking only the paths within the inerval [start, stop]
        stop = stop if stop != -1 else len(hdf5_paths)-1
        hdf5_paths = hdf5_paths[start:stop+1]

        generate_annotations(config=config, hdf5_paths=hdf5_paths, do_coco=do_coco, do_bop=do_bop, keyframes=None, batch_size=200, output_dir=output_dir)

    # Transform COCO annotations into YOLO annotations:
    if to_yolo:
        coco_dir = os.path.join(config['output_dir'], 'coco')
        category_mapping = config.get("coco_to_yolo_category_mapping")

        coco2yolo(coco_dir, category_mapping=category_mapping, verbose=True)        

         # --- Final time log ---
        _end_time = datetime.now()
        _elapsed = time.perf_counter() - _t0
        formatted_end = _end_time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_runtime = str(timedelta(seconds=int(_elapsed)))

        print("\n======================================")
        print(f"Annotation finished at: {formatted_end}")
        print(f"Total runtime: {formatted_runtime} ({_elapsed:.2f}s)")
        print("======================================")

        # Determine output directory
        out_dir = output_dir or config.get("output_dir")
        if out_dir:
            log_path = os.path.join(out_dir.rstrip("/"), "annotation_log.txt")
            try:
                os.makedirs(out_dir, exist_ok=True)
                with open(log_path, "a") as f:
                    f.write("\n======================================\n")
                    f.write(f"Annotation finished at: {formatted_end}\n")
                    f.write(f"Total runtime: {formatted_runtime} ({_elapsed:.2f}s)\n")
                    f.write(f"COCO: {do_coco}, BOP: {do_bop}, YOLO: {to_yolo}\n")
                    f.write(f"Config file: {config_path}\n")
                    f.write(f"Processed frames: {len(hdf5_paths) if (do_coco or do_bop) else 'N/A'}\n")
                    f.write("======================================\n")
                print(f"Annotation log written to: {log_path}")
            except Exception as e:
                print(f"⚠️ Could not write annotation log file: {e}")
        else:
            print("⚠️ No 'output_dir' found — skipping log file write.")