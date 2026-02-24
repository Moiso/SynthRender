import os
import subprocess
import argparse
import copy
import time   # <-- added

import yaml  # pip install pyyaml if needed


def run_batches(
    template_config: str,
    base_output_root: str,
    base_name: str,
    num_batches: int,
    frames_per_batch: int = 100,
    start_batch: int = 1,
    sleep_after: int = 20,   # seconds to wait after each run
):
    # Load the template once
    with open(template_config, "r") as f:
        base_cfg = yaml.safe_load(f)

    for batch_idx in range(start_batch, start_batch + num_batches):
        # Example:
        # /media/vrt/D/Datasets/Outputs/BATCHES_100/BASELINE_4k_Material_Randomization_RGB_EXP_PHYSICS_v1
        batch_out_dir = os.path.join(
            base_output_root,
            f"{base_name}_v{batch_idx}",
        )
        os.makedirs(batch_out_dir, exist_ok=True)

        # Copy config and modify
        cfg = copy.deepcopy(base_cfg)

        # Adjust to your actual keys if needed
        cfg["output_dir"] = batch_out_dir
        cfg["seed"] = batch_idx + 80  # or any deterministic mapping you prefer

        # Save config inside that batch folder
        batch_config_path = os.path.join(batch_out_dir, "config_batch.yaml")
        with open(batch_config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"\n=== Batch {batch_idx} ===")
        print(f"Output dir: {batch_out_dir}")
        print(f"Config:     {batch_config_path}")
        print(f"Seed:       {cfg['seed']}")

        # Build the main.py command
        cmd = [
            "python",
            "main.py",
            "-n", str(frames_per_batch),
            "-c", batch_config_path,
            "-dc",
            "-ty",
        ]
        print("Running:", " ".join(cmd))

        # Run and stop if anything fails
        subprocess.run(cmd, check=True)

        # --- Cooldown between runs so Blender/driver can clean up ---
        if sleep_after > 0:
            print(f"[run_batches] Sleeping {sleep_after} seconds before next batch...")
            time.sleep(sleep_after)


def parse_args():
    p = argparse.ArgumentParser("Run synthetic pipeline in batches")
    p.add_argument(
        "-t", "--template-config",
        required=True,
        help="Path to template config (YAML) to copy from"
    )
    p.add_argument(
        "--base-output-root",
        default="/media/vrt/D/Datasets/Outputs/BATCHES_100",
        help="Root folder under which batch folders will be created"
    )
    p.add_argument(
        "--base-name",
        default="BASELINE_4k_Material_Randomization_RGB_EXP_PHYSICS",
        help="Base name prefix before _v1, _v2, ..."
    )
    p.add_argument(
        "-b", "--num-batches",
        type=int,
        default=40,
        help="Number of batches to run"
    )
    p.add_argument(
        "-f", "--frames-per-batch",
        type=int,
        default=100,
        help="Number of frames per batch"
    )
    p.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="Starting batch index (useful if resuming)"
    )
    p.add_argument(
        "--sleep-after",
        type=int,
        default=20,
        help="Seconds to sleep after each batch (cooldown for Blender/GPU)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batches(
        template_config=args.template_config,
        base_output_root=args.base_output_root,
        base_name=args.base_name,
        num_batches=args.num_batches,
        frames_per_batch=args.frames_per_batch,
        start_batch=args.start_batch,
        sleep_after=args.sleep_after,
    )


# python run_batches.py \
#   -t config_wip_BASELINE_material_randomization.yaml \
#   -b 40 \
#   -f 100 \
#   --base-output-root /media/vrt/D/Datasets/Outputs/BATCHES_100_b9 \
#   --base-name BASELINE_4k_Material_Randomization_RGB_EXP_PHYSICS_b9 \
#   --sleep-after 30
