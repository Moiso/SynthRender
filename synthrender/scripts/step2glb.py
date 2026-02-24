import os
import subprocess
import argparse

from synthrender.src.step2glb import freecad_step2glb


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("step_file", type=str, help="Path to step file")
    parser.add_argument("output_glb", type=str, help="Path to output glb file")
    parser.add_argument("--skip-optim", '-s', action="store_true", help="Skips the optimization of the generated glb file.")
    
    args = parser.parse_args()

    return args.step_file, args.output_glb, args.skip_optim


if __name__ == "__main__":
    in_step, out_glb, skip_optim = parse_args()

    # Converting step file into glb.
    command = ["freecad", freecad_step2glb.__file__, "--pass", in_step, out_glb]
    print(f"[{" ".join(command)}]", flush=True)

    
    my_env = os.environ.copy()
    my_env["QT_QPA_PLATFORM"]       = "offscreen" # Force Qt to load in offscreen mode
    my_env["QT_OPENGL"]             = "software"
    my_env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    subprocess.run(command, env=my_env, check=True)

    print()

    # Improving and compressing exported glb.
    if os.path.isfile(out_glb) and not skip_optim:
        script = os.path.join(os.path.dirname(freecad_step2glb.__file__), "blender_glbOptimize.py")

        command = ["blenderproc", "run", script, out_glb, out_glb]
        print(f"[{" ".join(command)}]")

        subprocess.run(command, check=True)


    print("Finished!")