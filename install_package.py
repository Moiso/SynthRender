import blenderproc as bproc

import os
import subprocess

python_bin, packages_path, _, _ = bproc.SetupUtility.determine_python_paths(None, None)
env = dict(os.environ, PYTHONNOUSERSITE="0", PYTHONUSERBASE=packages_path)

# Install missing pkgs on blender:
missing_pkgs = ["tqdm", "numba", "ffmpeg-python"]
subprocess.Popen([python_bin, "-m", "pip", "install"] + missing_pkgs, env=env).wait()

# Install synthrender as editable on blender:
subprocess.Popen([python_bin, "-m", "pip", "install", "-e", "."], env=env).wait()