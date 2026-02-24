import blenderproc as bproc
import os
import bpy
import time

from synthrender.utils import hdf5_utils



############################################################
# FUNCTIONS USED FOR SETTING RENDER PARAMETERS (deprecated)#
############################################################

def save_output_deprecated(config:dict, data:dict, start, stop, start_frame, extra_data:dict, verbose, seed):
    # Save HDF5 results.
    if config["save_hdf5"]:
        output_dir_hdf5 = os.path.join(config["output_dir"], "hdf5")

        # Adding extra data:
        data["models_data"] = []

        env_text = bpy.context.scene.world.node_tree.nodes.get('Environment Texture', None)
        background_name = env_text.image.name if env_text else None

        for j in range(start-start_frame, stop+1-start_frame):
            if verbose:
                print(f"[update] Adding extra data for frame {start}/{stop-1})", flush=True)

            new_data = {key: extra_data[key][j+2] for key in extra_data.keys()}
            new_data["seed"] = seed
            new_data["background_img"] = background_name

            data["models_data"].append(new_data)

        bproc.writer.write_hdf5(output_dir_hdf5, data)

    # Save render results separatedly.
    if config["save_raw"]:
        timestamp = time.strftime("%Y-%m-%d-%H-%M")

        if rgb:=data.get('colors', None):
            output_dir_rgb = os.path.join(config["output_dir"],"raw", "rgb")
            hdf5_utils.export_rgb(rgb, output_dir=output_dir_rgb, index_start=start, timestamp=timestamp)

        if segmask:=data.get('category_id_segmaps', None):
            output_dir_segmasks = os.path.join(config["output_dir"], "raw", "segmasks")
            hdf5_utils.export_segmasks(segmask, output_dir=output_dir_segmasks, index_start=start, timestamp=timestamp)

        if depth:=data.get('depth', None):
            output_dir_depth = os.path.join(config["output_dir"], "raw", "depth")
            hdf5_utils.export_depth(depth, output_dir=output_dir_depth, index_start=start, timestamp=timestamp)

        if normals:=data.get('normals', None):
            output_dir_normals = os.path.join(config["output_dir"],"raw", "normals")
            hdf5_utils.export_normals(normals, output_dir=output_dir_normals, index_start=start, timestamp=timestamp)