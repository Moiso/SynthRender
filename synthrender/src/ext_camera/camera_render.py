import blenderproc as bproc

import os
import bpy
import numpy as np

from tqdm.auto import tqdm

from synthrender.utils.misc_utils import load_config
from synthrender.utils.bproc_utils import bproc_utils
from synthrender.utils import misc_utils
from synthrender.utils.blender_setup import scene_setter

class ExternalCameraRender:
    config:dict = None
    config_path:str = None
    camera:bpy.types.Object = None
    
    def __init__(self, config:dict|str=None):
        if config:
            self.set_config(config)

    def set_config(self, config:str|dict):
        if isinstance(config, dict):
            self.config = config
            return
        
        elif not isinstance(config, str):
            raise TypeError(f"Wrong type for config to load! {type(config)}")
        
        if not os.path.isfile(config):
            raise FileExistsError(f"Error: Could not find config file! '{config}'")
        
        self.config = load_config(config)
        self.config_path = config

    def sec2frames(self, n_sec:float):
        return int(bpy.context.scene.render.fps*n_sec)

    def set_up_renderer(self, desired_gpu=None, use_cycles=True, camera_id:int=1):
        output_dir = os.path.abspath(self.config["output_dir"] or "./output")

        # Deactivate the auto remove option from blenderproc.
        from blenderproc.python.writer.WriterUtility import _WriterUtility
        defaults = list(_WriterUtility.load_output_file.__defaults__)
        defaults[-1] = False
        _WriterUtility.load_output_file.__defaults__ = tuple(defaults)

        self.output_dir_rgb = os.path.join(output_dir, 'animations', f"animation_{camera_id}_frames/")
        os.makedirs(self.output_dir_rgb, exist_ok=True)
        bpy.context.scene.render.filepath = self.output_dir_rgb

        # Set render devices:
        if use_cycles:
            bproc.renderer.set_max_amount_of_samples(self.config["cycles_samples"]) # Number of samples taken by cycles while rendering.

            n_gpus = bpy.context.preferences.addons['cycles'].preferences.get_num_gpu_devices()
            if desired_gpu is not None and n_gpus < desired_gpu+1:
                desired_gpu = None
            bproc.renderer.set_render_devices(desired_gpu_ids=desired_gpu)
        else:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

        
        # Selecting camera:
        bpy.context.scene.camera = bpy.data.objects[bpy.data.cameras[camera_id].name]

    def render_scene(self, use_cycles=True, used_cameras:int=1, desired_gpu=None, verbose=False):

        if used_cameras in (1,0):
            cameras = [used_cameras]
        elif used_cameras == 2:
            cameras = [1, 0]
        else:
            raise ValueError(f"'used_cameras' got a not valid value: {used_cameras}")
        
        original_start_frame = bpy.context.scene.frame_start
        original_stop_frame = bpy.context.scene.frame_end

        for camera_id in cameras:

            self.set_up_renderer(desired_gpu, use_cycles=use_cycles, camera_id=camera_id)

            if camera_id == 0:
                frustum = bproc.object.convert_to_meshes([bpy.data.objects["Camera Frustum"]])[0]
                frustum.delete()

            bpy.context.scene.frame_start = original_start_frame
            bpy.context.scene.frame_end = original_stop_frame

            start_frame = bpy.context.scene.frame_start
            stop_frame = bpy.context.scene.frame_end
            num_frames = stop_frame - start_frame + 1

            backgrounds = []

            # Load backgrounds for rendering.
            if self.config["world"]["random_backgrounds"]:
                backgrounds = misc_utils.scan_folder(self.config["backgrounds_dir"], self.config["backgrounds_whitelist"], self.config["backgrounds_blacklist"])

            # If no background has been loaded, we set it as None.
            backgrounds = backgrounds or [None]

            # Split frames in intervals based on the number of backgrounds.
            intervals = misc_utils.split_interval(0, num_frames-1, len(backgrounds))

            for i, (interval, background_path) in enumerate(zip(intervals, backgrounds)):
                start, stop = interval

                # If interval is out of start-stop interval
                if start_frame > stop or stop_frame < start:
                    continue 

                start = max(start, start_frame) # If start_frame is in between an interval, fix start.
                stop = min(stop, stop_frame)    # If stop_frame is in between an interval, fix stop.

                # Set background if available
                if background_path:
                    scene_setter.set_background_texture(background_path, loadfile=True)

                # Get set background name
                env_text = bpy.context.scene.world.node_tree.nodes.get('Environment Texture', None)
                background_name = env_text.image.name if env_text else None

                print(f"\n> batch_interval: [{start:>{5}}, {stop:>{5}}], batch_size: ({stop-start+1:>{4}}), background_({i:>{2}})='{background_name}', gpu={desired_gpu if desired_gpu is not None else 'all'}")
                bproc.utility.set_keyframe_render_interval(start, stop+1)
                data = bproc.renderer.render(output_dir=self.output_dir_rgb, verbose=verbose)

            from blenderproc.python.utility.GlobalStorage import GlobalStorage
            GlobalStorage._storage_dict.pop("output")

            














