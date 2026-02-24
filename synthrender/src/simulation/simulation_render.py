import blenderproc as bproc

import bpy
import os
import yaml
import shutil
import numpy as np

from synthrender.utils.blender_setup import scene_setter, scene_loader
from synthrender.utils.bproc_utils import bproc_utils
from synthrender.utils import misc_utils
from synthrender.utils.bproc_utils import custom_blenderproc
from synthrender.utils.misc_utils import load_config




class KeyframesRenderer:
    config:dict = None
    config_path:str = None
    temp_dir:str = None

    
    output_dir_rgb:str = None
    output_dir_hdf5:str = None

    def __init__(self, config:dict|str=None, temp_dir=None):
        if config:
            self.set_config(config)

        if temp_dir:
            self.temp_dir = temp_dir

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
        
    def set_up_renderer(self, desired_gpu=None, verbose=False):
        output_dir = self.config["output_dir"]

        # Setting temp folder inside of output folder.
        if self.temp_dir:
            # temp_dir = os.path.join(output_dir,"temp")
            bproc.SetupUtility.setup_utility_paths(self.temp_dir)

        # Deactivate the auto remove option from blenderproc.
        from blenderproc.python.writer.WriterUtility import _WriterUtility
        defaults = list(_WriterUtility.load_output_file.__defaults__)
        defaults[-1] = not self.config["save_raw"]
        _WriterUtility.load_output_file.__defaults__ = tuple(defaults)

        bproc.renderer.set_max_amount_of_samples(self.config["cycles_samples"]) # Number of samples taken by cycles while rendering.
        # bpy.context.scene.view_settings.view_transform = 'Standard' # filmic

        # Setting light configuration so glass like objects are transparent.
        bpy.context.scene.cycles.max_bounces = 12
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4
        bpy.context.scene.cycles.transmission_bounces = 12

        self.output_dir_hdf5 = os.path.join(output_dir, "hdf5/")
        self.output_dir_rgb = None

        # Configure render modes (depth, normals and segmasks, save_raw)
        if "depth" in self.config["render_options"]:
            output_dir_depth = None

            if self.config["save_raw"]:
                output_dir_depth = os.path.join(output_dir, "raw", "depth")
                os.makedirs(output_dir_depth, exist_ok=True)
            
            bproc.renderer.enable_depth_output(activate_antialiasing=False, output_dir=output_dir_depth)

        if "normals" in self.config["render_options"]:
            output_dir_normals = None

            if self.config["save_raw"]:
                output_dir_normals = os.path.join(output_dir,"raw", "normals")
                os.makedirs(output_dir_normals, exist_ok=True)
                
            bproc.renderer.enable_normals_output(output_dir=output_dir_normals)

        if "segmasks" in self.config["render_options"]:
            output_dir_segmasks = None

            if self.config["save_raw"]:
                output_dir_segmasks = os.path.join(output_dir,"raw", "segmasks")
                os.makedirs(output_dir_segmasks, exist_ok=True)
            
            bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id": 0}, output_dir=output_dir_segmasks)

        if self.config["save_raw"]:
            self.output_dir_rgb = os.path.join(output_dir,"raw", "rgb/")
            os.makedirs(self.output_dir_rgb, exist_ok=True)
        
            bpy.context.scene.render.filepath = self.output_dir_rgb


        # Use modified version of write_hdf5 in case parallel mode is enabled.
        if verbose:
            bproc.writer.write_hdf5 = custom_blenderproc.modified_write_hdf5

        # Set render devices:
        n_gpus = bpy.context.preferences.addons['cycles'].preferences.get_num_gpu_devices()
        if desired_gpu is not None and n_gpus < desired_gpu+1:
            desired_gpu = None
        bproc.renderer.set_render_devices(desired_gpu_ids=desired_gpu)

    def render_scene3(self, num_frames:int, seed=0, start_frame=0, stop_frame=-1, desired_gpu=None, verbose=False, target_objects=None):
        self.set_up_renderer(desired_gpu, verbose)

        # Fix stop_frame in case it is out of scope.
        if stop_frame == -1 or stop_frame >= num_frames:
            stop_frame = num_frames-1

        # Load extra information from each model (position, visibility, energy, etc)
        extra_data = bproc_utils.fetch_keyframe_info()
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
            print("Rendering frame ----------------", i)

            # Save HDF5 results.
            if self.config["save_hdf5"]:
                # Adding extra data:
                data["models_data"] = []

                for j in range(start-start_frame, stop+1-start_frame):
                    if verbose:
                        print(f"[update] Adding extra data for frame {start}/{stop-1})", flush=True)
                        
                    new_data = {key: extra_data[key][j+2] for key in extra_data.keys()}
                    new_data["seed"] = seed
                    new_data["background_img"] = background_name

                    data["models_data"].append(new_data)

                bproc.writer.write_hdf5(self.output_dir_hdf5, data)

    def post_render(self):
        # Making copy of config file used for generating the data.
        copy_path = os.path.join(self.config["output_dir"], f"config_copy.yaml")

        if self.config_path is not None:
            shutil.copyfile(self.config_path, copy_path)
        else:
            with open(copy_path, "w") as f:
                yaml.safe_dump(self.config, f, indent=4)


    def render_scene(self, num_frames:int, seed=0, start_frame=0, stop_frame=-1, desired_gpu=None, verbose=False, target_objects=None):
            self.set_up_renderer(desired_gpu, verbose)

            # Fix stop_frame in case it is out of scope.
            if stop_frame == -1 or stop_frame >= num_frames:
                stop_frame = num_frames-1

            # Load extra information from each model (position, visibility, energy, etc)
            extra_data = bproc_utils.fetch_keyframe_info()
            backgrounds = []

            # Load backgrounds for rendering.
            if self.config["world"]["random_backgrounds"]:
                backgrounds = misc_utils.scan_folder(self.config["backgrounds_dir"], self.config["backgrounds_whitelist"], self.config["backgrounds_blacklist"])

            # If no background has been loaded, we set it as None.
            backgrounds = backgrounds or [None]

            # Split frames in intervals based on the number of backgrounds.
            # intervals = misc_utils.split_interval(0, num_frames-1, len(backgrounds))

            # Split frames in intervals based on the number of backgrounds.
            intervals = misc_utils.split_interval(0, num_frames-1, num_frames)

            rng = np.random.default_rng(seed=seed)  

            for i, interval in enumerate(intervals):
                start, stop = interval

                # If interval is out of start-stop interval
                if start_frame > stop or stop_frame < start:
                    continue 
                
                start = max(start, start_frame) # If start_frame is in between an interval, fix start.
                stop = min(stop, stop_frame)    # If stop_frame is in between an interval, fix stop.

                # Material randomization
                # Random material folder
                scene_loader.material_randomize(config=self.config, train_models=target_objects, rng=rng)

                if self.config['models']['material_randomization']:
                    background_path = rng.choice(backgrounds)
                    # Set background if available
                    if background_path:
                        scene_setter.set_background_texture(background_path, loadfile=True)

                # Get set background name
                env_text = bpy.context.scene.world.node_tree.nodes.get('Environment Texture', None)
                background_name = env_text.image.name if env_text else None

                print(f"\n> batch_interval: [{start:>{5}}, {stop:>{5}}], batch_size: ({stop-start+1:>{4}}), background_({i:>{2}})='{background_name}', gpu={desired_gpu if desired_gpu is not None else 'all'}")
                bproc.utility.set_keyframe_render_interval(start, stop+1)
                data = bproc.renderer.render(output_dir=self.output_dir_rgb, verbose=verbose)
                print("Rendering frame ----------------", i)

                # Save HDF5 results.
                if self.config["save_hdf5"]:
                    # Adding extra data:
                    data["models_data"] = []

                    for j in range(start-start_frame, stop+1-start_frame):
                        if verbose:
                            print(f"[update] Adding extra data for frame {start}/{stop-1})", flush=True)
                            
                        new_data = {key: extra_data[key][j+2] for key in extra_data.keys()}
                        new_data["seed"] = seed
                        new_data["background_img"] = background_name

                        data["models_data"].append(new_data)

                    bproc.writer.write_hdf5(self.output_dir_hdf5, data)