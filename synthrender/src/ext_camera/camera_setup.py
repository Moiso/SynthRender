import blenderproc as bproc

import os
import bpy
import numpy as np

from tqdm.auto import tqdm

from synthrender.utils.misc_utils import load_config
from synthrender.utils.bproc_utils import bproc_utils

class ExternalCameraSetup:
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

    def create_frustum(self):
        self.frustum = bproc.camera.get_camera_frustum_as_object(clip_end=1.4)

        self.frustum.blender_obj.display.show_shadows = False
        self.frustum.blender_obj.visible_diffuse = False
        self.frustum.blender_obj.visible_glossy = False
        self.frustum.blender_obj.visible_volume_scatter = False
        self.frustum.blender_obj.visible_transmission = False
        self.frustum.blender_obj.visible_shadow = False

        material = bproc.material.create("frustum_material")
        material.nodes["Principled BSDF"].inputs[4].default_value = 0.5

        self.frustum.add_material(material)

    def new_camera_setup(self):
        # Creating new empty object for External Camera.
        ref = bproc.object.create_empty("ex_cam_ref")
        ref.set_location([0,0,0.5])
        ref.blender_obj.hide_viewport = True
        ref.blender_obj.hide_render = True

        # Creating new External Camera.
        cam_data = bpy.data.cameras.new("Camera_ext")           # First create camera.
        self.camera = bpy.data.objects.new(cam_data.name, cam_data) # Create object for camera.
        bpy.context.scene.collection.objects.link(self.camera)      # Add camera object to the scene.
        bpy.context.scene.camera = self.camera
        self.camera.constraints.new("TRACK_TO")
        self.camera.constraints['Track To'].target = bpy.data.objects["ex_cam_ref"]

        # Setting new camera for render.
        bpy.context.scene.camera = self.camera

    def set_up_keyframes(self, num_layouts:int=10, interp_time:float=1, pause_time:float=2, camera_rps:float=0.05, cam_dist:float=6, cam_height:float=2):  
        assert self.config is not None, ValueError(f"Ensure that a configuration has been loaded before!")
        
        # Getting old camera and its frustum:
        self.old_camera = bpy.context.scene.camera
        self.create_frustum()

        interp_frames = self.sec2frames(interp_time)
        pause_frames  = self.sec2frames(pause_time)
        total_frames  = num_layouts*(interp_frames+pause_frames) - interp_frames

        # Use Blender's bpy to set keyframes for each frame
        for frame in tqdm(range(0, num_layouts), desc="Preparing new keyframes:"):
            # Load generated layouts:
            bpy.context.scene.frame_set(frame)

            # Set pose for frustum object:
            old_cam_pose = self.old_camera.matrix_world
            self.frustum.set_local2world_mat(old_cam_pose)

            # Keyframe scene:
            start_pause = (frame*(interp_frames+pause_frames)) + num_layouts
            stop_pause = start_pause + max(0, pause_frames-1)

            bproc_utils.save_keyframe(start_pause, bproc_utils.get_all_parents())
            bproc_utils.save_keyframe(stop_pause, bproc_utils.get_all_parents())

        # Setting up new ExternalCamera:
        self.new_camera_setup()

        fpr = self.sec2frames(1) / camera_rps # frames per revolution

        # Get camera trajectory:
        for frame in tqdm(range(0, total_frames, 5), disable=True):
            frame_reduced = frame % fpr

            location_cam = np.array([cam_dist * np.cos(frame_reduced/fpr * 2*np.pi), 
                                     cam_dist * np.sin(frame_reduced/fpr * 2*np.pi), 
                                     cam_height])
            
            self.camera.location = location_cam
            self.camera.keyframe_insert(data_path="location", frame=frame + num_layouts)

        # General config:
        bpy.context.scene.frame_set(-1)

        bpy.context.scene.frame_start = num_layouts
        bpy.context.scene.frame_end = total_frames + num_layouts -1

