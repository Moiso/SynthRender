import blenderproc as bproc

import bpy
import os

from tqdm.auto import tqdm

from synthrender.utils.blender_setup import scene_loader, scene_randomizer, scene_setter
from synthrender.utils.bproc_utils import bproc_utils
from synthrender.utils.misc_utils import load_config
from synthrender.utils.blender_setup.material_randomizer import add_material_keyframes_for_models

class KeyframeGenerator:
    config:dict = None
    config_path:str = None

    backgrounds:list[bpy.types.Image] = None
    default_scene:list = None
    empty:bproc.types.Entity = None
    train_models:list[bproc.types.MeshObject] = None
    fake_train_models:list[bproc.types.MeshObject] = None
    distractors:list[bproc.types.MeshObject] = None
    planes:list[bproc.types.MeshObject] = None
    lights:list[bproc.types.Light] = None

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

    def set_up_scene(self):
        assert self.config is not None, ValueError(f"Ensure that a configuration has been loaded before!")

        bproc.init()

        # Setting up camera:
        scene_loader.camera_setup(self.config)

        # Load a default blender scene:
        self.default_scene = scene_loader.load_default_scene(self.config)

        # Create empty object and assigning categories to models.
        self.empty = bproc.object.create_empty("empty", "arrows")

        # Load PBR materials
        scene_loader.load_pbr_materials(self.config)
    

        # Loading world backgrounds:
        # if self.config["models"]["material_randomization"]:
        self.backgrounds = scene_loader.load_backgrounds(self.config)

        # Load train models:
        self.train_models, self.fake_train_models = scene_loader.load_train_models(self.config)

        # Load distractors:
        self.distractors, self.fake_distractors = scene_loader.load_distractor_models(self.config)

        # Load planes and their materials:
        self.planes = scene_loader.load_plane_materials(self.config)

        # Configure the lights:
        self.lights = scene_loader.load_lights(self.config, self.empty)

        # Setting up rigid bodies for the first time for all the models:
        if self.config["physics"]["simulate_physics"]:
            print("Setting models as rigidbodies...")
            scene_setter.setup_models_physics(self.config, [*self.train_models, *self.fake_train_models], as_active=True, enable=False, default_config=self.config['models']['trains'])
            scene_setter.setup_models_physics(self.config, [*self.distractors, *self.fake_distractors], as_active=False, enable=False, default_config=self.config['models']['distractors'])

        # Create copies for train models:
        bproc_utils.create_duplicates(self.train_models, self.config, self.config["models"]["trains"])

        # Create copies for fake train models:
        bproc_utils.create_duplicates(self.fake_train_models, self.config, self.config["fake_models"]["trains"])

        # Create copies for distractors models:
        bproc_utils.create_duplicates(self.distractors, self.config, self.config["models"]["distractors"])

        # Create copies for fake distractors models:
        bproc_utils.create_duplicates(self.fake_distractors, self.config, self.config["fake_models"]["distractors"])


        return self.empty, self.train_models, self.distractors, self.lights, self.planes, self.fake_train_models, self.fake_distractors, self.default_scene

    def set_up_keyframes(self, num_keyframes:int=10, start_frame=0, stop_frame=-1, verbose=False):
        assert self.config is not None, ValueError(f"Ensure that a configuration has been loaded before!")

        stop_frame = num_keyframes if stop_frame == -1 else stop_frame

        # Setting up al the elements in the scene.
        self.set_up_scene()

        # Configuring frame -2 and -1: Hidden and not hidden.
        scene_loader.setup_restart_frames(models=[self.empty, *self.train_models, *self.distractors, *self.fake_train_models, *self.fake_distractors], planes=self.planes)

        # default_meshes = [mesh for mesh in self.default_scene if not mesh.get_name().startswith("#Ignore")] # Filter out meshes whose names start with "#Ignore"
        default_meshes = [mesh for mesh in self.default_scene if mesh.get_name() != "ARENA2036"] # Filter meshes from all the loaded scene.
        default_poses  = [(mesh.get_location(), mesh.get_rotation_euler()) for mesh in default_meshes]

        # Use Blender's bpy to set keyframes for each frame
        bpy.context.scene.frame_set(-2)
        for frame in tqdm(range(0, num_keyframes), desc="Preparing keyframes", unit=" keyframe", disable=verbose):
            # Generate random setup for the scene:
            # print("prev0", flush=True)

            back_strength                     = scene_randomizer.randomize_backg(self.config)
            plane                             = scene_randomizer.randomize_plane(self.planes)
            empty_pos, empty_rot              = scene_randomizer.randomize_empty(self.config)
            cam_pose, dof                     = scene_randomizer.randomize_camera(self.config, empty_pos)
            lights_energy, color_rgb          = scene_randomizer.randomize_lights(self.config, self.lights, num_keyframes, frame)
            set_train_models, set_train_poses = scene_randomizer.randomize_train_numba(self.config, self.train_models, empty_pos, self.fake_train_models, cam_pose, list(zip(default_meshes, default_poses)))
            set_distr_models, set_distr_poses = scene_randomizer.randomize_distr_numba(self.config, self.distractors, self.fake_distractors, list(zip([*set_train_models, *default_meshes], [*set_train_poses, *default_poses])))


            # Choose a stride so you don't create 4000 keyframes for every property:
            models_cfg = self.config.get("models", {})
            material_randomization_options = models_cfg.get("material_randomization_options", {})
            stride = material_randomization_options.get("material_anim", {}).get("stride", 10)   # every 10 frames
            seed   = material_randomization_options.get("seed", 0)
            ranges = material_randomization_options.get("material_anim", {}).get("ranges", None) # optional per-channel ranges

            # print("prev1", flush=True)

            if start_frame <= frame <= stop_frame:
                sampled_models = [*set_train_models, *set_distr_models, plane]  # Update models to be hidden to the current selected ones.

                # Set background light strength
                bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = back_strength

                # Set plane values
                scene_setter.set_plane(plane)

                # Set location of empty object used as origin for train models.
                scene_setter.set_empty(self.empty, empty_pos, empty_rot)

                # Set lights values
                scene_setter.set_lights(self.config, self.lights, lights_energy, color_rgb)

                # print("prev2", flush=True)

                # Set train models values
                scene_setter.set_models_pose(set_train_models, set_train_poses)

                # Randomize material properties
                if frame % stride == 0:
                    add_material_keyframes_for_models(sampled_models, frame=frame, seed=seed, ranges={
                        'roughness': (0.15, 0.85),
                        'metallic':  (0.0,  0.9),
                        'specular':  (0.1,  0.6),
                        'sat': (0.75, 1.6),
                        'val': (0.85, 1.5)
                    })


                # Set distractors models values
                scene_setter.set_models_pose(set_distr_models, set_distr_poses)

                # Set camera values
                bproc.camera.add_camera_pose(cam_pose, frame=frame)
                bproc.camera.add_depth_of_field(focal_point_obj=self.empty, fstop_value=dof)

                # Keyframe scene:
                bproc_utils.save_keyframe(frame, sampled_models)

                # print("prev3", flush=True)


                # Recalculate train objects position with physic simulation.
                scene_setter.do_physics(self.config, set_train_models, set_distr_models, frame, empty=self.empty)

                # Hide only models shown in previous iteration.
                bproc_utils.hide_render_view(sampled_models)                    

            if verbose:
                print(f"[update] Setting up keyframe {frame+1}/{num_keyframes}", flush=True)

        # General config:
        bpy.context.scene.frame_set(-1)


