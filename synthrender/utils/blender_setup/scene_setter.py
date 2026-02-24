import blenderproc as bproc
import bpy
import numpy as np
import os

from blenderproc.python.types.MeshObjectUtility import MeshObject, Entity
from blenderproc.python.types.LightUtility import Light
from blenderproc.python.utility import Utility

from synthrender.utils.bproc_utils import bproc_utils
from synthrender.utils.bproc_utils import custom_blenderproc
from synthrender import __file__ as package_init_path




############################################################
# FUNCTIONS USED FOR SETTING SIMULATION PARAMETERS         #
############################################################

def set_background_texture(texture_name:str, loadfile=False):
    if loadfile:
        background = bpy.data.images.load(texture_name, check_existing=True)
    else:
        background = bpy.data.images.get(texture_name)

    if background is None:
        print(f"Warning: Could not find background: {texture_name}")
        return


    # Creating background texture node:
    nodes = bpy.context.scene.world.node_tree.nodes
    if nodes.get('Environment Texture') is None:
        rotation_euler = [0.0, 0.0, 0.0]

        world = bpy.context.scene.world
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # add a texture node and load the image and link it
        texture_node = nodes.new(type="ShaderNodeTexEnvironment")

        # get the one background node of the world shader
        background_node = Utility.Utility.get_the_one_node_with_type(nodes, "Background")

        # link the new texture node to the background
        links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

        # Set the brightness of the background
        background_node.inputs["Strength"].default_value = 1

        # add a mapping node and a texture coordinate node
        mapping_node = nodes.new("ShaderNodeMapping")
        tex_coords_node = nodes.new("ShaderNodeTexCoord")

        #link the texture coordinate node to mapping node
        links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])

        #link the mapping node to the texture node
        links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

        mapping_node.inputs["Rotation"].default_value = rotation_euler

    # Filter data in the folder:
    env_texture = nodes['Environment Texture']
    env_texture.image = background

def set_plane(plane:MeshObject):
    """
    Enables the viewport and render option of the chosen plane.

    Parameters:
        plane (MeshObject): Chosen plane.
    """

    if not plane: return

    plane.blender_obj.hide_render = False
    plane.blender_obj.hide_viewport = False

def set_empty(empty:Entity, empty_pos:list|np.ndarray, empty_rot:list|np.ndarray):
    """
    Sets the position and rotation (in euler angles) of the empty object.
    It also enabled its viewport and render option.

    Parameters:
        empty_pos ([x,y,z]): Position to place the empty object.
        empty_rot ([rx,ry,rz]): Rotation in degrees of the empty object using euler angles.
    """

    empty.set_location(empty_pos)
    empty.set_rotation_euler(empty_rot)
    empty.blender_obj.hide_render = False
    empty.blender_obj.hide_viewport = False

def set_lights(config:dict, lights:list[Light], lights_energy:list[float], color_rgb:list[float]):
    """
    Sets energy and color for each light.

    Parameters:
        config (dict): Loaded configuration containing the contrast for each light.
        lights (list[Light]): List containing each light object.
        lights_energy (list[float]): List containing the energy value for each light object.
        color_rgb (list[float, float, float]): List containing the rgb values for the lights.
    """

    for light, energy in zip(lights, lights_energy):
        light.set_energy(energy*light.get_cp("contrast"))  # Energy value based on random intensity.

        if config["lights"]["randomize_color"]:
            light.set_color(color_rgb)

def set_models_pose(sample_models:list[MeshObject], placed_models:list[tuple]):
    """
    Sets the location and rotation of the models.
    It also enables their render and viewport options.

    Parameters:
        sample_models (list[MeshObject]): List of models
        placed_models (list[tuple]): List containig the positions for each model:
            - location (list[float]): List containing the (x,y,z) location.
            - rotation (list[float]): List containing the (rx, ry, rz) rotation in euler angles.
    """
    for model, pos in zip(sample_models, placed_models):
        for child in [model, *bproc_utils.get_all_child_meshes(model)]:
            child.blender_obj.hide_viewport = False
            child.blender_obj.hide_render = False

            if child.has_cp("skip_render"):
                child.blender_obj.hide_render = True
                child.blender_obj.hide_viewport = True

            if child.has_cp("bbox"):
                child.blender_obj.hide_viewport = False

        location, rotation = pos[:2]

        model.set_location(location) # random position inside of a sphere
        model.set_rotation_euler(rotation)


def do_convex_hull_decomposition(obj:MeshObject):
    # Perform convex decomposition and reassign collections for new children
    modules_path = os.path.join(os.path.dirname(package_init_path), 'modules')
    cache_path = os.path.join(os.path.dirname(package_init_path), 'resources', 'decomposition_cache')
    old_children = set(obj.get_children())
    obj.build_convex_decomposition_collision_shape(vhacd_path=modules_path, cache_dir=cache_path)
    hulls:list[MeshObject] = list(set(obj.get_children()) - old_children)

    for hull in hulls:
        # Remove child from current collection and link it to parent's collections
        # if bpy.context.collection in hull.blender_obj.users_collection:
        #     bpy.context.collection.objects.unlink(hull.blender_obj)
        
        try:
            if bpy.context.collection and hull.blender_obj.name in bpy.context.collection.objects:
                bpy.context.collection.objects.unlink(hull.blender_obj)
        except Exception as e:
            print(f"[WARN] Could not unlink {hull.get_name()} from context collection: {e}")
            
        for collection in obj.blender_obj.users_collection:
            if collection not in hull.blender_obj.users_collection:
                collection.objects.link(hull.blender_obj)
        
        hull.set_cp("skip_render", True)
        hull.set_name("#Collider_vhacd")

        hull.blender_obj.hide_viewport = True
        hull.blender_obj.hide_render = True
        hull.blender_obj.visible_camera = False
        hull.blender_obj.visible_diffuse = False
        hull.blender_obj.visible_glossy = False
        hull.blender_obj.visible_transmission = False
        hull.blender_obj.visible_volume_scatter = False
        hull.blender_obj.visible_shadow = False
        hull.blender_obj.display.show_shadows = False

    return hulls

def setup_models_physics(config: dict[str, dict], models: list[MeshObject], as_active:bool=False, enable:bool=False, default_config:dict = None):
    # Cache for discarded models
    if not hasattr(setup_models_physics, "discarded_models"):
        setup_models_physics.discarded_models = set()

    def setup_model_physics(obj:MeshObject, rigidbody_args:dict, activate:bool=False) -> bool:
        # Check if object is a mesh
        if obj.get_attr("type") != "MESH":
            name = obj.get_name()
            if name not in setup_models_physics.discarded_models:
                print(f"Warning: {name} has type '{obj.get_attr('type')}'. Physics cannot be set for it.")
                setup_models_physics.discarded_models.add(name)
            return False

        if not obj.has_rigidbody_enabled():
            # Temporarily unhide object to create rigid body
            original_hide = obj.blender_obj.hide_viewport
            obj.blender_obj.hide_viewport = False
            obj.enable_rigidbody(**rigidbody_args)
            # Restore original hide state
            obj.blender_obj.hide_viewport = original_hide

        # Activate or deactivate model's physics
        obj.blender_obj.rigid_body.enabled = activate
        obj.blender_obj.rigid_body.collision_collections = [activate] * 20

        sim_collection = bpy.data.collections.get("RigidBodyWorld")
        try:
            colls = list(obj.blender_obj.users_collection)
        except ReferenceError:
            print(f"[CRASH-PREVENT] {obj.get_name()} has an invalid users_collection pointer.")
            exit(-1)
        except Exception as e:
            print(f"[CRASH-PREVENT] Failed to access users_collection for {obj.get_name()}: {e}")
            exit(-1)

        if activate:
            if sim_collection not in colls:
                sim_collection.objects.link(obj.blender_obj)
        else:
            if sim_collection in colls:
                sim_collection.objects.unlink(obj.blender_obj)

        return True

    default_config = default_config or {}
    valid_models: list[MeshObject] = []
    rigidbody_args = {"active": as_active, "friction": 1, "mass": 5, "collision_shape": "COMPOUND", "angular_damping": 0.5, "linear_damping": 0.04} # 0.1, 0.04

    for model in models:
        if model.get_name() not in bpy.data.objects:
            print(f"[SKIP] {model.get_name()} no longer exists.")
            exit(-1)

        # Get configuration for this model
        model_config = {}
        model_config.update(default_config)
        model_config.update(config["custom_models"].get(model.get_name(), {}))
        
        # We can have simplify enabled or not, that's all. (if simplified is enabled, there is no point in calcullating the previous #Collider)
        # set parent as a compound and children colliders as hull.
        simplify_model = model_config.get("do_convex_hull_decompose", False)
        ret = setup_model_physics(model, rigidbody_args, enable)

        if ret:
            rigidbody_args_colliders = rigidbody_args.copy()
            rigidbody_args_colliders["collision_shape"] = "CONVEX_HULL"
            to_add_physics:list[MeshObject] = []

            # Use #Colliders instead of convex hull decomposition
            if not simplify_model:
                colliders = [child for child in model.get_children() if child.get_name().startswith("#Collider")]
    
                # If no "#Colliders" were provided, create a general one as convex hull:
                if not colliders:
                    children = model.get_children()
                    hull_collider = bproc_utils.create_convex_collider(children) # Creates one convex hull collider from a list of meshes.
                    colliders.append(hull_collider)

                    # Add new collider to the same collection :
                    try:
                        colls = list(model.blender_obj.users_collection)
                    except ReferenceError:
                        print(f"[CRASH-PREVENT] {obj.get_name()} has an invalid users_collection pointer.")
                        exit(-1)
                    except Exception as e:
                        print(f"[CRASH-PREVENT] Failed to access users_collection for {obj.get_name()}: {e}")
                        exit(-1)

                    bpy.context.collection.objects.unlink(hull_collider.blender_obj)
                    colls[0].objects.link(hull_collider.blender_obj)
                    
                    for mesh in colliders:
                        mesh.set_parent(model)

                to_add_physics.extend(colliders)
            
            # Use Convex hull decomposition:
            # Create colliders as convex hull decomposition.
            else:
                colliders = [child for child in model.get_children() if child.get_name().startswith("#Collider_vhacd")]

                if not colliders: # If not colliders, the model has not been decomposed.
                    meshes = [child for child in model.get_children() if not child.get_name().startswith("#Collider")]

                    for mesh in meshes:
                        colliders.extend(do_convex_hull_decomposition(mesh))

                    for hull in colliders:
                        hull.set_parent(model)

                to_add_physics.extend(colliders)

            # Setting model's physics:
            for obj in to_add_physics:
                if obj.get_name() not in bpy.data.objects:
                    print(f"[SKIP] {model.get_name()} no longer exists.")
                    exit(-1)
                setup_model_physics(obj, rigidbody_args_colliders, enable) 

            if ret:
                valid_models.append(model)  
                    
    return valid_models

def do_physics(config:dict[str, dict|list], active_models:list[MeshObject], passive_models:list[MeshObject], frame:int, empty):
    """
    Performs a physic simulation by first making models active/passive and then calculating their collitions.

    It keyframes the new location and rotation of the active objects.

    Once the simulation is done, it removes the rigidbody option from the passed models to avoid them from contribute in future simulations unless they are passed again.

    It also recalculates the camera position to be looking at the centroid of the actives.

    Parameters:
        config (dict): Loaded configuration for the physics, it sets:
            - max_simulation_time: Maximum time that the simulation will run in seconds.
            - check_intervals: Interval at which it will be checked whether the objects are not moving anymore.
            - stopped_location_threshold: Minimum movement per second to be considered as stopped.
            - stopped_rotation_threshold: Minimum rotation per second to be considered as stopped.
        actives (list[MeshObject]): List of objects to which the rigidbody is set as active. It can move and collides with other rigidbodies.
        passives (list[MeshObject]): List of objects to which the rigidbody i set as passive. It cannot move but other rigidbodies can collide with it.
        frame (int): Current frame used as starting point for performing the simulation.
    Notes:
        - Rigidbodies are set as CONVEX_HULL for faster simulation. Other options are:
            -  'BOX', 'SPHERE', 'CAPSULE', 'CYLINDER', 'CONE', 'CONVEX_HULL', 'MESH', 'COMPOUND'.
    """

    do_physics  = config.get('physics', {}).get('simulate_physics', False)
    do_actives  = config.get('physics', {}).get('simulate_actives', False)
    do_passives = config.get('physics', {}).get('simulate_passives', False)
    reorientate = config.get('physics', {}).get('reorientate_camera', False)

    # Make models actively participate in the simulation (they can move)
    actives = active_models if do_actives else []
    passives = passive_models if do_passives else []

    # Run the simulation and fix the poses at the end
    if do_physics:
        # print("phy1", flush=True)
        
        valid_actives = setup_models_physics(config, actives, as_active=True, enable=True)
        
        # print("phy2", flush=True)

        valid_passives = setup_models_physics(config, passives, as_active=False, enable=True)

        # print("phy3", flush=True)

        with Utility.stdout_redirected(enabled=True):
            check_object_interval = config['physics'].get('check_interval')
            min_simulation_time = check_object_interval
            max_simulation_time = config['physics'].get('max_simulation_time')
            object_stopped_location_threshold = config['physics'].get('stopped_location_threshold')
            object_stopped_rotation_threshold = config['physics'].get('stopped_rotation_threshold')
            custom_blenderproc.simulate_physics_and_fix_final_poses_v2(min_simulation_time, max_simulation_time, check_object_interval, object_stopped_location_threshold, object_stopped_rotation_threshold, frame_start=frame, verbose=True)
            # custom_blenderproc.simulate_physics_v2(min_simulation_time, max_simulation_time, check_object_interval, object_stopped_location_threshold, object_stopped_rotation_threshold, frame_start=frame)

        # Keyframe the new positions for the target objects.
        for obj in valid_actives:
            if obj.get_name() not in bpy.data.objects:
                print(f"[SKIP] {obj.get_name()} no longer exists.")
                exit(-1)
                
            obj.blender_obj.keyframe_insert(data_path="location", frame=frame)
            obj.blender_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

        # # Recalculate camera orientation to look towards centre of actives:
        # if reorientate and valid_actives:
        #     poi = bproc.object.compute_poi(valid_actives)
        #     location = bproc.camera.get_camera_pose(frame)[:3, 3]

        #     rotation = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=None)
        #     cam_pose:np.ndarray = bproc.math.build_transformation_mat(location, rotation)

        #     # Keyframe the new camera position:
        #     bproc.camera.add_camera_pose(cam_pose, frame=frame)

        # Recalculate camera orientation to look towards centre of actives:
        if reorientate and valid_actives:
            # TODO: Poi should be only the real train object, not fakes.
            poi = bproc.object.compute_poi(valid_actives)

            # Recalculate distance to be within the range.
            old_location = bproc.camera.get_camera_pose(frame)[:3, 3]

            distance = np.linalg.norm(empty.get_location()-old_location)

            vector = old_location - poi # Vector pointing from poi to old_camera_location
            normalized = vector / np.linalg.norm(vector) # normalized vector from center to camera.

            new_location = poi + normalized * distance
            new_rotation = bproc.camera.rotation_from_forward_vec(poi - new_location, inplane_rot=None)
            cam_pose:np.ndarray = bproc.math.build_transformation_mat(new_location, new_rotation)

            # Keyframe the new camera position:
            bproc.camera.add_camera_pose(cam_pose, frame=frame)

        # Dissable rigidbody so only future sampled objects are used for physic calculations.
        setup_models_physics(config, valid_actives, enable=False)
        setup_models_physics(config, valid_passives, enable=False)
