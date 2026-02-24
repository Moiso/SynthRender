import blenderproc as bproc
import bpy
import mathutils
import bmesh

import os
import numpy as np
import itertools
import random
import math

from mathutils import Vector
from blenderproc.python.types.MeshObjectUtility import MeshObject, Entity
from blenderproc.python.types.LightUtility import Light


from synthrender.utils import misc_utils




def set_camera_pose(poi_objects:list, location = (0,0,3)):
    # Find point of interest, all cam poses should look towards it
    poi = bproc.object.compute_poi(poi_objects)

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    
    return cam2world_matrix

def compute_centroid(objects: list[MeshObject]) -> np.ndarray:
    """
    Computes the centroid (mean location) of the bounding boxes of the selected objects.
    If an object is an empty, it computes the centroid using its child mesh objects.
    
    :param objects: The list of objects that should be considered (can include empty objects).
    :param get_all_childs: A function that returns a list of all child MeshObjects for an empty object.
    :return: The centroid (mean location) of the bounding boxes.
    """
    # Initialize a list to store the centers of bounding boxes
    bbox_centers = []

    for obj in objects:
        child_mesh_objects = get_all_child_meshes(obj)
        for child in child_mesh_objects:
            bb_points = child.get_bound_box()
            bb_center = np.mean(bb_points, axis=0)
            bbox_centers.append(bb_center)

    # Compute the centroid by taking the mean of all bounding box centers
    if bbox_centers:
        centroid = np.mean(bbox_centers, axis=0)
    else:
        raise ValueError("No valid bounding boxes found for the provided objects.")
    
    return centroid

def random_camera_pose(poi_objects: list[MeshObject], pos_min=(0,0,4), pos_max=(1,1,4), max_attempts=25):
    """
    Generates a random camera pose that ensures all poi_objects are visible. If not all objects are visible,
    the camera position is recalculated up to a maximum number of attempts.

    :param poi_objects: List of objects to ensure visibility in the camera view.
    :param max_attempts: Maximum number of attempts to reposition the camera if not all objects are visible.
    :return: Camera-to-world transformation matrix that ensures all poi_objects are visible.
    """
    attempt = 0
    # Find point of interest, all cam poses should look towards it
    # poi = compute_centroid(poi_objects)
    poi = poi_objects[0].get_location()
    
    while attempt < max_attempts:
        # Increment attempt counter
        attempt += 1
        

        # location = np.random.uniform(pos_min, pos_max)
        location = sample_rect_prism(poi, pos_min, pos_max)

        # Compute rotation based on vector going from location towards poi
        # inplane_rot = np.random.uniform(-np.pi, np.pi)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=None)

        # Add homogeneous cam pose based on location and rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        # Check visible objects from the camera's perspective
        visible_objects = bproc.camera.visible_objects(cam2world_matrix, 20)

        # Check if all POI objects are visible
        if all(obj in visible_objects for obj in poi_objects):
            # If all objects are visible, return the current camera transformation matrix
            return cam2world_matrix

    # If max_attempts reached and not all objects are visible, return the last transformation matrix
    print(f"Warning: Could not find a camera pose showing all objects after {max_attempts} attempts.")
    return cam2world_matrix

def sample_rect_prism(origin=(0, 0, 0), dist_min=(0, 0, 0), dist_max=(1, 1, 1)):
    """
    Samples a point inside a rectangular prism centered at origin, with given distance
    ranges along each axis.
    
    Parameters:
    - origin: The center of the rectangular prism (x, y, z).
    - dist_min: Minimum distances from the origin along each axis (x, y, z).
    - dist_max: Maximum distances from the origin along each axis (x, y, z).
    
    Returns:
    - A point (x, y, z) sampled within the rectangular prism.
    """
    # Ensure origin, dist_min, and dist_max are numpy arrays for easy manipulation
    origin = np.array(origin)
    dist_min = np.array(dist_min)
    dist_max = np.array(dist_max)

    # Sample uniformly within the defined minimum and maximum distances from the origin for each axis
    sampled_x = np.random.uniform(-dist_max[0], dist_max[0])
    sampled_y = np.random.uniform(-dist_max[1], dist_max[1])
    sampled_z = np.random.uniform(-dist_max[2], dist_max[2])

    # Ensure that the sampled point is within the minimum distance range as well
    # Correct by shifting the sampled point based on dist_min if it's within the min-max bounds
    if abs(sampled_x) < dist_min[0]:
        sampled_x = np.sign(sampled_x) * dist_min[0] + np.random.uniform(0, dist_max[0] - dist_min[0])

    if abs(sampled_y) < dist_min[1]:
        sampled_y = np.sign(sampled_y) * dist_min[1] + np.random.uniform(0, dist_max[1] - dist_min[1])

    if abs(sampled_z) < dist_min[2]:
        sampled_z = np.sign(sampled_z) * dist_min[2] + np.random.uniform(0, dist_max[2] - dist_min[2])

    # Adjust the sampled point by adding the origin
    sampled_point = origin + np.array([sampled_x, sampled_y, sampled_z])

    return sampled_point


def get_entities(discard:list[str]=()):
    all_objs = bproc.object.convert_to_meshes(bpy.data.objects)
    filtered_entities:list[Entity] = [obj for obj in all_objs if obj.get_attr("type") not in ("MESH", "EMPTY") and obj.get_name() not in discard]

    return filtered_entities

def get_parent(object:MeshObject):
    parent = object
    
    if object.get_parent():
        parent = get_parent(object.get_parent())

    return parent

def get_parent_mesh(object:MeshObject):
    parent = object
    upper = object.get_parent()
    
    if upper and upper.get_attr('type') == 'MESH':
        parent = get_parent_mesh(upper)

    return parent



def get_all_parents(discard:list[str]=()):
    all_objs = bproc.object.convert_to_meshes(bpy.data.objects)

    parents = [obj for obj in all_objs if not obj.get_parent()]
    filtered_parents = [obj for obj in parents if obj.get_attr("type") in ("MESH", "EMPTY") and obj.get_name() not in discard]

    return filtered_parents

def get_all_child_meshes(obj: MeshObject) -> list[MeshObject]:
    """ Recursively get all child meshes of an object, including the object itself if it's a mesh. """
    meshes = []
    if isinstance(obj, MeshObject) and obj.get_mesh() is not None:
        meshes.append(obj)
    for child in obj.get_children():
        meshes.extend(get_all_child_meshes(child))
    return meshes

def print_meshes_attr(meshes: list[MeshObject], attr:str="name"):
    assert isinstance(meshes, list), f"Meshes passed are not of type list but: {type(meshes)}"
    print(f"> There are {len(meshes)} objects")
    for i, mesh in enumerate(meshes):
        print("\t",i, mesh.get_attr(attr))

def set_category_to_meshes(meshes:list[MeshObject], models_config:dict[str, dict] = None):
    for mesh in bproc.object.get_all_mesh_objects():
        mesh.set_cp("category_id", 0)

    models_config = models_config or {}

    count = itertools.count(start=1) # Iterator that counts starting from 1.
    reserved_ids = set()

    # Getting all the reserved ids:
    parents_names = set(mesh.get_name() for mesh in meshes)
    for model_name, model_config in models_config.items():
        if model_name not in parents_names: continue

        for id in model_config.get('meshes_ids', {}).values():
            reserved_ids.add(id)
    
    # Set category to each mesh
    for mesh in meshes:
        parent_name = mesh.get_name()
        model_config = {}
        model_config.update(models_config.get("default_config", {}))
        model_config.update(models_config.get(parent_name, {}))
        
        whitelist = set(model_config.get("segment_whitelist", []))
        blacklist = set(model_config.get("segment_blacklist", []))

        for child_mesh in get_all_child_meshes(mesh):
            child_name = child_mesh.get_name()
            
            if child_mesh.has_cp("skip_render"): continue # Skipping #Colliders
            if whitelist and child_name not in whitelist: continue # child not in whitelist, not annotated.
            if blacklist and child_name in blacklist: continue # child in blacklist, not annotated.

            custom_id = model_config.get('meshes_ids', {}).get(child_name, None)

            # Setting reserved id or finding first free one.
            if custom_id:
                child_mesh.set_cp("category_id", custom_id)    # Set a unique category ID for each mesh.
            else:
                while (id:=next(count)) in reserved_ids: continue # Skip ids until we get a non-reserved one.

                child_mesh.set_cp("category_id", id)  # Set a unique category ID for each mesh.




# Lecacy function
def save_keyframe_old(frame:int, items:list[MeshObject|Entity|Light]=None, items_children:list[list[MeshObject|Entity]]=None):
    if items_children is None:
        objects:list[MeshObject|Entity] = get_all_parents()     # Mesh and Emptys
        entities:list[Entity|Light] = get_entities()            # Lights and Cameras
        items:list[MeshObject|Entity|Light] = [*objects, *entities]

        items_children = [[item, *get_all_child_meshes(item)] for item in items]

    for children in items_children:
        item = children[0]
        
        # Keyframe children visibility:
        for child in children:
            if child.has_cp("hull"): continue
            
            child.blender_obj.keyframe_insert(data_path="hide_render", frame=frame)
            child.blender_obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        
        # Keyframe parent pose:
        # if not item.blender_obj.hide_render or frame < 0 or item.has_cp("combined_mesh"):
        if not item.blender_obj.hide_render or frame < 0:
            item.blender_obj.keyframe_insert(data_path="location", frame=frame)
            item.blender_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Block for keyframing the material color (in case we change the material on each frame)
        # if item.get_attr("type") == "MESH" and len(item.blender_obj.material_slots): # Keyframe material colors.
        #     nodes = item.blender_obj.material_slots[0].material.node_tree.nodes
        #     if nodes.get("Principled BSDF"):
        #         nodes["Principled BSDF"].inputs["Base Color"].keyframe_insert(data_path="default_value", frame=frame)

        if item.get_attr("type") == "LIGHT":
            item.blender_obj.data.keyframe_insert(data_path="shadow_soft_size", frame=frame)
            item.blender_obj.data.keyframe_insert(data_path="energy", frame=frame)
            item.blender_obj.data.keyframe_insert(data_path="color", frame=frame)


    # Saving camera F-stop
    camera = bpy.data.objects['Camera']
    camera.data.keyframe_insert(data_path="dof.aperture_fstop", frame=frame)

    # Insert a keyframe for the strength at the given frame
    background_node = bpy.data.worlds["World"].node_tree.nodes["Background"]
    background_node.inputs["Strength"].keyframe_insert(data_path="default_value", frame=frame)

def save_keyframe(frame:int, parents:list[MeshObject|Entity|Light]=None):
    entities:list[Entity|Light] = get_entities()                 # Lights and Cameras
    items:list[MeshObject|Entity|Light] = [*get_all_parents(), *entities]
    parents = set(parents) if parents else set()

    for item in items:
        # Keyframe parent pose:
        if not item.is_hidden() or frame < 0 or item in parents:
            item.blender_obj.keyframe_insert(data_path="location", frame=frame)
            item.blender_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Keyframe children visibility:
        for child in get_all_child_meshes(item):
            child.blender_obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            child.blender_obj.keyframe_insert(data_path="hide_render", frame=frame)

        # Block for keyframing the material color (in case we change the material on each frame)
        # if item.get_attr("type") == "MESH" and len(item.blender_obj.material_slots): # Keyframe material colors.
        #     nodes = item.blender_obj.material_slots[0].material.node_tree.nodes
        #     if nodes.get("Principled BSDF"):
        #         nodes["Principled BSDF"].inputs["Base Color"].keyframe_insert(data_path="default_value", frame=frame)

        if item.get_attr("type") == "LIGHT":
            item.blender_obj.data.keyframe_insert(data_path="shadow_soft_size", frame=frame)
            item.blender_obj.data.keyframe_insert(data_path="energy", frame=frame)
            item.blender_obj.data.keyframe_insert(data_path="color", frame=frame)

    # Saving camera F-stop
    camera = bpy.data.objects['Camera']
    camera.data.keyframe_insert(data_path="dof.aperture_fstop", frame=frame)

    # Insert a keyframe for the strength at the given frame
    background_node = bpy.data.worlds["World"].node_tree.nodes["Background"]
    background_node.inputs["Strength"].keyframe_insert(data_path="default_value", frame=frame)


def hide_render_view(models:list[MeshObject]):
    for model in models:
        if model is None: continue # skip Nones

        for child in [model, *get_all_child_meshes(model)]:
            bpy_model = child.blender_obj

            bpy_model.hide_render = True
            bpy_model.hide_viewport = True



def fetch_keyframe_info() -> dict[str,list]:
    """
    Function to fetch keyframe data for any Blender data block that has animations.
    This includes objects, world properties, lights, materials, and more.
    The data is indexed by object_name -> list of frames with animated data.
    :return: {"object_name":[{"data1": value, "data2": value, ...}, {...}, ...], "object_name2":[{"data1": value, ...}, {...}, ...], ...}
    """
    animation_data_dict:dict[str,list[dict]] = {}

    # List of data blocks to check for animations
    data_blocks:list[bpy.types.BlendDataObjects] = [
        bpy.data.objects,  # All objects
        bpy.data.worlds,   # World settings
        bpy.data.lights,   # Lights
        bpy.data.cameras,  # Cameras
        bpy.data.materials, # Materials
        [bpy.data.worlds['World'].node_tree],   # World settings
    ]

    # Function to process animations on F-curves and populate the dictionary
    def process_fcurves(obj_name:str, action:bpy.types.Action, animation_dict:dict, data_label=""):
        for fcurve in action.fcurves:
            data_path = f"{data_label}{fcurve.data_path}"  # The property being animated
            array_index = fcurve.array_index  # X=0, Y=1, Z=2 for vector properties

            # Iterate through the keyframe points
            for keyframe in fcurve.keyframe_points:
                frame = int(keyframe.co.x)  # Frame number
                value = keyframe.co.y  # Value at that frame

                # Initialize a list for frames if not present
                frame_list:list = animation_dict.setdefault(obj_name, [])

                # Find the frame entry or create a new one
                frame_dict = next((item for item in frame_list if item["frame"] == frame), None)
                if frame_dict is None:
                    frame_dict = {"frame": frame}
                    frame_list.append(frame_dict)

                # If it's a vector property (location/rotation/color), store it in a tuple
                if "location" in data_path or "rotation_euler" in data_path or "color" in data_path:
                    frame_dict.setdefault(data_path, [None, None, None])[array_index] = value
                elif 'nodes["Principled BSDF"].inputs[0].default_value' in data_path:
                    frame_dict.setdefault(data_path, [None, None, None, None])[array_index] = value
                else:
                    # For single-value properties, store the value directly
                    frame_dict[data_path] = value
    
    # Iterate over each data block type (objects, worlds, etc.)
    for data_block in data_blocks:
        for obj in data_block:
            # Process object-level animations
            if obj.animation_data and obj.animation_data.action:
                process_fcurves(obj.name, obj.animation_data.action, animation_data_dict)

            # # Process data-level animations (e.g., lights, materials, etc.)
            # if hasattr(obj, 'data') and obj.data and obj.data.animation_data and obj.data.animation_data.action:
            #     process_fcurves(obj.name, obj.data.animation_data.action, animation_data_dict, data_label="data.")

    # Convert vector lists (X, Y, Z) to tuples
    for obj_name, frames in animation_data_dict.items():
        for frame_dict in frames:
            for data_path, values in frame_dict.items():
                if isinstance(values, list) and None not in values:
                    frame_dict[data_path] = tuple(values)


    # Saving also materials into objects as well as their colour:
    for obj in bpy.data.objects:
        if obj and len(obj.material_slots):
            material = obj.material_slots[0].material
            if material.node_tree.animation_data and material.node_tree.animation_data.action:
                process_fcurves(obj.name, material.node_tree.animation_data.action, animation_data_dict)
                
                for data in animation_data_dict[obj.name]:
                    data["material"] = material.name
                


    return animation_data_dict



def detect_overexposure(image_path:str=None, threshold=0.9, overexposure_limit=10, verbose=True):
    """
    Detects overexposure in a rendered image.
    
    Args:
    - image_path (str): The file path of the image to be analyzed.
    - threshold (float): The luminance threshold for detecting overexposed pixels. Default is 0.9.
    - overexposure_limit (float): The percentage threshold to classify the image as overexposed. Default is 5%.
    
    Returns:
    - percentage_overexposed (float): The percentage of overexposed pixels in the image.
    - overexposed (bool): True if the image is considered overexposed, False otherwise.
    """
    try:
        output_path = image_path or './examples/debugging/temp/rgb_render_exposure.png'

        
        # Ensure the render settings are configured to output an RGB image
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        
        # Render the image and save it
        bpy.ops.render.render(write_still=True)
        bpy.data.images['Render Result'].save_render(filepath=output_path)

        # Load the saved image back into Blender
        image = bpy.data.images.load(output_path)

        # Access the pixel data
        pixels = np.array(image.pixels[:]).reshape(-1, 4)  # Assuming RGBA format

        # Extract RGB values and compute luminance
        rgb_pixels = pixels[:, :3]
        luminance = np.mean(rgb_pixels, axis=1)

        # Handle NaN or invalid values in luminance
        luminance = np.nan_to_num(luminance, nan=0.0, posinf=0.0, neginf=0.0)

        # Count overexposed pixels (luminance > threshold)
        overexposed_pixels = np.sum(luminance > threshold)
        total_pixels = luminance.size
        percentage_overexposed = (overexposed_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        # Determine if the image is overexposed
        overexposed = percentage_overexposed > overexposure_limit

        if verbose:
            print(f"Percentage of overexposed pixels: {percentage_overexposed:.2f}%")
            if overexposed:
                print("Image is overexposed.")
            else:
                print("Image exposure is acceptable.")

        return percentage_overexposed, overexposed

    except Exception as e:
        print(f"Error loading image: {e}")
        return None, False



def load_models_folder(dir_path, whitelist:list[str] = None, blacklist:list[str] = None, pos_layout=None, collection=None, models_config:dict = None):
    loaded_models:list[MeshObject] = []

    # Load dict of complex models if any:
    models_config = models_config or {}

    # Load and filter models:
    paths = misc_utils.scan_folder(dir_path, whitelist, blacklist)

    for i, model_path in enumerate(paths):
        filename:str = os.path.basename(model_path)
        model_config = {}
        model_config.update(models_config.get("default_config", {}))
        model_config.update(models_config.get(filename, {}))

        # Loading model depending on its extension:
        extension = os.path.splitext(os.path.basename(model_path))[-1]
        
        if extension == ".blend":
            obj = bproc.loader.load_blend(model_path, obj_types=["mesh", "empty"])
        elif extension in (".ply", ".fbx", ".obj", ".glb"):
            obj = bproc.loader.load_obj(model_path)
        else:
            print(f"Warning: Unknown format for file: {model_path}, skipping it.", flush=True)
            continue

        # Processing loaded obj:
        parent_model = process_model(obj, name=os.path.basename(model_path), collection=collection, model_config=model_config)

        if pos_layout:
            parent_model.set_location(pos_layout(paths, i))

        loaded_models.append(parent_model)

    return loaded_models

# Lecacy function
def load_model_old(model_path:str, collection=None, model_config:dict = None):
    """
    Loads the model from a specific model_path. and adds the model to a collection if specified.

    It can also set the following configuration if a dict is passed with the following keys:

        - 'scale' : 1.0 - Scale factor of the loaded model.
        - 'keep_children' : True - Whether to keep the children meshes or combine them into a single mesh.
        - 'combined_parent' : True - Whether to have a combined mesh as a parent.
        - 'children_whitelist' : [] - Children meshes to be accepted.
        - 'children_blacklist' : [] - Children meshes to be removed.

    Parameters:
        model_path (str): Path to the model (.blend, .ply, .fbx, .obj, .glb).
        collection=None: Colleciton object in which include the loaded model.
        model_config (dict) = None: Model configuration.
    Returns:
        loaded_model: A reference to the parent of the loaded model
    """

    # Loading model depending on its extension:
    extension = os.path.splitext(os.path.basename(model_path))[-1]
    
    if extension == ".blend":
        models = bproc.loader.load_blend(model_path, obj_types=["mesh", "empty"])
    elif extension in (".ply", ".fbx", ".obj", ".glb"):
        models = bproc.loader.load_obj(model_path)
    else:
        return None

    # Default custom model settings
    model_config    = model_config or {}
    scale           = model_config.get("scale", 1.0)
    keep_children   = model_config.get("keep_children", True)
    combined_parent = model_config.get("combined_parent", True)
    whitelist       = set(model_config.get("children_whitelist", []) or [model.get_name() for model in models])
    blacklist       = set(model_config.get("children_blacklist", []))

    # Filter models based on whitelist and blacklist
    valid_model_names   = (set([model.get_name() for model in models]) & whitelist) - blacklist
    objects_to_delete   = [model for model in models if model.get_name() not in valid_model_names]
    models              = [model for model in models if model.get_name() in valid_model_names]

    # Process each model: adjust scale, get meshes, unparent them and update collection links if neccesary.
    meshes:list[MeshObject] = []
    to_delete:list[Entity] = []
    for model in models:
        if model.get_attr("type") == "MESH":
            model.clear_parent()
            model.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
            meshes.append(model)

            if collection:
                bpy.context.collection.objects.unlink(model.blender_obj)
                collection.objects.link(model.blender_obj)

        else:
            to_delete.append(model)

    
    # Now we handle parenting depending on whether we are joining the objects together or not.
    parent = meshes[0]

    if len(meshes) > 1:
        if keep_children:
            if combined_parent:
                parent = create_combined_parent(meshes)
            else:
                parent = bproc.object.create_with_empty_mesh("new_parent")
                # parent = bproc.object.merge_objects(meshes)
                parent.set_cp("combined_mesh", False)

            center = get_children_center(meshes)
            parent.set_location(center)

            if collection:
                bpy.context.collection.objects.unlink(parent.blender_obj)
                collection.objects.link(parent.blender_obj)

            # Setting common parent for all the meshes.
            for mesh in meshes:
                mesh.set_origin(center)
                mesh.set_parent(parent)

        else:
            parent.join_with_other_objects(meshes[1:])

    # Now we set the name of the parent to be the same as the file for easier clasification:
    if parent.get_attr("type") == "MESH":
        parent.set_origin(mode="CENTER_OF_MASS")
        
    parent.set_name(os.path.basename(model_path))

    bproc.object.delete_multiple(to_delete)
    bproc.object.delete_multiple(objects_to_delete)
    parent.set_scale(parent.get_scale()*scale)
    parent.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    # parent.blender_obj.show_name = True
        
    return parent

def load_model_v2_old(model_path:str, collection:bpy.types.Collection=None, model_config:dict = None):
    """
    Loads the model from a specific model_path. and adds the model to a collection if specified.

    Parameters:
        model_path (str): Path to the model (.blend, .ply, .fbx, .obj, .glb).
    Returns:
        loaded_model: A reference to the parent of the loaded model
    """

    # Loading model depending on its extension:
    extension = os.path.splitext(os.path.basename(model_path))[-1]
    
    if extension == ".blend":
        models = bproc.loader.load_blend(model_path, obj_types=["mesh", "empty"])
    elif extension in (".ply", ".fbx", ".obj", ".glb"):
        models = bproc.loader.load_obj(model_path)
    else:
        return None

    # Default custom model settings
    model_config    = model_config or {}
    scale           = model_config.get("scale", 1.0)
    keep_children   = model_config.get("keep_children", True)
    whitelist       = set(model_config.get("children_whitelist", []) or [model.get_name() for model in models])
    blacklist       = set(model_config.get("children_blacklist", []))

    # Filter models based on whitelist and blacklist
    valid_model_names   = (set([model.get_name() for model in models]) & whitelist) - blacklist
    remaining_models    = [model for model in models if model.get_name() not in valid_model_names]
    models              = [model for model in models if model.get_name() in valid_model_names]

    # Process each model: adjust scale, get meshes, unparent them and update collection links if neccesary.
    meshes:list[MeshObject] = []
    colliders:list[MeshObject] = []
    to_delete:list[Entity] = []
    for model in models:
        if model.get_attr("type") == "MESH":
            model.clear_parent()
            model.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
            
            if model.get_name().startswith("#Collider"):
                colliders.append(model)
                model.set_cp("skip_render", True)
                model.blender_obj.hide_render = True
            else:
                meshes.append(model)

            if collection:
                bpy.context.collection.objects.unlink(model.blender_obj)
                collection.objects.link(model.blender_obj)

        else:
            to_delete.append(model)

    # Create a parent from all the models:
    parent = create_bbox_mesh(meshes, name=os.path.basename(model_path))

    if collection:
        bpy.context.collection.objects.unlink(parent.blender_obj)
        collection.objects.link(parent.blender_obj)

    # Now we handle parenting depending on whether we are joining the objects together or not.
    if not keep_children:
        base = meshes.pop(0)
        if meshes:
            base.join_with_other_objects(meshes)
        meshes = [base]

    for mesh in meshes + colliders:
        mesh.set_parent(parent)

    parent.set_origin(mode="CENTER_OF_MASS")

    bproc.object.delete_multiple(to_delete)
    bproc.object.delete_multiple(remaining_models)
    parent.set_scale(parent.get_scale()*scale)
    parent.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
    for child in parent.get_children():
        child.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

        
    return parent

def process_model(models:list[MeshObject|Entity], name=None, collection:bpy.types.Collection=None, model_config:dict = None):
    """
    Process the model from a specific loaded obj and adds the model to a collection if specified.

    It can also set the following configuration if a dict is passed with the following keys:

        - 'scale' : 1.0 - Scale factor of the loaded model.
        - 'keep_children' : True - Whether to keep the children meshes or combine them into a single mesh.
        - 'children_whitelist' : [] - Children meshes to be accepted.
        - 'children_blacklist' : [] - Children meshes to be removed.

    Parameters:
        models (list[MeshObject|Entity]): List containing the loaded parts of the model.
        collection=None: Colleciton object in which include the loaded model.
        model_config (dict) = None: Model configuration.
    Returns:
        loaded_model: A reference to the parent of the loaded model
    """

    # Default custom model settings
    model_config    = model_config or {}
    scale           = model_config.get("scale", 1.0)
    keep_children   = model_config.get("keep_children", True)
    whitelist       = set(model_config.get("children_whitelist", []) or [model.get_name() for model in models])
    blacklist       = set(model_config.get("children_blacklist", []))

    # Filter models based on whitelist and blacklist
    valid_model_names   = (set([model.get_name() for model in models]) & whitelist) - blacklist
    remaining_models    = [model for model in models if model.get_name() not in valid_model_names]
    models              = [model for model in models if model.get_name() in valid_model_names]

    # Process each model: adjust scale, get meshes, unparent them and update collection links if neccesary.
    meshes:list[MeshObject] = []
    colliders:list[MeshObject] = []
    to_delete:list[Entity] = []
    for model in models:
        if model.get_attr("type") == "MESH":
            model.clear_parent()
            if model.blender_obj.data.users == 1:
                model.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
            
            if model.get_name().startswith("#Collider"):
                colliders.append(model)
                model.set_cp("skip_render", True)
                model.blender_obj.hide_render = True
            else:
                meshes.append(model)

            if collection:
                bpy.context.collection.objects.unlink(model.blender_obj)
                collection.objects.link(model.blender_obj)

        else:
            to_delete.append(model)

    # Create a parent from all the models:
    parent = create_bbox_mesh(meshes, name=name)

    if collection:
        bpy.context.collection.objects.unlink(parent.blender_obj)
        collection.objects.link(parent.blender_obj)

    # Now we handle parenting depending on whether we are joining the objects together or not.
    if not keep_children:
        base = meshes.pop(0)
        if meshes:
            base.join_with_other_objects(meshes)
        meshes = [base]

    for mesh in meshes + colliders:
        mesh.set_parent(parent)

    parent.set_origin(mode="CENTER_OF_MASS")

    bproc.object.delete_multiple(to_delete)
    bproc.object.delete_multiple(remaining_models)
    parent.set_scale(parent.get_scale()*scale)
    parent.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
    for child in parent.get_children():
        if child.blender_obj.data.users == 1:
            child.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

        
    return parent


def create_convex_collider(meshes:list[MeshObject], name="#Collider"):
    """Join meshes and generate a clean convex hull object."""
    # Duplicate and join meshes
    copies = [m.duplicate() for m in meshes]
    
    base:MeshObject = copies.pop(0)
    if copies:
        base.join_with_other_objects(copies)
    base.clear_all_cps()
    base.set_name(name)

    # Extract Hull
    mesh_data = base.blender_obj.data
    coords = [v.co.copy() for v in mesh_data.vertices]

    bm = bmesh.new()
    for c in coords:
        bm.verts.new(c)

    bm.verts.ensure_lookup_table()
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(mesh_data)
    bm.free()

    # cleanup
    mesh_data.materials.clear()
    base.blender_obj.display_type = 'WIRE'
    base.set_cp("skip_render", True)

    base.blender_obj.hide_render = True
    base.blender_obj.visible_camera = False
    base.blender_obj.visible_diffuse = False
    base.blender_obj.visible_glossy = False
    base.blender_obj.visible_transmission = False
    base.blender_obj.visible_volume_scatter = False
    base.blender_obj.visible_shadow = False
    base.blender_obj.display.show_shadows = False

    return base

def create_bbox_mesh(meshes:list[MeshObject], name=None):
    if not name:
        name = "bbox"

    min_co = Vector(( float('inf'),  float('inf'),  float('inf'))) # initialize “infinite” bounds
    max_co = Vector((-float('inf'), -float('inf'), -float('inf'))) # initialize “infinite” bounds

    for mesh in meshes:
        obj = mesh.blender_obj
        # ensure transforms are applied to the bound box
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_co.x = min(min_co.x, world_corner.x)
            min_co.y = min(min_co.y, world_corner.y)
            min_co.z = min(min_co.z, world_corner.z)
            max_co.x = max(max_co.x, world_corner.x)
            max_co.y = max(max_co.y, world_corner.y)
            max_co.z = max(max_co.z, world_corner.z)

    center = (min_co + max_co) / 2.0
    size   = max_co - min_co

    parent = bproc.object.create_primitive("CUBE", size=1.0, location=center, scale=size)
    parent.blender_obj.display_type = 'WIRE'
    parent.set_name(name)
    parent.set_cp("skip_render", True)
    parent.set_cp("bbox", True)
    parent.blender_obj.hide_render = True

    parent.blender_obj.visible_camera = False
    parent.blender_obj.visible_diffuse = False
    parent.blender_obj.visible_glossy = False
    parent.blender_obj.visible_transmission = False
    parent.blender_obj.visible_volume_scatter = False
    parent.blender_obj.visible_shadow = False
    parent.blender_obj.display.show_shadows = False



    return parent

# Legacy function
def copy_material_from_train_to_distr(train_models: list[MeshObject], distractors: list[MeshObject]):
    """
    Copies materials from training models to distractor objects.
    
    Args:
        train_models: List of training model objects
        distractors: List of distractor objects to receive the materials
    """
    # Collect all available materials from train models
    train_materials = []
    for model in train_models:
        for child in get_all_child_meshes(model):
            if child.blender_obj.material_slots:
                material = child.blender_obj.material_slots[0].material
                if material not in train_materials:
                    train_materials.append(material)

    if not train_materials:
        print("Warning: No materials found in training models to copy")
        return

    # Apply random materials from training models to distractors
    for distractor in distractors:
        for child in get_all_child_meshes(distractor):
            if child.blender_obj.material_slots:
                # Pick a random material from training models
                random_material = random.choice(train_materials)
                # Create a copy of the material to avoid modifying the original
                new_material = random_material.copy()
                # Apply the copied material to the distractor
                child.blender_obj.material_slots[0].material = new_material


def create_random_distractor(pos, collection=None, models_config:dict=None):
    models_config = models_config or {}

    modes = ["CUBE", "CYLINDER", "CONE", "SPHERE"]

    mode = np.random.choice(modes)

    simple_model = bproc.object.create_primitive(mode)
    simple_model.set_name(f"{mode}.000")
    model_name = simple_model.get_name()

    model_config = {}
    model_config.update(models_config.get("default_config", {}))
    model_config.update(models_config.get(model_name, {}))

    simple_model.set_scale(simple_model.get_scale()*np.random.uniform([0.05,0.05,0.05], [0.15,0.15,0.15]))
    simple_model.set_scale(simple_model.get_scale()*model_config.get('scale', 1))
    simple_model.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    simple_model.new_material("Simple_distractor_material.000")
    material = simple_model.get_materials()[0]
    material.set_principled_shader_value("Base Color", np.random.uniform([0,0,0,1], [1,1,1,1]))
    
    if collection:
        bpy.context.collection.objects.unlink(simple_model.blender_obj)
        collection.objects.link(simple_model.blender_obj)  

    # Create a parent from all the models:
    parent = create_bbox_mesh([simple_model], name=f"Simple_model.000")
    simple_model.set_parent(parent)

    if collection:
        bpy.context.collection.objects.unlink(parent.blender_obj)
        collection.objects.link(parent.blender_obj)

    parent.set_location(pos)


    return parent

def create_duplicates(models:list[MeshObject], config:dict[str, dict], default_config:dict):
    copies = []
    custom_models = config.get("custom_models", {})

    for model in models:
        model_name = model.get_name()

        model_config = {}
        model_config.update(default_config)
        model_config.update(custom_models.get(model_name, {}))

        n_copies = model_config['n_copies']

        for _ in range(n_copies):
            copy = model.duplicate(duplicate_children=True, linked=True) # Linked == True to share mesh data

            for child in get_all_child_meshes(copy):
                bpy.context.collection.objects.unlink(child.blender_obj)
                for collection in model.blender_obj.users_collection:
                    collection.objects.link(child.blender_obj)
            
            config["custom_models"][copy.get_name()] = model_config

            copies.append(copy)


    models.extend(copies)

    return copies


def displace_model_normalized(model:MeshObject=None, strength: float=1):
    """
    Applies a random displacement to the active mesh object.
    The displacement is normalized relative to the object's bounding box diagonal,
    making the effect consistent regardless of the object's scale.
    
    :param strength: A factor (between 0 and 1 is typical) that scales the maximum displacement.
    """
    if model is None:
        obj = bpy.context.active_object
    else:
        obj = model.blender_obj

    if obj is None or obj.type != 'MESH':
        raise Exception("Please select a mesh object!")
    
    # Access the mesh data using BMesh
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Compute the bounding box extents in local space
    xs = [vert.co.x for vert in bm.verts]
    ys = [vert.co.y for vert in bm.verts]
    zs = [vert.co.z for vert in bm.verts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    # Calculate the diagonal length of the bounding box
    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z
    diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)

    # Apply random displacement to every vertex using the diagonal as a scale factor
    for vert in bm.verts:
        vert.co.x += random.uniform(-strength, strength) * diagonal
        vert.co.y += random.uniform(-strength, strength) * diagonal
        vert.co.z += random.uniform(-strength, strength) * diagonal

    # Write the modified BMesh back to the mesh and update the scene
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    print("Normalized random displacement applied successfully.")

def twist_model_uniform(model:MeshObject=None, twist_axis: str='X', twist_strength: float=1):
    """
    Applies a uniform twist deformation to the active mesh object.
    The twist effect is normalized based on the object's bounding box along the twist axis.
    
    :param twist_axis: Axis to twist around ('X', 'Y', or 'Z')
    :param twist_strength: Maximum twist angle in radians.
    """
    print(model.get_name(), twist_axis, twist_strength)

    if model is None:
        obj = bpy.context.active_object
    else:
        obj = model.blender_obj
        
    if obj is None or obj.type != 'MESH':
        raise Exception("Please select a mesh object!")
    
    # Access the mesh data using BMesh
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Determine which index corresponds to the twist axis
    axis_index = {'X': 0, 'Y': 1, 'Z': 2}.get(twist_axis.upper(), None)
    if axis_index is None:
        raise ValueError("Invalid twist axis. Choose 'X', 'Y', or 'Z'.")
    
    # Compute the min, max, and center values along the twist axis for all vertices
    coords = [vert.co[axis_index] for vert in bm.verts]
    min_coord = min(coords)
    max_coord = max(coords)
    center_coord = (min_coord + max_coord) / 2.0
    range_coord = max_coord - min_coord
    
    if range_coord == 0:
        range_coord = 1  # Avoid division by zero if object is flat on this axis.
    
    # Apply the twist deformation uniformly:
    for vert in bm.verts:
        # Normalize the coordinate to the range [-0.5, 0.5]
        norm_val = (vert.co[axis_index] - center_coord) / range_coord
        # Calculate twist angle: vertices at the extremes get ±twist_strength
        angle = twist_strength * norm_val * 2.0  # Multiplied by 2 to span the full range
        
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        # Apply rotation based on the chosen twist axis
        if twist_axis.upper() == 'Z':
            original_x = vert.co.x
            original_y = vert.co.y
            # Rotate in the XY plane around Z
            vert.co.x = cos_angle * original_x - sin_angle * original_y
            vert.co.y = sin_angle * original_x + cos_angle * original_y
            
        elif twist_axis.upper() == 'X':
            original_y = vert.co.y
            original_z = vert.co.z
            # Rotate in the YZ plane around X
            vert.co.y = cos_angle * original_y - sin_angle * original_z
            vert.co.z = sin_angle * original_y + cos_angle * original_z
            
        elif twist_axis.upper() == 'Y':
            original_x = vert.co.x
            original_z = vert.co.z
            # Rotate in the XZ plane around Y
            vert.co.x = cos_angle * original_x - sin_angle * original_z
            vert.co.z = sin_angle * original_x + cos_angle * original_z
    
    # Write the modified BMesh back to the mesh and update the scene
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    print("Uniform twist deformation applied successfully.")

def create_similar_distractor(pos, model:MeshObject, collection=None, model_config:dict=None):
    model_config = model_config or {}

    copy:MeshObject|Entity = model.duplicate() # Copy not linked of ex_plane
    copy.clear_all_cps()

    # In case the model has childrens, simplify model as a single mesh.
    copy_children = [child for child in get_all_child_meshes(copy)]
    base_mesh = copy_children.pop(0)
    if copy_children:
        base_mesh.join_with_other_objects(copy_children)


    # In case a collection has ben passed:
    if collection:
        bpy.context.collection.objects.unlink(base_mesh.blender_obj)
        collection.objects.link(base_mesh.blender_obj)

    # Apply twist in every axis:
    # 0.25 0.75
    rand_strength = lambda min, max: np.random.randint(min, max+1) / 100 * np.random.choice([-1,1])
    twist_model_uniform(base_mesh,'X', rand_strength(50,70))
    twist_model_uniform(base_mesh,'Y', rand_strength(50,70))
    twist_model_uniform(base_mesh,'Z', rand_strength(50,70))

    # Apply vertex displacement:
    displace_model_normalized(base_mesh, 0.001)

    parent = get_parent(base_mesh)
    parent.set_location(pos)

    parent.set_scale(parent.get_scale()*model_config.get('scale', 1))
    parent.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    return parent


# Calculate the bounding box center from the child objects
def get_children_center(children:list[MeshObject]):
    # Initialize min/max vectors to extreme values
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((-float('inf'), -float('inf'), -float('inf')))
    
    for child in children:
        # Ensure we are using the evaluated object in the current context
        obj = child.blender_obj  
        # Each object's bounding box is in local space; convert to world coordinates
        for corner in obj.bound_box:
            # Convert the local coordinate to world coordinate
            v_world = obj.matrix_world @ Vector(corner)
            min_co.x = min(min_co.x, v_world.x)
            min_co.y = min(min_co.y, v_world.y)
            min_co.z = min(min_co.z, v_world.z)
            max_co.x = max(max_co.x, v_world.x)
            max_co.y = max(max_co.y, v_world.y)
            max_co.z = max(max_co.z, v_world.z)
    
    # Calculate center from the bounding box min and max
    center = (min_co + max_co) / 2
    return center
    
def create_combined_parent(meshes:list[MeshObject]):
    parent = bproc.object.create_with_empty_mesh("temp_parent")

    # Duplicate each mesh object so we don't alter the originals.
    dup_objects = []
    for mesh in meshes:
        dup = mesh.duplicate(duplicate_children=False, linked=False)
        dup_objects.append(dup)

    center = get_children_center(dup_objects)
    base:MeshObject = dup_objects[0]
    base.join_with_other_objects(dup_objects[1:])
    base.set_origin(center) 

    joined_bmesh = base.mesh_as_bmesh(return_copy=True)

    parent.set_location(base.get_location())
    parent.set_rotation_euler(base.get_rotation_euler())
    parent.set_scale(base.get_scale())
    parent.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
    parent.update_from_bmesh(bm=joined_bmesh)
    parent.set_cp("combined_mesh", True)

    bproc.object.delete_multiple([base])

    parent.blender_obj.hide_viewport = True
    parent.blender_obj.hide_render = True
    parent.blender_obj.visible_camera = False

    def simplify_with_decimate(obj: MeshObject, ratio: float = 0.1):
        # Access the underlying Blender object
        blender_obj = obj.blender_obj
        
        # Add a Decimate modifier
        modifier = blender_obj.modifiers.new(name="Decimate", type='DECIMATE')
        modifier.ratio = ratio  # e.g., 0.1 keeps 10% of the original geometry
        
        # Optionally apply the modifier immediately
        bpy.context.view_layer.objects.active = blender_obj
        blender_obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier="Decimate")
        blender_obj.select_set(False)

    # simplify_with_decimate(parent, ratio=0.001)


    return parent







