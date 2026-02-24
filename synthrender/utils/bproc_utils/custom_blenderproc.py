import bpy
import mathutils

import os
import numpy as np
import h5py
import json
import cv2
import sys

from tqdm.auto import tqdm
from typing import List, Dict
from typing import Optional, Dict, Tuple, List, Union
from mathutils import Matrix
from skimage import measure

from blenderproc.python.types.MeshObjectUtility import MeshObject, get_all_mesh_objects
from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects
from blenderproc.python.writer.WriterUtility import _WriterUtility
from blenderproc.python.utility import Utility, BlenderUtility
from blenderproc.python.writer import CocoWriterUtility
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.object.PhysicsSimulation import _PhysicsSimulation
from blenderproc.python.utility.Utility import UndoAfterExecution

from synthrender.utils.bproc_utils import bproc_utils


def modified_write_hdf5(output_dir_path: str, output_data_dict: Dict[str, List[Union[np.ndarray, list, dict]]],
               append_to_existing_output: bool = False, stereo_separate_keys: bool = False):
    """
    Saves the information provided inside of the output_data_dict into a .hdf5 container

    :param output_dir_path: The folder path in which the .hdf5 containers will be generated
    :param output_data_dict: The container, which keeps the different images, which should be saved to disc.
                             Each key will be saved as its own key in the .hdf5 container.
    :param append_to_existing_output: If this is True, the output_dir_path folder will be scanned for pre-existing
                                      .hdf5 containers and the numbering of the newly added containers, will start
                                      right where the last run left off.
    :param stereo_separate_keys: If this is True and the rendering was done in stereo mode, than the stereo images
                                 won't be saved in one tensor [2, img_x, img_y, channels], where the img[0] is the
                                 left image and img[1] the right. They will be saved in separate keys: for example
                                 for colors in colors_0 and colors_1.
    """

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    amount_of_frames = 0
    for data_block in output_data_dict.values():
        if isinstance(data_block, list):
            amount_of_frames = max([amount_of_frames, len(data_block)])

    # if append to existing output is turned on the existing folder is searched for the highest occurring
    # index, which is then used as starting point for this run
    if append_to_existing_output:
        frame_offset = 0
        # Look for hdf5 file with highest index
        for path in os.listdir(output_dir_path):
            if path.endswith(".hdf5"):
                index = path[:-len(".hdf5")]
                if index.isdigit():
                    frame_offset = max(frame_offset, int(index) + 1)
    else:
        frame_offset = 0

    if amount_of_frames != bpy.context.scene.frame_end - bpy.context.scene.frame_start:
        raise Exception("The amount of images stored in the output_data_dict does not correspond with the amount"
                        "of images specified by frame_start to frame_end.")

    frames = (bpy.context.scene.frame_start, bpy.context.scene.frame_end-1)
    for frame in tqdm(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end), desc=f"Saving hdf5 {frames}", unit="hdf5", ncols=6, disable=True):
        # for each frame a new .hdf5 file is generated
        hdf5_path = os.path.join(output_dir_path, str(frame + frame_offset) + ".hdf5")
        with h5py.File(hdf5_path, "w") as file:
            # Go through all the output types
            print(f"[update] Merging data into {frame}.hdf5 ({frame}/{bpy.context.scene.frame_end-1})", flush=True)

            adjusted_frame = frame - bpy.context.scene.frame_start
            for key, data_block in output_data_dict.items():
                if adjusted_frame < len(data_block):
                    # get the current data block for the current frame
                    used_data_block = data_block[adjusted_frame]
                    if stereo_separate_keys and (bpy.context.scene.render.use_multiview or
                                                 used_data_block.shape[0] == 2):
                        # stereo mode was activated
                        _WriterUtility.write_to_hdf_file(file, key + "_0", data_block[adjusted_frame][0])
                        _WriterUtility.write_to_hdf_file(file, key + "_1", data_block[adjusted_frame][1])
                    else:
                        _WriterUtility.write_to_hdf_file(file, key, data_block[adjusted_frame])
                else:
                    raise Exception(f"Error: There are more frames {adjusted_frame} then there are blocks of information "
                                    f" {len(data_block)} in the given list for key {key}.")
            blender_proc_version = BlenderUtility.Utility.get_current_version()
            if blender_proc_version is not None:
                _WriterUtility.write_to_hdf_file(file, "blender_proc_version", np.string_(blender_proc_version))

def modified_write_coco_annotations(output_dir: str, instance_segmaps: List[np.ndarray],
                           instance_attribute_maps: List[dict],
                           colors: List[np.ndarray], color_file_format: str = "PNG",
                           mask_encoding_format: str = "rle", supercategory: str = "coco_annotations",
                           append_to_existing_output: bool = True,
                           jpg_quality: int = 95, label_mapping = None,
                           file_prefix: str = "", indent: Optional[Union[int, str]] = None):
    """ Writes coco annotations in the following steps:
    1. Locate the seg images
    2. Locate the rgb maps
    3. Locate the seg mappings
    4. Read color mappings
    5. For each frame write the coco annotation

    :param output_dir: Output directory to write the coco annotations
    :param instance_segmaps: List of instance segmentation maps
    :param instance_attribute_maps: per-frame mappings with idx, class and optionally supercategory/bop_dataset_name
    :param colors: List of color images. Does not support stereo images, enter left and right inputs subsequently.
    :param color_file_format: Format to save color images in
    :param mask_encoding_format: Encoding format of the binary masks. Default: 'rle'. Available: 'rle', 'polygon'.
    :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset set
                          by 'bop_dataset_name' or any loaded object with specified 'cp_supercategory'
    :param append_to_existing_output: If true and if there is already a coco_annotations.json file in the output
                                      directory, the new coco annotations will be appended to the existing file.
                                      Also, the rgb images will be named such that there are no collisions.
    :param jpg_quality: The desired quality level of the jpg encoding
    :param label_mapping: The label mapping which should be used to label the categories based on their ids.
                          If None, is given then the `name` field in the csv files is used or - if not existing -
                          the category id itself is used.
    :param file_prefix: Optional prefix for image file names
    :param indent: If indent is a non-negative integer or string, then the annotation output
                   will be pretty-printed with that indent level. An indent level of 0, negative, or "" will
                   only insert newlines. None (the default) selects the most compact representation.
                   Using a positive integer indent indents that many spaces per level.
                   If indent is a string (such as "\t"), that string is used to indent each level.
    """

    if len(colors) > 0 and len(colors[0].shape) == 4:
        raise ValueError("BlenderProc currently does not support writing coco annotations for stereo images. "
                         "However, you can enter left and right images / segmaps separately.")

    # Create output directory
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    coco_annotations_path = os.path.join(output_dir, "coco_annotations.json")
    # Calculate image numbering offset, if append_to_existing_output is activated and coco data exists
    if append_to_existing_output and os.path.exists(coco_annotations_path):
        with open(coco_annotations_path, 'r', encoding="utf-8") as fp:
            existing_coco_annotations = json.load(fp)
        image_offset = max(image["id"] for image in existing_coco_annotations["images"]) + 1
    else:
        image_offset = 0
        existing_coco_annotations = None

    # collect all RGB paths
    new_coco_image_paths = []

    # for each rendered frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        color_rgb = colors[frame - bpy.context.scene.frame_start]

        # Reverse channel order for opencv
        color_bgr = color_rgb.copy()
        color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

        if color_file_format == 'PNG':
            target_base_path = f'images/{file_prefix}{frame + image_offset:06d}.png'
            target_path = os.path.join(output_dir, target_base_path)
            cv2.imwrite(target_path, color_bgr)
        elif color_file_format == 'JPEG':
            target_base_path = f'images/{file_prefix}{frame + image_offset:06d}.jpg'
            target_path = os.path.join(output_dir, target_base_path)
            cv2.imwrite(target_path, color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        else:
            raise RuntimeError(f'Unknown color_file_format={color_file_format}. Try "PNG" or "JPEG"')


        new_coco_image_paths.append(target_base_path)

    coco_output = CocoWriterUtility._CocoWriterUtility.generate_coco_annotations(instance_segmaps,
                                                               instance_attribute_maps,
                                                               new_coco_image_paths,
                                                               supercategory,
                                                               mask_encoding_format,
                                                               existing_coco_annotations,
                                                               label_mapping)

    with open(coco_annotations_path, 'w', encoding="utf-8") as fp:
        json.dump(coco_output, fp, indent=indent)




class check_collitions:
    @staticmethod
    def check_spacing_fast(obj_pos: np.ndarray, placed_objects: List[tuple[np.ndarray, np.ndarray]], min_distance: float, max_distance: float = np.inf) \
            -> bool:
        """ Check if object is not too close or too far from previous objects.

        :param obj_pos: Position of object for which the check is carried out.
        :param placed_objects: A list of locations for already placed objects that should be used for checking spacing.
        :param min_distance: Minimum distance to the closest other object from placed_objects. Center to center.
        :param max_distance: Maximum distance to the closest other object from placed_objects. Center to center.
        :return: True, if the spacing is correct
        """

        closest_distance = None
        for already_placed in placed_objects:
            distance = np.linalg.norm(already_placed[0] - obj_pos)
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance

            if not (min_distance <= closest_distance <= max_distance):
                return False

        return closest_distance is None or (min_distance <= closest_distance <= max_distance)

    @staticmethod
    def check_spacing(obj: MeshObject, placed_objects: List[MeshObject], min_distance: float, max_distance: float = np.inf) \
            -> bool:
        """ Check if object is not too close or too far from previous objects.

        :param obj: Object for which the check is carried out.
        :param placed_objects: A list of already placed objects that should be used for checking spacing.
        :param min_distance: Minimum distance to the closest other object from placed_objects. Center to center.
        :param max_distance: Maximum distance to the closest other object from placed_objects. Center to center.
        :return: True, if the spacing is correct
        """
        models = [obj, *bproc_utils.get_all_child_meshes(obj)]

        closest_distance = None
        for model in models:
            for already_placed in placed_objects:
                distance = np.linalg.norm(already_placed.get_local2world_mat()[:3,-1] - model.get_local2world_mat()[:3,-1])
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance

                if not (min_distance <= closest_distance <= max_distance):
                    return False

        return closest_distance is None or (min_distance <= closest_distance <= max_distance)   
    
    @staticmethod
    def check_intersections(obj: MeshObject, bvh_cache: Optional[Dict[str, mathutils.bvhtree.BVHTree]],
                            objects_to_check_against: List[MeshObject],
                            list_of_objects_with_no_inside_check: List[MeshObject]):
        """ Checks if an object intersects with any object given in the list.

        The bvh_cache adds all current objects to the bvh tree, which increases the speed.

        If an object is already in the cache it is removed, before performing the check.

        :param obj: Object which should be checked. Type: :class:`bpy.types.Object`
        :param bvh_cache: Dict of all the bvh trees, removes the `obj` from the cache before adding it again. \
                          Type: :class:`dict`
        :param objects_to_check_against: List of objects which the object is checked again \
                                         Type: :class:`list`
        :param list_of_objects_with_no_inside_check: List of objects on which no inside check is performed. \
                                                     This check is only done for the objects in \
                                                     `objects_to_check_against`. Type: :class:`list`
        :return: Type: :class:`bool`, True if no collision was found, false if at least one collision was found
        """

        no_collision = True
        # Now check for collisions
        for collision_obj in objects_to_check_against:
            # Do not check collisions with yourself
            if collision_obj == obj:
                continue
            # First check if bounding boxes collides
            intersection = CollisionUtility.check_bb_intersection(obj, collision_obj)
            # if they do
            if intersection:
                skip_inside_check = collision_obj in list_of_objects_with_no_inside_check
                # then check for more refined collisions
                intersection, bvh_cache = check_collitions.check_mesh_intersection(obj, collision_obj,
                                                                                   bvh_cache=bvh_cache,
                                                                                   skip_inside_check=skip_inside_check)
            if intersection:
                no_collision = False
                break
        return no_collision
    

    @staticmethod
    def check_mesh_intersection(obj1: MeshObject, obj2: MeshObject, skip_inside_check: bool = False,
                                bvh_cache: Optional[Dict[str, mathutils.bvhtree.BVHTree]] = None) \
            -> Tuple[bool, Dict[str, mathutils.bvhtree.BVHTree]]:
        """
        Checks if the two objects are intersecting.

        This will use BVH trees to check whether the objects are overlapping.

        It is further also checked if one object is completely inside the other.
        This check requires that both objects are watertight, have correct normals and are coherent.
        If this is not the case it can be disabled via the parameter skip_inside_check.

        :param obj1: object 1 to check for intersection, must be a mesh
        :param obj2: object 2 to check for intersection, must be a mesh
        :param skip_inside_check: Disables checking whether one object is completely inside the other.
        :param bvh_cache: Dict of all the bvh trees, removes the `obj` from the cache before adding it again.
        :return: True, if they are intersecting
        """

        if bvh_cache is None:
            bvh_cache = {}

        # If one of the objects has no vertices, collision is impossible
        if len(obj1.get_mesh().vertices) == 0 or len(obj2.get_mesh().vertices) == 0:
            return False, bvh_cache

        # create bvhtree for obj1
        if obj1.get_name() not in bvh_cache:
            obj1_BVHtree = obj1.create_bvh_tree()
            bvh_cache[obj1.get_name()] = obj1_BVHtree
        else:
            obj1_BVHtree = bvh_cache[obj1.get_name()]

        # create bvhtree for obj2
        if obj2.get_name() not in bvh_cache:
            obj2_BVHtree = obj2.create_bvh_tree()
            bvh_cache[obj2.get_name()] = obj2_BVHtree
        else:
            obj2_BVHtree = bvh_cache[obj2.get_name()]

        # Check whether both meshes intersect
        inter = len(obj1_BVHtree.overlap(obj2_BVHtree)) > 0

        # Optionally check whether obj2 is contained in obj1
        if not inter and not skip_inside_check:
            inter = CollisionUtility.is_point_inside_object(obj1, obj1_BVHtree,
                                                            Matrix(obj2.get_local2world_mat()) @
                                                            obj2.get_mesh().vertices[0].co)
            # if inter:
                # print("Warning: Detected that " + obj2.get_name() + " is completely inside " + obj1.get_name() +
                #       ". This might be wrong, if " + obj1.get_name() +
                #       " is not water tight or has incorrect normals. If that is the case, consider setting "
                #       "skip_inside_check to True.")

        # Optionally check whether obj1 is contained in obj2
        if not inter and not skip_inside_check:
            inter = CollisionUtility.is_point_inside_object(obj2, obj2_BVHtree, Matrix(obj1.get_local2world_mat())
                                                            @ obj1.get_mesh().vertices[0].co)
            # if inter:
            #     print("Warning: Detected that " + obj1.get_name() + " is completely inside " + obj2.get_name() +
            #           ". This might be wrong, if " + obj2.get_name() + " is not water tight or has incorrect "
            #                                                            "normals. If that is the case, consider "
            #                                                            "setting skip_inside_check to True.")

        return inter, bvh_cache

@staticmethod
def custom_binary_mask_to_polygon(binary_mask: np.ndarray, tolerance: int = 0) -> list:
    """Converts a binary mask to COCO polygon representation."""
    polygons = []

    # Pad mask to close contours of shapes that touch the edge of the image
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    
    # Get contours of the binary mask
    contours = measure.find_contours(padded_binary_mask, 0.5)

    # Process each contour
    for contour in contours:
        # Reverse padding effect (subtract the 1 added by padding)
        contour = contour - 1
        # Close the contour
        contour = CocoWriterUtility._CocoWriterUtility.close_contour(contour)

        # Approximate the contour by a polygon with tolerance (to simplify)
        polygon = measure.approximate_polygon(contour, tolerance)
        # Skip invalid polygons with fewer than 3 points
        if len(polygon) < 3:
            continue

        # Convert (y, x) to (x, y)
        polygon = np.flip(polygon, axis=1)
        # Flatten the contour into a list of points
        polygon = polygon.ravel().tolist()
        # Clip negative coordinates (created by padding offset)
        polygon = [max(0, point) for point in polygon]
        polygons.append(polygon)

    return polygons

##################################################################################################
# Ammendments for having physics simulation starting at a specific frame rather than at frame 1. #
##################################################################################################

def simulate_physics_and_fix_final_poses(min_simulation_time: float = 4.0, max_simulation_time: float = 40.0,
                                         check_object_interval: float = 2.0,
                                         object_stopped_location_threshold: float = 0.01,
                                         object_stopped_rotation_threshold: float = 0.1, substeps_per_frame: int = 10,
                                         solver_iters: int = 10, verbose: bool = False, use_volume_com: bool = False, frame_start:int=None):
    """ Simulates the current scene and in the end fixes the final poses of all active objects.

    The simulation is run for at least `min_simulation_time` seconds and at a maximum `max_simulation_time` seconds.
    Every `check_object_interval` seconds, it is checked if the maximum object movement in the last second is below a
    given threshold. If that is the case, the simulation is stopped.

    After performing the simulation, the simulation cache is removed, the rigid body components are disabled and the
    pose of the active objects is set to their final pose in the simulation.

    :param min_simulation_time: The minimum number of seconds to simulate.
    :param max_simulation_time: The maximum number of seconds to simulate.
    :param check_object_interval: The interval in seconds at which all objects should be checked if they are still
                                  moving. If all objects have stopped moving, then the simulation will be stopped.
    :param object_stopped_location_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param object_stopped_rotation_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param substeps_per_frame: Number of simulation steps taken per frame.
    :param solver_iters: Number of constraint solver iterations made per simulation step.
    :param verbose: If True, more details during the physics simulation are printed.
    :param use_volume_com: If True, the center of mass will be calculated by using the object volume.
                           This is more accurate than using the surface area (default), but requires a watertight mesh.
    """
    # Undo changes made in the simulation like origin adjustment and persisting the object's scale
    with UndoAfterExecution():
        # Run simulation and remember poses before and after
        obj_poses_before_sim = _PhysicsSimulation.get_pose()
        origin_shifts = simulate_physics(min_simulation_time, max_simulation_time, check_object_interval,
                                         object_stopped_location_threshold, object_stopped_rotation_threshold,
                                         substeps_per_frame, solver_iters, verbose, use_volume_com, frame_start)
        obj_poses_after_sim = _PhysicsSimulation.get_pose()

        # Make sure to remove the simulation cache as we are only interested in the final poses
        with bpy.context.temp_override(point_cache=bpy.context.scene.rigidbody_world.point_cache):
            bpy.ops.ptcache.free_bake()

    # Fix the pose of all objects to their pose at the end of the simulation (also revert origin shift)
    for obj in get_all_mesh_objects():
        if obj.has_rigidbody_enabled():
            # Skip objects that have parents with compound rigid body component
            has_compound_parent = obj.get_parent() is not None and isinstance(obj.get_parent(), MeshObject) \
                                  and obj.get_parent().get_rigidbody() is not None \
                                  and obj.get_parent().get_rigidbody().collision_shape == "COMPOUND"
            if obj.get_rigidbody().type == "ACTIVE" and not has_compound_parent:
                # compute relative object rotation before and after simulation
                R_obj_before_sim = mathutils.Euler(obj_poses_before_sim[obj.get_name()]['rotation']).to_matrix()
                R_obj_after = mathutils.Euler(obj_poses_after_sim[obj.get_name()]['rotation']).to_matrix()
                R_obj_rel = R_obj_before_sim @ R_obj_after.transposed()
                # Apply relative rotation to origin shift
                origin_shift = R_obj_rel.transposed() @ mathutils.Vector(origin_shifts[obj.get_name()])

                # Fix pose of object to the one it had at the end of the simulation
                obj.set_location(obj_poses_after_sim[obj.get_name()]['location'] - origin_shift)
                obj.set_rotation_euler(obj_poses_after_sim[obj.get_name()]['rotation'])

    # Deactivate the simulation so it does not influence object positions
    bpy.context.scene.rigidbody_world.enabled = False
    bpy.context.view_layer.update()

def simulate_physics(min_simulation_time: float = 4.0, max_simulation_time: float = 40.0,
                     check_object_interval: float = 2.0, object_stopped_location_threshold: float = 0.01,
                     object_stopped_rotation_threshold: float = 0.1, substeps_per_frame: int = 10,
                     solver_iters: int = 10, verbose: bool = False, use_volume_com: bool = False, frame_start:int=None) -> dict:
    """ Simulates the current scene.

    The simulation is run for at least `min_simulation_time` seconds and at a maximum `max_simulation_time` seconds.
    Every `check_object_interval` seconds, it is checked if the maximum object movement in the last second is below
    a given threshold. If that is the case, the simulation is stopped.

    The origin of all objects is set to their center of mass in this function which is necessary to achieve a realistic
    simulation in blender (see https://blender.stackexchange.com/questions/167488/physics-not-working-as-expected)
    Also the scale of each participating object is persisted as scale != 1 can make the simulation unstable.

    :param min_simulation_time: The minimum number of seconds to simulate.
    :param max_simulation_time: The maximum number of seconds to simulate.
    :param check_object_interval: The interval in seconds at which all objects should be checked if they are still
                                  moving. If all objects have stopped moving, then the simulation will be stopped.
    :param object_stopped_location_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param object_stopped_rotation_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param substeps_per_frame: Number of simulation steps taken per frame.
    :param solver_iters: Number of constraint solver iterations made per simulation step.
    :param verbose: If True, more details during the physics simulation are printed.
    :param use_volume_com: If True, the center of mass will be calculated by using the object volume.
                           This is more accurate than using the surface area (default), but requires a watertight mesh.
    :return: A dict containing for every active object the shift that was added to their origins.
    """
    # Shift the origin of all objects to their center of mass to make the simulation more realistic
    origin_shift = {}
    for obj in get_all_mesh_objects():
        if obj.has_rigidbody_enabled():
            prev_origin = obj.get_origin()
            new_origin = obj.set_origin(mode="ORIGIN_CENTER_OF_VOLUME" if use_volume_com else "CENTER_OF_MASS")
            origin_shift[obj.get_name()] = new_origin - prev_origin

            # Persist mesh scaling as having a scale != 1 can make the simulation unstable
            obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    # Configure simulator
    bpy.context.scene.rigidbody_world.substeps_per_frame = substeps_per_frame
    bpy.context.scene.rigidbody_world.solver_iterations = solver_iters

    # Perform simulation
    do_simulation(min_simulation_time, max_simulation_time, check_object_interval,
                                     object_stopped_location_threshold, object_stopped_rotation_threshold,
                                     verbose, frame_start)

    return origin_shift


def simulate_physics_and_fix_final_poses_v2(min_simulation_time: float = 4.0, max_simulation_time: float = 40.0,
                                         check_object_interval: float = 2.0,
                                         object_stopped_location_threshold: float = 0.01,
                                         object_stopped_rotation_threshold: float = 0.1, substeps_per_frame: int = 10,
                                         solver_iters: int = 10, verbose: bool = False, use_volume_com: bool = False, frame_start:int=None):
    """ Simulates the current scene and in the end fixes the final poses of all active objects.

    The simulation is run for at least `min_simulation_time` seconds and at a maximum `max_simulation_time` seconds.
    Every `check_object_interval` seconds, it is checked if the maximum object movement in the last second is below a
    given threshold. If that is the case, the simulation is stopped.

    After performing the simulation, the simulation cache is removed, the rigid body components are disabled and the
    pose of the active objects is set to their final pose in the simulation.

    :param min_simulation_time: The minimum number of seconds to simulate.
    :param max_simulation_time: The maximum number of seconds to simulate.
    :param check_object_interval: The interval in seconds at which all objects should be checked if they are still
                                  moving. If all objects have stopped moving, then the simulation will be stopped.
    :param object_stopped_location_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param object_stopped_rotation_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param substeps_per_frame: Number of simulation steps taken per frame.
    :param solver_iters: Number of constraint solver iterations made per simulation step.
    :param verbose: If True, more details during the physics simulation are printed.
    :param use_volume_com: If True, the center of mass will be calculated by using the object volume.
                           This is more accurate than using the surface area (default), but requires a watertight mesh.
    """
    # Run simulation and remember poses before and after
    simulate_physics_v2(min_simulation_time, max_simulation_time, check_object_interval,
                                        object_stopped_location_threshold, object_stopped_rotation_threshold,
                                        substeps_per_frame, solver_iters, verbose, use_volume_com, frame_start)
    obj_poses_after_sim = _PhysicsSimulation_get_pose()

    # Make sure to remove the simulation cache as we are only interested in the final poses
    with bpy.context.temp_override(point_cache=bpy.context.scene.rigidbody_world.point_cache):
        bpy.ops.ptcache.free_bake_all()

    # Fix the pose of all objects to their pose at the end of the simulation (also revert origin shift)
    for obj in get_all_mesh_objects():
        if obj.has_rigidbody_enabled() and obj.blender_obj.rigid_body.enabled:
            # Skip objects that have parents with compound rigid body component
            has_compound_parent = obj.get_parent() is not None and isinstance(obj.get_parent(), MeshObject) \
                                  and obj.get_parent().get_rigidbody() is not None \
                                  and obj.get_parent().get_rigidbody().collision_shape == "COMPOUND"
            
            if obj.get_rigidbody().type == "ACTIVE" and not has_compound_parent:
                # Fix pose of object to the one it had at the end of the simulation
                obj.set_location(obj_poses_after_sim[obj.get_name()]['location'])
                obj.set_rotation_euler(obj_poses_after_sim[obj.get_name()]['rotation'])

    # Deactivate the simulation so it does not influence object positions
    bpy.context.scene.rigidbody_world.enabled = False
    bpy.context.view_layer.update()

def simulate_physics_v2(min_simulation_time: float = 4.0, max_simulation_time: float = 40.0,
                     check_object_interval: float = 2.0, object_stopped_location_threshold: float = 0.01,
                     object_stopped_rotation_threshold: float = 0.1, substeps_per_frame: int = 10,
                     solver_iters: int = 10, verbose: bool = False, use_volume_com: bool = False, frame_start:int=None) -> dict:
    """ Simulates the current scene.

    The simulation is run for at least `min_simulation_time` seconds and at a maximum `max_simulation_time` seconds.
    Every `check_object_interval` seconds, it is checked if the maximum object movement in the last second is below
    a given threshold. If that is the case, the simulation is stopped.

    The origin of all objects is set to their center of mass in this function which is necessary to achieve a realistic
    simulation in blender (see https://blender.stackexchange.com/questions/167488/physics-not-working-as-expected)
    Also the scale of each participating object is persisted as scale != 1 can make the simulation unstable.

    :param min_simulation_time: The minimum number of seconds to simulate.
    :param max_simulation_time: The maximum number of seconds to simulate.
    :param check_object_interval: The interval in seconds at which all objects should be checked if they are still
                                  moving. If all objects have stopped moving, then the simulation will be stopped.
    :param object_stopped_location_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param object_stopped_rotation_threshold: The maximum difference per second and per coordinate in the rotation
                                              Euler vector that is allowed such that an object is still recognized
                                              as 'stopped moving'.
    :param substeps_per_frame: Number of simulation steps taken per frame.
    :param solver_iters: Number of constraint solver iterations made per simulation step.
    :param verbose: If True, more details during the physics simulation are printed.
    :param use_volume_com: If True, the center of mass will be calculated by using the object volume.
                           This is more accurate than using the surface area (default), but requires a watertight mesh.
    :return: A dict containing for every active object the shift that was added to their origins.
    """
    ##########
    # Origin is not moved because now objects are already loaded with the center_of_mass option setted,
    # This has been done because this loop was slowing down a lot the simulation.
    ###########
    # Shift the origin of all objects to their center of mass to make the simulation more realistic
    # origin_shift = {}
    # for obj in get_all_mesh_objects():
    #     if obj.has_rigidbody_enabled() and obj.get_rigidbody().type == "ACTIVE":
    #         prev_origin = obj.get_origin()
    #         new_origin = obj.set_origin(mode="ORIGIN_CENTER_OF_VOLUME" if use_volume_com else "CENTER_OF_MASS")
    #         origin_shift[obj.get_name()] = new_origin - prev_origin

    #         # Persist mesh scaling as having a scale != 1 can make the simulation unstable
    #         obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    # Configure simulator
    bpy.context.scene.rigidbody_world.substeps_per_frame = substeps_per_frame
    bpy.context.scene.rigidbody_world.solver_iterations = solver_iters

    # Perform simulation
    do_simulation(min_simulation_time, max_simulation_time, check_object_interval,
                                     object_stopped_location_threshold, object_stopped_rotation_threshold,
                                     verbose, frame_start)

    # return origin_shift

@staticmethod
def do_simulation(min_simulation_time: float, max_simulation_time: float, check_object_interval: float,
                      object_stopped_location_threshold: float, object_stopped_rotation_threshold: float,
                      verbose: bool = False, frame_start:int=None):
        """ Perform the simulation.

        This method bakes the simulation for the configured number of iterations and returns all object positions
        at the last frame.

        :param min_simulation_time: The minimum number of seconds to simulate.
        :param max_simulation_time: The maximum number of seconds to simulate.
        :param check_object_interval: The interval in seconds at which all objects should be checked if they are still
                                      moving. If all objects have stopped moving, then the simulation will be stopped.
        :param object_stopped_location_threshold: The maximum difference per second and per coordinate in the rotation
                                                  Euler vector that is allowed such that an object is still recognized
                                                  as 'stopped moving'.
        :param object_stopped_rotation_threshold: The maximum difference per second and per coordinate in the rotation
                                                  Euler vector that is allowed such that an object is still recognized
                                                  as 'stopped moving'.
        :param verbose: If True, more details during the physics simulation are printed.
        """
        # Make sure the RigidBody world is active
        bpy.context.scene.rigidbody_world.enabled = True

        # Run simulation
        point_cache = bpy.context.scene.rigidbody_world.point_cache
        point_cache.frame_start = 1 if frame_start is None else frame_start

        if min_simulation_time >= max_simulation_time:
            raise Exception("max_simulation_iterations has to be bigger than min_simulation_iterations")

        # Run simulation starting from min to max in the configured steps
        for current_time in np.arange(min_simulation_time, max_simulation_time, check_object_interval):
            simulated_frames = _PhysicsSimulation.seconds_to_frames(current_time)
            print("Running simulation up to " + str(current_time) + " seconds (" + str(simulated_frames) + " frames)")

            # Simulate current interval
            point_cache.frame_end = point_cache.frame_start + simulated_frames
            with Utility.stdout_redirected(enabled=not verbose):
                with bpy.context.temp_override(point_cache=point_cache):
                    bpy.ops.ptcache.bake(bake=True)

            # Go to second last frame and get poses
            bpy.context.scene.frame_set(point_cache.frame_end - _PhysicsSimulation.seconds_to_frames(1))
            old_poses =_PhysicsSimulation_get_pose()

            # Go to last frame of simulation and get poses
            bpy.context.scene.frame_set(point_cache.frame_end)
            new_poses =_PhysicsSimulation_get_pose()

            # If objects have stopped moving between the last two frames, then stop here
            if _PhysicsSimulation.have_objects_stopped_moving(old_poses, new_poses, object_stopped_location_threshold,
                                                              object_stopped_rotation_threshold):
                print("Objects have stopped moving after " + str(current_time) + "  seconds (" + str(
                    simulated_frames) + " frames)")
                break
            if current_time + check_object_interval >= max_simulation_time:
                print("Stopping simulation as configured max_simulation_time has been reached")
            else:
                # Free bake (this will not completely remove the simulation cache, so further simulations can
                # reuse the already calculated frames)
                with bpy.context.temp_override(point_cache=point_cache):
                    bpy.ops.ptcache.free_bake()

@staticmethod
def _PhysicsSimulation_get_pose() -> dict:
    """ Returns position and rotation values of all objects in the scene with ACTIVE rigid_body type.

    :return: Dict of form {obj_name:{'location':[x, y, z], 'rotation':[x_rot, y_rot, z_rot]}}.
    """
    objects_poses = {}
    objects_with_physics = [obj for obj in get_all_blender_mesh_objects() if obj.rigid_body is not None and obj.rigid_body.enabled]

    for obj in objects_with_physics:
        if obj.rigid_body.type == 'ACTIVE' and obj.rigid_body.collision_shape == "COMPOUND":
            location = bpy.context.scene.objects[obj.name].matrix_world.translation.copy()
            rotation = mathutils.Vector(bpy.context.scene.objects[obj.name].matrix_world.to_euler())
            objects_poses.update({obj.name: {'location': location, 'rotation': rotation}})

    return objects_poses