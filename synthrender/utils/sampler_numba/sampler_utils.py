import numpy as np

from typing import Callable
from numba import njit

from synthrender.utils.sampler_numba import collisions_utils
from synthrender.utils.sampler_numba import camera_utils
from synthrender.utils.sampler_numba import CACHE_NUMBA

@njit(cache=CACHE_NUMBA)
def sample_pose_numba(func:Callable, fun_args:tuple[np.ndarray], new_seed,
                      sampled_models_bbox:np.ndarray,
                      camera_pose:np.ndarray, projection_matrix:np.ndarray,
                      placed_locations:np.ndarray, placed_radius:np.ndarray, placed_aabb_corners:np.ndarray, placed_moved_bboxs:np.ndarray, placed_sat_axis:np.ndarray, placed_amount:int,
                      frustum_check:bool, sphere_check:bool, AABB_check:bool, SAT_check:bool, 
                      max_attempts:int=1_000, verbose:bool=False):

    np.random.seed(new_seed)

    n_sampled = len(sampled_models_bbox)

    new_placed = np.zeros((n_sampled), dtype=np.bool_)
    new_placed_locations = np.empty((n_sampled, 3), dtype=np.float64)
    new_placed_rotations = np.empty((n_sampled, 3), dtype=np.float64)

    for id in range(n_sampled):
        bbox = sampled_models_bbox[id]
        success = True

        for attempt in range(max_attempts):
            success = True

            location, rotation = func(id, *fun_args)

            # Checks whether the sampled location falls within the camera frustum or not.
            if success and frustum_check and not camera_utils.is_point_in_frustum_numba(camera_pose, location, projection_matrix):
                success = False

            # A more advance camera frustum check! (Checks whether all the corners of the bbox of the model are inside of the camera frustum):
            if success and frustum_check and not camera_utils.is_bbox_in_frustum_numba(camera_pose, bbox, location, rotation, projection_matrix, min_corners=6):
                success = False

            # Checks whether the sampled location intersects with other boundingboxes (SAT and AABB tests).
            moved_bbox   = collisions_utils.calculate_moved_bbox_position_numba(bbox, location, rotation)
            aabb_corners = collisions_utils.compute_AABB_numba(moved_bbox)
            radius       = collisions_utils.get_radius_numba(moved_bbox)
            sat_axis     = collisions_utils.get_box_axes_numba(moved_bbox)

            if success and collisions_utils.check_bbox_intersect_numba(location, radius, aabb_corners, moved_bbox, sat_axis, 
                                                                        placed_locations, placed_radius, placed_aabb_corners, placed_moved_bboxs, placed_sat_axis, placed_amount, 
                                                                        sphere_check, AABB_check, SAT_check):
                success = False
            
            if success:
                break

        if success:
            # Adding new model to placed cache:
            placed_locations[placed_amount] = location
            placed_radius[placed_amount] = radius
            placed_aabb_corners[placed_amount] = aabb_corners
            placed_moved_bboxs[placed_amount] = moved_bbox
            placed_sat_axis[placed_amount] = sat_axis
            placed_amount += 1
            
            # Saving result:
            new_placed[id] = True
            new_placed_locations[id] = location
            new_placed_rotations[id] = rotation
        
        elif verbose:
            print(f"Warning: Could not place model after {max_attempts} attempts!", id)


    return new_placed, new_placed_locations, new_placed_rotations