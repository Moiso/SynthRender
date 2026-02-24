import numpy as np

from typing import Callable

from blenderproc.python.types.MeshObjectUtility import MeshObject

from synthrender.utils.sampler import collisions_utils
from synthrender.utils.sampler import camera_utils


def sample_pose(func:Callable, func_args:tuple,
                sampled_models:MeshObject, placed_models:list[collisions_utils.PlacedModel], camera_pose:np.ndarray, 
                frustum_check:bool=False, sphere_check:bool=True, AABB_check:bool=True, SAT_check:bool=False, 
                max_attempts:int=1_000, verbose=False):
    
    samples_placed:list[collisions_utils.PlacedModel] = []
    new_placed_models:list[MeshObject] = []

    # Placement logic with a max attempt count to prevent infinite loops
    for i, model in enumerate(sampled_models):
        for attempt in range(max_attempts):
            success = True

            location, rotation = func(model, *func_args)

            # Checks whether the sampled location falls within the camera frustum or not.
            if success and frustum_check and not camera_utils.camera_frustum.is_point_in_frustum(camera_pose, location):
                success = False

            # A more advance camera frustum check! (Checks whether all the corners of the bbox of the model are inside of the camera frustum):
            if success and frustum_check and not camera_utils.camera_frustum.is_bbox_in_frustum(camera_pose, model, location, rotation, min_corners=6):
                success = False

            # Checks whether the sampled location intersects with other boundingboxes (SAT and AABB tests).
            obj = collisions_utils.PlacedModel(model, location, rotation)
            if success and collisions_utils.CollisionChecker.check_bbox_intersect(obj, [*placed_models, *samples_placed], sphere_check, AABB_check, SAT_check):
                success = False

            if success:
                break

        if success:
            samples_placed.append(obj)
            new_placed_models.append(obj)

        elif verbose:
            print(f"Warning: Could not place a model after {max_attempts} attempts!")

    return new_placed_models
