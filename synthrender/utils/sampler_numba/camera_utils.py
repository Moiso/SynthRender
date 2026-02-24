import numpy as np

from numba import njit

from synthrender.utils.sampler_numba.collisions_utils import calculate_moved_bbox_position_numba
from synthrender.utils.sampler_numba import CACHE_NUMBA

@njit(cache=CACHE_NUMBA)
def is_point_in_frustum_numba(camera_pose: np.ndarray, point: np.ndarray, projection_matrix: np.ndarray) -> bool:
    # Create the 4D point by concatenating a one-element array.
    point4d = np.concatenate((point, np.array([1.0])))
    
    # Apply the inverse camera pose transform.
    inv_cam_transpose = np.linalg.inv(camera_pose).T
    point4d = point4d @ inv_cam_transpose
    point4d = point4d / point4d[3]
    
    # Apply the projection matrix.
    point4d = point4d @ projection_matrix.T
    point4d = point4d / point4d[3]
    
    # Use the resulting coordinates to determine if the point is within the frustum.
    transformed = point4d[:3]
    return (transformed[0] < 1 and -1 < transformed[0] and 
            transformed[1] < 1 and -1 < transformed[1] and 
            transformed[2] < 1 and -1 < transformed[2])

@njit(cache=CACHE_NUMBA)
def is_bbox_in_frustum_numba(cam_pose:np.ndarray, bbox:np.ndarray, location:np.ndarray, rotation:np.ndarray, projection_matrix:np.ndarray, min_corners:int=8) -> bool:
    corners_inside:int = 8
    
    # Move bounding box into new position in world coordinates.
    moved_bbox = calculate_moved_bbox_position_numba(bbox, location, rotation)
    n = len(moved_bbox)

    for i in range(n):
        point = moved_bbox[i]
        if not is_point_in_frustum_numba(cam_pose, point, projection_matrix):
            corners_inside -= 1
        
        if corners_inside < min_corners:
            return False

    return True