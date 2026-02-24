import numpy as np

from numba import njit

from blenderproc.python.types.MeshObjectUtility import MeshObject
from synthrender.utils.sampler.collisions_utils import PlacedModel
from synthrender.utils.sampler_numba import CACHE_NUMBA

class PlacedModelsNumba:
    def __init__(self, max_models):
            """
            Pre-allocate storage for a fixed number of models.

            Parameters:
                max_models (int): Maximum number of models that will be added.
            """
            self.max_models = max_models
            self.count = 0  # Counter to track how many models have been added
            
            # Pre-allocate arrays.
            self.location = np.empty((max_models, 3), dtype=np.float64)
            self.radius = np.empty((max_models,), dtype=np.float64)
            self.aabb_corners = np.empty((max_models,) + (2,3), dtype=np.float64)
            self.moved_bbox = np.empty((max_models,) + (8, 3), dtype=np.float64)
            self.models_bbox = np.empty((max_models,) + (8, 3), dtype=np.float64)
            self.sat_axis = np.empty((max_models,) + (3,3), dtype=np.float64)

    def add_model(self, model:MeshObject, location, rotation):
        self.add_obj(PlacedModel(model, location, rotation))

    def add_obj(self, obj:PlacedModel):
        """
        Add an object to the pre-allocated arrays.

        Parameters:
            obj: An instance of PlacedModel that contains the data.
                Assumes:
                - obj.location is a (location_dim,) array-like.
                - obj.radius is a scalar.
                - obj.aabb_corners is an array-like with shape matching aabb_shape.
                - obj.moved_bbox is an array-like with shape matching moved_bbox_shape.
                - obj.sat_axis is an array-like with shape matching sat_axis_shape.
        """
        if self.count >= self.max_models:
            raise IndexError("Cannot add more objects than the allocated maximum.")

        # Set the values in the pre-allocated arrays.
        self.location[self.count] = np.array(obj.location)
        self.radius[self.count] = obj.radius
        self.aabb_corners[self.count] = np.array(obj.aabb_corners)
        self.moved_bbox[self.count] = np.array(obj.moved_bbox)
        self.models_bbox[self.count] = np.array(obj.model_bbox)
        self.sat_axis[self.count] = np.array(obj.sat_axis)
        self.count += 1

    def get_all_list(self) -> list:
        """
        :return: [self.location, self.radius, self.aabb_corners, self.moved_bbox, self.sat_axis, self.count]
        """
        return [self.location, self.radius, self.aabb_corners, self.moved_bbox, self.sat_axis, self.count]
        

@njit(cache=CACHE_NUMBA)
def euler_to_rotation_matrix_xyz_numba(rotation: np.ndarray) -> np.ndarray:
    """
    Computes the 3x3 rotation matrix from Euler angles in XYZ order.
    The rotations are applied as: R = Rz @ Ry @ Rx.
    """
    rx, ry, rz = rotation
    
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)
    
    # Compute the rotation matrix for Euler angles in XYZ order.
    # Using the convention R = Rz @ Ry @ Rx.
    R = np.empty((3, 3), dtype=np.float64)
    
    R[0, 0] = cz * cy
    R[0, 1] = -cx*sz + sx*sy*cz
    R[0, 2] = sx*sz + cx*sy*cz

    R[1, 0] = cy*sz
    R[1, 1] = cx*cz + sx*sy*sz
    R[1, 2] = -sx*cz + cx*sy*sz

    R[2, 0] = -sy               # -sy
    R[2, 1] = sx*cy             # -cy * sx correct
    R[2, 2] = cx*cy             # cy * cx  correct
    
    return R

@njit(cache=CACHE_NUMBA)
def calculate_moved_bbox_position_numba(model_bbox: np.ndarray, location: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # Compute the rotation matrix from Euler angles
    rot_matrix   = euler_to_rotation_matrix_xyz_numba(rotation)
    rotated_bbox = np.dot(model_bbox, rot_matrix.T) # Apply the rotation.
    moved_bbox   = rotated_bbox + location # Translate the rotated bounding box.

    return moved_bbox

@njit(cache=CACHE_NUMBA)
def get_radius_numba(moved_bbox: np.ndarray) -> float:
    """
    Computes the radius of a set of points given by moved_bbox.
    It calculates the center as the mean of the points along axis 0,
    then computes each point's Euclidean distance to the center,
    and returns the maximum distance (radius).

    Parameters:
        moved_bbox: A 2D array of shape (N, d), where N is the number
                    of points and d is the dimension (usually 3).

    Returns:
        The maximum Euclidean distance from the center to the points.
    """
    n, d = moved_bbox.shape

    # Compute mean (center) along axis 0 manually.
    center = np.zeros(d, dtype=moved_bbox.dtype)
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += moved_bbox[i, j]
        center[j] = s / n

    # Compute the maximum distance from the center.
    max_radius = 0.0
    for i in range(n):
        s = 0.0
        for j in range(d):
            diff = moved_bbox[i, j] - center[j]
            s += diff * diff
        # Calculate the Euclidean norm for row i.
        norm = np.sqrt(s)
        if norm > max_radius:
            max_radius = norm

    return max_radius

@njit(cache=CACHE_NUMBA)
def get_box_axes_numba(corners: np.ndarray) -> np.ndarray:
    """
    Derives the 3 local axes of a bounding box from its 8 corner coordinates.
    
    Assumes a consistent vertex ordering:
      - axis1: from corner 0 to corner 1
      - axis2: from corner 0 to corner 2
      - axis3: from corner 0 to corner 4
    
    The axes are normalized (if the computed length is too small, the unmodified vector is returned).
    
    Parameters:
       corners: A (8, 3) numpy array of the box's corners.
    
    Returns:
       A (3, 3) numpy array containing the three unit vectors.
    """
    # Prepare output array: three axes, each of dimension 3
    axes = np.empty((3, 3), dtype=corners.dtype)
    
    # The indices for computing the axes based on the specified vertex ordering.
    indices = [1, 2, 4]
    
    for i in range(3):
        # Compute the vector from corner 0 to the specified corner.
        v0 = corners[indices[i]] - corners[0]
        
        # Calculate the Euclidean norm (manual computation for Numba)
        norm_sq = 0.0
        for j in range(3):
            norm_sq += v0[j] * v0[j]
        norm_val = np.sqrt(norm_sq)
        
        # Normalize if the norm is sufficiently large; otherwise, just use the original vector.
        if norm_val < 1e-8:
            for j in range(3):
                axes[i, j] = v0[j]
        else:
            for j in range(3):
                axes[i, j] = v0[j] / norm_val
                
    return axes

@njit(cache=CACHE_NUMBA)
def compute_AABB_numba(moved_bbox: np.ndarray) -> np.ndarray:
    """
    Compute the axis-aligned bounding box (AABB) for an array of points.
    Assumes moved_bbox is of shape (N, d) and returns a (2, d) array,
    where the first row is the minimum coordinates and the second row is the maximum.
    """
    N, d = moved_bbox.shape
    # Initialize min and max with the first row.
    min_coords = moved_bbox[0].copy()
    max_coords = moved_bbox[0].copy()
    
    for i in range(1, N):
        for j in range(d):
            if moved_bbox[i, j] < min_coords[j]:
                min_coords[j] = moved_bbox[i, j]
            if moved_bbox[i, j] > max_coords[j]:
                max_coords[j] = moved_bbox[i, j]
                
    # Create an array with the min and max coordinates.
    aabb = np.empty((2, d), dtype=moved_bbox.dtype)
    aabb[0] = min_coords
    aabb[1] = max_coords
    return aabb

@njit(cache=CACHE_NUMBA)
def sphere_intersect_numba(location1, radius1, location2, radius2):
    return (radius1 + radius2) > np.linalg.norm(location1-location2)

@njit(cache=CACHE_NUMBA)
def AABB_intersect_numba(min1, max1, min2, max2):
    """
    Check if two AABBs intersect.
    
    :param min1: From aabb1
    :param max1: From aabb1
    :param min2: From aabb2
    :param max2: From aabb2
    :return bool: True if they intersect; False otherwise.
    """
    for i in range(3):
        if max1[i] < min2[i] or max2[i] < min1[i]:
            return False
    return True

@njit(cache=CACHE_NUMBA)
def SAT_intersect_numba(corners1, corners2, axes1, axes2):
    # Allocate space for up to 15 candidate axes (3 + 3 + 9)
    candidate_axes = np.empty((15, 3))
    axis_count = 0

    # Add the 3 axes from each box
    for i in range(3):
        candidate_axes[axis_count] = axes1[i]
        axis_count += 1
        candidate_axes[axis_count] = axes2[i]
        axis_count += 1

    # Add valid cross product axes (normalized)
    for i in range(3):
        for j in range(3):
            cross = np.cross(axes1[i], axes2[j])
            norm = np.linalg.norm(cross)
            if norm > 1e-6:
                candidate_axes[axis_count] = cross / norm
                axis_count += 1

    # Prepare projection ranges
    min1 = np.empty(axis_count)
    max1 = np.empty(axis_count)
    min2 = np.empty(axis_count)
    max2 = np.empty(axis_count)

    # Project both boxes onto each axis
    for k in range(axis_count):
        axis = candidate_axes[k]
        projections1 = corners1 @ axis
        projections2 = corners2 @ axis
        min1[k] = projections1.min()
        max1[k] = projections1.max()
        min2[k] = projections2.min()
        max2[k] = projections2.max()

    # Check for separating axis
    for k in range(axis_count):
        if max1[k] < min2[k] or max2[k] < min1[k]:
            return False  # Found separating axis

    return True  # No separating axis found → boxes intersect

@njit(cache=CACHE_NUMBA)
def check_bbox_intersect_numba(obj1_location, obj1_radius, obj1_aabb_corners, obj1_moved_bbox, obj1_sat_axis, 
                        placed_objs_location, placed_objs_radius, placed_objs_aabb_corners, placed_objs_moved_bbox, placed_objs_sat_axis, added_count,
                        sphere_check:bool, AABB_check:bool, SAT_check:bool):
    
    for i in range(added_count):
        obj2_location = placed_objs_location[i]
        obj2_radius = placed_objs_radius[i]
        obj2_aabb_corners = placed_objs_aabb_corners[i]
        obj2_moved_bbox = placed_objs_moved_bbox[i]
        obj2_sat_axis = placed_objs_sat_axis[i]

        # Check if the spheres intersect (Broad-Phase Detection):
        if sphere_check and not sphere_intersect_numba(obj1_location, obj1_radius, obj2_location, obj2_radius):
            continue  # No collision

        # if spheres intersect, use AABB test (Broad-Phase Detection):
        elif AABB_check and not AABB_intersect_numba(obj1_aabb_corners[0], obj1_aabb_corners[1], obj2_aabb_corners[0], obj2_aabb_corners[1]):
            continue  # No collision

        # If AABBs overlap, use the SAT test (Narrow-Phase Detection):
        elif SAT_check and not SAT_intersect_numba(obj1_moved_bbox, obj2_moved_bbox, obj1_sat_axis, obj2_sat_axis):
            continue  # No collision

        return True # An intersection was found.
    
    return False # No intersection detected.
