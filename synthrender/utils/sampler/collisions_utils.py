import numpy as np
import mathutils
from blenderproc.python.types.MeshObjectUtility import MeshObject

class PlacedModel:
    def __init__(self, model:MeshObject, location=None, rotation=None):
        self.model = model
        
        if location is not None and rotation is not None:
            self.set_up(location, rotation)

    def set_up(self, location, rotation):
        self.sat_proj_axis = {}
        self.location = location
        self.rotation = rotation

        self.model_bbox = np.array(self.model.blender_obj.bound_box)
        self.moved_bbox = CollisionChecker.calculate_moved_bbox_position(self.model_bbox, self.location, self.rotation)

        self.aabb_corners = AABBCollisionTest.compute_AABB(self.moved_bbox)
        center = np.mean(self.moved_bbox, axis=0)
        self.radius = np.linalg.norm(self.moved_bbox - center, axis=1).max()

        # Vectorized projection for SAT axes
        self.sat_axis = SATCollisionTest.get_box_axes(self.moved_bbox)
        for axis in self.sat_axis:
            min, max = SATCollisionTest.project_box_onto_axis(self.moved_bbox, axis)
            self.sat_proj_axis[tuple(axis)] = (min, max)


class CollisionChecker:

    @staticmethod
    def calculate_moved_bbox_position(model_bbox: np.ndarray, location: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        rot_matrix = np.array(mathutils.Euler(rotation, 'XYZ').to_matrix())
        return model_bbox @ rot_matrix.T + location
    
    @staticmethod
    def check_spacing_fast(obj_pos: np.ndarray, placed_objects: list[tuple[np.ndarray, np.ndarray]], min_distance: float, max_distance: float = np.inf) \
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
    def sphere_intersect(obj1:PlacedModel, obj2:PlacedModel):
        return (obj1.radius + obj2.radius) > np.linalg.norm(obj1.location-obj2.location)

    @staticmethod
    def check_bbox_intersect(obj1: PlacedModel, placed_models: list[PlacedModel], sphere_check=True, AABB_check=False, SAT_check=False) -> bool:
        """
        Checks whether the given object `obj1` intersects with any object in `placed_models`
        based on a series of collision tests.
        
        The tests are applied in order:
        1. Sphere intersection test (if enabled)
        2. AABB test (if enabled)
        3. SAT test (if enabled)
        
        An early exit occurs when an intersection is found.
        
        Parameters:
            obj1 (Placed_Models): The object to test against.
            placed_models (list[Placed_Models]): List of objects to test for intersection.
            sphere_check (bool): Whether to perform the sphere intersection test.
            AABB_check (bool): Whether to perform the AABB test (requires sphere_check to pass).
            SAT_check (bool): Whether to perform the SAT test (requires AABB_check to pass).
            
        Returns:
            bool: True if an intersection is found; otherwise, False.
        """

        if not any((sphere_check, AABB_check, SAT_check)):
            return False
        
        for obj2 in placed_models:
            # Check if the spheres intersect (Broad-Phase Detection):
            if sphere_check and not CollisionChecker.sphere_intersect(obj1, obj2):
                continue

            # if spheres intersect, use AABB test (Broad-Phase Detection):
            elif AABB_check and not AABBCollisionTest.AABB_intersect(obj1, obj2):
                continue

            # If AABBs overlap, use the SAT test (Narrow-Phase Detection):
            elif SAT_check and not SATCollisionTest.SAT_intersect(obj1, obj2):
                continue
            
            return True # An intersection was found.

        return False # No intersection detected.

## Tests used for checking bbox intersections:

class AABBCollisionTest:
    @staticmethod
    def compute_AABB(corners: np.ndarray) -> np.ndarray:
        """
        Compute the Axis-Aligned Bounding Box (AABB) from the 8 corners.
        
        :param corners: A NumPy array of shape (8,3)
        :return: Two 3-element arrays: (min_corner, max_corner)
        """
        return np.array([corners.min(axis=0), corners.max(axis=0)])
    
    @staticmethod
    def AABB_intersect(obj1: PlacedModel, obj2: PlacedModel) -> bool:
        """
        Check if two AABBs intersect.
        
        :param obj1: PlacedModel for first AABB.
        :param obj2: PlacedModel for second AABB.
        :return: True if they intersect; False otherwise.
        """
        min1, max1 = obj1.aabb_corners
        min2, max2 = obj2.aabb_corners
        
        # For each axis, check for separation.
        for i in range(3):
            if max1[i] < min2[i] or max2[i] < min1[i]:
                return False
        return True


class SATCollisionTest:
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Return the normalized vector of v."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return v / norm

    @staticmethod
    def get_box_axes(corners: np.ndarray) -> list:
        """
        Derives the 3 local axes of the bounding box based on its 8 corners.
        
        Assumes a consistent vertex ordering, e.g.:
          - axis1: from corner 0 to corner 1
          - axis2: from corner 0 to corner 2
          - axis3: from corner 0 to corner 4
          
        :param corners: A NumPy array of shape (8, 3) with the box's corners.
        :return: A list of three unit vectors.
        """
        axis1 = SATCollisionTest.normalize(corners[1] - corners[0])
        axis2 = SATCollisionTest.normalize(corners[2] - corners[0])
        axis3 = SATCollisionTest.normalize(corners[4] - corners[0])
        return np.array([axis1, axis2, axis3])

    @staticmethod
    def project_box_onto_axis(corners: np.ndarray, axis: np.ndarray) -> tuple:
        """
        Projects all corners of the box onto the given axis.
        
        :param corners: A NumPy array of shape (8, 3).
        :param axis: A 3-element unit vector.
        :return: A tuple (min_projection, max_projection).
        """
        dots = np.dot(corners, axis)
        return np.min(dots), np.max(dots)

    @staticmethod
    def SAT_intersect(obj1: PlacedModel, obj2: PlacedModel) -> bool:
        """
        Checks whether two oriented bounding boxes (OBBs) intersect using the
        Separating Axis Theorem (SAT).
        
        :param obj1: A PlacedModel object to check collisions.
        :param obj2: A PlacedModel object to check collisions.
        :return bool: True if the boxes intersect; False otherwise.
        """
        axes1 = SATCollisionTest.get_box_axes(obj1.moved_bbox)
        axes2 = SATCollisionTest.get_box_axes(obj2.moved_bbox)
        candidate_axes = list(axes1) + list(axes2)

        # Add axes from cross products of each axis from box1 and box2.
        for a in axes1:
            for b in axes2:
                cross = np.cross(a, b)
                if np.linalg.norm(cross) > 1e-6:
                    candidate_axes.append(SATCollisionTest.normalize(cross))
                    
        # Check each candidate axis.
        for axis in candidate_axes:
            min1, max1 = SATCollisionTest.project_box_onto_axis(obj1.moved_bbox, axis)
            min2, max2 = SATCollisionTest.project_box_onto_axis(obj2.moved_bbox, axis)
            if max1 < min2 or max2 < min1:
                return False  # Found a separating axis.
            
        return True  # No separating axis found; boxes intersect.