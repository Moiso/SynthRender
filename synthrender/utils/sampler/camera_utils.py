import numpy as np

from numba import njit

from blenderproc.python.camera.CameraUtility import get_projection_matrix
from blenderproc.python.types.MeshObjectUtility import MeshObject

from synthrender.utils.sampler.collisions_utils import CollisionChecker

class camera_frustum:

    @staticmethod
    def is_point_in_frustum(cam_pos:np.ndarray, point:np.ndarray|list) -> bool:
        """ Checks if a given 3D point lies inside the camera frustum.
        (Modified version from Blenderproc. Takes a camera position as argument instead of expecting the camera to have it already set).

        :param point: The point, which should be checked
        :param clip_start: The distance between the camera pose and the near clipping plane.
        :param clip_end: The distance between the camera pose and the far clipping plane.
        :param frame: The frame number whose assigned camera pose should be used. If None is give, the current frame is used.
        :return: True, if the point lies inside the camera frustum, else False
        """
        camera_pose = cam_pos

        point4d = np.insert(np.array(point), 3, 1, axis=0)
        point4d = point4d @ np.linalg.inv(camera_pose).transpose()
        point4d /= point4d[3]
        projection_matrix = get_projection_matrix(None, None)
        point4d = point4d @ projection_matrix.transpose()
        point4d /= point4d[3]
        point4d = point4d[:3]

        return np.all([point4d < 1, -1 < point4d])
    
    @staticmethod
    def is_bbox_in_frustum(cam_pose:np.ndarray, model:MeshObject, location:np.ndarray, rotation:np.ndarray, min_corners:int=8):
        """ Checks if a given model bounding box is inside the camera frustum after applying a transformation_matrix.

        :param cam_pose: Pose of the camera matrix.
        :param model: Model from which to check the bounding box.
        :param location: The distance between the camera pose and the far clipping plane.
        :param frame: The frame number whose assigned camera pose should be used. If None is give, the current frame is used.
        :param min_corners: Minimum number of corners that should be inside of the frustum to be consider as valid (up to 8).
        :return: True, if all the bounding box corners lie inside the camera frustum, else False
        """
        
        corners_inside = 8
        bbox = np.array(model.blender_obj.bound_box) # bbox in local coordinates.
        
        # Move bounding box into new position in world coordinates.
        moved_bbox = CollisionChecker.calculate_moved_bbox_position(bbox, location, rotation)


        for point in moved_bbox:
            if not camera_frustum.is_point_in_frustum(cam_pose, point):
                corners_inside -= 1
            
            if corners_inside < min_corners:
                return False

        return True

