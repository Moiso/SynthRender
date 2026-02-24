import blenderproc as bproc
import numpy as np
import random
import colorsys

from numba import njit


from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.types.LightUtility import Light
from blenderproc.python.camera import CameraUtility

from synthrender.utils import sampler, sampler_numba
from synthrender.utils.sampler_numba import CACHE_NUMBA


############################################################
# FUNCTIONS USED FOR RANDOMIZING SIMULATION PARAMETERS     #
############################################################

def randomize_backg(config:dict):
    """
    Randomizes the background light strength based on the provided configuration.
    Parameters:
        config (dict): Configuration dictionary containing world settings.
    Returns:
        float: Randomized strength value for the environmental light, ranging from 0.0 to 1.0 in 0.1 increments.
    """

    # Randomly set up the enviromental light:
    back_strength = np.array(config["world"]["background_light_strength"])*100
    back_strength = random.randint(*back_strength)/100.0 # Strength from 0.0 to 1.0 with 0.1 steps

    return back_strength

def randomize_plane(planes:list[MeshObject]=None):
    """
    Randomly chooses a plane from a list of available planes. If no list is passed, returns None.

    Parameters:
        planes (planes:list[MeshObject]): List of planes objects to select from.

    Returns:
        plane (MeshObject): A randomly selected plane (random.choice(planes))
    """

    if not planes: return None

    return random.choice(planes)

def randomize_lights(config:dict, lights:list[Light], n_frames:int=None, frame:int=None):
    """
    Randomizes energy and color for each light based on the provided configuration.
    If enabled in the config, it will calculate the energy with an exponential curve from 0 to n_frames.

    Parameters:
        config (dict): Configuration dictionary with light settings.
        lights (list[Light]): List of light objects in the scene.
        n_frames (int, optional): Total number of frames to render; used if exponential lights option is enabled.
        frame (int, optional): Current frame number; used if exponential lights option is enabled.

    Returns:
        tuple: A tuple containing:
            - lights_energy (list[float]): List with the energy value for each light.
            - color_rgb (tuple[float, float, float]): Randomized RGB color value.
    """

    hue_value = random.randint(0,10)/10                             # values from 0.0 to 1.0 with steps of 0.1 (11 colours + white)
    sat_value = random.choice([0.5, 0.8]) if hue_value < 1 else 0     # 0.5 or 1 if not white. else 0 (0.8 better colors)
    color_rgb = colorsys.hsv_to_rgb(hue_value, sat_value, 1)
    distance  = np.linalg.norm(lights[0].get_location())

    min_light, max_light = config["lights"]["light_intensity"]
    intensity = random.randint(min_light, max_light)

    if config['lights']['exponential_lights'] and n_frames is not None and frame is not None:
        exp = config['lights']['exponential_factor']
        fixed_nframes = max(n_frames, 2)
        get_intensity = lambda x: pow((x/(fixed_nframes-1)), exp)*max_light
        intensity = round(get_intensity(frame))

    state = random.getstate()
    light1_intensity = random.randint(0, intensity)
    light2_intensity = random.randint(0, intensity - light1_intensity)
    light3_intensity = intensity - light1_intensity - light2_intensity
    random.setstate(state)

    lights_energy = [intensity * distance**2 for intensity in [light1_intensity, light2_intensity, light3_intensity]]

    return lights_energy, color_rgb

def randomize_empty(config:dict):
    """
    Randomly samples a position and rotation for an empty object based on configuration parameters.

    Parameters:
        config (dict): Loaded configuration containing parameters for the empty object, including the position shell radius, elevation range, and center.
    Returns:
        tuple: A tuple containing:
            - empty_pos: The sampled location for the empty object.
            - empty_rot: The sampled rotation in Euler angles.
    """

    rad_min, rad_max = config["empty_object"]["pos_shell_radius"]
    elev_min, elev_max = config["empty_object"]["pos_shell_elevation"]
    center = config["empty_object"]["center"]

    if elev_min == elev_max:
        elev_min = elev_max - 0.0001

    empty_pos = bproc.sampler.shell(center=center, radius_min=rad_min, radius_max=rad_max, elevation_min=elev_min, elevation_max=elev_max)
    empty_rot = bproc.sampler.uniformSO3(around_x=False, around_y=False, around_z=True)

    return empty_pos, empty_rot

def randomize_train(config:dict, train_models:list[MeshObject], center_pos:list[float], fake_models:list[MeshObject]=None, camera_pose=None, already_placed:list[list[MeshObject, tuple[np.ndarray, np.ndarray]]]=None):
    """
    Randomly samples and places a set of train models, along with optional fake models, around a specified center position.

    Parameters:
        config (dict): Loaded configuration containing placement parameters for train and fake models.
        train_models (list[MeshObject]): List of train models available for placement.
        center_pos: Position around which the models will be spawned.
        fake_models (list, optional): List of fake train models available for placement.
        camera_pose (optional): Camera position used to check if a placed model is within the camera frustum.
        placed_models (optional): List of already placed models.

    Returns:
        tuple: A tuple containing:
            - placed_models (list[MeshObject]): Models that were successfully placed.
            - placed_positions (list[tuple]): Corresponding positions (location, rotation) for each placed model.
    """
    if not train_models and not fake_models: return [], []

    fake_models = fake_models or []
    train_models = train_models or []

    # Loading config parameters:
    reals_config        = config['models']['trains']
    fakes_config        = config['fake_models']['trains']
    
    pos_min, pos_max    = reals_config["pos_min"], reals_config["pos_max"]
    rot_min, rot_max    = reals_config["rot_min"], reals_config["rot_max"]
    dynamic_origin      = reals_config["dynamic_origin"]
    static_origin       = reals_config["static_origin"]
    train_sample_size   = reals_config["sample_size"]
    fakes_sample_size   = fakes_config["sample_size"]

    # Checking range of sample_size. If it is -1 or the sample is out of range, set the len of train_models.
    for i, _ in enumerate(train_sample_size):
        if train_sample_size[i] == -1 or train_sample_size[i] > len(train_models): # undefined or too big.
            train_sample_size[i] = len(train_models)

    for i, _ in enumerate(fakes_sample_size):
        if fakes_sample_size[i] == -1 or fakes_sample_size[i] > len(fake_models): # undefined or too big.
            fakes_sample_size[i] = len(fake_models)

    # Set up variables:
    already_placed  = already_placed or []
    origin          = center_pos if dynamic_origin else static_origin
    frustum_check   = camera_pose is not None
    trains_sample   = random.sample(train_models, random.randint(*train_sample_size))
    fakes_sample    = random.sample(fake_models, random.randint(*fakes_sample_size))
    sampled_models  = trains_sample + fakes_sample
    placed_models:list[sampler.PlacedModel] = []
    new_placed_models:list[MeshObject] = []
    new_placed_poses:list[tuple[np.ndarray, np.ndarray]] = []

    for placed, pose in already_placed:
        obj = sampler.PlacedModel(placed, pose[0], pose[1])
        placed_models.append(obj)

    # Placing objects in the scene.
    new_placed:list[sampler.PlacedModel] = sampler.sample_pose(randomize_pose_train, (pos_min, pos_max, rot_min, rot_max, origin),
                                                                                sampled_models, placed_models, camera_pose, 
                                                                                frustum_check, sphere_check=True, AABB_check=True, SAT_check=True, max_attempts=1_000)
    for obj in new_placed:
        new_placed_models.append(obj.model)
        new_placed_poses.append((obj.location, obj.rotation))


    return new_placed_models, new_placed_poses

def randomize_distr(config:dict, distr_models:list[MeshObject], fake_models:list[MeshObject] = None, already_placed:list[list[MeshObject, tuple[np.ndarray, np.ndarray]]]=None):
    """
    Samples and places a random selection of distractor models within a defined area.

    Parameters:
        config (dict): Loaded configuration containing distractor placement parameters.
        distr_models (list[MeshObject]): List of available distractor models.
        already_placed (list[list[MeshObject, tuple[np.ndarray, np.ndarray]]], optional): List of already placed models and their pose.

    Returns:
        tuple: A tuple containing:
            - sample_distr (list[MeshObject]): The sampled distractor models.
            - placed_distr (list[tuple]): The poses (location and rotation) where the models were successfully placed.
    """
    if not distr_models and not fake_models: return [], []

    fake_models = fake_models or []
    distr_models = distr_models or []

    # Loading config parameters:
    reals_config        = config['models']['distractors']
    fakes_config        = config['fake_models']['distractors']

    # Loading config parameters:
    pos_min, pos_max    = reals_config["pos_min"], reals_config["pos_max"]
    rot_min, rot_max    = reals_config["rot_min"], reals_config["rot_max"]
    distr_sample_size   = reals_config["sample_size"]
    fakes_sample_size   = fakes_config["sample_size"]

    # Checking range of sample_size. If it is -1 or the sample is out of range, set the len of train_models.
    for i, _ in enumerate(distr_sample_size):
        if distr_sample_size[i] == -1 or distr_sample_size[i] > len(distr_models): # undefined or too big.
            distr_sample_size[i] = len(distr_models)

    for i, _ in enumerate(fakes_sample_size):
        if fakes_sample_size[i] == -1 or fakes_sample_size[i] > len(fake_models): # undefined or too big.
            fakes_sample_size[i] = len(fake_models)

    # Set up variables:
    already_placed  = already_placed or []
    distr_sample    = random.sample(distr_models, random.randint(*distr_sample_size))
    fakes_sample    = random.sample(fake_models, random.randint(*fakes_sample_size))
    sampled_models  = distr_sample + fakes_sample
    placed_models:list[sampler.PlacedModel] = []
    new_placed_models:list[MeshObject] = []
    new_placed_poses:list[tuple[np.ndarray, np.ndarray]] = []

    for placed, pose in already_placed:
        obj = sampler.PlacedModel(placed, pose[0], pose[1])
        placed_models.append(obj)

    # Placing objects in the scene.
    new_placed:list[sampler.PlacedModel] = sampler.sample_pose(randomize_pose_distr, (pos_min, pos_max, rot_min, rot_max),
                                                                sampled_models, placed_models, camera_pose=None, 
                                                                frustum_check=False, AABB_check=True, SAT_check=True, max_attempts=1_000)
    
    for obj in new_placed:
        new_placed_models.append(obj.model)
        new_placed_poses.append((obj.location, obj.rotation))

    
    return new_placed_models, new_placed_poses

def randomize_train_numba(config:dict, train_models:list[MeshObject], center_pos:list[float], fake_models:list=None, camera_pose=None, already_placed:list[list[MeshObject, tuple[np.ndarray, np.ndarray]]]=None):
    """
    Randomly samples and places a set of train models, along with optional fake models, around a specified center position.

    Parameters:
        config (dict): Loaded configuration containing placement parameters for train and fake models.
        train_models (list[MeshObject]): List of train models available for placement.
        center_pos: Position around which the models will be spawned.
        fake_train_models (list, optional): List of fake train models available for placement.
        camera_pose (optional): Camera position used to check if a placed model is within the camera frustum.

    Returns:
        tuple: A tuple containing:
            - placed_models (list[MeshObject]): Models that were successfully placed.
            - placed_positions (list[tuple]): Corresponding positions (location, rotation) for each placed model.
    """
    if not train_models and not fake_models: return [], []

    fake_models = fake_models or []
    train_models = train_models or []

    # Loading config parameters:
    reals_config        = config['models']['trains']
    fakes_config        = config['fake_models']['trains']
    
    pos_min, pos_max    = reals_config["pos_min"], reals_config["pos_max"]
    rot_min, rot_max    = reals_config["rot_min"], reals_config["rot_max"]
    dynamic_origin      = reals_config["dynamic_origin"]
    static_origin       = reals_config["static_origin"]
    train_sample_size   = reals_config["sample_size"]
    fakes_sample_size   = fakes_config["sample_size"]

    # Checking range of sample_size. If it is -1 or the sample is out of range, set the len of train_models.
    for i, _ in enumerate(train_sample_size):
        if train_sample_size[i] == -1 or train_sample_size[i] > len(train_models): # undefined or too big.
            train_sample_size[i] = len(train_models)
    
        if fakes_sample_size[i] == -1 or fakes_sample_size[i] > len(fake_models): # undefined or too big.
            fakes_sample_size[i] = len(fake_models)

    # Set up variables:
    already_placed  = already_placed or []
    origin          = center_pos if dynamic_origin else static_origin
    trains_sample   = random.sample(train_models, random.randint(*train_sample_size))
    fakes_sample    = random.sample(fake_models, random.randint(*fakes_sample_size))
    sampled_models  = trains_sample + fakes_sample
    frustum_check   = camera_pose is not None
    new_placed_models:list[MeshObject] = []
    new_placed_poses:list[tuple[np.ndarray, np.ndarray]] = []

    # Set up variables for numba:
    pos_min, pos_max = np.asarray(pos_min), np.asarray(pos_max)
    rot_min, rot_max = np.asarray(rot_min), np.asarray(rot_max)
    projection_matrix = CameraUtility.get_projection_matrix(None, None)
    camera_pose = camera_pose if camera_pose is not None else np.zeros((4,4), dtype=np.float64)
    seed = np.random.randint(100_000)
    sampled_models_bbox = np.empty((len(sampled_models),) + (8, 3), dtype=np.float64)
    sampled_models_offsets = np.empty((len(sampled_models),) + (3,), dtype=np.float64)

    # Get data for numba of the sampled models:
    for i, model in enumerate(sampled_models):
        sampled_models_bbox[i]    = np.array(model.blender_obj.bound_box)
        sampled_models_offsets[i] = np.array(origin)

    # Get data for numba of the already placed models:
    already_placed_numba = sampler_numba.PlacedModelsNumba(max_models=len(already_placed) + len(sampled_models))
    for placed, pose in already_placed:
        already_placed_numba.add_model(placed, pose[0], pose[1])

    placed, locations, rotations = sampler_numba.sample_pose_numba(randomize_pose_train_numba, (sampled_models_offsets, pos_min, pos_max, rot_min, rot_max), seed,
                                                                sampled_models_bbox,
                                                                camera_pose, projection_matrix,
                                                                *already_placed_numba.get_all_list(),
                                                                frustum_check=frustum_check, sphere_check=True, AABB_check=True, SAT_check=True, max_attempts=1_000)

    for i, model in enumerate(sampled_models):
        if placed[i]:
            new_placed_models.append(model)
            new_placed_poses.append((locations[i], rotations[i]))
    
    return new_placed_models, new_placed_poses

def randomize_distr_numba(config:dict, distr_models:list[MeshObject], fake_models:list[MeshObject]=None, already_placed:list[list[MeshObject, tuple[np.ndarray, np.ndarray]]]=None):
    """
    Samples and places a random selection of distractor models within a defined area.

    Parameters:
        config (dict): Loaded configuration containing distractor placement parameters.
        distractors (list[MeshObject]): List of available distractor models.
        already_placed_pos (list, optional): List of already placed positions to maintain spacing.

    Returns:
        tuple: A tuple containing:
            - sample_distr (list[MeshObject]): The sampled distractor models.
            - placed_distr (list[tuple]): The positions (location and rotation) where the models were successfully placed.
    """
    if not distr_models and not fake_models: return [], []

    fake_models = fake_models or []
    distr_models = distr_models or []

    # Loading config parameters:
    reals_config        = config['models']['distractors']
    fakes_config        = config['fake_models']['distractors']

    # Loading config parameters:
    pos_min, pos_max    = reals_config["pos_min"], reals_config["pos_max"]
    rot_min, rot_max    = reals_config["rot_min"], reals_config["rot_max"]
    distr_sample_size   = reals_config["sample_size"]
    fakes_sample_size   = fakes_config["sample_size"]

    # Checking range of sample_size. If it is -1 or the sample is out of range, set the len of train_models.
    for i, _ in enumerate(distr_sample_size):
        if distr_sample_size[i] == -1 or distr_sample_size[i] > len(distr_models): # undefined or too big.
            distr_sample_size[i] = len(distr_models)

        if fakes_sample_size[i] == -1 or fakes_sample_size[i] > len(fake_models): # undefined or too big.
            fakes_sample_size[i] = len(fake_models)

    # Set up variables:
    already_placed = already_placed or [] # If you remove placed models it works fine!
    new_placed_models:list[sampler.PlacedModel] = []
    new_placed_poses:list[tuple[np.ndarray, np.ndarray]] = []
    distr_sample    = random.sample(distr_models, random.randint(*distr_sample_size))
    fakes_sample    = random.sample(fake_models, random.randint(*fakes_sample_size))
    sampled_models  = distr_sample + fakes_sample

    # Set up variables for numba:
    sampled_models_bbox = np.empty((len(sampled_models),) + (8, 3), dtype=np.float64)
    sampled_models_offsets = np.empty((len(sampled_models),) + (3,), dtype=np.float64)
    projection_matrix = np.zeros((4,4))
    camera_pose = np.zeros((4,4), dtype=np.float64)
    pos_min, pos_max = np.asarray(pos_min), np.asarray(pos_max)
    rot_min, rot_max = np.asarray(rot_min), np.asarray(rot_max)

    # Get data for numba of the sampled models:
    for i, model in enumerate(sampled_models):
        sampled_models_bbox[i] = np.array(model.blender_obj.bound_box)

        dist_orig_to_bottom = np.abs(sampled_models_bbox[i][0][2]) + 0.001
        sampled_models_offsets[i] = np.array([0, 0, dist_orig_to_bottom])

    # Get data for numba of the already placed models:
    already_placed_numba = sampler_numba.PlacedModelsNumba(max_models=len(already_placed) + len(sampled_models))
    for placed, pose in already_placed:
        already_placed_numba.add_model(placed, pose[0], pose[1])

    seed = np.random.randint(100_000)
    placed, locations, rotations = sampler_numba.sample_pose_numba(randomize_pose_distr_numba, (sampled_models_offsets, pos_min, pos_max, rot_min, rot_max), seed,
                                                                    sampled_models_bbox,
                                                                    camera_pose, projection_matrix,
                                                                    *already_placed_numba.get_all_list(),
                                                                    frustum_check=False, sphere_check=True, AABB_check=True, SAT_check=True, max_attempts=1_000)

    for i, model in enumerate(sampled_models):
        if placed[i]:
            new_placed_models.append(model)
            new_placed_poses.append((locations[i], rotations[i]))
    
    return new_placed_models, new_placed_poses
    
def randomize_pose_train(model:MeshObject, pos_min, pos_max, rot_min, rot_max, origin):
    location = np.random.uniform(pos_min, pos_max)
    rotation = np.random.uniform(rot_min, rot_max)

    # Adding origin offset
    offset = origin
    location += offset

    return location, rotation

def randomize_pose_distr(model:MeshObject, pos_min, pos_max, rot_min, rot_max):
    # Equivalent of location = np.random.uniform([],[]) since it is not suported by numba.
    location = np.empty(3, dtype=np.float64)
    for i in range(3):
        location[i] = np.random.uniform(pos_min[i], pos_max[i])

    # Equivalent of rotation = np.random.uniform([],[]) since it is not suported by numba.
    rotation = np.empty(3, dtype=np.float64)
    for i in range(3):
        rotation[i] = np.random.uniform(rot_min[i], rot_max[i])

    # Making distractors spawn using its bottom.
    offset = np.array([0,0, abs(model.blender_obj.bound_box[0][2]) + 0.001]) # security offset in case a model has no height.
    location += offset

    return location, rotation

@njit(cache=CACHE_NUMBA)
def randomize_pose_train_numba(id, offsets, pos_min, pos_max, rot_min, rot_max):
    # Equivalent of location = np.random.uniform([],[]) since it is not suported by numba.
    location = np.empty(3, dtype=np.float64)
    for i in range(3):
        location[i] = np.random.uniform(pos_min[i], pos_max[i])

    # Equivalent of rotation = np.random.uniform([],[]) since it is not suported by numba.
    rotation = np.empty(3, dtype=np.float64)
    for i in range(3):
        rotation[i] = np.random.uniform(rot_min[i], rot_max[i])

    location += offsets[id]

    return location, rotation

@njit(cache=CACHE_NUMBA)
def randomize_pose_distr_numba(id, offsets, pos_min, pos_max, rot_min, rot_max):
    # Equivalent of location = np.random.uniform([],[]) since it is not suported by numba.
    location = np.empty(3, dtype=np.float64)
    for i in range(3):
        location[i] = np.random.uniform(pos_min[i], pos_max[i])

    # Equivalent of rotation = np.random.uniform([],[]) since it is not suported by numba.
    rotation = np.empty(3, dtype=np.float64)
    for i in range(3):
        rotation[i] = np.random.uniform(rot_min[i], rot_max[i])

    location += offsets[id]

    return location, rotation

def randomize_camera(config:dict, focus_point:list[float]):
    """
    Samples a random camera position while orienting it towards a specified focus point, and randomizes the camera's depth of field (f-stop).

    Parameters:
        config (dict): Loaded configuration containing camera parameters.
        focus_point: The point that the camera will focus on.

    Returns:
        tuple: A tuple containing:
            - cam_pose (ndarray): The transformation matrix of the camera.
            - cam_dof (float): The randomized depth of field value (f-stop).
    """

    # Randomly set up the camera:
    radius_min, radius_max =config["camera"]["pos_shell_radius"]
    elev_min, elev_max = config["camera"]["pos_shell_elevation"]
    if elev_min == elev_max:
        elev_min = elev_max - 0.0001
    location = bproc.sampler.shell(center=focus_point, radius_min=radius_min, radius_max=radius_max, elevation_min=elev_min, elevation_max=elev_max)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(focus_point - location, inplane_rot=None)

    cam_pose:np.ndarray = bproc.math.build_transformation_mat(location, rotation_matrix)

    cam_dof:float = np.random.uniform(*config["camera"]["f-stop"])

    return cam_pose, cam_dof
