import blenderproc as bproc
import os
import bpy
import numpy as np
import itertools
import yaml
from collections.abc import Iterable

from blenderproc.python.types.MeshObjectUtility import MeshObject, Entity

from synthrender.utils.bproc_utils import bproc_utils
from synthrender.utils import misc_utils
from synthrender.utils.blender_setup import scene_setter
from synthrender.utils.blender_setup import material_randomizer 

############################################################
# FUNCTIONS USED FOR LOADING ELEMENTS ON THE BLENDER SCENE #
############################################################

def _to_bpy_objects(items):
    """Normalize mixed BlenderProc / bpy / names into bpy.types.Object list."""
    if items is None:
        return []
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
        items = [items]

    out = []
    for it in items:
        if isinstance(it, bpy.types.Object):
            out.append(it); continue

        # BlenderProc wrappers
        ob = None
        if hasattr(it, "blender_obj") and isinstance(it.blender_obj, bpy.types.Object):
            ob = it.blender_obj
        elif hasattr(it, "get_blender_obj"):
            try:
                ob = it.get_blender_obj()
            except Exception:
                ob = None
        elif hasattr(it, "get_name"):
            ob = bpy.data.objects.get(it.get_name())
        elif hasattr(it, "name"):
            ob = bpy.data.objects.get(it.name)
        elif isinstance(it, (str, bytes)):
            ob = bpy.data.objects.get(str(it))

        if isinstance(ob, bpy.types.Object):
            out.append(ob)
    return out


def _mesh_descendants_inclusive(root_obj):
    """Return {root if MESH} ∪ {all MESH children (recursive)}; unique set."""
    meshes = set()
    if getattr(root_obj, "type", None) == "MESH":
        meshes.add(root_obj)
    # children_recursive covers all depths
    for ch in getattr(root_obj, "children_recursive", []):
        if getattr(ch, "type", None) == "MESH":
            meshes.add(ch)
    return list(meshes)


def load_pbr_materials(config:dict[str, dict]):

    amt = config["models"]["material_randomization_options"]["material_limit"]
    lib_dir = config.get("material_randomization_dir")
    
    folders = material_randomizer._stable_subfolders(lib_dir)
    folder = os.path.join(lib_dir, np.random.choice(folders))
    

    for i in range(amt):

        folder = os.path.join(lib_dir, np.random.choice(folders))
        mat = material_randomizer.build_pbr_material_from_folder(folder)




def load_backgrounds(config:dict[str, dict]):
    """
    Loads the background images with format .exr and sets a default one if specified in the config file.

    Parameters:
        config (dict[str, dict]): Loaded configuration as a dictionary.

    Returns:
        loaded_backgrounds (list): List with all the loaded backgrounds
    """
    loaded_backgrounds:list[bpy.types.Image] = []

    # Filter data in the folder:
    if os.path.isdir(dir_path:=config["backgrounds_dir"]):
        paths = misc_utils.scan_folder(dir_path, config["backgrounds_whitelist"], config["backgrounds_blacklist"])

        for path in paths:
            background = bpy.data.images.load(path, check_existing=True)
            loaded_backgrounds.append(background)

            # Setting default background if any.
            if config["world"]["default_background_img"] == os.path.basename(path):
                scene_setter.set_background_texture(os.path.basename(path))
                
    elif dir_path:
        print(f"Warning: Backgrounds folder not found!: {dir_path}")
    
    return loaded_backgrounds

def camera_setup(config:dict[str, str]):
    """
    Loads the camera settings of the loaded configuration.

    This settings include the sensor_size, resolution and intrinsic_parameters of the camera.

    Parameters:
        config (dict[str, dict]): Loaded configuration as a dictionary.

    """
    # Setting camera sensor size:
    sensor_size = config["camera"]["sensor_size"]
    bpy.context.scene.camera.data.sensor_width = sensor_size

    # Setting camera resolution:
    width, height = config["camera"]["resolution"]
    bproc.camera.set_resolution(image_width=width, image_height=height)

    # Setting camera intrinsic parameters if any.
    k_path:str = config["camera"]["intrinsic_parameters_path"]
    if os.path.isfile(k_path):
        assert k_path.split(".")[-1].lower() == "yaml", "Error: Extension of camera_intrinsics file should be YAML!"
        with open(k_path, "r") as f:
            k_matrix = np.array(yaml.safe_load(f))

        bproc.camera.set_intrinsics_from_K_matrix(K=k_matrix, image_width=width, image_height=height)
    elif k_path:
        print(f"Warning: camera_intrinsics file not found in {k_path}")

def load_default_scene(config: dict[str, dict]) -> list:
    """
    Loads the default scene, optionally creating a physics ground.

    Parameters:
        config (dict[str, dict]): Configuration dictionary.

    Returns:
        default_scene (list): A list of objects representing the default scene.
    """
    # Physics options:
    create_ground:bool = config['physics']["simulate_physics"] and config['physics']["create_ground_plane"]
    do_physics:bool = config['physics']["simulate_physics"] and config['physics']["default_scene_as_passives"]

    # Creating model collection:
    collection = bpy.data.collections.new("Default_Scene")
    bpy.context.scene.collection.children.link(collection)  # Link the collection to the current scene

    default_scene_meshes:list = []
    loaded_meshes:list[MeshObject] = []
    loaded_data = []
    ground:MeshObject = None

    # Creates ground plane if setted in the physics:
    if create_ground:
        ground = bproc.object.create_primitive('PLANE')
        ground.blender_obj.dimensions = [config["plane"]["x_length"], config["plane"]["y_length"], 0]
        ground.set_name("Ground")
        # ground.enable_rigidbody(active=False, collision_shape='CONVEX_HULL')
        ground.blender_obj.hide_viewport = True
        ground.blender_obj.hide_render = True
        loaded_meshes.append(ground)

    # Loading defalut_scene file:
    if os.path.isfile(path:=config["default_scene"]):
        assert path.split(".")[-1].lower() == "blend", "Error: Extension of default_scene file should be BLEND!"
        loaded_data = bproc.loader.load_blend(path, data_blocks=['armatures', 'curves', 'images',
                                                                    'lights', 'materials', 'meshes', 'objects', 'textures'])
        
        for loaded_model in loaded_data:
            if hasattr(loaded_model, 'blender_obj') and loaded_model.blender_obj.type != 'MESH':
                bpy.context.collection.objects.unlink(loaded_model.blender_obj)
                collection.objects.link(loaded_model.blender_obj)

        loaded_meshes.extend([mesh for mesh in loaded_data if hasattr(mesh, "get_mesh")]) # Filter from all the loaded data the meshes

    elif path:
        print(f"Warning: could not find default_scene file in: {path}")


    if loaded_meshes:
        print("Setting up models of default scene...")

        mesh_parents:set[MeshObject] = set()
        for mesh_parent in loaded_meshes:

            mesh_parents.add(bproc_utils.get_parent_mesh(mesh_parent))
        
        for mesh_parent in mesh_parents:
            children = bproc_utils.get_all_child_meshes(mesh_parent)
            obj_meshes = [child for child in children if not child.get_name().startswith("#Ignore")]

            if not obj_meshes: 
                continue
            
            parent_name = mesh_parent.get_name()
            mesh_parent.set_name(f"d_{parent_name}") # default_name

            # Processing model to have colliders and bbox structure:
            parent = bproc_utils.process_model(obj_meshes, parent_name, collection=collection)

            # Add physics if set in the config:
            if do_physics:
                scene_setter.setup_models_physics(config, [parent], as_active=False, enable=True, default_config=None)
            
            # Hide colliders and bboxes:
            parent.blender_obj.hide_viewport = True
            for child in parent.get_children():
                if child.get_name().startswith("#Collider"):
                    child.blender_obj.hide_viewport = True

            # Now we just return the loaded data + the ground if any
            default_scene_meshes.append(parent)


    return default_scene_meshes

def material_randomize(config, train_models, rng=None):

    # --- Material randomization for *loaded train models* (after IDs are set) ---
 
        
    lib_dir = config.get("material_randomization_dir")

    if not lib_dir or not os.path.isdir(lib_dir):
        print(f"[MaterialRand] Skipped: invalid library dir: {lib_dir!r}")
    else:
        parents = train_models
        
        # Fallback: if nothing resolved, try by names explicitly
        # if not parents and hasattr(train_models[0], "get_name"):
        #     parents = [bpy.data.objects.get(m.get_name()) for m in train_models]
        #     parents = [p for p in parents if p]

        mesh_targets = []
        for p in parents:
            children = bproc_utils.get_all_child_meshes(p)
            obj_meshes = [child for child in children if not child.get_name().startswith(("#Ignore","#Collider"))]
            mesh_targets.extend(obj_meshes)

        # Still empty? Maybe the wrappers are the meshes themselves.
        if not mesh_targets:
            direct_meshes = _to_bpy_objects(train_models)
            mesh_targets = [ob for ob in direct_meshes if getattr(ob, "type", None) == "MESH"]

        if not mesh_targets:
            print("[MaterialRand] No MESH descendants or direct meshes found under train_models; skipped.")
        else:
            # print(f"[MaterialRand] Assigning random PBRs from '{lib_dir}' to {len(mesh_targets)} mesh children…")

            valids = [mat for mat in bpy.data.materials.values() if mat.name.startswith("PBR_")]
            # print(valids)


            for ob in mesh_targets:
                ob = ob.blender_obj

                mat_rand_choice = np.random.choice(valids)

                material_randomizer.force_single_material(ob, mat_rand_choice)                  # wipe slots and apply
                # assign_material_to_object(obj, mat)
                print(f"[PBR] Assigned '{mat_rand_choice.name}' to '{ob.name}'")
                
                # material_randomizer.assign_random_pbr_from_library(ob, lib_dir, config["seed"], rng=rng)

            # Debug: list final materials on first few targets
            # for ob in mesh_targets[:3]:
            #     mats = [for m in m.name if m else "<None>"] if ob.data.materials else []
            #     print(f"[MaterialRand][CHECK] {ob.name} slots → {mats}")
                
    # except Exception as e:
    #     print(f"[MaterialRand] Error: {e}")

def load_train_models(config:dict[str, dict]):
    """
    Loads the train models specified in the config file and sets a category_id to all of them so that segmentation masks can be obtained.

    It can also create fake train models which will spawn as if they were train objects but with no category_id:
        - Creates deformed copies of the train models if 'fake_similar_trains' is enabled.
        - Creates simple geometric shapes if 'fake_simple_trains' is enabled.

    Parameters:
        config (dict): Configuration dictionary containing the train settings.
    Returns:
        tuple: A tuple of train models (loaded, fakes)
            - train_models (list[MeshObject]): A list of objects representing the loaded train models.
            - fake_models (list[MeshObject]): A list of objects representing the fake train models.
    """

    train_models:list[MeshObject] = []
    fake_models:list[MeshObject] = []

    # Creating model collection:
    collection = bpy.data.collections.new("Train_Models")
    bpy.context.scene.collection.children.link(collection)  # Link the collection to the current scene

    if os.path.isdir(dir_path:=config["train_models_dir"]):
        pos_callback = lambda x, i: (range(-len(x)//2+1, len(x)//2+1, 1)[i], -5, 0) # Setting up initial position of models
        whitelist = config["train_models_whitelist"]
        blacklist = config["train_models_blacklist"]
        
        models_config = {"default_config": config["models"]["trains"]}
        models_config.update(config.get("custom_models", {}))

        train_models = bproc_utils.load_models_folder(dir_path, whitelist, blacklist, pos_callback, collection, models_config)

        # Setting category_id to train objects.
        bproc_utils.set_category_to_meshes(train_models, models_config)

        material_randomize(config, train_models=train_models)

    elif dir_path:
        print(f"Warning: Train folder not found!: {dir_path}")

    # Creates fake training models using deformed versions of the training models.
    if n_similar:=config['fake_models']["trains"].get("similar_trains"):
        pos_callback = lambda x, i: (range(-len(x)//2+1, len(x)//2+1, 1)[i], -6, 0) # Setting up initial position of models
        total_copies = n_similar * len(train_models)
        count = itertools.count()

        # Create n_similar copies of each train_model.
        for model in train_models:
            model_name = model.get_name()

            model_config = {}
            model_config.update(config["fake_models"]["trains"])
            model_config.update(config['custom_models'].get(model_name, {}))

            for _ in range(n_similar):
                pos = pos_callback(range(total_copies), next(count))
                similar_dis = bproc_utils.create_similar_distractor(pos, model, collection, model_config)
                fake_models.append(similar_dis)

    # Creates fake training models using simple geometric shapes.
    if n_fakes:=config['fake_models']["trains"].get("simple_trains"):
        pos_callback = lambda x, i: (range(-len(x)//2+1, len(x)//2+1, 1)[i], -7, 0) # Setting up initial position of models

        models_config = {"default_config": config["fake_models"]["trains"]}
        models_config.update(config.get("custom_models", {}))

        for i in range(n_fakes):
            pos = pos_callback(range(n_fakes), i)
            simple_model = bproc_utils.create_random_distractor(pos, collection, models_config)
            fake_models.append(simple_model)


    return train_models, fake_models

def load_distractor_models(config:dict):
    """
    Loads the distractor models specified in the config file.

    It can also create fake train models which will spawn as if they were disctractors.
        - Creates simple geometric shapes if 'simple_distractors' is enabled.

    Parameters:
        config (dict): Configuration dictionary containing the disctractors settings.
    """

    distractors:list[MeshObject] = []
    fake_distractors:list[MeshObject] = []



    # Creating model collection:
    collection = bpy.data.collections.new("Distraction_Models")
    bpy.context.scene.collection.children.link(collection)  # Link the collection to the current scene

    if os.path.exists(dir_path:=config["distractors_dir"]):
        pos_callback = lambda x, i: (range(-len(x)//2+1, len(x)//2+1, 1)[i], 5, 0) # Setting up initial position of models
        whitelist = config["distraction_models_whitelist"]
        blacklist = config["distraction_models_blacklist"]

        models_config = {"default_config": config["models"]["distractors"]}
        models_config.update(config.get("custom_models", {}))

        distractors = bproc_utils.load_models_folder(dir_path, whitelist, blacklist, pos_callback, collection, models_config)

    elif dir_path:
        print(f"Warning: Distractors folder not found!: {dir_path}")

    if n_simples:=config['fake_models']["distractors"].get("simple_distractors"):
        pos_callback = lambda x, i: (range(-len(x)//2+1, len(x)//2+1, 1)[i], 6, 0) # Setting up initial position of models

        for i in range(n_simples):
            pos = pos_callback(range(n_simples), i)

            models_config = {"default_config": config["fake_models"]["distractors"]}
            models_config.update(config.get("custom_models", {}))

            simple_model = bproc_utils.create_random_distractor(pos, collection, models_config)
            fake_distractors.append(simple_model)

    return distractors, fake_distractors

def load_plane_materials(config:dict):
    """
    Loads the materials stored in planes_dir and spawns planes with them. The planes share the same geometry to avoid consuming more memory.

    It can also create simple planes with random materials if option 'simple_planes' is set in the config file.

    Parameters:
        config (dict): Configuration dictionary containing the planes settings.
    """
    aux_plane = bproc.object.create_primitive('PLANE')
    aux_plane.blender_obj.dimensions = [config["plane"]["x_length"], config["plane"]["y_length"], 0]
    aux_plane.set_name("Plane.000")
    planes:list[MeshObject] = []
    materials:list = []
    plane = None

    # Creating model collection:
    collection = bpy.data.collections.new("Planes_Models")
    bpy.context.scene.collection.children.link(collection)  # Link the collection to the current scene

    bpy.context.collection.objects.unlink(aux_plane.blender_obj)   # Unlink from current collection
    collection.objects.link(aux_plane.blender_obj)               # Link to the new collection
    aux_plane.blender_obj.hide_viewport = True
    aux_plane.blender_obj.hide_render = True

    if os.path.isdir(dir_path:=config["planes_dir"]):
        mod_paths = misc_utils.scan_folder(dir_path, config["planes_whitelist"], config["planes_blacklist"])
        materials = [bproc.loader.load_blend(path, data_blocks=['materials'])[0] for path in mod_paths]
    elif dir_path:
        print(f"Warning: Plane materials folder not found!: {dir_path}")


    if n_simples:=config["plane"].get("simple_planes", 0):
        for _ in range(n_simples):
            new_material = bproc.material.create("Simple_plane_material.000")
            new_material.set_principled_shader_value("Base Color", np.random.uniform([0,0,0,1], [1,1,1,1]))
            materials.append(new_material.blender_obj)

    for material in materials:
        if plane is None:
            plane:MeshObject = aux_plane.duplicate()            # Copy not linked of ex_plane
            plane.clear_materials()                             # Clear all material data
            plane.blender_obj.data.materials.append(None)       # Add a material slot but empty
            plane.blender_obj.material_slots[0].link = 'OBJECT' # Set the material slot to keep material only on object
            copy:MeshObject = plane
        else:
            copy:MeshObject = plane.duplicate(linked=True)      # Copy linked to share the same mesh

        copy.blender_obj.material_slots[0].material = material  # Setting the material

        bpy.context.collection.objects.unlink(copy.blender_obj) # Unlink from current collection
        collection.objects.link(copy.blender_obj)               # Link to the new collection

        planes.append(copy)

    # Remove auxiliar plane.
    bproc.object.delete_multiple([aux_plane])

    return planes

def load_lights(config:dict, empty:Entity):
    """
    Creates lights as a three-point lightning setup wherein there is a key, fill and back light. Each of them looks towards the empty object and has a direction and distance defined in the config file:
        - lights_dir: unit vectors for each of the lights indicating their direction.
        - light_size: size of the area lights.
        - contrasts: scaling factor for the lights energy.
        - distance: distance each light will mantain to the emtpy object.

    Parameters:
        config (dict): Configuration dictionary with light settings.
        empty (Entity): Emtpy object used (empty object from blender) as parent (lights will track it).
    """
    lights:list[bproc.types.Light] = []

    # Creating model collection:
    collection = bpy.data.collections.new("Lights")
    bpy.context.scene.collection.children.link(collection)  # Link the collection to the current scene

    for i, name in enumerate(['key', 'fill', 'back']):
        dir_vector = np.array(config["lights"]["lights_dir"])[i]
        contrast = config["lights"]["contrasts"][i]
        distance = config["lights"]["distance"]

        # Create light and set position:
        area_light = bproc.types.Light("AREA", name=name)
        area_light.blender_obj.data.size = config["lights"]["light_size"]
        area_light.set_location(dir_vector*distance)
        area_light.set_cp("contrast", contrast)

        # Set light constrains to follow empty object.
        area_light.blender_obj.constraints.new("TRACK_TO")
        area_light.blender_obj.constraints['Track To'].target = empty.blender_obj

        area_light.set_parent(empty)
        # area_light.blender_obj.constraints.new('COPY_TRANSFORMS')
        # area_light.blender_obj.constraints['Copy Transforms'].mix_mode = 'AFTER_SPLIT'
        # area_light.blender_obj.constraints['Copy Transforms'].target = empty.blender_obj


        bpy.context.collection.objects.unlink(area_light.blender_obj)   # Unlink from current collection (if needed)
        collection.objects.link(area_light.blender_obj)        # Link to the new collection

        lights.append(area_light)

    return lights

def setup_restart_frames(models:list[MeshObject|Entity], planes:list[MeshObject]):
    for frame in [-2, -1]:
        for model in models:
            for child in [model, *bproc_utils.get_all_child_meshes(model)]:
                child.blender_obj.hide_render = True
                child.blender_obj.hide_viewport = frame == -2

        for plane in planes:
            plane.blender_obj.hide_render = True
            plane.blender_obj.hide_viewport = True

        bproc_utils.save_keyframe(frame)

def update_trains_sample_size(config:dict, train_models, fake_models):
    # Checking range of sample_size. If it is -1 or the sample is out of range, set the len of train_models.
    tmp = config["train_models"]["sample_size_trains"]
    n_models = len(train_models)
    config["train_models"]["sample_size_trains"] = [n if 0 <= n <= n_models else n_models for n in tmp]

    tmp = config["train_models"]["sample_size_fakes"]
    n_models = len(fake_models)
    config["train_models"]["sample_size_fakes"] = [n if 0 <= n <= n_models else n_models for n in tmp]
