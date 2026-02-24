import json
import yaml
import os

def prettier_json(path:str):
    def custom_json_dumps(obj, indent=None):
        # First, serialize the object with the indent but with compact separators
        json_str = json.dumps(obj, indent=indent, separators=(',', ': '))

        # Then flatten lists by removing line breaks within lists
        json_str = json_str.replace('[\n        ', '[').replace('\n      ]', ']').replace(',\n        ', ', ')
        
        return json_str
    
    with open(path, 'r') as f:
        data = json.load(f)

    with open(path, 'w') as f:
        f.write(custom_json_dumps(data, indent=2))

def fix_coco_json(path: str, target_elements:list):
    with open(path, 'r') as f:
        data = json.load(f)

    for cat, element in zip(data['categories'], target_elements):
        cat['name'] = element.get_name()

    
    with open(path, 'w') as f:
        json.dump(data, f)

def split_interval(start:int, end:int, n_batches:int):
    total = end - start + 1
    n_batches = n_batches or 1 # In case n_batches is 0, we set it to 1.


    # Determine base size of each batch and the remainder to distribute
    base_size, remainder = divmod(total, n_batches)
    
    boundaries = []
    current = start
    for i in range(n_batches):
        # For the first 'remainder' batches, add one extra element
        batch_size = base_size + (1 if i < remainder else 0)
        batch_start = current
        batch_end = current + batch_size - 1
        boundaries.append((batch_start, batch_end))
        current = batch_end + 1
        
    return boundaries

def scan_folder(dir_path:str, whitelist:list[str]=None, blacklist:list[str]=None):
    """
    Creates a sorted list with the paths all the files in a folder after applying a white and black list.

    Parameters:
        dir_path (str): Path to directory to scan.
        whitelist (list[str]): Filenames whitelist
        blacklist (list[str]): Filenames blacklist

    Returns:
        paths (list[str]): 
    """
    if not os.path.isdir(dir_path):
        return []

    # Filter data in the folder:
    all_files = set(os.listdir(dir_path))
    whitelist = set(whitelist or all_files)                             # Preprocessing whitelist.
    filenames = (all_files & whitelist) - set(blacklist or [])          # Excluding only models in the blacklist.
    paths = sorted(os.path.join(dir_path, name) for name in filenames)  # Setting absolute path for filtered models

    return paths

def load_config(config_path:str|dict):
    """
    Loads config data from a yaml file.

    Parameters:
        config (str): Path to config file.

    Returns:
        config (dict): Loaded config data.
    """

    if not os.path.isfile(config_path):
        raise FileExistsError(f"Error: Could not find config file! '{config_path}'")

    if config_path.split(".")[-1].lower() != "yaml":
        raise TypeError(f"Error: config file should be a YAML")

    with open(config_path, 'r') as f:
        config:dict = yaml.safe_load(f)

    return config


