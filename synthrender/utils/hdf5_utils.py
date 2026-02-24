import os
import numpy as np
import json
import h5py
import tifffile as tiff
import cv2

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def hdf5_to_dict(hdf5_file:str|h5py.File):
    """
    Takes a hdf5 file or the path to a file and returns it as a dict.
    :hdf5_file: h5py file or path to the hdf5 file.
    :return: dict with the data
    """
    def process_hdf5(hdf5):
        """Helper function to process an hdf5 file object."""
        values = list(hdf5.values())[1:]  # Skip first value (if intended)
        data = {value.name[1:]: [np.array(value, dtype=value.dtype)] for value in values}

        for key in data.keys():
            if data[key][0].dtype.kind == 'S':  # Detect byte strings
                for i, maps in enumerate(data[key]):
                    data[key][i] = json.loads(maps.item().decode('utf-8'))

        return data

    try:
        # If the input is a file path, open it and process it
        if isinstance(hdf5_file, str):
            with h5py.File(hdf5_file, 'r') as f:
                return process_hdf5(f)
        else:
            # Process the already opened hdf5 file object
            return process_hdf5(hdf5_file)

    except Exception as e:
        print(f"Error processing {hdf5_file}: {e}")
        return {}

def open_multiple_hdf5_as_dict(hdf5_paths: list[str], show_progres=False) -> dict[str, list]:
    """
    In order to access the result:
    output["hdf5_key"][i (frame)]
    """
    data = {}

    # Switch to ThreadPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor() as executor:
        # Step 1: Submit all tasks and collect futures
        futures = list(map(lambda path: executor.submit(hdf5_to_dict, path), hdf5_paths))

        # Step 2: Process futures and show progress with tqdm
        for i, future in enumerate(tqdm(futures, desc="Loading files", total=len(futures), unit=" file", disable=not show_progres)):
            hdf5_as_dict = future.result()

            # Merge the results into the data dictionary
            for key in hdf5_as_dict.keys():
                if i == 0 and key not in data:
                    data[key] = []
                data[key].extend(hdf5_as_dict[key])

    return data

def hdf5_to_dict_old(hdf5:h5py.File):
    try:

        values = list(hdf5.values())[1:]
        data = {value.name[1:] : [np.array(value, dtype = value.dtype)] for value in values}

        for key in data.keys():
            if data[key][0].dtype.kind == 'S':  # Detect byte strings
                for i, maps in enumerate(data[key]):
                    data[key][i] = json.loads(maps.item().decode('utf-8'))

        return data

    except Exception as e:
        print(f"Error processing {hdf5}: {e}")
        return {}

def open_multiple_hdf5_as_dict_old(hdf5_paths:list[str]) -> dict[str,list]:
    """
    In order to access the result:
    output["hdf5_key"][i (frame)]
    """
    data = {}
    for i, path in enumerate(tqdm(hdf5_paths, desc="Loading hdf5 files", unit=" file")):
        with h5py.File(path, 'r') as f:
            hdf5_as_dict = hdf5_to_dict_old(f)
            
            for key in hdf5_as_dict.keys():
                if i == 0:
                    data[key] = []
                data[key].extend(hdf5_as_dict[key])

    return data

def load_hdf5_fast(hdf5_file, data_name:str="models_data"):
    """
    Takes a hdf5 file or the path to a file and returns it as a dict. (Only loads one parameter)
    :hdf5_file: h5py file or path to the hdf5 file.
    :return: dict with the data
    """
    def process_hdf5(hdf5):
        """Helper function to process an hdf5 file object."""
        data = {hdf5.name[1:]: [np.array(hdf5, dtype=hdf5.dtype)]}

        for key in data.keys():
            if data[key][0].dtype.kind == 'S':  # Detect byte strings
                for i, maps in enumerate(data[key]):
                    data[key][i] = json.loads(maps.item().decode('utf-8'))

        return data

    try:
        # If the input is a file path, open it and process it
        if isinstance(hdf5_file, str):
            with h5py.File(hdf5_file, 'r') as f:
                return process_hdf5(f[data_name])
        else:
            # Process the already opened hdf5 file object
            return process_hdf5(hdf5_file[data_name])

    except Exception as e:
        print(f"Error processing {hdf5_file}: {e}")
        return {}
    
def open_multiple_hdf5_as_dict_fast(hdf5_paths: list[str], data_name:str, show_progres=False) -> dict[str, list]:
    """
    In order to access the result:
    output["hdf5_key"][i (frame)]
    (Only loads one parameter)
    """
    data = {}

    # Switch to ThreadPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor() as executor:
        # Step 1: Submit all tasks and collect futures
        futures = list(map(lambda path: executor.submit(load_hdf5_fast, path, data_name), hdf5_paths))

        # Step 2: Process futures and show progress with tqdm
        for i, future in enumerate(tqdm(futures, desc="Loading files", total=len(futures), unit=" file", disable=not show_progres)):
            hdf5_as_dict = future.result()

            # Merge the results into the data dictionary
            for key in hdf5_as_dict.keys():
                if i == 0 and key not in data:
                    data[key] = []
                data[key].extend(hdf5_as_dict[key])

    return data


def export_rgb(data:list, output_dir:str, index_start:int=0, timestamp:str=None):
    for i, img in enumerate(data, start=index_start):
        img_path = os.path.join(output_dir, f"{timestamp}_{i:06d}.png")
        cv2.imwrite(img_path, img[..., ::-1])  # Convert RGB to BGR for OpenCV

def export_segmasks(data:list, output_dir:str, index_start:int=0, timestamp:str=None):
    for i, img in enumerate(data, start=index_start):
        img_path = os.path.join(output_dir, f"{timestamp}_{i:06d}.png")
        img = (img*255).astype(np.uint8)
        assert len(np.unique(img)) <= 2
        cv2.imwrite(img_path, img)

def export_depth(data:list, output_dir:str, index_start:int=0, timestamp:str=None):
    for i, img in enumerate(data, start=index_start):
        img_path = os.path.join(output_dir, f"{timestamp}_{i:06d}.tiff")
        tiff.imwrite(img_path, img)

def export_normals(data:list, output_dir:str, index_start:int=0, timestamp:str=None):
    for i, img in enumerate(data, start=index_start):
        img_path = os.path.join(output_dir, f"{timestamp}_{i:06d}.tiff")
        tiff.imwrite(img_path, img)



def from_hdf5_export_rgb(hdf5:h5py.File|str, output_dir:str, index_start:int=0):
    """
    Takes a hdf5 file or the path to a file and saves the rgb image in colors.
    :hdf5: h5py File or path to the hdf5 file.
    """
    data = load_hdf5_fast(hdf5, 'colors')['colors']
    export_rgb(data, output_dir, index_start)

def from_hdf5_export_segmasks(hdf5:h5py.File|str, output_dir:str, index_start:int=0):
    """
    Takes a hdf5 file or the path to a file and saves the rgb image in colors.
    :hdf5: h5py File or path to the hdf5 file.
    """
    data = load_hdf5_fast(hdf5, 'category_id_segmaps')['category_id_segmaps']
    export_segmasks(data, output_dir, index_start)

def from_hdf5_export_depth(hdf5:h5py.File|str, output_dir:str, index_start:int=0):
    """
    Takes a hdf5 file or the path to a file and saves the rgb image in colors.
    :hdf5: h5py File or path to the hdf5 file.
    """
    data = load_hdf5_fast(hdf5, 'depth')['depth']
    export_depth(data, output_dir, index_start)

def from_hdf5_export_normals(hdf5:h5py.File|str, output_dir:str, index_start:int=0):
    """
    Takes a hdf5 file or the path to a file and saves the rgb image in colors.
    :hdf5: h5py File or path to the hdf5 file.
    """
    data = load_hdf5_fast(hdf5, 'normals')['normals']
    export_normals(data, output_dir, index_start)

