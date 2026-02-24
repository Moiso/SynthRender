import os
import shutil

from synthrender.utils import yolo_utils

def coco2yolo(coco_dir:str, category_mapping:dict=None, verbose:bool=False, annotation_type:str="object_detection"):
    """
    Converts coco annotations into yolo

    Parameters:
        coco_dir (str)
        category_mapping (dict): List containing each light object.
        verbose (bool): List containing the energy value for each light object.
        annotation_type (str): Type of annotation -> "object_detection" or "instance_segmentation".
    """

    yolo_dir = os.path.join(os.path.dirname(coco_dir), os.path.basename(coco_dir).replace("coco", "yolo"))

    if not os.path.isdir(coco_dir):
        print(f"Error! Cannot find COCO annotations at {coco_dir}")
        exit(-1)

    if os.path.isdir(yolo_dir):
        print(f"Warning: Yolo folder '{os.path.basename(yolo_dir)}' already exists, skipping..")
        return yolo_dir

    shutil.rmtree(yolo_dir, ignore_errors=True) # Removing old files:

    # Obtain yolo labels.
    if verbose:
        print("\nGetting YOLO annotations from COCO data...")
    yolo_utils.convert_coco_to_yolo(coco_dir, yolo_dir, category_mapping=category_mapping, annotation_type=annotation_type)
    
    # Clean unmatched images from the dataset.
    if verbose:
        print("Removing unmatched images from dataset")
    yolo_utils.clean_unmatched_imgs(yolo_dir)
    
    # Split dataset 80% train 20% test.
    if verbose:
        print("Splitting dataset into 80 / 20")
    yolo_utils.split_dataset(yolo_dir)

    return yolo_dir