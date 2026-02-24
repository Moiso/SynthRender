import os
import shutil

from synthrender.utils import yolo_utils

def coco2yolo(coco_dir:str, category_mapping:dict=None, verbose=False):

    yolo_dir = os.path.join(os.path.dirname(coco_dir), os.path.basename(coco_dir).replace("coco", "yolo"))

    if not os.path.isdir(coco_dir):
        print(f"Error! Cannot find COCO annotations at {coco_dir}")
        exit(-1)

    if os.path.isdir(yolo_dir):
        print(f"Warning: Yolo folder '{yolo_dir}' already exists, skipping..")
        return yolo_dir

    shutil.rmtree(yolo_dir, ignore_errors=True) # Removing old files:

    # Obtain yolo labels.
    if verbose:
        print("\nGetting YOLO annotations from COCO data...")
    yolo_utils.convert_coco_to_yolo(coco_dir, yolo_dir, category_mapping=category_mapping)
    
    # Clean unmatched images from the dataset.
    if verbose:
        print("Removing unmatched images from dataset")
    yolo_utils.clean_unmatched_imgs(yolo_dir)
    
    # Split dataset 80% train 20% test.
    if verbose:
        print("Splitting dataset into 80 / 20")
    yolo_utils.split_dataset(yolo_dir)

    return yolo_dir

# category_mapping = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "11": 10, "12": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "26": 25, "27": 26, "28": 27, "29": 28, "30": 29, "31": 30, "32": 31, "33": 32}
# coco2yolo(coco_dir="/home/vrt/Projects/Tom/Outputs/Physics_experiments/wip_1k_1024_with_physics_all_classes_corrected/coco", category_mapping=category_mapping, verbose=True)
# yolo_utils.visualize_yolo(yolo_dir="/home/vrt/Projects/Tom/Outputs/Physics_experiments/wip_1k_1024_with_physics_all_classes_corrected/yolo")