import os 
import json
import random
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


def convert_coco_to_yolo(coco_dir, yolo_dir, category_mapping:dict = None, hide_pbar=False, annotation_type="object_detection"):
    coco_json_path  = os.path.join(coco_dir, 'coco_annotations.json')
    output_dir      = os.path.join(yolo_dir, 'labels/train')

    if os.path.exists(coco_json_path):
        with open(coco_json_path, 'r') as f:
            coco_data:dict = json.load(f)
    else:
        print(f"Error! Coco annotations not found!: {coco_json_path}")
        exit(-1)

    os.rename(coco_dir, yolo_dir)
    os.makedirs(output_dir, exist_ok=True)

    if category_mapping:
        category_mapping = {int(key): value for key, value in category_mapping.items()}  # Casting keys from str to int.

    # Dictionary to map image_id to filenames:
    image_id_to_file = {image['id']: image['file_name'] for image in coco_data['images']}

    format = '{l_bar}{bar:25}{r_bar}'
    annotations = coco_data.get('annotations', [])
    with tqdm(total=len(annotations), desc="Converting COCO into YOLO", disable=hide_pbar, bar_format=format, ascii=" =") as pbar:
        for annotation in annotations:
            image_id = annotation['image_id']
            filename = image_id_to_file.get(image_id)

            image_width = annotation['width']
            image_height = annotation['height']

            if filename:
                # Convert to YOLO format: <class_index> <x_center> <y_center> <width> <height>
                if category_mapping:
                    class_index = category_mapping.get(annotation['category_id'], -1)  # Get the class index
                else:
                    class_index = annotation['category_id']-1 # Annotations start from 1 to n, shifting it to start from 0.
                    
                if class_index == -1:
                    continue  # Skip if class is not in mapping

                # Construct the label filename and write the annotation to it
                filename        = os.path.basename(filename)
                label_file_path = os.path.join(output_dir, filename.replace('.png', '.txt'))
                
                if annotation_type == "object_detection":
                    # Calculate the center and size in YOLO format
                    x_center = (annotation['bbox'][0] + annotation['bbox'][2] / 2) / image_width
                    y_center = (annotation['bbox'][1] + annotation['bbox'][3] / 2) / image_height
                    width = annotation['bbox'][2] / image_width
                    height = annotation['bbox'][3] / image_height
                
                    with open(label_file_path, 'a') as label_file:  # Append to the file
                        label_file.write(f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                elif annotation_type == "instance_segmentation":
                    # Normalize poygon containing detection
                    annotation['segmentation']

                    # Write the class_index
                    with open(label_file_path, 'a') as label_file:  # Append to the file
                        label_file.write(f"{class_index}")
                        
                        # Extract the segmentation polygons and normalize them
                        segmentation = annotation['segmentation']

                        segmentation.sort(key=len)
                        try:
                            polygon_big = segmentation[-1]

                            normalized_polygon_big = [
                                (float(x) / image_width, float(y) / image_height) for x, y in zip(polygon_big[::2], polygon_big[1::2])
                            ]
                            normalized_polygon_big = [coord for point in normalized_polygon_big for coord in point]
                            label_file.write(" " + " ".join(map(str, normalized_polygon_big)))

                            if len(segmentation) > 1:
                                polygon_small = segmentation[-2]
                                normalized_polygon_small = [
                                (float(x) / image_width, float(y) / image_height) for x, y in zip(polygon_small[::2], polygon_small[1::2])
                                ]
                                normalized_polygon_small = [coord for point in normalized_polygon_small for coord in point]
                                # print(normalized_polygon_small)
                                label_file.write(" " + " ".join(map(str, normalized_polygon_small)))
                            
                            label_file.write("\n")
                        except:
                            print(f'coco_json_path: {coco_json_path}')
                            print(f'segmentation: {segmentation} ({type(segmentation)})')
                            print(f'img_width: {image_width} ({type(image_width)})')

            pbar.update()

        pbar.colour = "green"
        pbar.refresh() 

def split_dataset(yolo_dir):
    images_path     = os.path.join(yolo_dir, 'images', 'train')       # Your current training images directory
    labels_path     = os.path.join(yolo_dir, 'labels', 'train')       # Your current training labels directory
    val_images_path = os.path.join(yolo_dir, 'images', 'val')
    val_labels_path = os.path.join(yolo_dir, 'labels', 'val')

    if not os.path.exists(yolo_dir):
        print(f"Error! Could not find {yolo_dir}")
        exit(-1)

    # Create folders if they don't exist (optimized)
    # Let's rename the data to a temp folder to avoid move each image:
    temp_dir = os.path.join(yolo_dir, "temp")                       # Temp folder for swapping paths.
    os.renames(old=os.path.join(yolo_dir, 'images'), new=temp_dir)  # Changing data path to temp folder
    os.renames(old=temp_dir, new=images_path)                       # Changing data path to dest folder

    # Creating the rest of the folders
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    # List all images
    images = os.listdir(images_path)

    # Shuffle and split into train and val (80% train, 20% val)
    random.shuffle(images)
    train_images = images[:int(0.8 * len(images))]
    val_images = images[int(0.8 * len(images)):]

    # Move validation images and corresponding label files
    for img in val_images:
        img_path = os.path.join(images_path, img)
        label_path = os.path.join(labels_path, img.replace('.png', '.txt'))

        # Move to validation directories
        val_label_name = os.path.join(val_labels_path, img.replace('.png', '.txt'))
        shutil.move(label_path, val_label_name)
        shutil.move(img_path, os.path.join(val_images_path, img))

def clean_unmatched_imgs(yolo_dir):
    image_folder    = os.path.join(yolo_dir, "images")        # Directory containing the images
    label_folder    = os.path.join(yolo_dir, "labels/train")        # Directory containing YOLO labels

    if not os.path.exists(yolo_dir):
        print(f"Error! Could not find {yolo_dir}")
        exit(-1)

    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Get list of image file names (without extensions)
    image_files = set(
        os.path.splitext(f)[0] 
        for f in os.listdir(image_folder) 
        if os.path.splitext(f)[1].lower() in image_extensions
    )

    # Get list of label file names (without extensions)
    label_files = set(
        os.path.splitext(f)[0] 
        for f in os.listdir(label_folder) 
        if f.endswith('.txt')  # Assuming labels are in YOLO format
    )
    
    # Find unmatched images (images that do not have corresponding labels)
    unmatched_images = image_files - label_files
    print(f"\t-Images removed ({len(unmatched_images)}): {list(unmatched_images)}")

    # Delete unmatched image files
    for image in unmatched_images:
        for ext in image_extensions:  # Try deleting all supported image extensions
            image_path = os.path.join(image_folder, f"{image}{ext}")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted unmatched image: {image_path}")

def visualize_yolo(yolo_dir:str, annotation_type:str):
    """
    Converts coco annotations into yolo

    Parameters:
        yolo_dir (str)
        annotation_type (str): Type of annotation -> "object_detection" or "instance_segmentation".
    """

    images_dir      = os.path.join(yolo_dir, "images/train")        # Directory containing the images
    labels_dir      = os.path.join(yolo_dir, "labels/train")        # Directory containing YOLO labels

    if not os.path.exists(yolo_dir):
        print(f"Error! Could not find {yolo_dir}")
        exit(-1)

    # Loop through each image:
    if annotation_type == "object_detection":
        filenames = [f for f in os.listdir(images_dir) if f.endswith(".png")]
        for name in filenames:
            label_file = name.replace('png','txt')
            image_path = os.path.join(images_dir, name)
            label_path = os.path.join(labels_dir, label_file)

            if os.path.exists(label_path):
                print(f"Displaying: {name}")
                _plot_image_with_bboxes(image_path, label_path)

            else:
                print(f"Label file not found for {name}, skipping.")

    elif annotation_type == "instance_detection":
        label_files = sorted(os.listdir(labels_dir), reverse=True)
        
        for num, name in enumerate(label_files):
            if os.path.exists(label_path):
                print(f"Displaying: {name}")
                visualize_segmentation(images_dir, labels_dir, num)
            else:
                print(f"Label file not found for {name}, skipping.")

# Function to plot image with bounding boxes
def _plot_image_with_bboxes(image_path, label_path, total_models=4):
    # Read the image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Read the YOLO label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Function to denormalize YOLO bounding boxes
    def denormalize_bbox(img_width, img_height, x_center, y_center, width, height):
        x_center *= img_width
        y_center *= img_height
        width    *= img_width
        height   *= img_height

        # Convert to x_min, y_min, x_max, y_max
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        return x_min, y_min, x_max, y_max

    # Loop through each bounding box in the label file
    
    for line in lines:
    
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        x_min, y_min, x_max, y_max = denormalize_bbox(img_width, img_height, x_center, y_center, width, height)
        
        # Get color based on class id:
        index = int(np.interp(class_id, [0, total_models], [0, 255]))
        temp = np.array([[index]], dtype=np.uint8)
        colored_img = cv2.applyColorMap(temp, cv2.COLORMAP_JET)
        color = tuple(int(c) for c in colored_img[0, 0]) # Extract the color as a tuple (B, G, R)

        print(color)

        # Draw the bounding box on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)  # Green box for visualization
        cv2.putText(img, f'Class: {int(class_id)}', (x_min, y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
     # Convert BGR to RGB for displaying in matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_segmentation(image, label_data):
    """
    Draw segmentation mask on the image based on YOLO format labels.
    
    :param image: Original image as a numpy array.
    :param label_data: List of label strings.
    :return: Image with segmentation mask overlayed.
    """
    height, width, _ = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)

    for line in label_data:
        parts = line.strip().split()
        class_id = int(parts[0])  # First value is the class ID
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        coords[:, 0] *= width  # Denormalize X
        coords[:, 1] *= height  # Denormalize Y
        coords = coords.astype(np.int32)

        # Draw polygon for the current label
        cv2.polylines(mask, [coords], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.fillPoly(mask, [coords], color=(0, 255, 0))

    # Blend the mask with the original image
    overlay = cv2.addWeighted(image, 0.8, mask, 0.5, 0)
    return overlay

def visualize_segmentation(image_dir, label_dir, num):
    """
    Visualize segmentation masks overlayed on images.
    
    :param image_dir: Path to the directory containing images.
    :param label_dir: Path to the directory containing YOLO labels.
    """
    image_files = sorted(os.listdir(image_dir), reverse=True)
    label_files = sorted(os.listdir(label_dir), reverse=True)

    for label_file in label_files[:num]:
        img_file = label_file.replace('.txt', '.png')
        if img_file in image_files:

            # Load image
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            rgb = image[:,:,:3]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Load labels
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as f:
                label_data = f.readlines()

            # Overlay segmentation
            overlayed_image = draw_segmentation(rgb, label_data)

            # Display the image (press 'Q' for next img)
            plt.figure(figsize=(10, 10))
            plt.imshow(overlayed_image)
            plt.axis('off')
            plt.title(f"Segmentation: {img_file}")
            plt.show()