import cv2
import os
import matplotlib.pyplot as plt

# --- Configuration ---
image_dir = "/media/vrt/D/Datasets/Outputs/BASELINE_4k_Material_Randomization_Corrected_v4/yolo/images/train"
label_dir = "/media/vrt/D/Datasets/Outputs/BASELINE_4k_Material_Randomization_Corrected_v4/yolo/labels/train"
output_dir = "/home/vrt/Projects/Tom/Outputs/BASELINE_4k_Material_Randomization/yolo/Output"
show_images = True
save_output = False
label_digits = 6
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Map class IDs -> names (index 0..31)
class_names = [
    "Ball_big",
    "Ball_small",
    "Clamp_big",
    "Clamp_small",
    "Collar_big",
    "Collar_small",
    "FestoI",
    "FestoT",
    "FestoV",
    "FestoX",
    "FestoY",
    "Festo_torch",
    "Hexagon",
    "Nut",
    "O_ring_big",
    "O_ring_medium",
    "O_ring_small",
    "Plastic_washer_big",
    "Plastic_washer_small",
    "Screw_flat",
    "Screw_large",
    "Screw_small",
    "Screw_threaded",
    "Silencer_big",
    "Silencer_small",
    "Socket_screw",
    "Spring",
    "Steel_bar_big",
    "Steel_bar_small",
    "Washer_big",
    "Washer_small",
    "Wing",
]

# --- Ensure output directory exists ---
if save_output and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# --- Process each image ---
for filename in sorted(os.listdir(image_dir)):
    if not filename.endswith(".png"):
        continue

    base_name = os.path.splitext(filename)[0]
    try:
        index = int(base_name)
    except ValueError:
        print(f"Skipping non-integer image filename: {filename}")
        continue

    padded_label_name = f"{index:0{label_digits}}.txt"
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, padded_label_name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    image_height, image_width = image.shape[:2]

    if not os.path.exists(label_path):
        print(f"No label file for {filename} (expected: {padded_label_name})")
        continue

    with open(label_path, 'r') as f:
        label_lines = f.readlines()

    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # Convert YOLO normalized format to pixel coordinates
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        color = colors[class_id % len(colors)]
        # Use class name if valid; fall back to numeric ID if out of range
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if save_output:
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

    if show_images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(filename)
        plt.axis("off")
        plt.show()
