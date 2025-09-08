import json
import os
from collections import defaultdict
from PIL import Image

# ---------------------- CONFIG ----------------------
data_dir = "data/ORAL"
images_dir = os.path.join(data_dir, "images")
labels_dir = os.path.join(data_dir, "labels")
annotation_file = os.path.join(data_dir, "Annotation.json")

# Folder name → YOLO class ID mapping (4 classes)
folder_class_map = {
    "Healthy": 0,   # Healthy / Normal
    "Benign": 1,    # Benign
    "OPMD": 2,      # Oral Potentially Malignant Disorder
    "OCA": 3        # Oral Cancer
}
# -----------------------------------------------------

# Load annotation.json
with open(annotation_file) as f:
    coco = json.load(f)

# Create a dict: image_id -> image info
images_info = {img['id']: img for img in coco['images']}

# Group annotations by image_id
ann_dict = defaultdict(list)
for ann in coco['annotations']:
    ann_dict[ann['image_id']].append(ann)

# Helper function to convert COCO bbox [x, y, w, h] → YOLO format
def convert_bbox(size, bbox):
    img_w, img_h = size
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

# Process train and val splits
for split in ["train", "val"]:
    split_dir = os.path.join(images_dir, split)
    label_split_dir = os.path.join(labels_dir, split)
    os.makedirs(label_split_dir, exist_ok=True)

    # Traverse class subfolders
    for folder_name in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name not in folder_class_map:
            print(f"Warning: {folder_name} not in class map. Skipping.")
            continue
        class_id = folder_class_map[folder_name]

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            file_path = os.path.join(folder_path, file_name)

            # Get image dimensions
            with Image.open(file_path) as img:
                width, height = img.size

            # Find image_id from JSON
            img_id = None
            for id_, info in images_info.items():
                if info['file_name'] == file_name:
                    img_id = id_
                    break
            if img_id is None:
                print(f"Warning: {file_name} not found in annotation.json. Skipping.")
                continue

            # Prepare YOLO label lines
            lines = []
            for ann in ann_dict.get(img_id, []):
                bbox = ann['bbox']
                x_c, y_c, w_n, h_n = convert_bbox((width, height), bbox)
                lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

            # Write label file
            label_file_path = os.path.join(label_split_dir, os.path.splitext(file_name)[0] + ".txt")
            with open(label_file_path, "w") as f:
                f.write("\n".join(lines))

print("✅ Conversion complete! YOLO label files are ready in labels/train/ and labels/val/")
