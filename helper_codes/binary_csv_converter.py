import os
import csv

# Paths to your YOLO dataset
images_dir = "data/ORAL/images"
labels_dir = "data/ORAL/labels"

# Output CSV path
output_csv = "normal_abnormal_labels.csv"

# Class IDs
NORMAL_CLASS_ID = "0"  # Healthy
ABNORMAL_CLASS_IDS = {"1", "2", "3"}  # Benign, OPMD, OCA

# Prepare CSV rows
rows = [["image_path", "label"]]

for split in ["train", "val"]:
    img_folder = os.path.join(images_dir, split)
    lbl_folder = os.path.join(labels_dir, split)
    
    for img_file in os.listdir(img_folder):
        if not img_file.lower().endswith((".jpg", ".png")):
            continue
        
        # Corresponding label file
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(lbl_folder, lbl_file)
        
        # Default to Normal if something goes wrong
        image_label = "Normal"
        
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                # If any object is not Healthy, mark as Abnormal
                for line in lines:
                    class_id = line.strip().split()[0]
                    if class_id in ABNORMAL_CLASS_IDS:
                        image_label = "Abnormal"
                        break
        
        # Add to CSV
        rows.append([os.path.join(img_folder, img_file), image_label])

# Write CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Normal/Abnormal CSV saved to: {output_csv}")
