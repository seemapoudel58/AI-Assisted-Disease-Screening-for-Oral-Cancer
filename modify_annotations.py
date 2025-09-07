import json
import os
import argparse
import shutil

def find_image_in_folders(image_filename, data_root):
   
    classes = ['Healthy', 'Benign', 'OPMD', 'OCA']
    splits = ['Training', 'Validation']
    
    for split in splits:
        for class_name in classes:
            class_path = os.path.join(data_root, split, class_name)
            if os.path.exists(class_path):
                image_path = os.path.join(class_path, image_filename)
                if os.path.exists(image_path):
                    return class_name
    return None

def count_oral_cavity_boxes(annotations, image_id):
    
    count = 0
    for ann in annotations:
        if ann['image_id'] == image_id and ann['category_id'] == 2: 
            count += 1
    return count

def modify_annotations(annotation_file, data_root, output_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    backup_file = annotation_file.replace('.json', '_backup.json')
    shutil.copy2(annotation_file, backup_file)
    print(f" Backup created: {backup_file}")
    
    available_images = set()
    image_to_class = {}
    
    for img in data['images']:
        filename = img['file_name']
        class_name = find_image_in_folders(filename, data_root)
        if class_name:
            available_images.add(img['id'])
            image_to_class[img['id']] = class_name
            print(f"  Found: {filename} -> {class_name}")
        else:
            print(f"  Missing: {filename}")
    
    print(f" Found {len(available_images)} available images out of {len(data['images'])}")
    
    print(" Filtering images and annotations...")
    
    filtered_images = [img for img in data['images'] if img['id'] in available_images]
    
    filtered_annotations = []
    updated_categories = []
    
    new_category_mapping = {
        'Healthy': 0,
        'Benign': 1,
        'OPMD': 2,
        'OCA': 3
    }
    
    for class_name, class_id in new_category_mapping.items():
        updated_categories.append({
            'id': class_id,
            'name': class_name,
            'supercategory': 'disease'
        })
    
    for img in filtered_images:
        img_id = img['id']
        class_name = image_to_class[img_id]
        
        oral_cavity_count = count_oral_cavity_boxes(data['annotations'], img_id)
        
        if oral_cavity_count == 1 and len([ann for ann in data['annotations'] if ann['image_id'] == img_id]) == 1:
            final_class = 'Healthy'
            print(f"  {img['file_name']}: Single Oral Cavity box -> Healthy")
        else:
            final_class = class_name
            print(f"  {img['file_name']}: Multiple boxes or other -> {final_class}")
        
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                new_ann = ann.copy()
                new_ann['category_id'] = new_category_mapping[final_class]
                filtered_annotations.append(new_ann)
    
    modified_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': updated_categories
    }
    
    print(f" Saving modified annotation file to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(modified_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Modify annotation file for oral cancer detection')
    parser.add_argument('--annotation_file', required=True, help='Path to original annotation file')
    parser.add_argument('--data_root', required=True, help='Root directory containing Training and Validation folders')
    parser.add_argument('--output_file', required=True, help='Path to save modified annotation file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotation_file):
        print(f" Error: Annotation file not found: {args.annotation_file}")
        return
    
    if not os.path.exists(args.data_root):
        print(f" Error: Data root directory not found: {args.data_root}")
        return
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        modify_annotations(args.annotation_file, args.data_root, args.output_file)
           
    except Exception as e:
        print(f" Error during modification: {str(e)}")

if __name__ == "__main__":
    main()
