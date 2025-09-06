import os, json, random, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# --------- CONFIG ---------
ANNOTATIONS = "/Users/saakar/Downloads/playground/aannotations.json"   # your JSON with folder-based lesion labels
IMAGES_ROOT = "/Users/saakar/Downloads/playground/ORAL"                 # root that contains train/validation/<class>/...
NUM_SAMPLES = 4
RANDOM_SEED = 42
# --------------------------

def find_image_path_by_basename(root, basename):
    """Search recursively for the first file named `basename` under `root`."""
    for dp, _, files in os.walk(root):
        if basename in files:
            return os.path.join(dp, basename)
    return None

def iter_polygons(segmentation):
    """Yield Nx2 arrays of polygon points from COCO 'segmentation'."""
    if not segmentation:
        return
    # handle flat list or list-of-lists
    polys = segmentation if (isinstance(segmentation[0], list)) else [segmentation]
    for seg in polys:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        yield pts

def main():
    random.seed(RANDOM_SEED)

    with open(ANNOTATIONS, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}

    # map image_id -> list of annotations
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # choose random images that actually exist on disk
    present = []
    for img in images:
        p = find_image_path_by_basename(IMAGES_ROOT, img["file_name"])
        if p:
            present.append((img, p))
    if not present:
        print("No images found on disk under:", IMAGES_ROOT)
        return

    sample = random.sample(present, min(NUM_SAMPLES, len(present)))
    print(f"Visualizing {len(sample)} images...")

    # simple color map per category id (consistent, but you can customize)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['r','g','b','m','c','y'])
    cat_ids_sorted = sorted(set(a["category_id"] for a in annotations))
    cat_to_color = {cid: color_cycle[i % len(color_cycle)] for i, cid in enumerate(cat_ids_sorted)}

    for img_meta, img_path in sample:
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to read:", img_path)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.set_title(f"{img_meta['file_name']}  (image_id={img_meta['id']})")
        ax.axis('off')

        anns = ann_by_image.get(img_meta["id"], [])
        for ann in anns:
            cat_id = ann.get("category_id")
            cat_name = categories.get(cat_id, f"id:{cat_id}")
            color = cat_to_color.get(cat_id, 'r')

            # draw bbox
            if "bbox" in ann and ann["bbox"]:
                x, y, w, h = ann["bbox"]
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x, max(0, y - 5), cat_name,
                        fontsize=10, color='white',
                        bbox=dict(facecolor='black', alpha=0.6, pad=2))

            # draw segmentation polygons (outline + translucent fill)
            seg = ann.get("segmentation")
            if seg:
                for pts in iter_polygons(seg):
                    poly = Polygon(pts, closed=True, linewidth=2, edgecolor=color, facecolor=color, alpha=0.25)
                    ax.add_patch(poly)

        # optional legend (one entry per category present in this image)
        present_cats = sorted(set(a["category_id"] for a in anns))
        handles = [patches.Patch(edgecolor=cat_to_color[c], facecolor=cat_to_color[c], alpha=0.25, label=categories.get(c, str(c)))
                   for c in present_cats]
        if handles:
            ax.legend(handles=handles, loc='lower right')

        plt.show()

if __name__ == "__main__":
    main()
