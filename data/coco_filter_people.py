import os
import json
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='coco_filter_people')
parser.add_argument('--root_dir', default='/ssd_data/gohar/coco',
                    type=str, help='Root dir where val2014 and annotations dir are located')

args = parser.parse_args()

if __name__ == "__main__":


    coco_dir = args.root_dir
    annotations_dir = os.path.join(coco_dir, "annotations")
    per_val = "person_keypoints_val2014.json"

    val_dir = os.path.join(coco_dir, "val2014")
    val_people_dir = os.path.join(coco_dir, "val2014_people")
    os.makedirs(val_people_dir, exist_ok=True)

    with open(os.path.join(annotations_dir, per_val), 'r') as f:
        per_val_json = json.load(f)

    image_ids = []
    for ann in tqdm(per_val_json["annotations"]):
        image_ids.append(ann["image_id"])
    image_ids = set(image_ids)

    count = 0
    for im in tqdm(per_val_json["images"]):
        if im["id"] in image_ids:
            file_name = im["file_name"]
            shutil.copy(os.path.join(val_dir, file_name), val_people_dir)
            count += 1
    
    print(f"Images with people: {count}")
    print(f"Location: {val_people_dir}")