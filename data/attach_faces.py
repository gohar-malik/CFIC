import os
from os.path import join
import argparse
import cv2
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description='attach_faces')
parser.add_argument('--root_dir', default='/ssd_data/gohar/coco',
                    type=str, help='Root dir where val2014 and annotations dir are located')

args = parser.parse_args()

def sqaure_enlarge_bbox(bbox, im_shape, pct=0.50):
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin

    cx = xmin+w//2
    cy = ymin+h//2
    cr  = max(w,h)//2

    r = cr+(cr*pct)

    xmin = max(int(round(cx-r)), 0)
    xmax = min(int(round(cx+r)), im_shape[1])
    ymin = max(int(round(cy-r)), 0)
    ymax = min(int(round(cy+r)), im_shape[0])

    return (xmin, ymin, xmax, ymax)

def resize_face(face, bbox):
    face_w = face.shape[1]
    face_h = face.shape[0]
    xmin, ymin, xmax, ymax= bbox
    w = xmax - xmin
    h = ymax - ymin
    if w==h:
        return cv2.resize(face, (w,h), cv2.INTER_AREA), False
    
    if h>w:
        diff = h-w
        diff = round(128/w*diff)
        face = face[:, diff//2:face_h-(diff//2)]
    elif w>h:
        diff = w-h
        diff = round(128/w*diff)
        face = face[diff//2:face_w-(diff//2), :]

    # assert face.shape[0]/face.shape[1] == h/w

    return cv2.resize(face, (w,h), cv2.INTER_AREA), True



if __name__ == "__main__":
    mod_faces_dir = join(args.root_dir, "stgan/val2014_faces_dark")
    im_dir = join(args.root_dir, "val2014_people")
    bbox_dir = join(args.root_dir, "val2014_faces_txt")

    att = mod_faces_dir.split("/")[-1].split("_")[-1]
    orig_results_dir = join(args.root_dir, f"stgan/val2014_{att}_orig")
    mod_results_dir = join(args.root_dir, f"stgan/val2014_{att}_mod")
    os.makedirs(orig_results_dir, exist_ok=True)
    os.makedirs(mod_results_dir, exist_ok=True)

    for face_file in tqdm(os.listdir(mod_faces_dir)):
        face = cv2.imread(join(mod_faces_dir, face_file))
        name = "_".join(face_file.split(".")[0].split("_")[0:-1])
        det_i = int(face_file.split(".")[0].split("_")[-1])
        # print(name)

        im = cv2.imread(join(im_dir, f"{name}.jpg"))

        bbox_file = f"{name}.txt"
        with open(join(bbox_dir, bbox_file), "r") as f:
            det_result = f.read().split("\n")
            num_faces = int(det_result[1])
            dets = det_result[2:]
        det = dets[det_i]
        x, y, w, h = det.split()[0:4]

        bbox = (int(x), int(y), int(x)+int(w), int(y)+int(h))
        bbox = sqaure_enlarge_bbox(bbox, im.shape)

        face, padding = resize_face(face, bbox)
        xmin, ymin, xmax, ymax= bbox
        im[ymin:ymax, xmin:xmax, :] = face

        cv2.imwrite(join(mod_results_dir, f"{name}.jpg"), im)
        shutil.copy(join(im_dir, f"{name}.jpg"), join(orig_results_dir, f"{name}.jpg"))
        # if not padding:
        #     print(name)
        #     break


