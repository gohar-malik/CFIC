import os
from os.path import join
import argparse
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='filter_faces')
parser.add_argument('--root_dir', default='/ssd_data/gohar/coco',
                    type=str, help='Root dir where val2014 and annotations dir are located')

args = parser.parse_args()


def resize_image(img, size=(128,128)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

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

if __name__ == "__main__":

    txt_dir = os.path.join(args.root_dir, "val2014_faces_txt")
    image_dir = os.path.join(args.root_dir, "val2014_people")
    face_dir = os.path.join(args.root_dir, "val2014_faces")
    # face_multiple_dir = "/ssd_data/gohar/coco/val2014_facesmultiple"
    os.makedirs(face_dir, exist_ok=True)
    # os.makedirs(face_multiple_dir, exist_ok= True)

    single_face_ims = 0
    multi_face_ims = 0
    for txt_file in tqdm(os.listdir(txt_dir)):
        name = txt_file.split(".")[0]
        im = cv2.imread(join(image_dir, f"{name}.jpg"))
        im_area = im.shape[0] * im.shape[1]

        with open(join(txt_dir, txt_file), "r") as f:
            det_result = f.read().split("\n")
            num_faces = int(det_result[1])
            dets = det_result[2:]
        
        valid_face_bboxes = []
        valid_face_indices = []
        for i in range(num_faces):
            det = dets[i]
            x, y, w, h, conf, le_x, le_y, re_x, re_y, n_x, n_y, ll_x, ll_y, rl_x, rl_y = det.split()

            bbox_area = (int(w) * int(h)) / im_area
            keypoint_area = ((float(re_x) - float(le_x)) * (float(rl_y) - float(le_y))) / im_area

            if float(conf) > 0.99 and bbox_area > 0.01:
                valid_face_bboxes.append((int(x), int(y), int(x)+int(w), int(y)+int(h)))
                valid_face_indices.append(i)
        
        if len(valid_face_bboxes) == 1:
            (x, y, x1, y1) = sqaure_enlarge_bbox(valid_face_bboxes[0], im.shape)
            im_cropped = im[y:y1, x:x1]
            im_cropped = resize_image(im_cropped)
            cv2.imwrite(join(face_dir, f"{name}_{valid_face_indices[0]}.jpg"), im_cropped)
            single_face_ims += 1
        # elif len(valid_face_bboxes) > 1:
        #     for i, bbox in enumerate(valid_face_bboxes):
        #         (x, y, x1, y1) = sqaure_enlarge_bbox(valid_face_bboxes[i], im.shape)
        #         im_cropped = im[y:y1, x:x1]
        #         im_cropped = resize_image(im_cropped)
        #         cv2.imwrite(join(face_multiple_dir, f"{name}_{valid_face_indices[i]}.jpg"), im_cropped)
        #     multi_face_ims += 1
            # break

    print(f"Face Images: {single_face_ims}")
    # print(f"Multiple Face Images: {multi_face_ims}")