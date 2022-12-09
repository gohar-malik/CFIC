from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import os
from os.path import join
from tqdm import tqdm

import imlib as im
import numpy as np
import tensorflow as tf
import tflib as tl

import models

def load_img(path, img_resize, sess):
    offset_h = 26
    offset_w = 3
    img_size = 170

    img = tf.read_file(path)
    img = tf.image.decode_png(img, 3)
    # img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
    
    # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
    # or
    img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
    img = tf.clip_by_value(img, 0, 255) / 127.5 - 1

    img = sess.run(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == "__main__":
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default=128, help='experiment_name')
    parser.add_argument('--gpu', type=str, default='all', help='gpu')
    parser.add_argument('--root_dir', type=str, default='/ssd_data/gohar/coco')
    parser.add_argument('--orig_att', type=str, default='dark')
    args_ = parser.parse_args()
    with open('./output/%s/setting.txt' % args_.experiment_name) as f:
        args = json.load(f)

    if args_.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args_.gpu

    n_att = len(args['atts'])

    ################################# tf graphs ###############################
    sess = tl.session()
    # model
    Genc = partial(models.Genc, dim=args['enc_dim'], n_layers=args['enc_layers'], multi_inputs=args['multi_inputs'])
    Gdec = partial(models.Gdec, dim=args['dec_dim'], n_layers=args['dec_layers'], shortcut_layers=args['shortcut_layers'],
                   inject_layers=args['inject_layers'], one_more_conv=args['one_more_conv'])
    Gstu = partial(models.Gstu, dim=args['stu_dim'], n_layers=args['stu_layers'], inject_layers=args['stu_inject_layers'],
                   kernel_size=args['stu_kernel_size'], norm=args['stu_norm'], pass_state=args['stu_state'])
    # input
    xa_sample = tf.placeholder(tf.float32, shape=[None, args['img_size'], args['img_size'], 3])
    _b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
    raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
    # sample
    test_label = _b_sample - raw_b_sample if args['label'] == 'diff' else _b_sample
    if args['use_stu']:
        x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                             test_label, is_training=False), test_label, is_training=False)
    else:
        x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)
    # init
    ckpt_dir = './output/%s/checkpoints' % args_.experiment_name
    tl.load_checkpoint(ckpt_dir, sess)

    ############################### data settings ###############################
    with open(f"val2014_faces_{args_.orig_att}.txt", "r") as f:
        face_file_list = f.read().split()

    face_dir = join(args_.root_dir, "val2014_faces")
    save_dir = join(args_.root_dir, "stgan", f"val2014_faces_{args_.orig_att}")
    os.makedirs(save_dir, exist_ok=True)

    # att input
    att_idx_dict = {"male":     [7,-1.5],
                    "female":   [7,0.5],
                    "dark":     [11,0.5]}
    _b_sample_ipt = np.zeros((1,n_att), dtype=np.float32)
    _b_sample_ipt[0, att_idx_dict[args_.orig_att][0]] = att_idx_dict[args_.orig_att][1]

    raw_a_sample_ipt = np.zeros((1,n_att), dtype=np.float32) + 0.5
    raw_a_sample_ipt[0, att_idx_dict[args_.orig_att][0]] = -0.5

    ############################## inference loop #################################
    for i, file in enumerate(tqdm(face_file_list)):
        # print(file)
        path = join(face_dir, file.replace("png", "jpg"))
        img = load_img(path, args["img_size"], sess)

        output = sess.run(x_sample, feed_dict={xa_sample: img,
                                                _b_sample: _b_sample_ipt,
                                                raw_b_sample: raw_a_sample_ipt})
        
        im.imwrite(output.squeeze(0), join(save_dir, file))
        # if i >= 10:
        #     break



