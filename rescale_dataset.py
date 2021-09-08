# code refer to https://github.com/ZhihengCV/Bayesian-Crowd-Counting/blob/c81c45d50405c36cdcd339006876a04faa742373/preprocess_dataset.py
from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
import glob
import cv2
import argparse

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def generate_data(im_path, dataset):
    im = Image.open(im_path)
    im_w, im_h = im.size

    if dataset == 'ucf-qnrf':
        mat_path = im_path.replace('.jpg', '_ann.mat')
        points = loadmat(mat_path)['annPoints'].astype(np.float32)
    elif dataset == 'st-a':
        mat_path = im_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')
        points = loadmat(mat_path)['image_info'][0,0][0,0][0]

    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='rescale')
    parser.add_argument(
        '--data-dir',
        default='',
        help='dataset directory ucf-qnrf: /UCF-QNRF_ECCV18, shanghai-tech-part*: /ShanghaiTech/part_*'
    )
    parser.add_argument(
        '--dataset',
        default='',
        help='dataset name [ucf-qnrf, st-a]'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = os.path.join(os.path.dirname(args.data_dir), 'rescale-' + os.path.basename(args.data_dir))
    dataset = args.dataset
    min_size = 512
    max_size = 2048

    if os.path.isdir(save_dir):
        print('already exists')
        
    else:
        if dataset == 'ucf-qnrf':
            for phase in ['Train', 'Test']:
                sub_dir = os.path.join(args.data_dir, phase)
                sub_save_dir = os.path.join(save_dir, phase)

                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                
                for i, img_path in enumerate(glob.glob(os.path.join(sub_dir,'*.jpg'))):
                    name = os.path.basename(img_path)
                    print(img_path)

                    img, points = generate_data(img_path, dataset=dataset)

                    im_save_path = os.path.join(sub_save_dir, name)
                    img.save(im_save_path)

                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    np.save(gd_save_path, points)


        elif dataset == 'st-a':
            for phase in ['train_data', 'test_data']:
                sub_dir = os.path.join(args.data_dir, phase, 'images')
                sub_save_dir = os.path.join(save_dir, phase, 'images')

                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                    os.makedirs(sub_save_dir.replace('images','ground_truth'))
                
                for i, img_path in enumerate(glob.glob(os.path.join(sub_dir,'*.jpg'))):
                    name = os.path.basename(img_path)
                    print(img_path)

                    img, points = generate_data(img_path, dataset=dataset)

                    im_save_path = os.path.join(sub_save_dir, name)
                    img.save(im_save_path)

                    gd_save_path = im_save_path.replace('jpg', 'npy').replace('images','ground_truth')
                    np.save(gd_save_path, points)