"""ground truth の密度マップを npy ファイルとして指定したディレクトリに保存
"""
import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加
import glob
import argparse

from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

from datasets.data_util import (
    judge_dataset,
    load_image,
    load_gt,
    mapping_gt,
    load_bbox,
    bbox_to_point
)

def parse_args():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--data-dir', type=str, default='', help='data directry')
    parser.add_argument('--sigma', type=int, default='15', help='sigma: gaussian filter param')
    args = parser.parse_args()
    
    return args

def create_density(gt_map, sigma):
    ## ガウシアンフィルタリング
    gt_map = np.array(gt_map)
    gt_density = gaussian_filter(gt_map, sigma)
    gt_density = Image.fromarray(gt_density)

    return gt_density

if __name__ == '__main__':
    args = parse_args()

    dataset = judge_dataset(args.data_dir)

    # train_im_dir = os.path.join(args.data_dir, 'train_data/train_img')
    # train_gt_dir = os.path.join(args.data_dir, 'train_data/train_gt')
    
    # train_im_list = glob.glob(os.path.join(train_im_dir, '*.png'))
    # train_gt_list = glob.glob(os.path.join(train_gt_dir, '*.mat'))

    # for gt_path in tqdm(train_gt_list, ncols=60):

    #     im_path = gt_path.replace('.mat', '.png').replace('GT', 'IMG').replace('train_gt', 'train_img')
    #     im = load_image(im_path)
    #     im_size = im.size

    #     points = load_gt(gt_path)
    #     head_map = mapping_gt(im_size, points)
    #     density = create_density(head_map, args.sigma)

    #     save_foldername = 'train_den_' + str(args.sigma)
    #     save_path = os.path.dirname(os.path.join(gt_path))
    #     save_path = save_path.replace('train_gt', save_foldername)
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)

    #     np_gt_path = os.path.join(save_path, os.path.basename(gt_path.replace('.mat', '.npy')))
    #     np.save(np_gt_path, density)

    train_im_dir = os.path.join(args.data_dir, 'test_data/test_img')
    train_gt_dir = os.path.join(args.data_dir, 'test_data/test_gt_np')
    
    train_im_list = glob.glob(os.path.join(train_im_dir, '*.png'))
    train_gt_list = glob.glob(os.path.join(train_gt_dir, '*.npy'))

    for gt_path in tqdm(train_gt_list, ncols=60):

        im_path = gt_path.replace('.npy', '.png').replace('GT', 'IMG').replace('test_gt_np', 'test_img')
        im = load_image(im_path)
        im_size = im.size

        points = load_gt(gt_path)
        head_map = mapping_gt(im_size, points)
        density = create_density(head_map, args.sigma)

        save_foldername = 'test_den_' + str(args.sigma)
        save_path = os.path.dirname(os.path.join(gt_path))
        save_path = save_path.replace('test_gt', save_foldername)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np_gt_path = os.path.join(save_path, os.path.basename(gt_path))
        np.save(np_gt_path, density)

    # test_im_dir = os.path.join(args.data_dir, 'test_data/test_img')
    # test_gt_dir = os.path.join(args.data_dir, 'test_data/test_bbox_anno')
    
    # test_im_list = glob.glob(os.path.join(test_im_dir, '*.png'))
    # test_gt_list = glob.glob(os.path.join(test_gt_dir, '*.mat'))

    # for gt_path in tqdm(test_gt_list, ncols=60):
        
    #     bbox = load_bbox(gt_path)
    #     gt = bbox_to_point(bbox)

    #     save_foldername = 'test_gt_np'
    #     save_path = os.path.dirname(os.path.join(gt_path))
    #     save_path = save_path.replace('test_bbox_anno', save_foldername)
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)

    #     np_gt_path = os.path.join(save_path, os.path.basename(gt_path.replace('.mat', '.npy').replace('BBOX', 'GT')))
    #     np.save(np_gt_path, gt)