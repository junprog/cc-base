import os
import glob
import argparse

from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from datasets.data_util import *

def parse_args():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--data-dir', type=str, default='', help='data directry')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()

    test_im_dir = os.path.join(args.data_dir, 'test_data/test_img')
    test_gt_dir = os.path.join(args.data_dir, 'test_data/test_bbox_anno')
    
    test_im_list = glob.glob(os.path.join(test_im_dir, '*.png'))
    test_gt_list = glob.glob(os.path.join(test_gt_dir, '*.mat'))

    for gt_path in tqdm(test_gt_list, ncols=60):
        
        bbox = load_bbox(gt_path)
        gt = bbox_to_point(bbox)

        save_foldername = 'test_gt_np'
        save_path = os.path.dirname(os.path.join(gt_path))
        save_path = save_path.replace('test_bbox_anno', save_foldername)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np_gt_path = os.path.join(save_path, os.path.basename(gt_path.replace('.mat', '.npy').replace('BBOX', 'GT')))
        #print(np_gt_path)
        np.save(np_gt_path, gt)