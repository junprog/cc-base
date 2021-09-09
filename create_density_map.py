"""ground truth の密度マップを npy ファイルとして指定したディレクトリに保存
"""
## TODO
# ShanghaiTech partA: done
# ShanghaiTech partB: done
# UCF-QNRF: done
# ShanghaiTechRGBD: done
# RGBT-CC

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
    output_dir_sufix = '_den_{}'.format(args.sigma)

    phases = ['train', 'test']
    for phase in phases:
        image_dirname, image_format = dirname_parser(dataset, phase, 'image')
        gt_dirname, gt_format = dirname_parser(dataset, phase, 'gt')

        phase_dirname = os.path.join(args.data_dir, image_dirname)
        image_path_list = glob.glob(os.path.join(phase_dirname, image_format))

        if dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b':
            phase_dirname = os.path.dirname(phase_dirname)

        for im_path in tqdm(image_path_list, ncols=60):
            im = load_image(im_path)
            im_size = im.size

            gt_path = create_gt_path(im_path, dataset, phase)
            points = load_gt(gt_path)
            head_map = mapping_gt(im_size, points)
            density = create_density(head_map, args.sigma)

            if dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b':
                save_foldername = 'den_' + str(args.sigma)
            elif dataset == 'shanghai-tech-rgbd':
                save_foldername = phase + '_den_' + str(args.sigma)
            elif dataset == 'ucf-qnrf':
                save_foldername = ''

            save_path = os.path.join(phase_dirname, save_foldername)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if dataset == 'ucf-qnrf':
                np_gt_path = os.path.join(save_path, os.path.basename(gt_path.replace('ann.mat', 'den.npy')))
            else:
                np_gt_path = os.path.join(save_path, os.path.basename(gt_path.replace('.mat', '.npy')))

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