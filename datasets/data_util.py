import os
import math
import json
import random

import numpy as np
from PIL import Image
import scipy.io as io

Image.MAX_IMAGE_PIXELS = 1000000000

"""
key: データセットのフォルダ名のキーワード
value: データセット名
"""
dataset_dict = {
    'part_A': 'shanghai-tech-a', 
    'part_B': 'shanghai-tech-b', 
    'RGBD': 'shanghai-tech-rgbd', 
    'UCF-QNRF': 'ucf-qnrf',
    'UCF_CC_50': 'ucf-cc-50',
    'NWPU': 'nwpu-crowd', 
    'RGBT': 'rgbt-cc'
}

def judge_dataset(data_dir):
    """データセットのディレクトリ構造からデータセット名を取得し、返す

    Args:
        data_dir        --- データセットのパス
    Return:
        dataset         --- データセット名
    """
    dataset = None

    for keyword in dataset_dict.keys():
        if keyword in data_dir:
            dataset = dataset_dict[keyword]

    return dataset

def load_image(image_path) -> Image:
    """画像パスから画像(PIL.Image)を返す

    Args:
        image_path      --- 画像のパス
    Return:
        image           --- 画像
    """
    with open(image_path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')

def load_depth(image_path) -> Image:
    """画像パスから深度画像(PIL.Image)を返す

    Args:
        image_path      --- 画像のパス
    Return:
        image           --- 画像
    """
    with open(image_path, 'rb') as f:
        depth = io.loadmat(f)['depth']

        index1 = np.where(depth == -999)
        index2 = np.where(depth > 20000)
        depth[index1] = 30000
        depth[index2] = 30000
        depth = depth.astype(np.float32) / 20000.

        return depth

def load_temperature(image_path) -> Image:
    """画像パスから温度画像(PIL.Image)を返す

    Args:
        image_path      --- 画像のパス
    Return:
        image           --- 画像
    """
    with open(image_path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('L')

def load_bbox(bbox_path) -> np.ndarray:
    """bboxパスからbboxを返す

    Args:
        image_path      --- 画像のパス
    Return:
        bbox            --- バウンディングボックス (x1, y1, x2, y2)
    """
    with open(bbox_path, 'rb') as f:
        return io.loadmat(f)['bbox']

def bbox_to_point(bboxes) -> np.ndarray:
    """bboxes [NUM, (x1, y1, x2, y2)] から 頭部ポイント [NUM, (x, y)]に変換
    
    Args:
        bboxes          --- バウンディングボックス (x1, y1, x2, y2)
    Return:
        pointes         --- 頭部ポイント (x, y)
    """
    points = []
    for bbox in bboxes:
        points.append(np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]))

    return np.asarray(points)

def load_geometry_images(image_path) -> list:
    """RGB画像パスからgeometry画像(PIL.Image)のリストを返す

    Args:
        image_path      --- RGB画像のパス
    Return:
        images          --- geometry画像のリスト
    """
    images = []
    if 'train' in image_path:
        geo_image_dir = image_path.replace('train_data/images', 'train_data_geometry').replace('.jpg', '_geo')
    elif 'test' in image_path:
        geo_image_dir = image_path.replace('test_data/images', 'test_data_geometry').replace('.jpg', '_geo')
    geo_image_list = os.listdir(geo_image_dir)

    for geo_img in geo_image_list:
        geo_image_path = os.path.join(geo_image_dir, geo_img)
    
        with open(geo_image_path, 'rb') as f:
            with Image.open(f) as image:
                images.append(image.convert('L'))
        
    return images

def create_gt_path(image_path, dataset, phase) -> str:
    """画像パスからground truthのパスを返す

    Args:
        image_path      --- 画像のパス
        dataset         --- データセットの種類
        phase           --- train, val, test

    Return:
        gt_path         --- ground truthのパス
    """
    if phase == 'train' or phase == 'val':
        if dataset == 'ucf-qnrf':
            gt_path = image_path.replace('.jpg', '_ann.mat')

        elif dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b':
            gt_path = image_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')

        if dataset == 'shanghai-tech-rgbd':
            gt_path = image_path.replace('img/IMG', 'gt/GT').replace('.png','.mat')

    elif phase == 'test':
        if dataset == 'ucf-qnrf':
            gt_path = image_path.replace('.jpg', '_ann.mat')

        elif dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b':
            gt_path = image_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')

        if dataset == 'shanghai-tech-rgbd':
            gt_path = image_path.replace('img/IMG', 'gt_np/GT').replace('.png','.npy')
    
    return gt_path

def create_density_path(image_path, dataset, phase) -> str:
    """画像パスから density のパスを返す

    Args:
        image_path      --- 画像のパス
        dataset         --- データセットの種類
        phase           --- train, val, test

    Return:
        density_path         --- ground truthのパス
    """
    if phase == 'train' or phase == 'val':
        if dataset == 'ucf-qnrf':
            density_path = image_path.replace('.jpg', '_den.npy')

        elif dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b':
            density_path = image_path.replace('.jpg','.npy').replace('images', 'den_15').replace('IMG_','GT_IMG_')

        if dataset == 'shanghai-tech-rgbd':
            density_path = image_path.replace('train_img/IMG', 'train_den_15/GT').replace('.png','.npy')
    
    elif phase == 'test':
        if dataset == 'ucf-qnrf':
            density_path = image_path.replace('.jpg', '_den.npy')

        elif dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b':
            density_path = image_path.replace('.jpg','.npy').replace('images', 'den_15').replace('IMG_','GT_IMG_')

        if dataset == 'shanghai-tech-rgbd':
            density_path = image_path.replace('test_img/IMG', 'test_den_15/GT').replace('.png','.npy')

    return density_path

def create_depth_path(image_path, dataset, phase) -> str:
    """ShanghaiTechRGBD のみに使用
    """
    if phase == 'train' or phase == 'val':
        # /mnt/hdd02/ShanghaiTechRGBD/train_data/train_img/IMG_0000.png
        # /mnt/hdd02/ShanghaiTechRGBD/train_data/train_depth/DEPTH_0000.mat
        depth_path = image_path.replace('train_img/IMG', 'train_depth/DEPTH').replace('.png','.mat')

    elif phase == 'test':
        # /mnt/hdd02/ShanghaiTechRGBD/test_data/test_img/IMG_0000.png
        # /mnt/hdd02/ShanghaiTechRGBD/test_data/test_depth/DEPTH_0000.mat
        depth_path = image_path.replace('test_img/IMG', 'test_depth/DEPTH').replace('.png','.mat')
    
    return depth_path

def load_gt(gt_path) -> np.ndarray:
    """ ground truthのパスから頭部位置座標(np.array)を返す

    Args:
        gt_path         --- ground truthのパス
    Return:
        np.ndarray      --- 頭部位置座標
    """
    if os.path.exists(gt_path):
        if '.mat' in os.path.basename(gt_path):

            # UCF-QNRF
            if 'UCF-QNRF' in gt_path:
                with open(gt_path, 'rb') as f:
                    return io.loadmat(f)['annPoints']

            # ShanghaiTech A, B
            elif ('part_A' in gt_path) or ('part_B' in gt_path):
                with open(gt_path, 'rb') as f:
                    return io.loadmat(f)['image_info'][0,0][0,0][0]

            # ShanghaiTech RGBD
            elif 'RGBD' in gt_path:
                with open(gt_path, 'rb') as f:
                    return io.loadmat(f)['point']

            # other (custom npy file)

        elif 'npy' in os.path.basename(gt_path):
            with open(gt_path, 'rb') as f:
                return np.load(f)

    else:
        print('gt_file: {} is not exists'.format(gt_path))
        return None

def load_json(json_path) -> list:
    """jsonパスからリストを返す

    Args:
        json_path       --- jsonのパス
    Return:
        json            --- json内のリスト
    """
    with open(json_path, 'r') as json_data:
        return json.load(json_data)

def mapping_gt(img_size: tuple, location) -> Image:
    """location -> imageサイズの空配列に1でマッピングし2D map(np.array)を返す

    Args:
        img_size        --- 画像サイズ (W, H)
        location        --- 頭部座標
    Return:
        head_map        --- 2D 頭部マップ
    """
    zeropad = np.zeros(img_size)

    for i in range(0,len(location)):
        if int(location[i][0]) < img_size[0] and int(location[i][1]) < img_size[1]:
            zeropad[int(location[i][0]),int(location[i][1])] = 1
    head_map = Image.fromarray(zeropad.T)

    return head_map

def decide_crop_area(img_size: tuple, crop_size: tuple):
    """画像とGTマップでクロップする箇所を揃えるための箇所の決定をする

    Args:
        img_size        --- 画像サイズ (W, H)
        crop_size       --- クロップサイズ (W, H)
    Return:
        top             --- クロップ箇所の左上 width座標 
        left            --- クロップ箇所の左上 height座標
    """
    w, h = img_size
    c_w, c_h = crop_size

    area_w = w - c_w
    area_h = h - c_h

    left = random.randint(0, area_w)
    top = random.randint(0, area_h)

    return top, left

def split_image_by_size(image: Image, max_width: int, max_height: int) -> list:
    """指定したサイズに画像を分割し、リストにして返す (1px 幅 or 高さの画像に分割される可能性あり)

    Args:
        image           --- 画像
        max_width       --- 幅に対する分割指定サイズ
        max_height      --- 高さに対する分割指定サイズ
    Return:
        images          --- 分割した画像のリスト
    """
    images = []
    w, h = image.size

    np_img = np.array(image)

    ## オリジナルのサイズが指定最大サイズと一致 or それ以下の時の例外処理
    if w <= max_width:
        width_flag = 1
    else:
        width_flag = 0

    if h <= max_height:
        height_flag = 1
    else:
        height_flag = 0
               
    if width_flag == 1 and height_flag == 1:
        images.append(image)

    else:
        wid_cnt = math.ceil(w / max_width)
        hgt_cnt = math.ceil(h / max_height)

        wid_remain = 0
        for i in range( wid_cnt ):
            if wid_cnt == 0:
                clip_width = w
            else:
                tmp_clip_width = w / wid_cnt
                clip_width = int(tmp_clip_width)

                wid_remain += tmp_clip_width - clip_width

            hgt_remain = 0
            for j in range( hgt_cnt ):
                if hgt_cnt == 0:
                    clip_height = h
                else:
                    tmp_clip_height = h / hgt_cnt
                    clip_height = int(tmp_clip_height)

                    hgt_remain += tmp_clip_height - clip_height

                if i == wid_cnt - 1:
                    width_remain = round(wid_remain)
                else:
                    width_remain = 0

                if j == hgt_cnt - 1:
                    height_remain = round(hgt_remain)
                else:
                    height_remain = 0

                if np_img.ndim == 3:
                    splitted_np_img = np_img[clip_height*j:clip_height*j+clip_height+height_remain, clip_width*i:clip_width*i+clip_width+width_remain, :]
                if np_img.ndim == 2:
                    splitted_np_img = np_img[clip_height*j:clip_height*j+clip_height+height_remain, clip_width*i:clip_width*i+clip_width+width_remain]
                splitted_pil_img = Image.fromarray(splitted_np_img)

                images.append(splitted_pil_img)
        
    return images

def split_image_by_num(image: Image, width_patch_num: int, height_patch_num: int) -> list:
    """指定したパッチ数に画像を分割し、リストにして返す

    Args:
        image               --- 画像
        width_patch_num     --- 幅に対する分割パッチ数
        height_patch_num    --- 高さに対する分割パッチ数
    Return:
        images              --- 分割した画像のリスト
    """
    images = []
    w, h = image.size
    np_img = np.array(image)
               
    # パッチ数 1x1のとき
    if width_patch_num == 1 and height_patch_num == 1:
        images.append(image)

    else:
        wid_remain = 0
        for i in range(  width_patch_num ):
            tmp_clip_width = w / width_patch_num
            clip_width = int(tmp_clip_width)

            wid_remain += tmp_clip_width - clip_width

            hgt_remain = 0
            for j in range( height_patch_num ):
                tmp_clip_height = h / height_patch_num
                clip_height = int(tmp_clip_height)

                hgt_remain += tmp_clip_height - clip_height

                if i == width_patch_num - 1:
                    width_remain = round(wid_remain)
                else:
                    width_remain = 0

                if j == height_patch_num - 1:
                    height_remain = round(hgt_remain)
                else:
                    height_remain = 0

                if np_img.ndim == 3:
                    splitted_np_img = np_img[clip_height*j:clip_height*j+clip_height+height_remain, clip_width*i:clip_width*i+clip_width+width_remain, :]
                if np_img.ndim == 2:
                    splitted_np_img = np_img[clip_height*j:clip_height*j+clip_height+height_remain, clip_width*i:clip_width*i+clip_width+width_remain]
                splitted_pil_img = Image.fromarray(splitted_np_img)

                images.append(splitted_pil_img)
        
    return images

def dirname_parser(dataset, phase, style):
    """データセット名から style: [image, depth, tempareture, gt, density, bbox] パスを返す
    ShanghaiTech/
        ├ part_A/
        │   ├ train_data/
        │   │   ├ images/
        │   │   │   └ IMG_*.jpg: image
        │   │   ├ (own) den_15/
        │   │   │   └ GT_IMG_*.npy: density
        │   │   └ ground_truth/
        │   │       └ GT_IMG_*.mat: gt   
        │   │
        │   └ test_data/
        │       ├ images/
        │       │   └ IMG_*.jpg: image
        │       ├ (own) den_15/
        │       │   └ GT_IMG_*.npy: density
        │       └ ground_truth/
        │           └ GT_IMG_*.mat: gt  
        │ 
        └ part_B/
            ├ train_data/
            │   ├ images/
            │   │   └ IMG_*.jpg: image
            │   ├ (own) den_15/
            │   │   └ GT_IMG_*.npy: density
            │   └ ground_truth/
            │       └ GT_IMG_*.mat: gt   
            │
            └ test_data/
                ├ images/
                │   └ IMG_*.jpg: image
                ├ (own) den_15/
                │   └ GT_IMG_*.npy: density
                └ ground_truth/
                    └ GT_IMG_*.mat: gt  

    ShanghaiTechRGBD/
        ├ train_data/
        |   ├ train_img/
        |   │   └ IMG_*.png: image
        |   ├ train_gt/
        │   │   └ GT_*.mat: gt
        |   ├ train_depth/
        |   |   └ DEPTH_*.mat: depth
        |   ├ train_bbox/
        |   |   └ BBOX_*.mat: bbox
        |   └ (own) train_den_15/
        |       └ GT_*.npy: density
        |
        └ test_data/
            ├ test_img/
            │   └ IMG_*.png: image
            ├ (own) test_gt_np/
            │   └ GT_*.npy: gt
            ├ test_depth/
            |   └ DEPTH_*.mat: depth
            ├ test_bbox/
            |   └ BBOX_*.mat: bbox
            └ (own) test_den_15/
                └ GT_*.npy: density

    UCF-QNRF_ECCV18/
        ├ Train/
        │   ├ img_*.jpg: image
        │   ├ img_*_ann.mat: gt
        |   └ (own) img_*_den.npy: density
        │
        └ Test/
            ├ img_*.jpg: image
            ├ img_*_ann.mat: gt
            └ (own) img_*_den.npy: density

    styles = ['image', 'depth', 'tempareture', 'gt', 'density', 'bbox']

    Args:
        dataset
        phase
        style

    Return:
        target_dirname
        target_format
    """

    ## 例外処理
    have_style_flag = judge_have_style(dataset, style)
    assert have_style_flag, '{} does not have {}'.format(dataset, style)

    ## train, test スプリット
    if dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b' or dataset == 'shanghai-tech-rgbd':
        if phase == 'train':
            split_dirname = 'train_data'
        elif phase == 'test':
            split_dirname = 'test_data'
    elif dataset == 'ucf-qnrf':
        if phase == 'train':
            split_dirname = 'Train'
        elif phase == 'test':
            split_dirname = 'Test'

    ## スタイル
    if dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b':
        if style == 'image':
            style_dirname = 'images'
            target_format = '*.jpg'
        elif style == 'gt':
            style_dirname = 'ground_truth'
            target_format = '*.mat'
        elif style == 'density':
            style_dirname = 'den_15'
            target_format = '*.npy'

    elif dataset == 'shanghai-tech-rgbd':
        if style == 'image':
            style_dirname = phase + '_img'
            target_format = '*.png'
        elif style == 'gt':
            if phase == 'train':
                style_dirname = phase + '_gt'
                target_format = '*.mat'
            elif phase == 'test':
                style_dirname = phase + '_gt_np'
                target_format = '*.npy'
        elif style == 'density':
            if phase == 'train':
                style_dirname = phase + '_den_15'
                target_format = '*.npy'
            elif phase == 'test':
                style_dirname = phase + '_den_15'
                target_format = '*.npy'
        elif style == 'depth':
            style_dirname = phase + '_depth'
            target_format = '*.mat'
        elif style == 'bbox':
            style_dirname = phase + '_bbox'
            target_format = '*.mat'

    elif dataset == 'ucf-qnrf':
        if style == 'image':
            style_dirname = ''
            target_format = '*.jpg'
        elif style == 'gt':
            style_dirname = ''
            target_format = '*.mat'
        elif style == 'density':
            style_dirname = ''
            target_format = '*.npy'

    target_dirname = os.path.join(split_dirname, style_dirname)

    return target_dirname, target_format

def judge_have_style(dataset, style):
    flag = True
    if dataset == 'shanghai-tech-a' or dataset == 'shanghai-tech-b' or dataset == 'ucf-qnrf':
        not_have_style = ['depth', 'tempareture', 'bbox']
        if style in not_have_style:
            flag = False
    elif dataset == 'shanghai-tech-rgbd':
        not_have_style = ['tempareture']
        if style in not_have_style:
            flag = False

    return flag