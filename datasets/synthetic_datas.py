import random
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms

from datasets.data_util import *
from datasets.calc_scale import *

def create_density(gt_map, sigma):
    gt_map = np.array(gt_map)
    density = gaussian_filter(gt_map, sigma)
    density = Image.fromarray(density)

    return density

class SyntheticDataset(data.Dataset):
    """Synthetic Dataset
    Args:
        dataset                 --- データセット名
        arch                    --- モデルアーキテクチャ名

        json_path               --- 画像のパスリストが記載されてるjsonパス
        crop_size               --- 学習時のクロップサイズ
        phase                   --- tarinかvalのフェーズ指定

        rescale                 --- 512 < min(size) < 2048 変換をしているか否か (STA のみ)

        sigma                   --- ガウシアンフィルタのパラメータ
        pool_num                --- 解像度が下がる回数(Pooling, Conv(stride>1)の層数)
        up_scale                --- 特徴マップにかけるアップスケール
        
    Return:
        image                   --- 画像
        target                  --- GT
        num                     --- 画像内の人数 (train: GT density mapのsum, val: len(location))
    """

    def __init__(
            self,
            dataset: str,
            arch: str,
            json_path: str,
            crop_size: tuple,
            phase='train',
            rescale=False,
            sigma=15,
            pool_num=3,
            up_scale=1
        ):

        self.dataset = dataset
        self.arch = arch
        self.crop_size = crop_size
        self.phase = phase
        self.rescale = rescale

        self.sigma = sigma
        self.pool_num = pool_num
        self.up_scale = up_scale

        self.img_path_list = load_json(json_path)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        ## 画像読み込み
        img_path = self.img_path_list[idx]
        image = load_image(img_path)
        self.img_size = image.size

        ## 密度マップ読み込み
        # den_path = create_density_path(img_path, dataset=self.dataset, phase=self.phase)
        # density = np.load(den_path)
        # density = Image.fromarray(density)

        ## gt 頭部マップ読み込み
        gt_path = create_gt_path(img_path, dataset=self.dataset, phase=self.phase)
        points = load_gt(gt_path)
        gt_map = mapping_gt(self.img_size, points)

        if not '2d' in self.dataset:
            gt_map = ImageOps.flip(gt_map)

        ## 変形処理
        if self.phase == 'train':
            ## crop size より小さい画像をパディング
            if self.img_size[0] < self.crop_size[0] or self.img_size[1] < self.crop_size[1]:
                image, gt_map = self._padding(image, gt_map)

            ## ランダムクロップ
            image, gt_map = self._random_crop(image, gt_map)
            
            ## 50%で反転
            if random.random() > 0.5:
                image, gt_map = map(F.hflip, [image, gt_map])

            ## ガウシアンブラー
            density = create_density(gt_map, self.sigma)

            ## モデルのアウトプットサイズに合わせて density リサイズ
            in_size = density.size
            out_size = calc_out_size(self.arch, in_size, self.pool_num, self.up_scale)
            density = density.resize(out_size, resample=Image.BICUBIC)

            ## リサイズ時の補間による数値的な変化を戻すためのスケーリング
            density = np.asarray(density)
            scale_factor = calc_scale_factor(in_size, out_size)
            density = density * (scale_factor[0]*scale_factor[1])

            ## 頭部数のカウント
            num = torch.from_numpy(np.asarray(density).copy()).clone()
            num = torch.sum(num, dim=0)
            num = torch.sum(num, dim=0)
            
            ## torch.Tensor化、正規化
            image = self.trans(image)
            target = self.to_tensor(np.array(density))
            
            return image, target, num, img_path

        elif self.phase == 'val' or self.phase == 'test':
            gt_path = create_gt_path(img_path, self.dataset, self.phase)
            points = load_gt(gt_path)

            ## ガウシアンブラー
            density = create_density(gt_map, self.sigma)

            ## モデルのアウトプットサイズに合わせて density リサイズ
            in_size = density.size
            out_size = calc_out_size(self.arch, in_size, self.pool_num, self.up_scale)
            density = density.resize(out_size, resample=Image.BICUBIC)

            ## リサイズ時の補間による数値的な変化を戻すためのスケーリング
            density = np.asarray(density)
            scale_factor = calc_scale_factor(in_size, out_size)
            density = density * (scale_factor[0]*scale_factor[1])

            ## 頭部数のカウント
            num = torch.tensor(float(len(points))).clone()

            image = self.trans(image)
            target = self.to_tensor(np.array(density))

            return image, target, num, img_path

    def _padding(self, image, gt_map):
        np_gt_map = np.asarray(gt_map)

        if self.img_size[0] < self.crop_size[0]:
            image = F.pad(image, (self.crop_size[0] - self.img_size[0], 0))
            np_gt_map = np.pad(np_gt_map, ((0,0), (self.crop_size[0] - np_gt_map.shape[1], self.crop_size[0] - np_gt_map.shape[1])), 'constant')
        if self.img_size[1] < self.crop_size[1]:
            image = F.pad(image, (0, self.crop_size[1] - self.img_size[1]))
            np_gt_map = np.pad(np_gt_map, ((self.crop_size[1] - np_gt_map.shape[0], self.crop_size[1] - np_gt_map.shape[0]), (0,0)), 'constant')

        gt_map = Image.fromarray(np_gt_map)
        self.img_size = image.size
        
        return image, gt_map

    def _random_crop(self, image, gt_map):
        ## ランダムクロップのパラメタライズ(クロップ座標を画像とGTマップで固定するため)
        self.top, self.left = decide_crop_area(self.img_size, self.crop_size)

        ## クロップ
        image = F.crop(image, self.top, self.left, self.crop_size[0], self.crop_size[1])
        gt_map = F.crop(gt_map, self.top, self.left, self.crop_size[0], self.crop_size[1])

        return image, gt_map