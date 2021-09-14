import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms

from datasets.data_util import *

class ShanghaiTechRGBD(data.Dataset):
    """ShanghaiTechRGBD dataet
    Args:
        dataset                 --- データセット名
        json_path               --- 画像のパスリストが記載されてるjsonパス
        crop_size               --- 学習時のクロップサイズ
        phase                   --- tarinかvalのフェーズ指定

        model_scale             --- モデルの出力のダウンスケール
        up_scale                --- 特徴マップにかけるアップスケール
        
    Return:
        image                   --- 画像
        target                  --- GT
        num                     --- 画像内の人数 (train: GT density mapのsum, val: len(location))
    """

    def __init__(
            self,
            dataset: str,
            json_path: str,
            crop_size: tuple,
            phase='train',
            model_scale=16,
            up_scale=1
        ):

        self.dataset = dataset
        self.crop_size = crop_size
        self.phase = phase

        self.model_scale = model_scale
        self.up_scale = up_scale

        self.img_path_list = load_json(json_path)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.441], std=[0.329]),
        ])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        ## 画像読み込み
        img_path = self.img_path_list[idx]
        image = load_image(img_path)
        self.img_size = image.size

        ##深度読み込み
        depth_path = create_depth_path(img_path, dataset=self.dataset, phase=self.phase)
        depth = load_depth(depth_path)
        depth = Image.fromarray(depth)

        ## 密度マップ読み込み
        den_path = create_density_path(img_path, dataset=self.dataset, phase=self.phase)
        density = np.load(den_path)
        density = Image.fromarray(density)

        if self.phase == 'train':
            ## ランダムクロップ
            image, density, depth = self._random_crop(image, density, depth)

            ## 頭部数のカウント
            num = torch.from_numpy(np.asarray(density).copy()).clone()
            num = torch.sum(num, dim=0)
            num = torch.sum(num, dim=0)
            
            ## 50%で反転
            if random.random() > 0.5:
                image, density, depth = map(F.hflip, [image, density, depth])

            ## モデルのアウトプットサイズに合わせて density リサイズ
            in_size = density.size
            self.scale_factor = self.model_scale / self.up_scale
            out_size = (int(in_size[0] / self.scale_factor), int(in_size[1] / self.scale_factor))
            density = density.resize(out_size, resample=Image.BICUBIC)

            ## リサイズ時の補間による数値的な変化を戻すためのスケーリング
            density = np.asarray(density)
            density = density * (self.scale_factor*self.scale_factor)
            
            ## torch.Tensor化、正規化
            image = self.trans(image)
            depth = self.depth_trans(np.array(depth))
            target = self.to_tensor(np.array(density))
            
            return image, depth, target, num

        elif self.phase == 'val' or self.phase == 'test':
            gt_path = create_gt_path(img_path, self.dataset, self.phase)
            points = load_gt(gt_path)

            ## 頭部数
            num = torch.tensor(float(len(points))).clone()

            ## モデルのアウトプットサイズに合わせて density リサイズ
            in_size = density.size
            self.scale_factor = self.model_scale / self.up_scale
            out_size = (int(in_size[0] / self.scale_factor), int(in_size[1] / self.scale_factor))
            density = density.resize(out_size, resample=Image.BICUBIC)

            ## リサイズ時の補間による数値的な変化を戻すためのスケーリング
            density = np.asarray(density)
            density = density * (self.scale_factor*self.scale_factor)

            image = self.trans(image)
            depth = self.depth_trans(np.array(depth))
            target = self.to_tensor(np.array(density))

            return image, depth, target, num

    def _random_crop(self, image, gt_map, depth):
        ## ランダムクロップのパラメタライズ(クロップ座標を画像とGTマップで固定するため)
        self.top, self.left = decide_crop_area(self.img_size, self.crop_size)

        ## クロップ
        image = F.crop(image, self.top, self.left, self.crop_size[0], self.crop_size[1])
        gt_map = F.crop(gt_map, self.top, self.left, self.crop_size[0], self.crop_size[1])
        depth = F.crop(depth, self.top, self.left, self.crop_size[0], self.crop_size[1])

        return image, gt_map, depth