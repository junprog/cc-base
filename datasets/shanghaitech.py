import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms

from datasets.data_util import *

class ShanghaiTechA(data.Dataset):
    """ShanghaiTech partA dataet
    Args:
        dataset                 --- データセット名
        json_path               --- 画像のパスリストが記載されてるjsonパス
        crop_size               --- 学習時のクロップサイズ
        phase                   --- tarinかvalのフェーズ指定

        rescale                 --- 512 < size < 2048 変換をしているか否か (STA のみ)

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
            rescale=False,
            model_scale=16,
            up_scale=4
        ):

        self.dataset = dataset
        self.crop_size = crop_size
        self.phase = phase
        self.rescale = rescale

        self.model_scale = model_scale
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
        den_path = create_density_path(img_path, dataset=self.dataset, phase=self.phase)
        density = np.load(den_path)
        density = Image.fromarray(density)

        ## 変形処理
        if self.phase == 'train':
            ## crop size より小さい画像をパディング
            if self.img_size[0] < self.crop_size[0] or self.img_size[1] < self.crop_size[1]:
                image, density = self._padding(image, density)

            ## ランダムクロップ
            image, density = self._random_crop(image, density)

            ## 頭部数のカウント
            num = torch.from_numpy(np.asarray(density).copy()).clone()
            num = torch.sum(num, dim=0)
            num = torch.sum(num, dim=0)
            
            ## 50%で反転
            if random.random() > 0.5:
                image, density = map(F.hflip, [image, density])

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
            target = self.to_tensor(np.array(density))
            
            return image, target, [], num

        elif self.phase == 'val' or self.phase == 'test':
            gt_path = create_gt_path(img_path, self.dataset, self.phase)
            points = load_gt(gt_path)

            ## 頭部数のカウント
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
            target = self.to_tensor(np.array(density))

            return image, target, [], num

    def _padding(self, image, gt_map):
        np_gt_map = np.asarray(gt_map)

        if self.img_size[0] < self.crop_size[0]:
            image = F.pad(image, (self.crop_size[0] - self.img_size[0], 0))
            np_gt_map = np.pad(np_gt_map, ((0,0), (self.crop_size[0] - np_gt_map.shape[1], self.crop_size[0] - np_gt_map.shape[1])), 'constant')
        if self.img_size[1] < self.crop_size[1]:
            image = F.pad(image, (0, self.crop_size[1] - self.img_size[1]))
            np_gt_map = np.pad(np_gt_map, ((self.crop_size[1] - np_gt_map.shape[0], self.crop_size[1] - np_gt_map.shape[0]), (0,0)), 'constant')

        gt_map = Image.fromarray(np_gt_map)

        ## self.img_size 書き換え
        self.img_size = image.size
        
        return image, gt_map

    def _random_crop(self, image, gt_map):
        ## ランダムクロップのパラメタライズ(クロップ座標を画像とGTマップで固定するため)
        self.top, self.left = decide_crop_area(self.img_size, self.crop_size)

        ## クロップ
        image = F.crop(image, self.top, self.left, self.crop_size[0], self.crop_size[1])
        gt_map = F.crop(gt_map, self.top, self.left, self.crop_size[0], self.crop_size[1])

        return image, gt_map


class ShanghaiTechB(data.Dataset):
    """ShanghaiTech partB dataet
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
            up_scale=4
        ):

        self.dataset = dataset
        self.crop_size = crop_size
        self.phase = phase

        self.model_scale = model_scale
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
        den_path = create_density_path(img_path, dataset=self.dataset, phase=self.phase)
        density = np.load(den_path)
        density = Image.fromarray(density)

        ## 変形処理
        if self.phase == 'train':
            ## ランダムクロップ
            image, density = self._random_crop(image, density)

            ## 頭部数のカウント
            num = torch.from_numpy(np.asarray(density).copy()).clone()
            num = torch.sum(num, dim=0)
            num = torch.sum(num, dim=0)
            
            ## 50%で反転
            if random.random() > 0.5:
                image, density = map(F.hflip, [image, density])

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
            target = self.to_tensor(np.array(density))
            
            return image, target, [], num

        elif self.phase == 'val' or self.phase == 'test':
            gt_path = create_gt_path(img_path, self.dataset, self.phase)
            points = load_gt(gt_path)

            ## 頭部数のカウント
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
            target = self.to_tensor(np.array(density))

            return image, target, [], num

    def _random_crop(self, image, gt_map):
        ## ランダムクロップのパラメタライズ(クロップ座標を画像とGTマップで固定するため)
        self.top, self.left = decide_crop_area(self.img_size, self.crop_size)

        ## クロップ
        image = F.crop(image, self.top, self.left, self.crop_size[0], self.crop_size[1])
        gt_map = F.crop(gt_map, self.top, self.left, self.crop_size[0], self.crop_size[1])

        return image, gt_map