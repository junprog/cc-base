import random
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from torch._C import dtype

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms

from datasets.data_util import *

class ShanghaiTech(data.Dataset):
    """ShanghaiTech dataet
    Args:
        json_path               --- 画像のパスリストが記載されてるjsonパス
        part                    --- ST Part (a or b)
        crop_size               --- 学習時のクロップサイズ
        phase                   --- tarinかvalのフェーズ指定
        rescale                 --- 512 < size < 2048 変換をしているか否か

        target_over_crop        --- ラベルノイズに対応するためにあえて大きくクロップ -> ガウシアンフィルタ -> もともとのクロップサイズにクロップ
        over_crop_len           --- オーバークロップの長さ(操作後のサイズ: over_clop_len + crop_size + over_crop_len となる)

        gaussian_std            --- ガウシアンフィルタのstd値
        model_scale             --- モデルの出力のダウンスケール
        up_scale                --- 特徴マップにかけるアップスケール
        
    Return:
        image                   --- tuple (等倍画像、1/2画像)
        target                  --- tuple (等倍GT、1/2GT)
        num                     --- 画像内の人数 (train: GT density mapのsum, val: len(location))
    """

    def __init__(
            self,
            json_path: str,
            part: str,
            crop_size: tuple,
            phase='train',
            rescale=False,
            target_over_crop=True,
            over_crop_len=10,
            gaussian_std=15,
            model_scale=16,
            up_scale=4
        ):

        self.dataset = 'st-' + part
        self.crop_size = crop_size
        self.phase = phase
        self.rescale = rescale

        self.target_over_crop = target_over_crop
        self.over_crop_len = over_crop_len

        self.gaussian_std = gaussian_std
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

        ## GTマップ読み込み
        gt_path = create_gt_path(img_path, dataset=self.dataset, rescale=self.rescale)
        location = load_gt(gt_path)
        gt_map = mapping_gt(self.img_size, location)

        ## 変形処理
        if self.phase == 'train':
            ## crop size より小さい画像をパディング
            if self.img_size[0] < self.crop_size[0] or self.img_size[1] < self.crop_size[1]:
                image, gt_map = self._padding(image, gt_map)

            ## ランダムクロップ
            image, gt_map = self._random_crop(image, gt_map)

            ## point map -> blur map
            target = self._create_target(gt_map)

            ## 頭部数のカウント
            num = torch.from_numpy(np.asarray(target).copy()).clone()
            num = torch.sum(num, dim=0)
            num = torch.sum(num, dim=0)
            
            ## 50%で反転
            if random.random() > 0.5:
                image, target = map(F.hflip, [image, target])
            
            ## torch.Tensor化、正規化
            image = self.trans(image)
            target = np.array(target)
            target = self.to_tensor(target)
            
            return image, target, [], num

        elif self.phase == 'val':
            ## point map -> blur map
            target = self._create_target(gt_map)

            ## 頭部数のカウント
            num = torch.tensor(float(len(location))).clone()

            if (not self.rescale) and (max(self.img_size) > 5000):
                images = split_image_by_num(image, 2, 2)

                images = [self.trans(im) for im in images]
            else:
                images = []

                images.append(self.trans(image))

            target = np.array(target)
            target = self.to_tensor(target)

            return images, target, [], num

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
        
        if self.target_over_crop:
            ## あえて大きくクロップする処理
            over_top, over_left, over_crop_size = self._decide_over_area(self.top, self.left)
            gt_map = F.crop(gt_map, over_top, over_left, over_crop_size[0], over_crop_size[1])
        else:
            gt_map = F.crop(gt_map, self.top, self.left, self.crop_size[0], self.crop_size[1])

        return image, gt_map

    def _decide_over_area(self, top, left):
        ## あえて大きくクロップするときのエリアの決定
        self.over_top_len = self.over_crop_len
        over_under_len = self.over_crop_len
        
        self.over_left_len = self.over_crop_len
        over_right_len = self.over_crop_len

        if top == 0:
            over_top = top
            self.over_top_len = 0
        elif self.over_crop_len > top:
            over_top = 0
            self.over_top_len = top
        else:
            over_top = top - self.over_crop_len
            if self.img_size[1] - (top + self.crop_size[1]) >= self.over_crop_len:
                pass
            else:
                over_under_len = self.img_size[1] - (top + self.crop_size[1])
        
        if left == 0:
            over_left = left
            self.over_left_len = 0
        elif self.over_crop_len > left:
            over_left = 0
            self.over_left_len = left
        else:
            over_left = left - self.over_crop_len
            if self.img_size[0] - (left + self.crop_size[0]) >= self.over_crop_len:
                pass
            else:
                over_right_len = self.img_size[0] - (left + self.crop_size[0])

        over_crop_height_size = self.crop_size[1] + self.over_top_len + over_under_len
        over_crop_width_size = self.crop_size[0] + self.over_left_len + over_right_len

        over_crop_size = (over_crop_height_size, over_crop_width_size)

        return over_top, over_left, over_crop_size

    def _create_target(self, gt_map):
        ## ガウシアンフィルタリング
        gt_map = np.array(gt_map)
        gt_density = gaussian_filter(gt_map, self.gaussian_std)
        gt_density = Image.fromarray(gt_density)

        ## あえて大きくクロップした場合のみ、もともとのクロップを実行
        if self.target_over_crop and self.phase == 'train':
            gt_density = F.crop(gt_density, self.over_top_len, self.over_left_len, self.crop_size[0], self.crop_size[1])

        ## モデルのアウトプットサイズに合わせる
        in_size = gt_density.size
        self.scale_factor = self.model_scale / self.up_scale
        out_size = (int(in_size[0] / self.scale_factor), int(in_size[1] / self.scale_factor))
        gt_density = gt_density.resize(out_size, resample=Image.BICUBIC)

        ## リサイズ時の補間による数値的な変化を戻すためのスケーリング
        gt_density = np.array(gt_density)
        gt_density = gt_density * (self.scale_factor*self.scale_factor)

        return Image.fromarray(gt_density)