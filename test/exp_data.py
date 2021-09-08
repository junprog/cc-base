import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.ucf_qnrf import UCF_QNRF

def display_imgs(img_list: list):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    row = 2
    col = 2
    plt.figure(figsize=(10,10))

    for i, img in enumerate(img_list):
        c, w, h = img.size()
        if c == 1:
            img = img.squeeze()
            img = np.array(img)
            num = np.sum(img)
            pos_w = int(w * 0.75)
            pos_h = int(h * 0.95)
        elif c == 3:
            img = np.array(img)
            img = img.transpose(1,2,0)
            img = img*std+mean
            img = Image.fromarray((img * 255).astype(np.uint8))
            num = None

        plt.subplot(row, col, i+1)
        if num is not None:
            plt.text(pos_w, pos_h, str(num), color='white')
        plt.imshow(img)

    plt.show()

import random
random.seed(765)

dataset = UCF_QNRF('json/ucf-qnrf/train.json', (512, 512), phase='train', target_over_crop=False, over_crop_len=50)

for i in range(9, len(dataset)):
    print('{} / {} th image'.format(i, len(dataset)))
    img, img_hf, tar, tar_hf, num = dataset[i]

    print(num)
    display_imgs([img, img_hf, tar, tar_hf])