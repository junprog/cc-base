import os

import torch
import numpy as np
from PIL import Image

import matplotlib
from torch.serialization import save
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class GraphPlotter:
    def __init__(self, save_dir, metrics: list, phase):
        self.save_dir = save_dir
        self.graph_name = 'result_{}.png'.format(phase)
        self.metrics = metrics

        self.epochs = []
        
        self.value_dict = dict()
        for metric in metrics:
            self.value_dict[metric] = []

    def __call__(self, epoch, values: list):
        assert (len(values) == len(self.value_dict)), 'metrics and values length shoud be same size.'
    
        self.epochs.append(epoch)

        fig, ax = plt.subplots()
    
        for i, metric in enumerate(self.metrics):
            self.value_dict[metric].append(values[i])
            ax.plot(self.epochs, self.value_dict[metric], label=metric)

        ax.legend(loc=0)
        fig.tight_layout()  # レイアウトの設定
        fig.savefig(os.path.join(self.save_dir, self.graph_name))

        plt.title(self.metrics)
        plt.close()


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def inputs_convert(img, N):
    _, c, _, _ = img.size()
    if c > 3:
        img = img[N,0:3,:,:].to('cpu').detach().numpy().copy()
    else:
        img = img[N,:,:,:].to('cpu').detach().numpy().copy()
    img = img.transpose(1,2,0)
    img = img*std+mean
    return img

def outputs_convert(img, N):
    img = img[N,:,:].to('cpu').detach().numpy().copy().squeeze()
    return img

def divide_im_geo(img, geo_num, N):
    rgb_img = img[N,0:3,:,:].to('cpu').detach().numpy().copy()
    rgb_img = rgb_img.transpose(1,2,0)
    rgb_img = rgb_img*std+mean

    geo_imgs = []
    for i in range(geo_num):
        geo_img = img[N,3+i,:,:].to('cpu').detach().numpy().copy()
        geo_imgs.append(geo_img)
    
    return rgb_img, geo_imgs

class Plotter:
    def __init__(self, opts, vis_num, save_dir):
        # vis_num is number of batch to display
        self.opts = opts
        self.vis_num = vis_num
        self.save_dir = save_dir

    @torch.no_grad()
    def __call__(self, epoch, inputs, outputs, target, phase, num=None):
        b, _, _, _ = inputs.size()
        if b < self.vis_num:
            self.vis_num = b

        in_im = []
        out_im = []
        targets = []
        for n in range(self.vis_num):
            in_im.append(inputs_convert(inputs, n))
            out_im.append(outputs_convert(outputs, n))
            targets.append(outputs_convert(target, n))

        self._display_images(epoch, in_im, out_im, targets, phase, num)

    def _display_images(self, epoch, images1, images2, targets, phase, num, label_font_size=8):

        if not (images1 and images2):
            print("No images to display.")
            return 

        plt.figure()
        i = 1

        for (im1, im2, tar) in zip(images1, images2, targets):
            im1 = Image.fromarray(np.uint8(im1*255))

            plt.subplot(3, self.vis_num, i)
            plt.title('Input Image', fontsize=10) 
            plt.imshow(im1)
            plt.subplot(3, self.vis_num, i+self.vis_num)
            if num is not None:
                plt.title('Ground Truth {}'.format(int(num)), fontsize=10) 
            else:
                plt.title('Ground Truth {:.2f}'.format(tar.sum()), fontsize=10) 
            plt.imshow(tar, cmap='jet')
            plt.subplot(3, self.vis_num, i+(self.vis_num*2))
            plt.title('Prediction {:.2f}'.format(im2.sum()), fontsize=10) 
            plt.imshow(im2, cmap='jet')
            i += 1
        
        plt.tight_layout()
        output_img_name = self.opts.dataset + '_{}_{}.png'.format(phase, epoch)
        plt.savefig(os.path.join(self.save_dir, 'images', output_img_name))
        plt.close()
  