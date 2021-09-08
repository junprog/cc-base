import os
import random
import argparse

import numpy as np
import torch

from engine.count_trainer import CountTrainer

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--save-dir', default='',
                        help='directory to save models.')
    parser.add_argument('--exp', default='',
                        help='result dir prefix')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--gaussian-std', type=int, default=15,
                        help='std of gaussian filter')

    # dataset
    parser.add_argument('--dataset', default='shanghai-tech-rgbd',
                        help='select dataset [ucf-qnrf, st-a, st-b]')     
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--rgb', action='store_true',
                        help='use rgb images')
    parser.add_argument('--depth', action='store_true',
                        help='use depth images')

    # dataloader
    parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    # model
    parser.add_argument('--arch', type=str, default='vgg19',
                        help='the model architecture [vgg19, vgg19_bn, resnet16, resnet50, resnet101, csrnet, mcnn]')
    parser.add_argument('--pool-num', type=int, default=3,
                        help='pool num')
    parser.add_argument('--up-scale', type=int, default=1,
                        help='up scale num')               

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='the weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='the momentum')                    

    # model save
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save')
    parser.add_argument('--check-point', type=int, default=10,
                        help='milestone of save model checkpoint')

    # epoch
    parser.add_argument('--max-epoch', type=int, default=501,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=10,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')

    args = parser.parse_args()
    return args

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    fix_seed(765)
    args = parse_args()
    
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip('-')  # set vis gpu

    trainer = CountTrainer(args)
    trainer.setup()
    trainer.train()