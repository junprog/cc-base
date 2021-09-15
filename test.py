import torch
import os
import numpy as np

from utils.visualizer import Plotter
from utils.helper import setlogger

from datasets.shanghaitech_rgbd import ShanghaiTechRGBD

from models.vgg import VGG
from models.resnet import ResNet
from models.mcnn import MCNN
from models.csrnet import CSRNet

import argparse
import logging

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')

    # dataset
    parser.add_argument('--dataset', default='shanghai-tech-rgbd',
                        help='select dataset [ucf-qnrf, rescale-ucf-qnrf, shanghai-tech-a, shanghai-tech-b]')
    parser.add_argument('--rgb', action='store_true',
                        help='use rgb images')
    parser.add_argument('--depth', action='store_true',
                        help='use depth images')

    # model
    parser.add_argument('--arch', type=str, default='vgg19',
                        help='the model architecture [vgg19, vgg19_bn, resnet16, resnet50, resnet101, csrnet, mcnn]')
    parser.add_argument('--pool-num', type=int, default=3,
                        help='pool num')
    parser.add_argument('--up-scale', type=int, default=1,
                        help='up scale num')            

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip('-')  # set vis gpu

    setlogger(os.path.join(args.save_dir, 'test.log'))  # set logger
    if args.rgb and args.depth:
        mode = 'both'
        in_ch = 4
    elif args.rgb:
        mode = 'rgb'
        in_ch = 3
    elif args.depth:
        mode = 'depth'
        in_ch = 1

    device = torch.device('cuda')

    if 'vgg19' in args.arch: 
        model = VGG(in_ch=in_ch, pool_num=args.pool_num, model=args.arch, up_scale=args.up_scale, pretrained=False)
    elif 'resnet' in args.arch:
        model = ResNet(in_ch=in_ch, pool_num=args.pool_num, model=args.arch, up_scale=args.up_scale, pretrained=False)
    elif 'mcnn' in args.arch:
        model = MCNN(in_ch=in_ch)
        args.pool_num = 0
        args.up_scale = 1
    elif 'csrnet' in args.arch:
        model = CSRNet(in_ch=in_ch, pretrained=False)
        args.pool_num = 0
        args.up_scale = 1

    model.to(device)
    print(model)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    model.eval()

    datasets = ShanghaiTechRGBD(
        json_path=os.path.join('json', args.dataset, 'test.json'),
        dataset=args.dataset,
        crop_size=None,
        phase='test',
        model_scale=2**3,
        up_scale=1
    )

    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)

    epoch_minus = []

    plotter = Plotter(args, 4, save_dir=args.save_dir)

    top_str = 'itr\tdiff\tgt_num\testimate'
    logging.info(top_str)
    # Iterate over data.
    for steps, (image, depth, target, num) in enumerate(dataloader):

        tmp_res = 0
        if mode == 'rgb':
            inputs = image.to(device)

        elif mode == 'depth':
            inputs = depth.to(device)

        elif mode == 'both':
            inputs = torch.cat([image, depth], dim=1)
            inputs = inputs.to(device)

        # inputs are images with different sizes
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            tmp_res += torch.sum(outputs).item()

        if steps % 10 == 0:
            plotter(steps, inputs, outputs[0], target[0], 'test', num)

        temp_minu = num[0].item() - tmp_res
        logging.info('{}/{}\t{}\t{}\t{}'.format(steps, len(dataloader), temp_minu, num[0].item(), tmp_res))
        epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    logging.info(log_str)