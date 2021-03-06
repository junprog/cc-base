import torch
import os
import numpy as np

from utils.visualizer import Plotter
from utils.helper import setlogger

from datasets.shanghaitech_rgbd import ShanghaiTechRGBD
from datasets.shanghaitech import ShanghaiTechA, ShanghaiTechB
from datasets.ucf_qnrf import UCF_QNRF
from datasets.synthetic_datas import SyntheticDataset

from models.vgg import VGG
from models.resnet import ResNet
from models.mcnn import MCNN
from models.csrnet import CSRNet
from models.bagnet import BagNet
from models.fusion_model import BagResNet

from metrics.error import game

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
    parser.add_argument('--sigma', type=int, default=15,
                        help='a gaussian filter parameter')

    # model
    parser.add_argument('--arch', type=str, default='vgg19',
                        help='the model architecture [vgg19, vgg19_bn, resnet16, resnet50, resnet101, bagnet33, bagnet17, bagnet9]')
    parser.add_argument('--pool-num', type=int, default=3,
                        help='pool num')
    parser.add_argument('--up-scale', type=int, default=1,
                        help='up scale num')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip('-')  # set vis gpu

    preffix = 'transfer_'

    log_file_name = args.dataset + '_' + preffix + 'test.log'

    setlogger(os.path.join(args.save_dir, log_file_name))  # set logger

    device = torch.device('cuda')

    if 'vgg19' in args.arch: 
        model = VGG(in_ch=3, arch=args.arch, pool_num=args.pool_num, up_scale=args.up_scale, pretrained=False)
    elif 'resnet' in args.arch:
        model = ResNet(in_ch=3, arch=args.arch, pool_num=args.pool_num, up_scale=args.up_scale, pretrained=False)
    elif 'bagnet' in args.arch:
        model = BagNet(in_ch=3, arch=args.arch, pool_num=args.pool_num, up_scale=args.up_scale, pretrained=False)
    elif 'fusionnet' in args.arch:
        model = BagResNet(pool_num=args.pool_num, pretrained=False)

    elif 'mcnn' in args.arch:
        model = MCNN(in_ch=3, up_scale=args.up_scale)
    elif 'csrnet' in args.arch:
        model = CSRNet(in_ch=3, up_scale=args.up_scale, pretrained=False)

    model.to(device)
    print(model)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, '400_ckpt.tar'), device)['model_state_dict'])
    model.eval()

    if args.dataset == 'shanghai-tech-a':
        datasets = ShanghaiTechA(
            dataset=args.dataset,
            arch=args.arch,
            json_path=os.path.join('json', args.dataset, 'test.json'),
            crop_size=None,
            phase='test',
            rescale=False,
            sigma=args.sigma,
            pool_num=args.pool_num,
            up_scale=args.up_scale
        )
    
    elif args.dataset == 'shanghai-tech-b':
        datasets = ShanghaiTechB(
            dataset=args.dataset,
            arch=args.arch,
            json_path=os.path.join('json', args.dataset, 'test.json'),
            crop_size=None,
            phase='test',
            sigma=args.sigma,
            pool_num=args.pool_num,
            up_scale=args.up_scale
        )

    elif args.dataset == 'shanghai-tech-rgbd':
        datasets = ShanghaiTechRGBD(
            json_path=os.path.join('json', args.dataset, 'test.json'),
            dataset=args.dataset,
            crop_size=None,
            phase='test',
            model_scale=2**3,
            up_scale=1
        )

    elif args.dataset == 'ucf-qnrf':
        datasets = UCF_QNRF(
            dataset=args.dataset,
            arch=args.arch,
            json_path=os.path.join('json', args.dataset, 'test.json'),
            crop_size=None,
            phase='test',
            rescale=False,
            sigma=args.sigma,
            pool_num=args.pool_num,
            up_scale=args.up_scale
        )
    
    elif 'rescale-ucf-qnrf' == args.dataset:
        datasets = UCF_QNRF(
            dataset=args.dataset,
            arch=args.arch,
            json_path=os.path.join('json', args.dataset, 'test.json'),
            crop_size=None,
            phase='test',
            rescale=True,
            sigma=args.sigma,
            pool_num=args.pool_num,
            up_scale=args.up_scale
        )

    elif args.dataset == 'synthetic-dataset' or args.dataset == 'synthetic-dataset-v2' or args.dataset == 'synthetic-dataset-2d' or args.dataset == 'synthetic-dataset-2d-bg':
        datasets = SyntheticDataset(
            dataset=args.dataset,
            arch=args.arch,
            json_path=os.path.join('json', args.dataset, 'test.json'),
            crop_size=None,
            phase='test',
            sigma=args.sigma,
            pool_num=args.pool_num,
            up_scale=args.up_scale
        )

    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)

    epoch_minus = []

    plotter = Plotter(args, 4, save_dir=args.save_dir)

    top_str = 'path\titr\tdiff\tgt_num\testimate'
    logging.info(top_str)

    res_game = [0, 0, 0, 0]
    # Iterate over data.
    for steps, (image, target, num, path) in enumerate(dataloader):

        tmp_res = 0

        inputs = image.to(device)

        # inputs are images with different sizes
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            tmp_res += torch.sum(outputs).item()

        for L in range(4):
            abs_error, square_error = game(outputs, target, L)
            res_game[L] += abs_error

        if steps % 10 == 0:
            plotter(steps, inputs, outputs[0], target[0], 'test', num)

        temp_minu = num[0].item() - tmp_res
        logging.info('{}\t{}/{}\t{}\t{}\t{}'.format(path, steps, len(dataloader), temp_minu, num[0].item(), tmp_res))
        epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    N = len(dataloader)
    res_game = [m / N for m in res_game]
    log_str = 'Final Test: mae {}, game1 {}, game2 {}, game3 {}, mse {}'.format(mae, res_game[1], res_game[2], res_game[3], mse)
    logging.info(log_str)