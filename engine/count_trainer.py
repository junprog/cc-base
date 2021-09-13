# code refer to https://github.com/ZhihengCV/Bayesian-Crowd-Counting/blob/master/utils/regression_trainer.py
import argparse
import os
from re import S
import time
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from engine.trainer import Trainer
from utils.visualizer import GraphPlotter, Plotter
from utils.helper import worker_init_fn, Save_Handle, AverageMeter

from datasets.ucf_qnrf import UCF_QNRF
from datasets.shanghaitech import ShanghaiTechA, ShanghaiTechB
from datasets.shanghaitech_rgbd import ShanghaiTechRGBD

from models.vgg import VGG
from models.resnet import ResNet
from models.mcnn import MCNN
from models.csrnet import CSRNet

class CountTrainer(Trainer):
    def setup(self):
        """initialize the datasets, model, loss and optimizer"""
        args = self.args
        self.loss_graph = GraphPlotter(self.save_dir, ['Loss'], 'loss')
        self.tr_graph = GraphPlotter(self.save_dir, ['MAE', 'MSE'], 'train')
        self.vl_graph = GraphPlotter(self.save_dir, ['MAE', 'MSE'], 'val')

        crop_size = (args.crop_size, args.crop_size)
        if args.rgb and args.depth:
            self.mode = 'both'
            in_ch = 4
        elif args.rgb:
            self.mode = 'rgb'
            in_ch = 3
        elif args.depth:
            self.mode = 'depth'
            in_ch = 1

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        if 'vgg19' in args.arch: 
            self.model = VGG(in_ch=in_ch, pool_num=args.pool_num, model=args.arch, up_scale=args.up_scale, pretrain=True)
        elif 'resnet' in args.arch:
            self.model = ResNet(in_ch=in_ch, pool_num=args.pool_num, model=args.arch, up_scale=args.up_scale, pretrain=True)
        elif 'mcnn' in args.arch:
            self.model = MCNN(in_ch=in_ch, up_scale=args.up_scale)
        elif 'csrnet' in args.arch:
            self.model = CSRNet(in_ch=in_ch, up_scale=args.up_scale, pretrained=True)

        self.model.to(self.device)
        print(self.model)

        if 'shanghai-tech-rgbd' == args.dataset:
            self.datasets = {x: ShanghaiTechRGBD(
                dataset=args.dataset,
                json_path=os.path.join('json', args.dataset, x +'.json'),
                crop_size=crop_size,
                phase=x,
                model_scale=2 ** args.pool_num,
                up_scale=args.up_scale
            ) for x in ['train', 'val']}
        elif 'shanghai-tech-a' == args.dataset:
            self.datasets = {x: ShanghaiTechA(
                dataset=args.dataset,
                json_path=os.path.join('json', args.dataset, x +'.json'),
                crop_size=crop_size,
                phase=x,
                rescale=False,
                model_scale=2 ** args.pool_num,
                up_scale=args.up_scale
            ) for x in ['train', 'val']}
        elif 'shanghai-tech-b' == args.dataset:
            self.datasets = {x: ShanghaiTechB(
                dataset=args.dataset,
                json_path=os.path.join('json', args.dataset, x +'.json'),
                crop_size=crop_size,
                phase=x,
                model_scale=2 ** args.pool_num,
                up_scale=args.up_scale
            ) for x in ['train', 'val']}

        self.dataloaders = {x: DataLoader(self.datasets[x],
            batch_size=(args.batch_size if x == 'train' else 1),
            shuffle=(True if x == 'train' else False),
            num_workers=args.num_workers*self.device_count,
            pin_memory=(True if x == 'train' else False),
            worker_init_fn=worker_init_fn
        ) for x in ['train', 'val']}

        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(args.max_epoch / 2)], gamma=0.1)
        #self.scheduler = None

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        plotter = Plotter(self.args, 4, save_dir=self.save_dir)

        # Iterate over data.
        for steps, (image, depth, target, num) in enumerate(tqdm(self.dataloaders['train'], ncols=100)):

            if self.mode == 'rgb':
                inputs = image.to(self.device)

            elif self.mode == 'depth':
                inputs = depth.to(self.device)

            elif self.mode == 'both':
                inputs = torch.cat([image, depth], dim=1)
                inputs = inputs.to(self.device)
            
            target = target.to(self.device)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - num.detach().cpu().numpy()
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

            if steps == 0 and self.epoch % 2 == 0:
                plotter(self.epoch, inputs, outputs, target, 'tr')

        logging.info('Epoch {} Train, Loss: {:.8f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))

        self.loss_graph(self.epoch, [epoch_loss.get_avg()])
        self.tr_graph(self.epoch, [epoch_mae.get_avg(), np.sqrt(epoch_mse.get_avg())])

        if self.epoch % self.args.check_point == 0:
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)

            self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []

        plotter = Plotter(self.args, 1, save_dir=self.save_dir)

        # Iterate over data.
        for steps, (image, depth, target, num) in enumerate(tqdm(self.dataloaders['val'], ncols=100)):

            tmp_res = 0
            if self.mode == 'rgb':
                inputs = image.to(self.device)

            elif self.mode == 'depth':
                inputs = depth.to(self.device)

            elif self.mode == 'both':
                inputs = torch.cat([image, depth], dim=1)
                inputs = inputs.to(self.device)

            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                tmp_res += torch.sum(outputs).item()

            res = num[0].item() - tmp_res
            epoch_res.append(res)

            if steps == 0:
                plotter(self.epoch, inputs, outputs, target, 'vl', num)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        self.vl_graph(self.epoch, [mae, mse])

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))