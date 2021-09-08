# code refer to https://github.com/ZhihengCV/Bayesian-Crowd-Counting/blob/c81c45d50405c36cdcd339006876a04faa742373/utils/trainer.py
import os
import json
import logging
from datetime import datetime

from utils.helper import setlogger

class Trainer(object):
    def __init__(self, args):
        #sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        # if args.rgb and args.depth:
        #     mode = 'both'
        # elif args.rgb:
        #     mode = 'rgb'
        # elif args.depth:
        #     mode = 'depth'

        #sub_dir = args.exp + '-{}'.format(mode)
        #sub_dir = sub_dir + '-{}-{}-p{}-u{}'.format(args.dataset, args.arch, args.pool_num, args.up_scale)
        sub_dir = args.exp

        self.save_dir = os.path.join(args.save_dir, sub_dir)
        assert not os.path.exists(self.save_dir), 'result path: {} has already exist.'.format(self.save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not os.path.exists(os.path.join(self.save_dir, 'images')):
            os.makedirs(os.path.join(self.save_dir, 'images'))

        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger

        with open(os.path.join(self.save_dir, 'args.json'), 'w') as opt_file:
            json.dump(vars(args), opt_file)

        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
            
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        raise NotImplementedError

    def train(self):
        """training one epoch"""
        raise NotImplementedError