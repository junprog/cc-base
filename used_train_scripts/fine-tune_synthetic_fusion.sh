#!bin/bash

## fusion stream は学習する。よって他のfine-tuneモデルたちより学習パラメータ多。

# sta
#python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/fine-tune-only-reg --exp sta-fusionnet-finetune-only-reg --arch fusionnet --pool-num 5 --dataset shanghai-tech-a --feat-freeze --lr 1e-5 --max-epoch 51

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/fine-tune-only-reg --exp stb-fusionnet-finetune-only-reg --arch fusionnet --pool-num 5 --dataset shanghai-tech-b --feat-freeze --lr 1e-5 --max-epoch 51 --batch-size 2

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/fine-tune-only-reg --exp re_qnrf-fusionnet-finetune-only-reg --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf --feat-freeze --lr 1e-5 --max-epoch 51

# sta
#python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/fine-tune-only-reg/sta-fusionnet-finetune-only-reg  --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/fine-tune-only-reg/stb-fusionnet-finetune-only-reg  --arch fusionnet --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/fine-tune-only-reg/re_qnrf-fusionnet-finetune-only-reg  --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf