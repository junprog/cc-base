#!bin/bash

# sta
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp sta-bagnet33-finetune-only-reg --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-bagnet33-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp sta-resnet50-finetune-only-reg --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp sta-vgg19-finetune-only-reg --arch vgg19 --pool-num 4 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp sta-vgg19_bn-finetune-only-reg --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19_bn-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
#未対応 python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp sta-fusionnet-finetune-only-reg --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp stb-bagnet33-finetune-only-reg --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-bagnet33-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp stb-resnet50-finetune-only-reg --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp stb-vgg19-finetune-only-reg --arch vgg19 --pool-num 4 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp stb-vgg19_bn-finetune-only-reg --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19_bn-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp re_qnrf-bagnet33-finetune-only-reg --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-bagnet33-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51 
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp re_qnrf-resnet50-finetune-only-reg --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp re_qnrf-vgg19-finetune-only-reg --arch vgg19 --pool-num 4 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg --exp re_qnrf-vgg19_bn-finetune-only-reg --arch vgg19_bn --pool-num 4 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19_bn-scratch/best_model.pth --feat-freeze --lr 1e-5 --max-epoch 51

# sta
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/sta-bagnet33-finetune-only-reg --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/sta-resnet50-finetune-only-reg --arch resnet50 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/sta-vgg19-finetune-only-reg --arch vgg19 --pool-num 4 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/sta-vgg19_bn-finetune-only-reg --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-a
#python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/sta-fusionnet-finetune-only-reg  --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/stb-bagnet33-finetune-only-reg --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/stb-resnet50-finetune-only-reg --arch resnet50 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/stb-vgg19-finetune-only-reg --arch vgg19 --pool-num 4 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/stb-vgg19_bn-finetune-only-reg --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-b
#python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/stb-fusionnet-finetune-only-reg  --arch fusionnet --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/re_qnrf-bagnet33-finetune-only-reg --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/re_qnrf-resnet50-finetune-only-reg --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/re_qnrf-vgg19-finetune-only-reg --arch vgg19 --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/re_qnrf-vgg19_bn-finetune-only-reg --arch vgg19_bn --pool-num 4 --dataset rescale-ucf-qnrf
#python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/fine-tune-only-reg/re_qnrf-fusionnet-finetune-only-reg  --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf