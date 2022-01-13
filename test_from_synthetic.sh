#!bin/bash

# test (no fine-tuning)
# sta
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-vgg19-scratch --arch vgg19 --pool-num 4 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-fusionnet-scratch  --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-vgg19-scratch --arch vgg19 --pool-num 4 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-fusionnet-scratch  --arch fusionnet --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-vgg19-scratch --arch vgg19 --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-fusionnet-scratch  --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf

