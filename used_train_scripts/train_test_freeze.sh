#!bin/bash

python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last --exp sta-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --feat-freeze --dataset shanghai-tech-a --batch-size 4
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last --exp stb-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --feat-freeze --dataset shanghai-tech-b --batch-size 2

python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/sta-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --dataset shanghai-tech-a