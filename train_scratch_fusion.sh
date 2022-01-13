#!bin/bash

#sta
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp sta-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset shanghai-tech-a --batch-size 2

#stb
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp stb-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset shanghai-tech-b --batch-size 1

#ucf
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp re_qnrf-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf --batch-size 2



# sta
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/sta-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/stb-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/re_qnrf-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf