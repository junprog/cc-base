#!bin/bash

# train
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-v2
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset-v2
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-vgg19-scratch --arch vgg19 --pool-num 4 --dataset synthetic-dataset-v2
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset synthetic-dataset-v2
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset synthetic-dataset-v2 --batch-size 4

# test
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-v2
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset-v2
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19-scratch --arch vgg19 --pool-num 4 --dataset synthetic-dataset-v2
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset synthetic-dataset-v2
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-fusionnet-scratch  --arch fusionnet --pool-num 5 --dataset synthetic-dataset-v2