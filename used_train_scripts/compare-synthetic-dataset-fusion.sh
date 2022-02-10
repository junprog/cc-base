#!bin/bash

# # synthetic-dataset-2d scratch
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d --exp syn-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/syn-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d

# synthetic-dataset-2d IMNet pretrained
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d --exp syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d --pretrained --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d


# synthetic-dataset-2d-bg scratch
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg --exp syn-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d-bg --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d-bg

# synthetic-dataset-2d-bg IMNet pretrained
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg --exp syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d-bg --pretrained --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset-2d-bg


# synthetic-dataset scratch
# done

# synthetic-dataset IMNet pretrained
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic --exp syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset --pretrained --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset


# synthetic-dataset-v2 scratch
# done

# synthetic-dataset-v2 IMNet pretrained
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset-v2 --pretrained --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset synthetic-dataset-v2