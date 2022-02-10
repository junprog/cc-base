# cc-base

2022/02/10 内田

「境域・広域受容野の融合モデルの群衆カウント」の実装

## 1. 依存ライブラリ (requirements)

* torch
* torchvision
* numpy
* tqdm
* PIL
* scipy

## 2. 使用手順

### 2.1 解像度が非常に大きいデータセット (UCF-QNRF, ShanghaiTech Part A等) の一括リスケール

* (短辺が 512 ~ 2048 に収まる) ∧ (アスペクト比はリスケール前と不変) となるようにリスケール

コマンド
```bash
$ python rescale_dataset.py --data-dir DATA_DIR --dataset DATASET_NAME
```

引数
```yaml
DATA_DIR: データセットの所在するディレクトリのパス
DATASET_NAME: データセット識別子 (詳しくは下部のデータセット識別子を参照) 
```

### 2.2 使用するデータセットの train / val / test のスプリット、 json ファイル作成

コマンド
```bash
$ python create_json.py --data-dir DATA_DIR --val-rate VAL_RATE(default: 4)
```

引数
```yaml
DATA_DIR: (str) データセットの所在するディレクトリのパス
VAL_RATE: (int) バリデーションデータの割合 (1 / VAL_RATE となるようにスプリット) 
```

### 2.3 学習

コマンド
```bash
$ python train.py ...
```

* 引数は `train.py` を参照
* 融合モデルの学習は、各モデルのアーキテクチャを指定し、ロードするウェイトをソース内に指定してください

### 2.4 テスト

コマンド
```bash
$ python test.py ...
```

* 引数は `test.py` を参照


## 識別子

* データセット識別子

```yaml
ShanghaiTech Part A: shanghai-tech-a 
ShanghaiTech Part B: shanghai-tech-b 
ShanghaiTech RGBD: shanghai-tech-rgbd
UCF-QNRF: ucf-qnrf
2D-synthetic: synthetic-dataset-2d
2D-texture-synthetic:  synthetic-dataset-2d-bg
3D-synthetic: synthetic-dataset
3D-texture-synthetic:  synthetic-dataset-v2
```

* モデル識別子

```yaml
VGG19: vgg19
VGG19_bn: vgg19_bn
ResNet50: resnet50
BagNet-9: bagnet9
BagNet-17: bagnet17
BagNet-33: bagnet33
RLGNet: fusionnet
```
