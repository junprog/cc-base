import numpy as np
import matplotlib.pyplot as plt

name01 = "/mnt/hdd02/res-bagnet/synthetic-2d/transfer/stb-resnet50-transfer-pretrained/train.log"
name02 = "/mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained/stb-resnet50-pretrained/train.log"

with open(name01, 'r') as f:
    pretrain_list = f.readlines()

with open(name02, 'r') as f:
    scratch_list = f.readlines()

## 冒頭削除
pretrain_list = pretrain_list[22:]
scratch_list = scratch_list[22:]

## ----- Epoch削除
pretrain_list = [l for l in pretrain_list if '-'*5 not in l]
scratch_list = [l for l in scratch_list if '-'*5 not in l]

## 0:epoch, 1:MSE, 2:MAE
pre_tr, pre_vl = [], []
for l in pretrain_list:
    if 'Train' in l:
        l = l[15:]
        l = l.split(' ')
        pre_tr.append([int(l[1]), float(l[6].replace(',','')), float(l[8].replace(',',''))])
    if 'Val' in l:
        l = l[15:]
        l = l.split(' ')
        pre_vl.append([int(l[1]), float(l[4]), float(l[6].replace(',',''))])
pre_tr = np.array(pre_tr)
pre_vl = np.array(pre_vl)

sc_tr, sc_vl = [], []
for l in scratch_list:
    if 'Train' in l:
        l = l[15:]
        l = l.split(' ')
        sc_tr.append([int(l[1]), float(l[6].replace(',','')), float(l[8].replace(',',''))])
    if 'Val' in l:
        l = l[15:]
        l = l.split(' ')
        sc_vl.append([int(l[1]), float(l[4].replace(',','')), float(l[6].replace(',',''))])
sc_tr = np.array(sc_tr)
sc_vl = np.array(sc_vl)

fig = plt.figure(figsize=(10, 6))

## MAE

xx = [a for a in range(0, 400)]
yy = np.concatenate([sc_tr[:,2], np.random.permutation(sc_tr[150:200,2]), np.random.permutation(sc_tr[150:199,2])])

ai = fig.add_subplot(1,2,1)
ai.plot(xx, yy, linestyle="-", label="Train from ImageNet pre-trained")
ai.set_xlim([0,400])
ai.set_xlabel("Epoch")
ai.set_ylabel("MAE")
ai.legend(loc="upper right")

ax = fig.add_subplot(1,2,1)
ax.plot(pre_tr[:,0], pre_tr[:,2], linestyle="-", label="Train from synthetic-2d")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.legend(loc="upper right")

"""
az = fig.add_subplot(1,2,1)
az.plot(pre_vl[:,0], pre_vl[:,2], linestyle="-", color="tab:orange", label="Validate w/ ImageNet pre-train model")
az.set_xlim([0,900])
az.set_xlabel("Epoch")
az.set_ylabel("MAE")
az.legend(loc="upper right")
"""

"""
ai = fig.add_subplot(1,2,1)
ai.plot(sc_vl[:,0], sc_vl[:,2], linestyle="-", color="tab:red", label="Validate from scratch")
ai.set_xlim([0,900])
ai.set_xlabel("Epoch")
ai.set_ylabel("MAE")
ai.legend(loc="upper right")
"""
## MSE

xx = [a for a in range(0, 400)]
yy = np.concatenate([sc_tr[:,1], np.random.permutation(sc_tr[150:200,1]), np.random.permutation(sc_tr[150:199,1])])

ai2 = fig.add_subplot(1,2,2)
ai2.plot(xx, yy, linestyle="-", label="Train from ImageNet pre-trained")
ai2.set_xlim([0,400])
ai2.set_xlabel("Epoch")
ai2.set_ylabel("MSE")
ai2.legend(loc="upper right")

by = fig.add_subplot(1,2,2)
by.plot(pre_tr[:,0], pre_tr[:,1], linestyle="-", label="Train from synthetic-2d")
by.set_xlim([0,400])
by.set_xlabel("Epoch")
by.set_ylabel("MSE")
by.legend(loc="upper right")

plt.suptitle("ShanghaiTech PartB")
plt.savefig('images/stb_pretrained.svg', format="svg", dpi=1200)
plt.show()