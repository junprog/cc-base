import numpy as np
import matplotlib.pyplot as plt

name010 = "/mnt/hdd02/res-bagnet/synthetic-2d/transfer/stb-resnet50-transfer-pretrained/train.log"
name011 = "/mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/stb-resnet50-transfer-pretrained/train.log"
name012 = "/mnt/hdd02/res-bagnet/synthetic/transfer/stb-resnet50-transfer-pretrained/train.log"
name013 = "/mnt/hdd02/res-bagnet/synthetic-v2/transfer/stb-resnet50-transfer-pretrained/train.log"

name02 = "/mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained/stb-resnet50-pretrained-400ep/train.log"

with open(name010, 'r') as f:
    syn_2d_list = f.readlines()
with open(name011, 'r') as f:
    syn_2d_bg_list = f.readlines()
with open(name012, 'r') as f:
    syn_3d_list = f.readlines()
with open(name013, 'r') as f:
    syn_3d_bg_list = f.readlines()

with open(name02, 'r') as f:
    scratch_list = f.readlines()

## 冒頭削除
syn_2d_list = syn_2d_list[22:]
syn_2d_bg_list = syn_2d_bg_list[22:]
syn_3d_list = syn_3d_list[22:]
syn_3d_bg_list = syn_3d_bg_list[22:]
scratch_list = scratch_list[22:]

## ----- Epoch削除
syn_2d_list = [l for l in syn_2d_list if '-'*5 not in l]
syn_2d_bg_list = [l for l in syn_2d_bg_list if '-'*5 not in l]
syn_3d_list = [l for l in syn_3d_list if '-'*5 not in l]
syn_3d_bg_list = [l for l in syn_3d_bg_list if '-'*5 not in l]
scratch_list = [l for l in scratch_list if '-'*5 not in l]

## 0:epoch, 1:MSE, 2:MAE
s2_tr, s2_vl = [], []
for l in syn_2d_list:
    if 'Train' in l:
        l = l[15:]
        l = l.split(' ')
        s2_tr.append([int(l[1]), float(l[6].replace(',','')), float(l[8].replace(',',''))])
    if 'Val' in l:
        l = l[15:]
        l = l.split(' ')
        s2_vl.append([int(l[1]), float(l[4]), float(l[6].replace(',',''))])
s2_tr = np.array(s2_tr)
s2_vl = np.array(s2_vl)

s2b_tr, s2b_vl = [], []
for l in syn_2d_bg_list:
    if 'Train' in l:
        l = l[15:]
        l = l.split(' ')
        s2b_tr.append([int(l[1]), float(l[6].replace(',','')), float(l[8].replace(',',''))])
    if 'Val' in l:
        l = l[15:]
        l = l.split(' ')
        s2b_vl.append([int(l[1]), float(l[4]), float(l[6].replace(',',''))])
s2b_tr = np.array(s2b_tr)
s2b_vl = np.array(s2b_vl)

s3_tr, s3_vl = [], []
for l in syn_3d_list:
    if 'Train' in l:
        l = l[15:]
        l = l.split(' ')
        s3_tr.append([int(l[1]), float(l[6].replace(',','')), float(l[8].replace(',',''))])
    if 'Val' in l:
        l = l[15:]
        l = l.split(' ')
        s3_vl.append([int(l[1]), float(l[4]), float(l[6].replace(',',''))])
s3_tr = np.array(s3_tr)
s3_vl = np.array(s3_vl)

s3b_tr, s3b_vl = [], []
for l in syn_3d_bg_list:
    if 'Train' in l:
        l = l[15:]
        l = l.split(' ')
        s3b_tr.append([int(l[1]), float(l[6].replace(',','')), float(l[8].replace(',',''))])
    if 'Val' in l:
        l = l[15:]
        l = l.split(' ')
        s3b_vl.append([int(l[1]), float(l[4]), float(l[6].replace(',',''))])
s3b_tr = np.array(s3b_tr)
s3b_vl = np.array(s3b_vl)

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

# xx = [a for a in range(0, 400)]
# yy = np.concatenate([sc_tr[:,2], np.random.permutation(sc_tr[250:300,2]), np.random.permutation(sc_tr[250:299,2])])
# print(yy.shape)

ai = fig.add_subplot(1,2,1)
ai.plot(sc_tr[:,0], sc_tr[:,2], linestyle="-", label="Train from IN")
ai.set_xlim([0,400])
ai.set_xlabel("Epoch")
ai.set_ylabel("MAE")
ai.legend(loc="upper right")

ax = fig.add_subplot(1,2,1)
ax.plot(s2_tr[:,0], s2_tr[:,2], linestyle="-", label="Train from IN → 2d")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.legend(loc="upper right")

ax = fig.add_subplot(1,2,1)
ax.plot(s2b_tr[:,0], s2b_tr[:,2], linestyle="-", label="Train from IN → 2d-tex")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.legend(loc="upper right")

ax = fig.add_subplot(1,2,1)
ax.plot(s3_tr[:,0], s3_tr[:,2], linestyle="-", label="Train from IN → 3d")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.legend(loc="upper right")

ax = fig.add_subplot(1,2,1)
ax.plot(s3b_tr[:,0], s3b_tr[:,2], linestyle="-", label="Train from IN → 3d-tex")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.legend(loc="upper right")

## MSE
# xx = [a for a in range(0, 400)]
# yy = np.concatenate([sc_tr[:,1], np.random.permutation(sc_tr[250:275,1]), np.random.permutation(sc_tr[250:275,1]), np.random.permutation(sc_tr[250:299,1])])

ai2 = fig.add_subplot(1,2,2)
ai2.plot(sc_tr[:,0], sc_tr[:,1], linestyle="-", label="Train from IN")
ai2.set_xlim([0,400])
ai2.set_xlabel("Epoch")
ai2.set_ylabel("RMSE")
ai2.legend(loc="upper right")

by = fig.add_subplot(1,2,2)
by.plot(s2_tr[:,0], s2_tr[:,1], linestyle="-", label="Train from IN → 2d")
by.set_xlim([0,400])
by.set_xlabel("Epoch")
by.set_ylabel("RMSE")
by.legend(loc="upper right")

ax = fig.add_subplot(1,2,2)
ax.plot(s2b_tr[:,0], s2b_tr[:,1], linestyle="-", label="Train from IN → 2d-tex")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE")
ax.legend(loc="upper right")

ax = fig.add_subplot(1,2,2)
ax.plot(s3_tr[:,0], s3_tr[:,1], linestyle="-", label="Train from IN → 3d")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE")
ax.legend(loc="upper right")

ax = fig.add_subplot(1,2,2)
ax.plot(s3b_tr[:,0], s3b_tr[:,1], linestyle="-", label="Train from IN → 3d-tex")
ax.set_xlim([0,400])
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE")
ax.legend(loc="upper right")

plt.suptitle("ShanghaiTech Part B")
plt.tight_layout()
plt.savefig('images/stb_pretrained_all.svg', format="svg", dpi=1200)
plt.show()