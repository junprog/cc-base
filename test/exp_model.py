import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加

import torch
from models.fusion_model import LinearFusionModel

inputs = torch.rand(8, 3, 256, 256)
model = LinearFusionModel()

outputs = model(inputs, inputs)

print(model)
print(outputs.shape)