# cc-base

gt: 頭部アノテーション
density: 密度マップ
bbox: バウンディングボックス

image: RGB画像
depth: 深度マップ
temperature: 温度マップ

```python
dataset_dict = {
    'part_A': 'shanghai-tech-a', 
    'part_B': 'shanghai-tech-b', 
    'RGBD': 'shanghai-tech-rgbd', 
    'UCF-QNRF': 'ucf-qnrf',
    'UCF_CC_50': 'ucf-cc-50',
    'NWPU': 'nwpu-crowd', 
    'RGBT': 'rgbt-cc'
}
```