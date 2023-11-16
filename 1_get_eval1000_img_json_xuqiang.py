import os
import shutil

""" 说明： 从2022标注的车牌数据中随机获取1000帧图片和json格式的标签，用作车牌检测和字符识别准确率的计算 """

imgae_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2022年送标车牌数据53645_车牌检测模型训练/2022images_labels/JPEGImages'
new_img = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2022年送标车牌数据53645_车牌检测模型训练/2022images_labels/eval1000img'
names = os.listdir(imgae_dir)
n = 0
# 1. 获取1000张2022年标注的车牌图片
for i in names:
    if n < 1000 and i.startswith('R'):
        old_name = os.path.join(imgae_dir, i)
        new_name = os.path.join(new_img, i)
        n += 1
        shutil.copy(old_name, new_name)
        print(f"PROCESS n:{n}")
    else:
        pass

# 2. 获取1000张2022年标注的车牌图片对应的json标签
old_json_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2022年送标车牌数据53645_车牌检测模型训练/2022images_labels/json_label53645'
new_json_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2022年送标车牌数据53645_车牌检测模型训练/2022images_labels/eval1000json'
m = 0
for j in os.listdir(new_img):
    txt_name = j[:-3] + 'json'
    old_json_name = os.path.join(old_json_dir, txt_name)
    new_json_name = os.path.join(new_json_dir, txt_name)
    m += 1
    shutil.copy(old_json_name, new_json_name)
    print(f"json_copy:{m}")
