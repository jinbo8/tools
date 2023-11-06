# name: JinboWang
# Dev time: 2022/9/15

import os

path_train = ""
path_val = ""
path_test = ""


def train_val_wr(path_train, path_val, path_test):
    for img_name in os.listdir(path_train):
        abs_path = os.path.join(path_train, img_name)
        print(abs_path)
        # print(type(abs_path))
        with open('./coco/train2017.txt', 'a', encoding='utf-8') as f:
            f.write(abs_path)
            f.write('\n')


    for img_name in os.listdir(path_val):
        abs_path = os.path.join(path_val, img_name)
        print(abs_path)
        # print(type(abs_path))
        with open('./coco/val2017.txt', 'a', encoding='utf-8') as f:
            f.write(abs_path)
            f.write('\n')

    for img_name in os.listdir(path_test):
        abs_path = os.path.join(path_test, img_name)
        print(abs_path)
        # print(type(abs_path))
        with open('./coco/test2017.txt', 'a', encoding='utf-8') as f:
            f.write(abs_path)
            f.write('\n')


if __name__ == '__main__':
    train_val_wr(path_train, path_val, path_test)


# python train.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
#