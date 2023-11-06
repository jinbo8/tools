#!/usr/bin/python
# coding: utf-8
# @File    : voc_label.py
# @Time    : 2021-08-02 13:51
# @Author  : Zhang Li
# @Email   : lizhang671@gmail.com


import json
import os
from os import getcwd
import cv2
import numpy as np

sets = ['train', 'val', 'test']
classes = ["single", "double", "null", "diy"]
# classes = ["car", "bus", "person", "truck", "motorcycle"]
# classes = ["car", "bus", "pedestrian", "cyclist", "cone", "truck", "van"]


def convert_box(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_corner(size, corner):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    corner[::2]  = (np.array(corner[::2]) * dw).tolist()
    corner[1::2] = (np.array(corner[1::2]) * dh).tolist()
    # from （01左上，23左下，45右下，67右上）
    # to（01左上，23右上，45右下，67左下）
    new_corner = [corner[0], corner[1], corner[6], corner[7], corner[4], corner[5], corner[2], corner[3]]
    return new_corner

# "type": "van",
# "box2d": [
#     1232.5306712405,
#     420.3531487189,
#     1261.2460644612,
#     430.7725570581
# ],
# "corner": [
#     1232.5306712405,
#     420.3531487189,
#     1232.5926915283,
#     430.7725570581,
#     1261.1840441734,
#     430.7725570581,
#     1261.2460644612,
#     420.5392095821
# ],
# "plate_number": "null",
# "plate_color": "2",
# "plate_layer": "1"

# label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
# 关键点依次是（左上，右上，右下，左下） 坐标都是经过归一化，x,y是中心点除以图片宽高，w,h是框的宽高除以图片宽高，ptx，pty是关键点坐标除以宽高

def json2yolo(json_id):
    json_path = os.path.join(dataset_path, "batch123_json", json_id + ".json")
    json_file = open(json_path)
    annos = json.load(json_file)

    img_path = os.path.join(dataset_path, "JPEGImages", json_id+".jpg")
    # img_path = os.path.join(dataset_path, "JPEGImages", json_id.replace("json", "jpg"))
    img = cv2.imread(img_path)
    w = img.shape[1]
    h = img.shape[0]

    out_file = open(dataset_path + '/labels/%s.txt' % json_id, 'w')
    # print(json_path)

    for anno in annos:
        # if anno["plate_layer"] == "null":
        #     anno["plate_layer"] = "1"
        if anno["plate_layer"] and anno["plate_layer"] != "0":
            box = anno["box2d"]
            corner = anno["corner"]
            # plate_color = anno["plate_color"]
            # plate_number = anno["plate_number"]

            if anno["plate_layer"] != "null":
                cls_id = int(anno["plate_layer"])
            else:
                if "_" in anno["plate_number"]:
                    cls_id = 2
                else:
                    cls_id = 1

            if anno["type"] == "other":
                cls_id = 0

            # if anno["type"] == "other":
            #     if anno["plate_color"] == "0":
            #         cls_id = 10
            #     if anno["plate_color"] == "1":
            #         cls_id = 11
            #     if anno["plate_color"] == "2":
            #         cls_id = 12
            #     if anno["plate_color"] == "3":
            #         cls_id = 13
            #     if anno["plate_color"] == "4":
            #         cls_id = 14
            # elif anno["plate_layer"] == "1":
            #     if anno["plate_color"] == "0":
            #         cls_id = 0
            #     if anno["plate_color"] == "1":
            #         cls_id = 1
            #     if anno["plate_color"] == "2":
            #         cls_id = 2
            #     if anno["plate_color"] == "3":
            #         cls_id = 3
            #     if anno["plate_color"] == "4":
            #         cls_id = 4
            # elif anno["plate_layer"] == "2":
            #     if anno["plate_color"] == "0":
            #         cls_id = 5
            #     if anno["plate_color"] == "1":
            #         cls_id = 6
            #     if anno["plate_color"] == "2":
            #         cls_id = 7
            #     if anno["plate_color"] == "3":
            #         cls_id = 8
            #     if anno["plate_color"] == "4":
            #         cls_id = 9
            # else:
            #     continue

            # print(type(box), "\n", box)
            # cls_id = 0
            # cls_id = classes.index(cls)
            # 标注越界修正
            if box[1] > w:
                box[1] = w
            if box[3] > h:
                box[3] = h
            bb = convert_box((w, h), box)
            cc = convert_corner((w,h), corner)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(a) for a in cc]) + '\n')
        else:
            print("no plate")

dataset_path = '/home/zhangli/data-hdd/ccp-lpr_plate_dataset'
img_dir = os.path.join(dataset_path, "JPEGImages")
if __name__ == "__main__":
    for image_set in sets:
        if not os.path.exists(dataset_path + '/labels'):
            os.makedirs(dataset_path + '/labels')
        image_ids = open(dataset_path + '/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
        list_file = open(dataset_path + '/%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write(dataset_path + f'/JPEGImages/{image_id}.jpg\n')
            # if os.path.exists(dataset_path + f'/labels/{image_id}.txt'):
            #     continue
            # else:
            json2yolo(image_id)
        list_file.close()

    # if not os.path.exists(dataset_path + '/labels'):
    #     os.makedirs(dataset_path + '/labels')
    # json_ids = os.listdir("/home/zhangli/data-hdd/ccp-lpr_plate_dataset/batch1_json_raw_modified")
    # for json_id in json_ids:
    #     json2yolo(json_id)

    # count = {0:0, 1:0, 2:0, 3:0}
    # for fn in os.listdir('/home/zhangli/data-hdd/ccp-lpr_batch2/JPEGImages'):
    #     if fn.endswith("txt"):
    #         txt = open(os.path.join('/home/zhangli/data-hdd/ccp-lpr_batch2/JPEGImages', fn))
    #         lines = txt.readlines()
    #         for line in lines:
    #             layer_num = int(line.split(" ")[0])
    #             count[layer_num] += 1
    # print(count)

    # batch2_20221220:
    # layer_num {0: 286, 1: 54212, 2: 1063, 3: 2795} 0层 1层 2层 3其他自定义车牌
    # color: {'blue': 39842, 'green': 13098, 'yellow': 2257, 'white': 2064, 'black': 30, 'null': 1065, 'xxx': 0}
    #
    # added new data from ccpd and crpd
    # layer_num {0: 286, 1: 72700, 2: 2264, 3: 2795}




