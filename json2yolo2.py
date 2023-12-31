import json
import os
import cv2


layer_dict = {"other": 0, "single": 1, "double": 2}


def convert_box(size, plate_box2d):
    """ json格式标签转成yolov7格式 """
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (plate_box2d[0] + plate_box2d[2]) / 2.0 - 1
    y = (plate_box2d[1] + plate_box2d[3]) / 2.0 - 1
    w = plate_box2d[2] - plate_box2d[0]
    h = plate_box2d[3] - plate_box2d[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def json2yolo(image_dir, json_dir, yolo_dir):
    """
        车牌层数: 0自定义车牌other, 1单层车牌single, 2双层车牌double

        每个图像对应一个txt文件，文件每一行为一个目标的信息，包括class, x_center, y_center, width, height格式。格式如下：

        Args:
            image_dir: 图片路径
            json_dir: 标注json标签路径
            yolo_dir: yolo格式的txt路径
    """

    if not os.path.exists(yolo_dir):
        os.mkdir(yolo_dir)
    jsons = os.listdir(json_dir)
    total = len(jsons)
    n = 0
    for j in jsons:
        n += 1
        print(f"process:{n}/{total}")
        # 获取图片的宽度和高度信息
        img_path = os.path.join(image_dir, j[:-4] + "jpg")
        img = cv2.imread(img_path)
        width = img.shape[1]
        height = img.shape[0]
        size = (width, height)
        # 获取标签信息
        json_path = os.path.join(json_dir, j)
        json_file = open(json_path)
        json_datas = json.load(json_file)
        yolo_txt = os.path.join(yolo_dir, j[:-4] + "txt")
        # json转yolov7格式并保存
        with open(yolo_txt, 'a', encoding='utf-8') as f:
            for data in json_datas:
                print(yolo_txt)
                if data["plate_box2d"] != '' and data["plate_layer"] != '':
                    plate_box2d = data["plate_box2d"]
                    plate_layer = layer_dict[data["plate_layer"]]
                    x, y, w, h = convert_box(size, plate_box2d)  # json格式标签转成yolov7格式
                    info = str(plate_layer) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) +'\n'
                    f.write(info)


if __name__ == "__main__":
    image_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/联合标注_景联文/jlwbatch123/batch1/batch1image'
    json_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/联合标注_景联文/jlwbatch123/batch1/batch1label'
    yolo_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/联合标注_景联文/jlwbatch123/batch1/yolov7Det'

    #  景联文：json转yolov7格式并保存
    json2yolo(image_dir, json_dir, yolo_dir)



