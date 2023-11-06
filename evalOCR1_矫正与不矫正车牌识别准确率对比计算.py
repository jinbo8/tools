import argparse
import time
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from paddleocr import PaddleOCR


def lp_ocr_rec(ocr, img, image_name, sum_correct, show_input_lp=False):
    """
    功能： 矫正前/后的车牌字符识别准确行统计
    """

    # 输入ocr的图片可视化
    if show_input_lp:
        new= Image.fromarray(np.uint8(img))
        new.show()

    plate_chars = ocr.ocr(img, cls=False, det=False)  # 默认即可， 也可以单独注释本行使用以下plate_rec_color_model(torch.Tensor(ims).to(device)) 进行车牌字符识别

    # ocr预测结果：字符+字符串置信度
    # for line in plate_chars:
    #     print(f"line lp::{line}")

    pred_plate_license = plate_chars[0][0][0]
    pred_lp_conf = plate_chars[0][0][1]

    label_true = str(image_name[:-4])
    if pred_plate_license == label_true:
        sum_correct += 1
        # print(f"\nocr_predict_true：anno_lp={pred_plate_license}:{label_true}")
    else:  # 打印识别错误的车牌
        # print(f"\nocr_predict_err：anno_lp={pred_plate_license}:{label_true}")
        pass
    return sum_correct




def ocr_pred(ocr, img, image_name, sum_correct, show_input_lp=False):
    """
    功能： 矫正前/后的车牌字符识别准确行统计
    """

    # 输入ocr的图片可视化
    if show_input_lp:
        new= Image.fromarray(np.uint8(img))
        new.show()

    plate_chars = ocr.ocr(img, cls=False, det=False)  # 默认即可， 也可以单独注释本行使用以下plate_rec_color_model(torch.Tensor(ims).to(device)) 进行车牌字符识别

    # ocr预测结果：字符+字符串置信度
    # for line in plate_chars:
    #     print(f"line lp::{line}")

    pred_plate_license = plate_chars[0][0][0]
    pred_lp_conf = plate_chars[0][0][1]

    label_true = str(image_name[:-4])
    if pred_plate_license == label_true:
        sum_correct += 1
        # print(f"\nocr_predict_true：anno_lp={pred_plate_license}:{label_true}")
    else:  # 打印识别错误的车牌
        # print(f"\nocr_predict_err：anno_lp={pred_plate_license}:{label_true}")
        pass
    return sum_correct


def order_points(pts):
    """ 功能：关键点按照（左上，右上，右下，左下）排列
        pts: 车牌在图像上的角点坐标
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    功能： 通过透视变换，对倾斜车牌进行矫正
        image：输入的原始图片
        pts：车牌在图像上的四个角点坐标

    """
    # new = Image.fromarray(np.uint8(image))
    # new.show()
    rect = order_points(pts)  # 排好序的坐标，关键点按照（左上，右上，右下，左下）排列
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # 四边形下边宽度
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # 四边形上边宽度
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # 四边形左边高度
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # 四边形右边高度
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)  # 获得矫正函数参数
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # new2 = Image.fromarray(np.uint8(warped))
    # new2.show()
    return warped  # 返回的是矫正后车牌字符区域图片


def read_json(json_path):
    file = open(json_path, "r", encoding='utf-8')
    points = json.load(file)['shapes'][0]['points']
    points = np.asarray(points)
    return points


def read_img(img_path):
    # 图像的读取与可视化
    # img_path = '/home/dell/桌面/ccp-lpr_v2/data/plate_license_3840_1280_test/1920_1080_plate.png'
    img = Image.open(img_path)
    # img.show()
    img = np.asarray(img)
    img = torch.tensor(img)
    img.permute(2, 0, 1)
    img = np.asarray(img)
    img = img[:,:,:3]
    # new= Image.fromarray(np.uint8(img))
    # new.show()
    # print(f"\n图片名称：{img_path}")
    return img


def ocr_rec_eval(ocr, correction=False):
    path_src = opt.path_src # 包含 json/lp_img的文件夹
    img_path = os.path.join(path_src,'images')
    json_path = os.path.join(path_src,'json')

    img_path_src = img_path
    json_path_src = json_path

    images_name = os.listdir(img_path)
    sum_correct = 0
    t1 = time.time()

    for image_name in images_name:
        img_path = os.path.join(img_path, image_name)   # img path
        j = image_name[:-3]+'json'
        json_path = os.path.join(json_path, j) # json path
        image = read_img(img_path)

        # 是否使用矫正后测车牌进行字符识别
        if correction:
            pts = read_json(json_path)  # 4个角点
            img_correct = four_point_transform(image, pts)  # 车牌矫正
        else:
            img_correct = image  # 使用未矫正的图片进行字符识别打开注释

        sum_correct = ocr_pred(ocr, img_correct, image_name, sum_correct, show_input_lp=False)
        img_path = img_path_src
        json_path = json_path_src

    print(f"Correction：{correction}：{sum_correct}，acc：{sum_correct}{'%'}, run time：{time.time()-t1}{'s'}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color_0.991589_epoth_119_model.pth', help='model.pt path(s)')  # 车牌颜色与单双层识别模型
    parser.add_argument('--path_src', type=str, default='/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/evalTestData/lpOCR', help='eval lp path(s)')  # 评估车牌路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    print(opt)

    """ 初始模型加载， PaddleOCR模型，字符识别 """
    ocr = PaddleOCR(rec_model_dir='/home/dell/桌面/ccp-lpr_v2/weights/paddleocr_src',
                    rec_char_dict_path='../weights/paddleocr_deploy/chinese_plate_dict.txt',
                    use_angle_cls=False,
                    use_gpu=False)  # use_gpu=False 不使用GPU 进行代码调试，调试结束后再设为True

    # 车牌OCR模型评估方式1：根据标注文件json和以标签命名的车牌图像进行ocr识别模型性能评估
    ocr_rec_eval(ocr, correction=True)






