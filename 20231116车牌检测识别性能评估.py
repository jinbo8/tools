import os
import json
import numpy as np


def calculate_iou(box1, box2):
    """
    功能：以中心坐标格式为例计算 标注bbox1 和预测车牌框bbox2之间IOU
    Args:
        box1: 标注的标签车牌框bbox左上角/右下角坐标
        box2: 模型检测到的车牌框bbox左上角/右下角坐标
    Returns: 两个框的IOU

    """
    x1, y1, w1, h1 = int(box1[0]),int(box1[1]),int(box1[2]),int(box1[3])
    x2, y2, w2, h2 = int(box2[0]),int(box2[1]),int(box2[2]),int(box2[3])

    # 计算每个框的上下左右边线的坐标
    y1_max = y1 + h1 / 2
    x1_max = x1 + w1 / 2
    y1_min = y1 - h1 / 2
    x1_min = x1 - w1 / 2

    y2_max = y2 + h2 / 2
    x2_max = x2 + w2 / 2
    y2_min = y2 - h2 / 2
    x2_min = x2 - w2 / 2

    # 上取小下取大，右取小左取大
    xx1 = np.max([x1_min, x2_min])
    yy1 = np.max([y1_min, y2_min])
    xx2 = np.min([x1_max, x2_max])
    yy2 = np.min([y1_max, y2_max])

    # 计算各个框的面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算相交的面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算IoU
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def label_boxes(label_path):
    """ 功能：获取标注的车牌框坐标、车牌字符串 """
    label_list = []
    lp_chars_list = []
    not_null_bbox_num = 0

    with open(label_path, 'r', encoding='utf-8') as label_file:
        label_datas = json.load(label_file)
        if len(label_datas) != 0:           # 获取标注的非空车牌信息
            for label_data in label_datas:  # 遍历标注的标签
                # 统计标注2D框
                label_box2d = label_data["box2d"]
                label_list.append(label_box2d)

                # 统计非空车牌
                lp_char = label_data['plate_number']
                if lp_char != 'null':
                    not_null_bbox_num += 1
                    lp_chars_list.append(lp_char)

    return label_list, lp_chars_list, not_null_bbox_num


def pred_boxs(pred_path, char_len_thred=7):
    """ 功能：获取模型预测得到的车牌框坐标、车牌字符集 """
    pred_2dbbox_list = []           # 模型预测得到的车牌框坐标集
    pred_lp_chars_list = []  # 模型预测得到的车牌集

    with open(pred_path, 'r', encoding='utf-8') as pred_file:
        pred_data = json.load(pred_file)
        if len(pred_data) != 0:
            for lp in pred_data.values():  # 获取预测的非空车牌信息
                # （1）预测车牌框
                if len(lp['box']) > 0:     # 车牌框坐标非空
                    pred_box2d = lp['box']
                    pred_2dbbox_list.append(pred_box2d)
                # （2）车牌字符长度大于阈值 char_len_thred
                if len(lp['plate_char']) >= char_len_thred:  # 车牌字符长度大于阈值，统计车牌字符
                    pred_lp_char = lp['plate_char']
                    pred_lp_chars_list.append(pred_lp_char)

    return pred_2dbbox_list, pred_lp_chars_list


def calculate_bbox_det_acc_reacall(label_dir, pred_dir, pred_iou_thred=0.5, char_len_thred=7):
    label_jsons = os.listdir(label_dir)

    label_bbox_total_nums = 0    # 标注bbox总数量
    label_lp_total_num = 0       # 标注车牌字符总数量
    label_not_null_bbox_num = 0  # 同时标注bbox+车牌字符的框数量

    pred_bbox_total_nums = 0     # 预测bbox总数量
    pred_lp_char_total_num = 0   # 预测得到的车牌长度符合条件的车牌总数量

    pred_box_right_num = 0       # 车牌框bbox检测结果正确的总数量
    lp_ocr_right = 0             # 车牌OCR正确数量

    pred_ocr_all_right = 0       # 车牌字符OCR识别正确的总数量


    for label_json in label_jsons:
        label_path = os.path.join(label_dir, label_json)
        pred_path = os.path.join(pred_dir, label_json)

        # label_boxs_list：标注框总量， label_lp_chars_list：非空字符串集，label_not_null_bbox_num_tmp：非空字符串集对应bbox集
        label_boxs_list, label_lp_chars_list, label_not_null_bbox_num_tmp = label_boxes(label_path)
        label_bbox_total_nums += len(label_boxs_list)   # 标注框总量
        label_lp_total_num += len(label_lp_chars_list)  # label:带字符的车牌数量
        label_not_null_bbox_num += label_not_null_bbox_num_tmp  # 非空字符串集对应bbox数量

        # pred_boxs_list:预测车牌bbox，pred_lp_chars_list：预测的车牌字符长度大于阈值的车牌字符集
        pred_boxs_list, pred_lp_chars_list = pred_boxs(pred_path, char_len_thred=char_len_thred)
        pred_bbox_total_nums += len(pred_boxs_list)         # predit bbox 数量
        pred_lp_char_total_num += len(pred_lp_chars_list)   # predict：车牌字符长度符合条件的车牌数量


        # (1) 当标签和预测得到的2D bbox都不为空时，计算bbox检测的准确率, (1) 车牌框预测统计
        if len(label_boxs_list) != 0 and len(pred_boxs_list) != 0:
            for box1 in label_boxs_list:
                for box2 in pred_boxs_list:
                    iou = calculate_iou(box1, box2)
                    if iou > pred_iou_thred:
                        pred_box_right_num += 1   # 计算车牌框预测正确数量

        # (2) 当标签和预测得到的车牌字符都不为空时，计算车牌字符识别的准确率。
        if len(label_lp_chars_list) != 0 and len(pred_lp_chars_list) != 0:
            for label_char in label_lp_chars_list:
                for pred_char in pred_lp_chars_list:
                    if label_char == pred_char:
                        lp_ocr_right += 1   # 计算车牌OCR正确数量

        # (3) 计算车牌检测和识别全正确的准确率, 标注结果字典， key:车牌字符， value: 车牌bbox坐标
        label_dict = {}
        for k, v in zip(label_lp_chars_list, label_boxs_list):
            label_dict[k] = v

        # (4) 预测结果字典， key:车牌字符， value: 2D bbox
        pred_dict = {}
        for kp, vp in zip(pred_lp_chars_list, pred_boxs_list):
            pred_dict[kp] = vp

        for lb_char, lb_bbox in label_dict.items():       # 遍历标注标签
            for prd_char, prd_bbox in pred_dict.items():  # 遍历预测保存结果
                iou = calculate_iou(lb_bbox, prd_bbox)    # box1: 标注车牌框bbox坐标标签
                if iou > pred_iou_thred:
                    if lb_char == prd_char:
                        pred_ocr_all_right += 1

    print(f"********************************************* 车牌框bbox *********************************************************")
    print(f"标注bbox总量:{label_bbox_total_nums}, 标注非空bbox:{label_not_null_bbox_num}，空box:{label_bbox_total_nums-label_not_null_bbox_num};\n预测bbox总量:{pred_bbox_total_nums}, 预测正确bbox数量:{pred_box_right_num},")
    print(f"车牌框检测 Acc:{pred_box_right_num / pred_bbox_total_nums}; 车牌框检测 Recall:{pred_box_right_num / label_bbox_total_nums}")


    print(f"********************************************* 车牌字符识别 *******************************************************")
    print(f"带字符车牌标签总量:{label_lp_total_num}, \n预测车牌总量（长度大于7）：{pred_lp_char_total_num}，车牌字符识别正确数量:{lp_ocr_right}；\n车牌框+车牌字符识别都正确数量：{pred_ocr_all_right}")
    print(f"车牌字符识别 Acc:{lp_ocr_right/pred_lp_char_total_num}，Recall {lp_ocr_right / label_lp_total_num}")
    print(f"检测+识别正确Acc:{pred_ocr_all_right/pred_lp_char_total_num}, Recall:{pred_ocr_all_right / label_lp_total_num}")
    print("**" * 56)


if __name__ == '__main__':
    label_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/license_plate_eval/许强评估数据/v2_苏州标注车牌数据1000/eval1000json'
    pred_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/license_plate_eval/许强评估数据/v2_苏州标注车牌数据1000/jwangPredict/detOCRPredictResJSON'

    calculate_bbox_det_acc_reacall(label_dir, pred_dir, pred_iou_thred=0.70, char_len_thred=7)


# ********************************************* 车牌框bbox *********************************************************
# 标注bbox总量:3608, 标注非空bbox:1616，空box:1992;
# 预测bbox总量:2381, 预测正确bbox数量:2362,
# 车牌框检测 Acc:0.9920201595968081; 车牌框检测 Recall:0.6546563192904656
# ********************************************* 车牌字符识别 *******************************************************
# 带字符车牌标签总量:1616,
# 预测车牌总量（长度大于7）：1066，车牌字符识别正确数量:649；
# 车牌框+车牌字符识别都正确数量：478
# 车牌字符识别 Acc:0.6088180112570356，Recall 0.40160891089108913
# 检测+识别正确Acc:0.44840525328330205, Recall:0.2957920792079208
# ****************************************************************************************************************

