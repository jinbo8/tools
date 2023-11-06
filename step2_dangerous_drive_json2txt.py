
import json
import os


""" 危险驾驶行为检测：清汽研标注的json数据z转换成yolov7训练需要的txt格式 """
det_res_dict = {'driver_call': 0, 'driver_smoke': 1, 'driver_nobelt': 2, 'driver_handsoff': 3, 'driver_ordinary': 4, 'driver_unknown': 5, 'front_windshield': 6, "driver_empty": 7}
                 # 'driver_call',   'driver_smoke',    'driver_nobelt',     'driver_handsoff',   'driver_ordinary',    'driver_unknown',    'front_windshield',     'driver_empty'
# key_class_dicts = {0: '6001',        1: '6002',          2: '6003',         3: '6004',             4: '6005',           5: '6006',            6: '6007'                           }  # 事件库编号


def xyxy2cx_cy(class_name, xyxy, img_w, img_h):
    """ 1.输出归一化后yolo坐标格式 ;
        2.驾驶行为映射成yolov7训练的数字
     """
    dw = 1. / (img_w)
    dh = 1. / (img_h)
    x = (xyxy[0] + xyxy[2]) / 2.0  # c_x
    y = (xyxy[1] + xyxy[3]) / 2.0  # c_y
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    c_x = x * dw
    box_w = w * dw
    c_y = y * dh
    box_h = h * dh

    class_num = det_res_dict[class_name]  # 使用字典映射成编号: 0-7

    return class_num, c_x, c_y, box_w, box_h


def windshield(box2dcorner, img_w, img_h, front_windshield='front_windshield'):
    """  1.输出归一化后yolo坐标格式; 2.挡风玻璃转换成YOLOv7训练的数据格式标签 """

    dw = 1. / (img_w)
    dh = 1. / (img_h)
    x = (box2dcorner[0] + box2dcorner[4]) / 2.0
    y = (box2dcorner[1] + box2dcorner[5]) / 2.0
    w = box2dcorner[4] - box2dcorner[0]
    h = box2dcorner[5] - box2dcorner[1]
    c_x = x * dw
    box_w = w * dw
    c_y = y * dh
    box_h = h * dh
    class_num = det_res_dict[front_windshield]
    return class_num, c_x, c_y, box_w, box_h


def decode_json2txt(json_path, json_name, label_txt_path):
    """ 将json格式的标注标签全部写入到txt文件，包含前挡风玻璃 """
    json2txt_res = []
    win = []

    txt_name = json_name[0:-5] + '.txt'  # 生成txt文件你想存放的路径
    txt_name = os.path.join(label_txt_path, txt_name)
    json_path = os.path.join(json_path, json_name)

    with open(txt_name, 'a', encoding='utf-8') as f:
        data = json.load(open(json_path, 'r', encoding='utf-8'))
        print(f'data:{data}')
        per_img_labels_leng = len(data)

        # 遍历每张图片的所有标签
        for i in range(per_img_labels_leng):
            class_name = data[i]['type']
            img_w = data[i]['width']
            img_h = data[i]['height']
            xyxy = data[i]['box2d']
            box2dcorner = data[i]['box2dcorner']  # 挡风玻璃坐标

            if class_name in ['多选&&&&driver_unknown', 'driver_callanddriver_handsoff', 'driver_callanddriver_smoke', 'driver_smokeanddriver_handsoff', 'driver_handsoffanddriver_nobelt', 'driver_callanddriver_smokeanddriver_handsoff',
                              'driver_callanddriver_handsoffanddriver_nobelt', 'driver_callanddriver_nobelt', 'driver_smokeanddriver_nobelt', 'driver_callanddriver_smokeanddriver_nobelt']:
                break

            if class_name != "driver_empty":
                clss, c_x, c_y, w, h = xyxy2cx_cy(class_name, xyxy, img_w, img_h)  # 1.输出归一化后yolo坐标格式; 2.驾驶行为映射成yolov7训练的数字
                json2txt_res.append(clss)
                json2txt_res.append(c_x)
                json2txt_res.append(c_y)
                json2txt_res.append(w)
                json2txt_res.append(h)
                f.write(" ".join([str(a) for a in json2txt_res]) + '\n')

                json2txt_res = []

                win_clss, win_c_x, win_c_y, win_w, win_h = windshield(box2dcorner, img_w, img_h, front_windshield='front_windshield')
                win.append(win_clss)
                win.append(win_c_x)
                win.append(win_c_y)
                win.append(win_w)
                win.append(win_h)
                f.write(" ".join([str(m) for m in win]) + '\n')
                win = []


def json2txt(json_path, txt_path):

    """ create new dir"""
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)

    json_names = os.listdir(json_path)
    for json_name in json_names:
        decode_json2txt(json_path, json_name, txt_path)    # 2.生成包含挡风玻璃的YOLOv7训练集格式的标签


if __name__ == "__main__":
    """ 将标注的危险驾驶行为json格式的标注文件转换成yolov7训练的txt相对坐标格式 """

    json_path = "/media/dell/sata4t/jwang/datasets/TYJTdatalabel2023/dangerousDrive/dangerDriver_20230731/label"  # 危险驾驶标注的json文件夹路径
    txt_path = "/media/dell/sata4t/jwang/datasets/TYJTdatalabel2023/dangerousDrive/dangerDriver_20230731/label_txt"   # 转换后的txt标签文件存储的文件夹路径

    # 危险驾驶苏州本地：20231026数据格式转换
    json2txt(json_path, txt_path)

