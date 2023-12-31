import argparse
import time
import os
import cv2
import torch
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


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



def pure_lp_pred_eval():
    """   作用：使用车牌图像名称作为标签评估模型字符识别性能 """
    license_plate_dir = opt.license_plate
    images = os.listdir(license_plate_dir)
    images_num = len(images)
    sum_correct = 0
    t1 = time.time()
    process_n = 0
    for image_name in images:
        process_n += 1
        image_path = os.path.join(license_plate_dir, image_name)
        image = read_img(image_path)

        plate_chars = ocr.ocr(image, cls=False,
                              det=False)  # 默认即可， 也可以单独注释本行使用以下plate_rec_color_model(torch.Tensor(ims).to(device)) 进行车牌字符识别

        # ocr预测结果：字符+字符串置信度
        # for line in plate_chars:
        #     print(f"line lp::{line}")

        pred_lp_char = plate_chars[0][0][0]
        pred_lp_conf = plate_chars[0][0][1]
        if "_" in image_name:
            new_lp = image_name.split('_')
            lp_true_anno = new_lp[0]
        else:
            lp_true_anno = str(image_name[:-4])

        if pred_lp_char == lp_true_anno:
            sum_correct += 1
            # print(f"\nocr_pred_right：anno_lp=={pred_lp_char}:{lp_true_anno}")
        else:  # 打印识别错误的车牌
            print(f"\nocr_pred_err：anno_lp=={pred_lp_char}:{lp_true_anno}")
            # pass

        print(f"compute process:{process_n}/{images_num}")
    print(f"ocr_rec_acc:{sum_correct / images_num},{sum_correct}/{images_num},{time.time() - t1}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # V1 stable version
    # parser.add_argument('--rec_model', type=str, default='/home/dell/桌面/ccp-lpr_v2/weights/paddleocr_src', help='model.pt path(s)')  # 字符识别模型
    # parser.add_argument('--ocr_char_dict', type=str, default='/home/dell/桌面/ccp-lpr_new_v3/weights/paddleocr_deploy/chinese_plate_dict.txt', help='character dict list')  # 字符识别模型

    # 稳定版
    parser.add_argument('--rec_model', type=str, default='/home/dell/桌面/ccp-lpr_new_v3/weights/paddleocr_deploy', help='model.pt path(s)')  # 字符识别模型
    parser.add_argument('--ocr_char_dict', type=str, default='/home/dell/桌面/ccp-lpr_new_v3/weights/paddleocr_deploy/chinese_plate_dict.txt', help='character dict list')  # 字符识别模型



    parser.add_argument('--license_plate', type=str, default='/media/dell/Elements/eval_data/eval_2/less100', help='eval lp path(s)')  # 评估车牌路径 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    print(opt,'\n')

    """ PaddleOCR模型，字符识别 """
    ocr = PaddleOCR(rec_model_dir=opt.rec_model,
                    rec_char_dict_path=opt.ocr_char_dict,
                    use_angle_cls=False,
                    use_gpu=False)  # use_gpu=False 不使用GPU 进行代码调试，调试结束后再设为True

    # 作用：使用车牌图像名称作为标签评估模型字符识别性能
    pure_lp_pred_eval()