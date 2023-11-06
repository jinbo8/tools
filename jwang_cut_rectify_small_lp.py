import os
import json
import time
import numpy as np
import cv2


def creat_dirs(save_dirs):
    for i in save_dirs:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            pass

def order_points(pts):  # 关键点按照（左上，右上，右下，左下）排列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):  #透视变换
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def get_split_merge(img):
    h,w,c = img.shape
    img_upper = img[0:int(5/12*h),:]
    img_lower = img[int(1/3*h):,:]
    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
    new_img = np.hstack((img_upper,img_lower))
    return new_img


def get_small_lp(json_dir, image_dir, yellow_2layer, rectify_yellow_2layer, rectify_small_common_lp):

    jsons = os.listdir(json_dir)
    total = len(jsons)
    n = 0
    for j in jsons:
        n += 1
        json_path = os.path.join(json_dir, j)
        json_file = open(json_path)
        json_datas = json.load(json_file)
        for data in json_datas:
            # 第一批车牌标注：颜色数字对应关系 {蓝牌：0, 绿牌：1,黄牌:2, 白牌:3, 黑牌:4}
            #  plate_layer:0:无牌车, 1:单层, 2：双层
            pic_ = os.path.join(image_dir, j[:-4] + 'jpg')
            img = cv2.imread(pic_)  # 得到BGR图像
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # BGR
            # cv2.imshow('Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            corners = data['corner']
            landmarks_np = np.asarray(corners).reshape((4, 2))

            # 获取根据四个角点校正车牌
            lp_correction = four_point_transform(img, landmarks_np)
            lp_save_name = data['plate_number']

            #截取双层车牌矫正后保存
            if data['plate_layer']=='2' and data['plate_color']=='2' and data['plate_number']!="null":
                suffix = str(int(1000*time.time()))[-5:]
                if (lp_save_name+'.jpg') not in os.listdir(yellow_2layer):
                    lp_save_path = os.path.join(yellow_2layer, lp_save_name+'.jpg')
                    cv2.imwrite(lp_save_path, lp_correction)
                else:
                    lp_save_path = os.path.join(yellow_2layer, lp_save_name+"_"+suffix+'.jpg')
                    cv2.imwrite(lp_save_path, lp_correction)

                # 将矫正后的双层车牌进行上下层拼接后保存
                correction2splicing = get_split_merge(lp_correction)
                if (lp_save_name+'.jpg') not in os.listdir(rectify_yellow_2layer):
                    correction2splicing_save_path = os.path.join(rectify_yellow_2layer, lp_save_name+'.jpg')
                    cv2.imwrite(correction2splicing_save_path, correction2splicing)
                else:
                    correction2splicing_save_path = os.path.join(rectify_yellow_2layer, lp_save_name+"_"+suffix+'.jpg')
                    cv2.imwrite(correction2splicing_save_path, correction2splicing)
            else:
                #对单层非黄色车牌进行保存
                suffix = str(int(1000 * time.time()))[-5:]
                if (lp_save_name + '.jpg') not in os.listdir(rectify_small_common_lp):
                    lp_save_path = os.path.join(rectify_small_common_lp, lp_save_name + '.jpg')
                    cv2.imwrite(lp_save_path, lp_correction)
                else:
                    lp_save_path = os.path.join(rectify_small_common_lp, lp_save_name + "_" + suffix + '.jpg')
                    cv2.imwrite(lp_save_path, lp_correction)

        print(f"process:{n}/{total}")

if __name__ == '__main__':
    """  
        功能：
            双层黄色车牌截取、矫正、拼接与保存
    """
    # json格式标签/大图片文件夹
    json_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/tyjt20222lp/jsons'
    image_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/tyjt20222lp/images'

    # 截取的双层车牌矫正后/拼接后车牌保存文件夹
    yellow_2layer = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/tyjt20222lp/cut_res/yellow_2layer'
    rectify_yellow_2layer = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/tyjt20222lp/cut_res/rectify_yellow_2layer'
    # 截取单层车牌矫正后保存的文件夹
    rectify_small_common_lp = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/tyjt20222lp/cut_res/rectify_small_common_lp'

    # 创建保存图片的空文件夹
    save_dirs =  [yellow_2layer, rectify_yellow_2layer, rectify_small_common_lp]


    # 根据json标签，截取车牌并进行矫正
    get_small_lp(json_dir, image_dir, yellow_2layer, rectify_yellow_2layer, rectify_small_common_lp)
