import os
import json
import numpy as np
import cv2


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


def get_yellow_lp(json_dir, image_dir, lp_correction_dir, correction_splice):

    jsons = os.listdir(json_dir)
    n = 0
    for j in jsons:
        n += 1
        json_path = os.path.join(json_dir, j)
        json_file = open(json_path)
        json_datas = json.load(json_file)
        for data in json_datas:
            # 第一批车牌标注：颜色数字对应关系 {蓝牌：0, 绿牌：1,黄牌:2, 白牌:3,黑牌:4}
            #  plate_layer:0:无牌车, 1:单层, 2：双层
            if data['plate_layer']=='2' and data['plate_color']=='2':
                print(data)
                pic_ = os.path.join(image_dir, j[:-4]+'jpg')
                img = cv2.imread(pic_)  # 得到BGR图像
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # BGR
                # cv2.imshow('Image', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                corners = data['corner']
                landmarks_np = np.asarray(corners).reshape((4,2))

                # 获取根据四个角点校正后的双层车牌
                lp_correction = four_point_transform(img, landmarks_np)
                lp_save_name = data['plate_number']+'.jpg'
                if lp_save_name not in os.listdir(lp_correction_dir):
                    lp_save_path = os.path.join(lp_correction_dir, lp_save_name)
                    cv2.imwrite(lp_save_path, lp_correction)

                # 将矫正后的双层车牌进行拼接
                correction2splicing = get_split_merge(lp_correction)
                if lp_save_name not in os.listdir(correction_splice):
                    correction2splicing_save_path = os.path.join(correction_splice, lp_save_name)
                    cv2.imwrite(correction2splicing_save_path, correction2splicing)

        print(n, '*'*50)

if __name__ == '__main__':
    """  
        功能：
            双层黄色车牌截取、矫正、拼接与保存
    """

    json_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/tyjt_label1/jsons'
    image_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/tyjt_label1/images'
    # 截取的矫正后双层车牌保存文件夹
    lp_correction_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/tyjt_label1/lp_correction_dir'
    # 截取的矫正_拼接后的双层车牌保存文件夹
    correction_splice = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/tyjt_label1/correction_splice'

    get_yellow_lp(json_dir, image_dir, lp_correction_dir,correction_splice)
