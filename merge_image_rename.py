import os
import shutil


def images_rename(old_dir, new_dir, camera='R20_Aw_CamN'):
    names = os.listdir(old_dir)
    for name in names:
        new_name = camera + '_' + name
        old_img = os.path.join(old_dir, name)
        new_img = os.path.join(new_dir, new_name)
        shutil.copy(old_img, new_img)


def labels_rename(old_dir, new_dir, camera='R20_Aw_CamN'):
    names = os.listdir(old_dir)
    for name in names:
        new_name = camera + '_' + name
        old_label = os.path.join(old_dir, name)
        new_label = os.path.join(new_dir, new_name)
        shutil.copy(old_label, new_label)


def copy_image(old_dir_img, new_dir_image):
    names = os.listdir(old_dir_img)
    for name in names:
        old_img = os.path.join(old_dir_img, name)
        new_img = os.path.join(new_dir_image, name)
        print(f"old_img:{old_img}")
        print(f"new_img:{new_img}")
        shutil.copy(old_img, new_img)


if __name__ == '__main__':

    """
     "R20_Aw_CamN": 4892
     "R22_Bn_CamE": 10204
     "R24_Ds_CamW": 5951
     "R2_Aw_CamN": 7795
     "R2_Bn_CamE": 5191
     "R3_Aw_CamN": 984,
     "R3_Bn_CamE": 5600
     "R5_Aw_CamN": 14407
     "R5_Aw_CamS": 260
     "R5_Bn_CamE": 4339
     "R6_Aw_CamN": 1519
     """
    # 1.标注的图片标签文件重命名
    old_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerousDrive20230831/dangerDriver_20230831'
    new_dir_image = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerousDrive20230831/dangerDriver_20230831/image'
    new_dir_label = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerousDrive20230831/dangerDriver_20230831/label'

    for camera_name in ["R20_Aw_CamN", "R22_Bn_CamE","R24_Ds_CamW","R2_Aw_CamN","R2_Bn_CamE","R3_Aw_CamN","R3_Bn_CamE","R5_Aw_CamN","R5_Aw_CamS","R5_Bn_CamE","R6_Aw_CamN"]:
        print(f"process:{camera_name}")
        old_dir = os.path.join(old_dir, camera_name)

        old_dir_img = os.path.join(old_dir, 'image')
        old_dir_label = os.path.join(old_dir, 'label')

        images_rename(old_dir_img, new_dir_image, camera=camera_name)
        labels_rename(old_dir_label, new_dir_label, camera=camera_name)
        old_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerousDrive20230831/dangerDriver_20230831'


    # 2. 验证车牌和标签名称是否完全一样
    # image_names = os.listdir(new_dir_image)
    # labels_name = os.listdir(new_dir_label)
    # images_list = []
    # labels_list = []
    # for img in image_names:
    #     images_list.append(img[:-4])
    #
    # for label in labels_name:
    #     if label[:-5] not in images_list:
    #         labels_list.append(label[:-5])
    # print(f"labels_list:{labels_list}")
    # print(f"labels_list_len:{len(labels_list)}")

    # 3.将不同文件夹的图片进行合并
    # old_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerousDrive20230831/dangerDriver_20230731'
    # new_dir_image = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerousDrive20230831/dangerDriver_20230731/image'
    #
    # for camera_name in ["R20_Aw_CamN", "R22_Bn_CamE", "R24_Ds_CamW", "R2_Aw_CamN", "R2_Bn_CamE", "R3_Aw_CamN",
    #                     "R3_Bn_CamE", "R5_Aw_CamN", "R5_Aw_CamS", "R5_Bn_CamE", "R6_Aw_CamN"]:
    #     old_dir = os.path.join(old_dir, camera_name)
    #     old_dir_img = os.path.join(old_dir, 'image')
    #
    #     copy_image(old_dir_img, new_dir_image)
    #     old_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerousDrive20230831/dangerDriver_20230731'
    #
