import os
import random
import shutil
import shutil


def mk_dirs(dir_path):
    """ create new dir"""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    train_path = os.path.join(dir_path, 'train2017')
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    val_path = os.path.join(dir_path, 'val2017')
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    test_path = os.path.join(dir_path, 'test2017')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    return train_path, val_path, test_path



def split_images2train_test_val(src_path, train_path, val_path, test_path, train_per=0.7, val_per=0.2, test_per=0.1):

    src_file_names = os.listdir(src_path)  #（图片文件夹）
    num_file = len(src_file_names)
    index_list = list(range(num_file))
    random.shuffle(index_list)  # 打乱顺序

    num = 0
    for i in index_list:
        file_path = os.path.join(src_path, src_file_names[i])  #（图片文件夹）+图片名=图片地址
        if num < num_file * train_per:   # 0.7 --> 7:1:2
            shutil.move(file_path, train_path)

        elif num < num_file * (train_per+val_per):  # val占比：0.2 = 0.9-0.7
            shutil.move(file_path, val_path)

        else:
            shutil.move(file_path, test_path)  # test占比：0.1
        num += 1
        print(f"process:{num}/{num_file}")


def split_labels2train_test_val(src_path_labels, train_path, val_path, test_path, train_path_label, val_path_label, test_path_label):
    train_imgs = os.listdir(train_path)
    test_imgs = os.listdir(test_path)
    val_imgs = os.listdir(val_path)

    for img_train in train_imgs:
        label_name_train = img_train[:-4] + '.txt'
        train_label_path = os.path.join(src_path_labels, label_name_train)
        shutil.move(train_label_path, train_path_label)

    for img_val in val_imgs:
        label_name_val = img_val[:-4] + '.txt'
        val_label_path = os.path.join(src_path_labels, label_name_val)
        shutil.move(val_label_path, val_path_label)

    for img_test in test_imgs:
        label_name_test = img_test[:-4] + '.txt'
        test_label_path = os.path.join(src_path_labels, label_name_test)
        shutil.move(test_label_path, test_path_label)

    print("---finish---")


if __name__ == '__main__':
    """ 同时完成将训练图片、标签进行train/test/val数据集的划分，通过传入参数设置比例划分。需要修改img/labels的文件夹路径 """

    # 图片和标签文件存储夹
    src_path_imgs = "/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch123/image"
    src_path_labels = "/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch123/label"

    # 图片和标签文件分帧保存文件夹
    dest_path_img = "/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch123"
    dest_path_label = "/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch123"

    train_path, val_path, test_path = mk_dirs(dest_path_img)  # 若保存文件的文件夹不存在，则创建空文件夹
    train_path_label, val_path_label, test_path_label = mk_dirs(dest_path_label)  # 若保存文件的文件夹不存在，则创建空文件夹

    # 训练图片、标签进行train/test/val划分
    split_images2train_test_val(src_path_imgs, train_path, val_path, test_path, train_per=0.7, val_per=0.2, test_per=0.1)
    split_labels2train_test_val(src_path_labels, train_path, val_path, test_path, train_path_label, val_path_label, test_path_label)
