import os
import shutil


def move_file(src_path, dst_path):
    """功能：将多个文件夹的文件移动到一个新的文件夹内
            src_path：原始要移动文件的文件夹
            dst_path：保存的目的文件夹

     """
    dirs_file = os.listdir(src_path)

    for file in dirs_file:
        src_dir_file_path = os.path.join(src_path, file)
        for img_name in os.listdir(src_dir_file_path):
            src_img = os.path.join(src_dir_file_path, img_name)
            shutil.move(src_img, dst_path)

    print("----moving--end----")


def dele_json_file(img_path, label_path):
    """功能：图片和标注文件json保持一致,删除没有标注的图片
            src_path：图片文件夹
            dst_path：标注文件夹
     """
    img_names = [i[:-4] for i in os.listdir(img_path)]   # JSON 取5
    label_names = [j[:-5] for j in os.listdir(label_path)] # JSON 取5
    n = 0
    for img_name in img_names:
        if img_name not in label_names:
            img_name_jpg = img_name + '.jpg'
            remove_img = os.path.join(img_path, img_name_jpg)
            os.remove(remove_img)
            n += 1
    print(f"去除的无效img数量:{n}")


def dele_txt_label(img_path, label_path):
    """功能：图片和标注文件txt保持一致,删除没有图片对应的标签
            src_path：图片文件夹
            dst_path：标注文件夹
     """
    img_names = [i[:-4] for i in os.listdir(img_path)]
    label_names = [j[:-4] for j in os.listdir(label_path)]  # txt取4
    n = 0
    for label_name in label_names:
        if label_name not in img_names:
            label_name_ful = label_name + '.txt'
            remove_label = os.path.join(label_path, label_name_ful)
            os.remove(remove_label)
            n += 1
    print(f"去除的无效label数量:{n}")


def dele_img_file(img_path, label_path):
    """功能：图片和标注文件txt保持一致,删除没有标注的图片
            src_path：图片文件夹
            dst_path：标注文件夹
     """
    img_names = [i[:-4] for i in os.listdir(img_path)]
    label_names = [j[:-4] for j in os.listdir(label_path)]
    n = 0
    for img_name in img_names:
        if img_name not in label_names:
            img_name_jpg = img_name + '.jpg'
            remove_img = os.path.join(img_path, img_name_jpg)
            os.remove(remove_img)
            n += 1
    print(f"去除的无效img数量:{n}")


if __name__ == '__main__':
    """ 代码功能：
        第一步：将文件夹内的多个文件夹的子文件移动到一个新的文件夹内
        第二步：统一img/label问价夹内的文件数量，使得两个文件夹内图片和标签一一对应
    """

    # 第一步：功能：将多个文件夹的文件移动到一个新的文件夹内
    src_path = r"D:\coco\ty_驾驶行为20221221"  # 原始要移动文件文件夹
    dst_path = r"D:\coco\labels"  # 保存的目的文件夹
    # move_file(src_path, dst_path)  # src文件夹内的子文件夹内的文件合并到dst文件夹

    # 第二步：功能:统一img/label问价夹内的文件数量，使得两个文件夹内图片和标签一一对应
    img_path = "/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch2/images"    # 图片文件夹
    label_path = "/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch2/batch2_txt"  # 图片对应的标签文件夹

    # 以下两行调研函数代码可以同时执行，打印双向验证时删除的图片数量
    dele_txt_label(img_path, label_path)
    dele_img_file(img_path, label_path)