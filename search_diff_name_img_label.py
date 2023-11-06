import os


def compare_img_label(images, labels, json_value=5):
    """
    功能： 比较两个文件夹的文件名是否完全一致
    Args:
        images: 图片文件夹
        labels: 标签文件夹

    Returns:

    """

    image_name = os.listdir(images)
    label_name = os.listdir(labels)

    image_names = []
    label_names = []
    for i in image_name:
        image_names.append(i[:-4])
    for j in label_name:
        if j[:-5] not in image_names:
            label_names.append(j)
    print(f"不存在与之对应的图片，标签名称：{len(label_names)}，{label_names}")


    image_names2 = []
    label_names2 = []
    for j in label_name:
        label_names2.append(j[:-5])
    for i in image_name:
        if i[:-4] not in label_names2:
            image_names2.append(i)
    print(f"不存在与之对应的标签的图片名称：{len(image_names2)}，{image_names2}")



if __name__ == '__main__':
    images = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch2/images'
    labels = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch2/batch2json'

    compare_img_label(images, labels)