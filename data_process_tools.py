
import os
import shutil

if __name__ == '__main__':
    src_path = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/images/val2017'
    dst_path = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/labels/val2017'

    names = os.listdir(src_path)
    for i in names:
        if i.endswith('txt'):
            src_abs = os.path.join(src_path, i)
            dstc_abs = os.path.join(dst_path, i)
            print(src_abs)
            shutil.move(src_abs, dstc_abs)



