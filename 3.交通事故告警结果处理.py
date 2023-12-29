import os


def list_dir(dirpath):
    """ 功能： 统计/计算文件夹数量 """
    abs_path_list = []
    names = os.listdir(dirpath)
    numbers = len(names)
    for name in names:
        abs_name = os.path.join(dirpath, name)
        abs_path_list.append(abs_name)

    return names, abs_path_list, numbers


def remove_empty_dir(res_dir):
    """ 功能： 去除空文件夹 """
    res_dirs = os.listdir(res_dir)
    for i in res_dirs:
        res_dir = os.path.join(res_dir, i)
        files = os.listdir(res_dir)
        print(files)
        if len(files) == 0:
            os.removedirs(res_dir)



def Merge_multiple_file2one(dir_path, save_path, save_name='normal.txt'):
    """ 功能：多文件夹文件合并：将多个文件夹内的文件内的内容写入到同一个文件内 """
    save_path_file = os.path.join(save_path, save_name)
    with open(save_path_file, 'a', encoding='utf-8') as fw:
        dirs = os.listdir(dir_path)
        # 遍历文件夹内的子文件夹
        for dir in dirs:
            file_dir = os.path.join(dir_path, dir)
            files = os.listdir(file_dir)
            # 遍历子文件夹内的文件
            for file in files:
                file_abs_path = os.path.join(file_dir, file)
                with open(file_abs_path, 'r', encoding='utf-8') as fr:
                    data = fr.readlines()
                    fw.write(str(data[0]))
                    print(data)


def merge_acc_key_value2one(det_res_path, stdxy, new_res_merge_path):
    """ 功能：合并告警结果重要参数和stdxy到一个文件内 """
    n = 0
    with open(det_res_path, 'r', encoding='utf-8') as fr:
        res_data = fr.readlines()
        for data in res_data:
            data_list = data.split(',')
            camera = data_list[1].split('：')[-1]
            alarmcls = data_list[2].split(':')[-1]
            conf = data_list[3].split('，')[0].split('：')[-1]
            with open(stdxy, 'r', encoding='utf-8') as fr2:
                res_data = fr2.readlines()
                for data in res_data:
                    dir_name = data.split(' ')[0]
                    x = int((data.split(' ')[-2]))
                    y = int((data.split(' ')[-1])[:-1])
                    if camera==dir_name:
                        n += 1
                        info = str(n) + ' ' + camera + ' ' + conf + ' ' + alarmcls + ' ' + str(x)+' '+str(y)+'\n'
                        print(info)
                        with open(new_res_merge_path, 'a', encoding='utf-8') as fw:
                            fw.write(info)



def nor_acc_write2file(acc_video_dir, alarm_all_res, acc_processed, nor_processed):
    """
    功能： 根据事故视频文件夹将事故检测结果分开保存到不同的文件内

        acc_video_dir:交通事故视频文件夹路径
        alarm_all_res:检测识别正确/错误的结果,文件内的每条数据都包含一个视频的重要信息
        acc_processed:检测正确样本
        nor_processed:误检视频
    """

    acc_list = []
    acc_video_names = os.listdir(acc_video_dir)
    for i in acc_video_names:
        acc_list.append(i[:-4])
    acc_video_names_lens = len(acc_list)
    # print(f"acc_len:{acc_video_names_lens}, acc_list：{acc_list}")
    acc_alarm_num = 0
    with open(alarm_all_res, 'r', encoding='utf-8') as fr:
        res_data = fr.readlines()
        for datas in res_data:
            data_list = datas.split(' ')
            name = data_list[1]
            conf = data_list[2]
            cls = data_list[3]
            stdx = data_list[4]
            stdy = int(data_list[-1][:-1])

            # 正确识别交通事故事件写入到文件中
            if name in acc_list:
                acc_alarm_num += 1
                with open(acc_processed, 'a', encoding='utf-8') as fw:
                    fw.write(datas)
            else:
                with open(nor_processed, 'a', encoding='utf-8') as fw:
                    fw.write(datas)

    print(f"acc_alarm_number:{acc_alarm_num}")
    print("---END---")


def alarm_num_stastic(file, threshold=0.6):
    acc_alarm_num = 0
    with open(file, 'r', encoding='utf-8') as fr:
        res_data = fr.readlines()
        for datas in res_data:
            data_list = datas.split(' ')
            name = data_list[1]
            conf = float(data_list[2])
            cls = data_list[3]
            stdx = int(data_list[4])
            stdy = int(data_list[-1][:-1])

            # 统计符合条件的数量
            if conf>threshold:  # stdx <15 and stdy<15
            # if conf>threshold and (stdx <5 and stdy<10):
                acc_alarm_num += 1

    return acc_alarm_num


def alarm_accurate_recall_compute(acc_processed, nor_processed, normal_video_num=78, acc_video_num=124, threshold_value=0.67):
    """ 计算交通事故告警准确率和召回率 """

    acc_num = alarm_num_stastic(acc_processed, threshold=threshold_value)
    error_num = alarm_num_stastic(nor_processed, threshold=threshold_value)

    predict_acc = acc_num / (acc_num + error_num)
    recall = acc_num / acc_video_num
    print(
        f"事故视频数量：{acc_video_num}，无事故视频数量：{normal_video_num}，正确识别数量：{acc_num}，误检数量：{error_num}")
    print(f"predict_acc:{predict_acc}， recall:{recall}")





# # 获取stdxy检测结果
# stdxy = "/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/stdxy_txt"
# stdxy_names = []
# xs = []
# ys = []
# xmid = 15
# ymid = 15
# num_less_xmid = 0  # 105
# num_less_ymid = 0  # 149
#
# accx = []
# accy = []
# norx = []
# nory = []
#
#
# with open(stdxy, 'r', encoding='utf-8') as fr:
#     res_data = fr.readlines()
#     for data in res_data:
#         dir_name = data.split(' ')[0]
#         x = int((data.split(' ')[-2]))
#         y = int((data.split(' ')[-1])[:-1])
#         stdxy_names.append(dir_name)
#         xs.append(x)
#         ys.append(y)
#         if x<xmid:
#             num_less_xmid +=1
#         if y<ymid:
#             num_less_ymid+=1
#
#         # 事故stdxy
#         print(dir_name)
#         print(traffic_video_names_res)
#         if dir_name in traffic_video_names_res:
#             accx.append(x)
#             accy.append(y)
#         # 正常stdxy
#         if dir_name in normal_video_names_res:
#             norx.append(x)
#             nory.append(y)
# print(f"lenx:{len(accx)}, max:{max(accx)},accx:{sorted(accx)}")
# print(f"leny:{len(accy)}, max:{max(accy)},accy:{sorted(accy)}")
# print(f"lennorx{len(norx)}, min:{min(norx)},norx:{sorted(norx)}")
# print(f"lennory{len(nory)}, min:{min(nory)},nory:{sorted(nory)}")

# print(stdxy_names)
# print(xs)
# print(ys)
# print(num_less_xmid, num_less_ymid)


if __name__ == '__main__':

    """
        # 0.性能评估，检测测试视频参数,2023年年终考核版本参数
        parser.add_argument('--weight', nargs='+', type=str, default="/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/weights/20230918/traffic_best.pt",  help='model.pt path(s)')
        parser.add_argument('--conf_thres', type=float, default=0.35, help='object confidence threshold')      # NMS的
        parser.add_argument('--iou_thres', type=float, default=0.45, help='object confidence threshold')       # NMS的IOU阈值
        parser.add_argument('--bbox_conf_thre', type=float, default=0.35, help=' object confidence threshold')  # 交通事故检测框的置信度，用作统计缓存的检测结果中交通事故数量
        parser.add_argument('--acc_avg_thres', type=float, default=0.20, help='traffic accident average threshold')  # 20230904 版本告警阈值0.85： 使用平均值作为该类型事故的概率衡量标准，告警条件avg_conf>该值才告警
        parser.add_argument('--cache_frames', type=int, default=300, help='cache detection frames')  # 缓存帧数，默认缓存帧数300 5min，相机时间，每秒1帧
        parser.add_argument('--traffic_acc_alarm_frames_thres', type=int, default=210, help='traffic accident alarm frames threshold')  # 告警阈值2：缓存的交通事故检测结果>该阈值才告警
        parser.add_argument('--alarm_interval_time', type=int, default=600, help='twice alarm time gap ')  # 两次告警时间间隔
    """


    #
    # 交通事故视频
    traffic_video_dir = '/media/dell/sata4t/jwang/datasets/AlarmFromEventCenter/trafficAccident/trafficEval/traffic1min'
    # 交通事故视频检测结果
    traffic_video_dir_res = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/traffic'

    # 无交通事故视频
    normal_video_dir = '/media/dell/sata4t/jwang/datasets/AlarmFromEventCenter/trafficAccident/trafficEval/normal'
    # 无交通事故视频结果
    normal_video_dir_res = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/normal'

    # 1.使用推理脚本进行推理，对交通事故告警结果进行处理
    # res_dir = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/normal'
    # remove_empty_dir(res_dir)

    # 2.将多个文件夹内的文件内的内容写入到同一个文件内
    # Merge_multiple_file2one(dir_path='/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/traffic', save_path='/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt')
    # Merge_multiple_file2one(dir_path='/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/normal', save_path='/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt')

    # 3.合并告警结果重要参数和stdxy到一个文件内
    # stdxy = "/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/stdxy_txt"
    # det_res_path = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/acc_nor_merge.txt'
    # new_res_merge_path = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/merge_new_res.txt'
    # merge_acc_key_value2one(det_res_path, stdxy, new_res_merge_path)

    # 4.将交通事故识别正确/错误结果分别写入到2个不同文件内
    # 处理后的事故告警每条数据格式：82 34R27_Ce_CamN_102_gpu23_carbike15 0.4708961123511905 8002 1 5
    # 交通事故视频文件夹
    # acc_video_dir = '/media/dell/sata4t/jwang/datasets/AlarmFromEventCenter/trafficAccident/trafficEval/traffic1min'
    # alarm_all_res = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/merge_new_res.txt'
    # acc_processed = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/acc_processed.txt'
    # nor_processed = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/nor_processed.txt'
    # nor_acc_write2file(acc_video_dir,  alarm_all_res, acc_processed, nor_processed)

    # 5.计算交通事故识别准确率与召回率
    acc_video_num = 124
    normal_video_num = 78
    threshold_value = 0.67  # 年终考核参数 0.67 acc=0.882 recall=0.36

    acc_processed = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/acc_processed'
    nor_processed = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/nor_processed'
    alarm_accurate_recall_compute(acc_processed, nor_processed, normal_video_num=78, acc_video_num=124, threshold_value=0.67)



