import os






def file_num_stastic(file, threshold=0.6):
    acc_det_num = 0
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
                acc_det_num += 1

    return acc_det_num




if __name__ == '__main__':
    """
        # 0.性能评估，检测测试视频参数
        parser.add_argument('--weight', nargs='+', type=str, default="/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/weights/20230918/traffic_best.pt",  help='model.pt path(s)')
        parser.add_argument('--conf_thres', type=float, default=0.35, help='object confidence threshold')      # NMS的
        parser.add_argument('--iou_thres', type=float, default=0.45, help='object confidence threshold')       # NMS的IOU阈值
        parser.add_argument('--bbox_conf_thre', type=float, default=0.35, help=' object confidence threshold')  # 交通事故检测框的置信度，用作统计缓存的检测结果中交通事故数量
        parser.add_argument('--acc_avg_thres', type=float, default=0.20, help='traffic accident average threshold')  # 20230904 版本告警阈值0.85： 使用平均值作为该类型事故的概率衡量标准，告警条件avg_conf>该值才告警
        parser.add_argument('--cache_frames', type=int, default=300, help='cache detection frames')  # 缓存帧数，默认缓存帧数300 5min，相机时间，每秒1帧
        parser.add_argument('--traffic_acc_alarm_frames_thres', type=int, default=210, help='traffic accident alarm frames threshold')  # 告警阈值2：缓存的交通事故检测结果>该阈值才告警
        parser.add_argument('--alarm_interval_time', type=int, default=600, help='twice alarm time gap ')  # 两次告警时间间隔
    """


    # 计算交通事故识别准确率与召回率
    acc_processed = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/acc_processed'
    nor_processed = '/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/nor_processed'
    threshold_value = 0.68
    acc_video_num = 124
    acc_num = file_num_stastic(acc_processed, threshold=threshold_value)
    error_num = file_num_stastic(nor_processed, threshold=threshold_value)

    predict_acc = acc_num/(acc_num+error_num)
    recall = acc_num/acc_video_num
    print(f"事故视频数量：78，无事故视频数量：124，正确识别数量：{acc_num}，误检数量：{error_num}")
    print(f"predict_acc:{predict_acc}， recall:{recall}")






