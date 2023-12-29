import argparse
import os.path
import time

import numpy
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device
from matplotlib import pyplot as plt
import cv2





def ret_acc_cls(cls_caches, acc_dict, conf_caches):
    """ 作用：计算交通事故的类别和该类别的平均概率
        1.若缓存的检测结果种4种结果出现的频次相同，则认为是轻微碰撞，并返回轻微碰撞的概率；否则按照实际情况进行计算
        2.按照发生事故的频次最大值作为判断依据，取事故类型发生最多的类型作为事故发生的类型
    """
    num_cls = list((cls_caches.count(0), cls_caches.count(1), cls_caches.count(2), cls_caches.count(3)))
    if num_cls[0] == num_cls[1] == num_cls[2] == num_cls[3]:
        avg_conf = 0
        for i, j in enumerate(cls_caches):
            if j == 2:
                avg_conf += conf_caches[i]
        avg_conf = avg_conf/cls_caches.count(2)
        acc_cls_name = acc_dict[2]  # 若次数相同，则返回第2个索引
        return acc_cls_name, avg_conf
    else:
        # 过滤事故类型为正常（8001）的误报
        if num_cls[0] > max(list((cls_caches.count(1), cls_caches.count(2), cls_caches.count(3)))): # 第一个值是正常，不是事故；若第一个值出现的频次为最大，则认为是误检
            return None, None
        else:
            acc_cls = num_cls.index(max(num_cls))
            avg_conf = 0
            for i, j in enumerate(cls_caches):
                if j == acc_cls:
                    avg_conf += conf_caches[i]
            avg_conf = avg_conf/max(num_cls)
            acc_cls_name = acc_dict[acc_cls]  # 若次数相同，则返回第2个索引
            return acc_cls_name, avg_conf


def img_save(img):
    """ 测试输入图像通道顺序 """
    save_img = img.transpose(1,2,0)
    plt.imshow(save_img[:,:,:])  # （w,h,channel）通道反转实现BGR转RGB
    plt.savefig("src.jpg")  # 图片存储
    plt.imshow(save_img[:,:,::-1])  # （w,h,channel）通道反转实现BGR转RGB
    plt.savefig("src_exchange_channels.jpg")  # 图片存储
    cv2.waitKey(0)





def bbox_mean(max_conf_bbox_caches):
    data = numpy.array(max_conf_bbox_caches)
    top_x = []
    top_y = []
    bottom_x = []
    bottom_y = []
    for v in data:
        top_x.append(v[0])
        top_y.append(v[1])
        bottom_x.append(v[2])
        bottom_y.append(v[3])
    top_x_mean = int(np.mean(top_x))
    top_y_mean = int(np.mean(top_y))
    bottom_x_mean = int(np.mean(bottom_x))
    bottom_y_mean = int(np.mean(bottom_y))
    result = str([top_x_mean, top_y_mean, bottom_x_mean, bottom_y_mean])
    return result


def detect(max_conf_bbox_caches, time_cache, conf_caches, cls_caches, acc_dict):
    source, imgsz, bbox_conf_thre, weights, cache_frames, traffic_acc_alarm_frames_thres, alarm_interval_time, alarm_camera_info, alarm_result_abs, acc_avg_thres = \
        opt.source, opt.img_size, opt.bbox_conf_thre, opt.weight, opt.cache_frames, opt.traffic_acc_alarm_frames_thres, opt.alarm_interval_time, opt.alarm_camera_info, opt.alarm_results, opt.acc_avg_thres
    # 首次告警，创建保存的文件
    write_alarm_name = source.split('/')[-1][:-4]


    alarm_results = os.path.join(alarm_result_abs, source.split('/')[-1][:-4])
    if not os.path.exists(alarm_results):
        os.makedirs(alarm_results)
    res_txt = os.path.join(alarm_results, 'alarm_results.txt')

    alarm_camera_info = str(alarm_camera_info)  # 相机编号id, 数据类型转成str格式， 用作定位事故发生地点
    alarm_num = 0   # 告警编号
    start_time = 0  # 碰撞事件发生时间
    end_time = 0    # 碰撞事件结束时间
    alarm_time = {"start_time": start_time, "end_time": end_time}

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 模型加载
    model = attempt_load(weights, map_location=device)  # load FP32 modelconf_thres
    stride = int(model.stride.max())  # model stride 32
    imgsz = check_img_size(imgsz, s=stride)  # check img_size，检查是否为32的整数倍，若不是则向上求32的最临近整数值

    if half:
        model.half()  # to FP16
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    # 对视频/图片数据加载, dataset返回四个值，使用四个变量进行接收
    for path, img, im0s, vid_cap in dataset:
        """  path: 图片路径； img： resize后的图片尺寸【3，xxx，1280】，图像通道顺序； im0s:resize前的图片尺寸    """
        # img_save(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:  # 图片升维度
            img = img.unsqueeze(0)

        # 对图片进行推理 Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        """ 
            Apply NMS：pred：模型预测结果；conf_thres：过滤置信度阈值；iou_thres：过滤重叠的bbox阈值； classes：保留的类别；agnostic_nms：类别无关非极大值抑制
            pred结果是tensor格式：[x, y, w, h, conf, cls]
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
        # ---------------以下代码：将交通事故检测结果添加到list()中进行缓存-----------------
        for result in pred:  # 得到一张图的predict结果
            """ result是返回的每帧图像的检测结果为tensor格式 """
            max_conf_bbox_caches_len = len(max_conf_bbox_caches)
            if len(list(result.cpu().data.numpy())) != 0:  # 不等于0表示当前图片有检测结果
                result_numpy = result.cpu().data.numpy()   # 转换成<class 'numpy.ndarray'>格式，获取present frame max conf Boxes
                max_conf_coordinate = np.where(result_numpy == (np.max(result_numpy, axis=0)[-2]))  # axis=0，计算本列的最大值;用where得到最大值的索引，前面的array对应行数，后者对应列数
                max_conf_bbox = result_numpy[max_conf_coordinate[0]][0]  # [x1,y1,x2,y2,conf,cls] 一张图片中的最大置信度的车辆检测碰撞bbox,取出最大置信度对应的bbox,conf, cls
                """ 若小于缓存帧数阈值，则对检测结果进行缓存， 否则删除最早缓存的结果，再在尾部添加最新的检测结果"""
                # 交通事故检测到的事故类型结果添加到缓存中
                if (max_conf_bbox_caches_len == 0) or (max_conf_bbox_caches_len > 0 and max_conf_bbox_caches_len < cache_frames):
                    max_conf_bbox_caches.append(max_conf_bbox)  # max_conf_bbox_caches存储了bbox, conf., cls 三种结果
                    conf_caches.append(max_conf_bbox[-2])  # 缓存置信度
                    time_cache.append(time.time())         # 缓存当前时间
                    cls_caches.append(int(max_conf_bbox[-1]))  # 交通事故类型缓存
                else:
                    del max_conf_bbox_caches[0]
                    max_conf_bbox_caches.append(max_conf_bbox)
                    del conf_caches[0]
                    conf_caches.append(max_conf_bbox[-2])  # 缓存置信度
                    del time_cache[0]
                    time_cache.append(time.time())
                    del cls_caches[0]
                    cls_caches.append(int(max_conf_bbox[-1]))
                    
        # ------------------------以下代码对缓存结果进行告警逻辑实现； 满足告警条件：则对告警结果进行保存---------------------
        # print(f"max_conf_bbox_caches:{max_conf_bbox_caches}")

        acc_gt_conf_times = np.sum(np.asarray(conf_caches) > bbox_conf_thre)  # 统计大于交通事故阈值的总数量，缓存的检测总数量大于设定阈值才进行告警，初步筛查发生事故
        # 告警条件1：缓存的检测结果中存在大于既定阈值的帧数traffic_acc_alarm_frames_thres的交通事故
        if acc_gt_conf_times >= traffic_acc_alarm_frames_thres:   # 确认告警频次满足条件，缓存的总数中有大于该次数的事故检测结果
            if alarm_time["start_time"] == 0:  # 首次交通事故告警
                # print(f"Warning: traffic accident has occurred, please deal with it in time, occurred time：{time.strftime('%Y-%m-%d %H:%M:%S')}")
                alarm_time["start_time"] = time.time()
                acc_cls_name, avg_conf = ret_acc_cls(cls_caches, acc_dict, conf_caches)  # 计算告警的交通事故类型和交通事故的平均概率

                # 告警条件2-1：首次告警：(1)告警类型不对正常的检测结果告警（2）告警的交通事故类型的平均置信度大于预设的置信度
                # if acc_cls_name !=None and avg_conf > acc_avg_thres :  # 事故发生的平均值阈值
                bbox_std_move_key = bbox_stand_move_value(max_conf_bbox_caches, source)  # 20230904代码修改
                if acc_cls_name !=None and avg_conf > acc_avg_thres and bbox_std_move_key:  # 20230906 update, ‘8001’为正常，不告警

                    avg_bbox_position = bbox_mean(max_conf_bbox_caches)  # 20230906 update deploy

                    with open(res_txt, 'a', encoding='utf-8') as save_alarm_res:
                        alarm_num += 1
                        # traffic_acc_record = f"Warning：{alarm_num}, accident class:{acc_cls_name}, avg_conf：{avg_conf}，max_conf_bbox_caches：{max_conf_bbox_caches}，time_cache：{time_cache}，time：{time.strftime('%Y-%m-%d %H:%M:%S')}, Camera INFO：{alarm_camera_info}" + '\n'

                        traffic_accident_start_time = time_cache[-1]  # 20230906 update
                        source_name_video = source.split('/')[-1][:-4]
                        traffic_acc_record = f"Warning：{alarm_num}, Camera INFO：{source_name_video}, Accident class:{acc_cls_name}, avg_conf：{avg_conf}，avg_bbox_position：{avg_bbox_position}，traffic_accident_start_time：{traffic_accident_start_time}，time：{time.strftime('%Y-%m-%d %H:%M:%S')}, " + '\n'  # 20230906 update
                        save_alarm_res.write(traffic_acc_record)
                    # return traffic_acc_record
            else:
                end_time = time.time()
                time_gap = end_time - alarm_time["start_time"]
                # print(f"Time interval from last traffic accident alarm:{time_gap}")

                # 告警条件2-2：非首次告警：（1）再次告警的与上次告警的时间间隔大于设定的时间间隔(2)告警类型不对正常的检测结果告警（3）告警的交通事故类型的平均置信度大于预设的置信度
                if time_gap >= alarm_interval_time:  # 20230906 update  两次交通碰撞事故告警大于预设时间间隔，防止短时间内重复进行告警
                    acc_cls_name, avg_conf = ret_acc_cls(cls_caches, acc_dict, conf_caches)
                    # if acc_cls_name != None and avg_conf > acc_avg_thres:   # 事故发生的平均值阈值
                    bbox_std_move_key = bbox_stand_move_value(max_conf_bbox_caches)  # 20230904代码修改
                    # if acc_cls_name != None and acc_cls_name !='8001' and avg_conf > acc_avg_thres and bbox_std_move_key:  # 20230906 update
                    if acc_cls_name != None and avg_conf > acc_avg_thres and bbox_std_move_key:  # 20230906 update

                        avg_bbox_position = bbox_mean(max_conf_bbox_caches)   # 20230906 update
                        with open(res_txt, 'a', encoding='utf-8') as save_alarm_res:
                            alarm_num += 1
                            traffic_accident_start_time = time_cache[-1]  # 20230906 update
                            source_name_video = source.split('/')[-1][:-4]
                            # traffic_acc_record = f"Warning：{alarm_num}, accident class:{acc_cls_name}, avg_conf：{avg_conf}，max_conf_bbox_caches：{max_conf_bbox_caches}，time_cache：{time_cache}，time：{time.strftime('%Y-%m-%d %H:%M:%S')}, Camera INFO：{alarm_camera_info}" + '\n'
                            traffic_acc_record = f"Warning：{alarm_num}, Camera INFO：{source_name_video}, Accident class: {acc_cls_name}, avg_conf：{avg_conf}，avg_bbox_position：{avg_bbox_position}，traffic_accident_start_time：{traffic_accident_start_time}，time：{time.strftime('%Y-%m-%d %H:%M:%S')}" + '\n'

                            save_alarm_res.write(traffic_acc_record)
                        alarm_time["start_time"] = time.time()
                        # return traffic_acc_record


def bbox_stand_move_value(max_conf_bbox_caches, source):
    std_data = numpy.array(max_conf_bbox_caches)
    max_conf_bbox_std_x = []
    max_conf_bbox_std_y = []
    for std in std_data:
        std_x_pos = int((std[2] - std[0])/2)
        std_y_pos = int((std[3] - std[1])/2)
        max_conf_bbox_std_x.append(std_x_pos)
        max_conf_bbox_std_y.append(std_y_pos)
    std_x = int(np.std(max_conf_bbox_std_x))
    std_y = int(np.std(max_conf_bbox_std_y))
    # print(f"std_x:{std_x}, std_y:{std_y}")

    with open('/alarm_txt/stdxy_txt', 'a', encoding='utf-8') as f:
        source_name = source.split('/')[-1][:-4]
        value = source_name+' '+str(std_x)+' '+str(std_y) +'\n'
        f.write(value)

    # if std_x < 100 and std_y < 50:  # 20230904版本参数值设为：std_x<2，std_y<1
    if std_x < 20 and std_y < 20:  #16 16
        return True
    else:
        return False


if __name__ == '__main__':
    """
        # 可以直接输入图片或者视频进行性能测试， 每秒1帧
    
        性能测试：
        缓存帧数：300                            时间5min*60s*1（frames/s）=300 ， 每秒1帧
        缓存帧数中有70%检测结果发生交通事故：210    300*0.7=210帧
        两次告警间隔时间：300s                    5min*60s=300s
        缓存结果最大时间间隔：300                  5min*60s=300s
        
        # 测试参数
        cache_frames=300;
        traffic_acc_alarm_frames_thres=210  # 300*0.7=210
        alarm_interval_time=300
        acc_avg_thres=0.85
        
    """

    """ 
        20230904代码更新
        增加功能：根据使用端反馈的误报问题，降低交通事故误报过滤条件。
        
        20230906代码更新
        
        功能：
          (1)修改输出到kafka的缓存bbox结果为平均框的坐标，输出事件开始时间为事件的判断时间，真实时间应该往前推5min
          (2)更新了逻辑判断的时间，阈值，具体参数见：https://tyjt.yuque.com/almcm6/project_doc/puo2v1d4t1qy3isn#YgH4h
        
    """


    """
        # 在linux环境下，使用bash 批量执行python 检测脚本
        for file in $(ls /media/dell/sata4t/jwang/datasets/AlarmFromEventCenter/trafficAccident/trafficEval/traffic1min) # 保存视频集的文件夹
        do
            conda activate yolo7
            abs_path="/media/dell/sata4t/jwang/datasets/AlarmFromEventCenter/trafficAccident/trafficEval/traffic1min/$file"
            python /home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/2.infer_deploy-opt-0904_1227考核评估.py --source  "$abs_path" --alarm_results  "/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/traffic "
        done

    
    """

    parser = argparse.ArgumentParser()
    # 年终考核代码路径：/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/
    parser.add_argument('--source', type=str, default='/media/dell/sata4t/jwang/datasets/AlarmFromEventCenter/trafficAccident/accident/1_5minR5_Bn_CamE_carwhiteVan/1min/R5_Bn_CamE_carWhiteVan5.mp4', help='source')  # file/folder path, 0 for webcam
    parser.add_argument('--alarm_results', default='/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt',help='ave results to *.txt')  # 告警结果保存路径
    # parser.add_argument('--weight', nargs='+', type=str, default="/home/dell/tyjtitem/trafficaccident/traffic_accident_train/runs/train/exp2/weights/traffic_best.pt",  help='model.pt path(s)')
    parser.add_argument('--weight', nargs='+', type=str, default="/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/weights/20230918/traffic_best.pt",  help='model.pt path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.35, help='0.35, object confidence threshold')      # NMS的
    parser.add_argument('--iou_thres', type=float, default=0.45, help='0.45, object confidence threshold')       # NMS的IOU阈值
    parser.add_argument('--bbox_conf_thre', type=float, default=0.35, help='0.35, object confidence threshold')  # 交通事故检测框的置信度，用作统计缓存的检测结果中交通事故数量
    parser.add_argument('--acc_avg_thres', type=float, default=0.20, help='0.5, traffic accident average threshold')  # 20230904 版本告警阈值0.85： 使用平均值作为该类型事故的概率衡量标准，告警条件avg_conf>该值才告警
    parser.add_argument('--cache_frames', type=int, default=300, help='1800=1min*60s*30（frames/s）, cache detection frames')  # 缓存帧数，默认缓存帧数300 5min，相机时间，每秒1帧
    parser.add_argument('--traffic_acc_alarm_frames_thres', type=int, default=210, help='1080帧=1800*0.6, traffic accident alarm frames threshold')  # 210 告警阈值2：缓存的交通事故检测结果>该阈值才告警
    parser.add_argument('--alarm_interval_time', type=int, default=600, help='60*10s, unit:second, twice alarm time gap ')  # 两次告警时间间隔：300帧=10s


    parser.add_argument('--alarm_camera_info', type=str, default="R7_Ds_CamW-192.168.16.41", help='camera information')  # 相机信息，用作定位事故发生地点
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')  # 检测图片的尺寸：1280， 640
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 计算设备选择， 默认GPU:0
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # 是否使用类别无关的非极大值抑制，默认为False。
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 1 2 3')  # 筛选保存的检测结果类别
    # parser.add_argument('--alarm_results', default='/home/dell/tyjtitem/trafficaccident/traffic_acc_det_deploy/alarm_txt/R5_Aw_CamS_102_gpu052', help='ave results to *.txt')  # 告警结果存储文件
    opt = parser.parse_args()
    # print(opt)

    max_conf_bbox_caches = list()  # 初始化空列表，缓存每帧的检测结果, 数据类型为list
    time_cache = list()  # 对缓存的检测结果进行记录时间，最初缓存结果与最后缓存结果间隔时间不能太长
    conf_caches = list()
    cls_caches = list()
    # acc_dict = {0: "normal", 1: "scratch", 2: "slight", 3: "serious"}
    # 8001：normal； 8002：scratch， 8003：slight， 8004：serious
    # 事件编号对应关系：https://tyjt.yuque.com/almcm6/project_doc/puo2v1d4t1qy3isn#YgH4h
    acc_dict = {0: "8001", 1: "8002", 2: "8003", 3: "8004"}
    detect(max_conf_bbox_caches, time_cache, conf_caches, cls_caches, acc_dict)


