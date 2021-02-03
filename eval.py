import time
import cv2
import torch
from yolo_tiny.utils import resize, nms, return2RealPosition
from yolo_tiny.detector import Detector
from utils import init_settings
import argparse
import datetime


def Eval(path, net_path=None, img=None):
    """
    网络的使用函数
    :param img: 可选参数，可传入图片数据
    :param path: 图片文件路径
    :param net_path: 网络文件路径
    :return: 侦测到的物体的位置、IOU、和分类信息
    """
    if img:
        img, expand, scale = resize(img)
    else:
        img, expand, scale = resize(cv2.imread(path))

    # 数据转换， 转换图像数组数据为tensor且数据缩放到-0.5~0.5
    data = torch.tensor(img.transpose([2, 0, 1]) / 255 - 0.5,
                        dtype=torch.float32).unsqueeze(dim=0)
    # 初始化设置
    setting = init_settings()
    # 检查是否有试着网络路径
    if net_path:
        setting['net_path'] = net_path
        print('[{}]指定的网络文件路径已起效'.format(datetime.datetime.now()))
    # 获取侦测类对象
    yolo = Detector(setting['net_path'])

    startTime = time.time()
    # 调用侦测类中的侦测处理函数
    boxes = yolo.detect(data, 0.5, setting['anchors'])  # Box <- [batch,confi,cx,cy,w,h,cls]
    # 数据取到cpu（如果本来就在CPU也不会有问题）
    boxes = boxes.cpu()
    stopTime = time.time()
    print('* ------------------------------------------ *')
    print('* PROCESSING TIME COST : {}'.format(stopTime - startTime))

    if boxes.size()[0] == 0:
        print('* NO THINGS CAUGHT')
        return boxes.numpy()

    # 合并重合框
    frame = nms(boxes, 0.5, True).cpu().detach().numpy()  # box_idx, [N, IOU, CX, CY, W, H, CLS]
    # 反算416*416下的位置到图片的实际位置
    frame[:, 2:6] = return2RealPosition(frame[:, 2:6], expand, scale)
    print('* NUM OF BOXES : {} / {}'.format(frame.shape[0], boxes.size()[0]))
    return frame


if __name__ == '__main__':
    # 控制台参数输入设置
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--path', nargs='?', type=str, help='文件路径', required=True)
    parser.add_argument('--net_path', nargs='?', default=None, type=str, help='网络路径', required=False)
    parser.set_defaults(tboard=False)
    args = parser.parse_args()
    boxes = Eval(args.path, args.net_path)
    img = cv2.imread(args.path)
    # 画图展示
    for n, iou, x, y, w, h, *cls in boxes:
        print("* Object IOU : {}".format(iou))
        print("* Cls : {}".format(cls))
        x, y, w, h = list(map(int, [x, y, w, h]))
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.imshow('detection', img)
    cv2.waitKey(0)
