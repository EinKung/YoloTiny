import json
import os
import datetime


def init_settings(fixed_config_path=r'./config.json'):
    """
    初始化设置
    :param fixed_config_path: 固定的设置文件路径
    :return:设置dict数据
    """
    if os.path.exists(fixed_config_path):
        print("[{}]已找到设置文件".format(datetime.datetime.now()))
    else:
        print("[{}]未找到设置文件，正在初始化原始设置...".format(datetime.datetime.now()))
        return format_missing_settings()
    try:
        with open(fixed_config_path, 'r') as f:
            settings = json.load(f)
            # 转换anchors的键类型为int
            settings['anchors'] = dict([(int(item[0]), item[1]) for item in settings['anchors'].items()])
            # 计算每个anchors对应的areas值
            settings['areas'] = calc_area(settings['anchors'])
            print("[{}]设置读取完成".format(datetime.datetime.now()))
        return settings
    except Exception as e:
        print("[{}]设置文件读取发生错误".format(datetime.datetime.now()))
        print("*--------------------------------------------*")
        print(e)
        print("*--------------------------------------------*")
        raise e


def format_missing_settings():
    """
    格式化设置，当设置文件不存在时会使用程序内固定的设置重新生成
    :return:
    """
    fixed_format_settings = {
      "dataset_dir": "./dataset/train",
      "valid_dir": "./dataset/valid",
      "test_dir": "./dataset/test",
      "net_path": "./model/yolo-tiny.pth",
      "anchors": {
        "13": [[116, 90], [156, 198], [373, 326]],
        "26": [[30, 61], [62, 45], [59, 119]]
      },
      "epochs": 500,
      "batch_size": 2,
      "launch_mode": "train",
      "is_new": False,
      "log_dir": "./log",
      "plot_interval": 3,
      "save_loss_plot": False,
      "plot_pause": 0.001,
      "plot_loss": False,
      "optimizer": "Adam"
    }
    with open(r'./config.json', 'a+') as f:
        json.dump(fixed_format_settings, f)
        print("[{}]已初始化原始设置，设置文件已保存".format(datetime.datetime.now()))
    fixed_format_settings['anchors'] = dict([(int(item[0]), item[1]) for item in fixed_format_settings['anchors'].items()])
    fixed_format_settings['areas'] = calc_area(fixed_format_settings['anchors'])
    return fixed_format_settings


def calc_area(anchors):
    """
    计算areas
    :param anchors:
    :return:
    """
    areas = {}
    for feature_size in anchors:
        areas[feature_size] = [x * y for x, y in anchors[feature_size]]
    return areas


def apply_new_settings(changed_settings, original_settings):
    """
    应用新的设置
    :param changed_settings:
    :param original_settings:
    :return:
    """
    for settings_key in changed_settings:
        original_settings[settings_key] = changed_settings[settings_key]
    return original_settings


def print_settings(settings):
    """
    打印设置
    :param settings:
    :return:
    """
    print("*---------------------------------------------*")
    for settings_key in settings:
        print("* [{}]: {}".format(settings_key, settings[settings_key]))
    print("*---------------------------------------------*")
