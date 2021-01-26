import json
import os


def init_settings(fixed_config_path=r'./config.json'):
    if check_settings_file(fixed_config_path):
        print()
    else:
        print()
    try:
        with open(fixed_config_path, 'r') as f:
            settings = json.load(f)
            settings['anchors'] = calc_area(settings['anchors'])
        return settings
    except Exception as e:
        print()
        raise e


def check_settings_file(config_path):
    if not os.path.exists(config_path):
        print()
        return False
    else:
        print()
        return True


def check_missing_settings():
    pass


def calc_area(anchors):
    areas = {}
    for feature_size in anchors:
        areas[feature_size] = [x * y for x, y in anchors[feature_size]]
    return areas
