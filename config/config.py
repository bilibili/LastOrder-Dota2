import yaml
import os


# 读取配置文件
def read_config(filepath):
    res = None
    with open(filepath, "rt", encoding="utf8") as f:
        res = yaml.safe_load(f)
    return res[enviroment()]


def enviroment():
    return "production" if is_production() else "development"


def is_production():
    return not is_development()


def is_development():
    return "LOCALTEST" in os.environ.keys()
