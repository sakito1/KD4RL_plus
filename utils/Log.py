import logging
import argparse
import time
import os


def create_logger(logger_file_path):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)

    log_name = '{}.log'.format(time.strftime('%m-%d'))
    final_log_file = os.path.join(logger_file_path, log_name)

    # 【修改点1】使用 logger_file_path 作为唯一名称
    # 避免获取到全局 Root Logger，从而隔离不同进程/模块的日志配置
    logger = logging.getLogger(logger_file_path)

    logger.setLevel(logging.INFO)

    # 【修改点2】防止日志向上传播到父 Logger (Root Logger)
    # 这能避免如果 Root Logger 也有 Handler 时出现的重复打印
    logger.propagate = False

    # 【修改点3】检查是否已经添加过 Handler
    # 如果该 Logger 已经有 Handler（说明已被初始化过），则直接返回，避免重复添加
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(final_log_file, mode='a')  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设置控制台输出格式

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


