#!/usr/bin/env python3
import logging
import logging.handlers
import os
import threading
from multiprocessing import Queue
from colorlog import ColoredFormatter


# 定义格式化器设置函数，handler=日志处理器，with_color=是否彩色
def __set_default_formatter(handler, with_color=True):
    if with_color:
        if os.getenv("eink_screen") == "1":
            with_color = False
    # 时间 + 日志级别 + 进程名 + 文件名:行号 + 日志消息
    format_str = "%(asctime)s %(levelname)s {%(processName)s} [%(filename)s => %(lineno)d] : %(message)s"
    if with_color:
        formatter = ColoredFormatter(
            "%(log_color)s" + format_str,
            log_colors={
                "DEBUG": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            style="%",
        )
    else:
        formatter = logging.Formatter(
            format_str,
            style="%",
        )

    handler.setFormatter(formatter)


# 定义消费线程函数，q=日志队列，logger=实际日志器，logger_lock=线程锁
def __worker(q, logger, logger_lock):
    while True:
        try:
            record = q.get()
            if record is None:
                return
            with logger_lock:
                logger.handle(record)
        except EOFError:
            return


# 初始化核心彩色日志器 __colored_logger
__logger_lock = threading.RLock()
__colored_logger = logging.getLogger("colored_logger")
if not __colored_logger.handlers:
    __colored_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    __set_default_formatter(handler, with_color=True)
    __colored_logger.addHandler(handler)
    __colored_logger.propagate = False


# 初始化多进程安全的日志器 __stub_colored_logger
__stub_colored_logger = logging.getLogger("colored_multiprocess_logger")
if not __stub_colored_logger.handlers:
    __stub_colored_logger.setLevel(logging.INFO)
    q = Queue()
    __stub_colored_logger.addHandler(logging.handlers.QueueHandler(q))
    __stub_colored_logger.propagate = False
    __background_thd = threading.Thread(
        target=__worker, args=(q, __colored_logger, __logger_lock)
    )
    __background_thd.start()


# 停止日志器
def stop_logger():
    q.put(None)
    __background_thd.join()


# 添加文件处理器，filename=日志文件路径
def add_file_handler(filename):
    filename = os.path.normpath(os.path.abspath(filename))
    with __logger_lock:
        for handler in __colored_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename == filename:
                    return handler
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(filename)
        __set_default_formatter(handler, with_color=False)
        __colored_logger.addHandler(handler)
        return handler


# 设置日志级别，level=日志级别（如logging.DEBUG）
def set_level(level):
    with __logger_lock:
        __stub_colored_logger.setLevel(level)


# 全局设置格式化器，formatter=自定义格式化器
def set_formatter(formatter):
    with __logger_lock:
        for handler in __colored_logger.handlers:
            handler.setFormatter(formatter)


# 获取当前日志配置
def get_logger_setting():
    setting = {}
    with __logger_lock:
        setting["level"] = __stub_colored_logger.level
        setting["handlers"] = []
        for handler in __colored_logger.handlers:
            handler_dict = {"formatter": handler.formatter}
            if isinstance(handler, logging.FileHandler):
                handler_dict["type"] = "file"
                handler_dict["filename"] = handler.baseFilename
            elif isinstance(handler, logging.StreamHandler):
                handler_dict["type"] = "stream"
            else:
                raise NotImplementedError()
            setting["handlers"].append(handler_dict)
    return setting


# 定义应用配置，setting=配置字典（同get_logger_setting返回格式）
def apply_logger_setting(setting):
    with __logger_lock:
        set_level(setting["level"])
        for handler_info in setting["handlers"]:
            if handler_info["type"] == "stream":
                handler = __colored_logger.handlers[0]
                assert isinstance(handler, logging.StreamHandler)
            elif handler_info["type"] == "file":
                handler = add_file_handler(handler_info["filename"])
            else:
                raise NotImplementedError()
            handler.setFormatter(handler_info["formatter"])


def get_logger():
    return __stub_colored_logger
