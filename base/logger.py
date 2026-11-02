import logging
import os
from config import Config

# todo 1.计算日志文件的最终保存路径.(工程中能够动态调整)
# 1.获取当前脚本文件的绝对路径.
module_path=os.path.abspath(__file__)
# print(module_path)         # E:\LLM_ubuntu\edu_rag_Project\logging_lesson\utils\logger.py
# 2.获取当前脚本所在目录的路径
# dirname(当前路径)  获取（当前路径的）上级目录
base_path=os.path.dirname(module_path)
# print(base_path)                # E:\LLM_ubuntu\edu_rag_Project\logging_lesson\utils
# 3.获取当前脚本所在目录的父目录的路径.
current_path=os.path.dirname(base_path)
# print(current_path)             # E:\LLM_ubuntu\edu_rag_Project\logging_lesson
# 4.拼接日志文件的完整路径.     上级目录+Logs文件夹+app.Log文件.
# log_file = os.path.join(current_path, 'logs/app.log')
log_file = os.path.join(current_path, Config().LOG_FILE)        #效果同上.
# print(f'log_file: {log_file}')

# todo 2.封装日志配置函数 一 可重复使用的双输出日志.
def setup_logger(log_flie=log_file):
    """
    函数功能:创建并返回1个日志记录器，支持日志同时输出到控制台和文件，且避免重复添加处理器。
    :param log_file:日志文件的保存路径
    :return: 日志记录器
    """
    # 1.确保日志目录存存(实际开发不需要确认，日志文件一定会手动建好存在)
    # 参1:获取日志的所在目录.参2: 如果目录已存在，不报错，如果不存在，则创建.
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    # 2.创建日志记录器对象 ->统一管理日志处理器，并设置日志级别.
    logger = logging.getLogger('EDU_RAG')
    logger.setLevel(logging.INFO)  # 后续处理器可设置更高级别.实现:记录器收全，处理器筛选。

    if not logger.handlers:
        # 3.创建控制台输出器 -> 用于将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level=logging.INFO)

        # 4.创建文件输出器 ->用于将日志输出到文件.
        # 默认就是追加模式
        file_handler = logging.FileHandler(filename=log_flie,encoding='utf-8')
        file_handler.setLevel(level=logging.INFO)

        # 5.创建日志输出格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        # 6.为两个处理器分别绑定日志格式
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 7.为日志记录器添加处理器，核心:判断处理器是否存在，避免多次调用函数导致重复添加
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# todo 3.初始化日志器 ->项目启动时自动执行，其它模块导入此Logger日志器即可直接使用
logger = setup_logger()
