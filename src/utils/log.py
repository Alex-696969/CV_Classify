"""
Desc:日志模块
"""
import os.path
import time
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, output_dir):
        self.output_dir = os.path.join(output_dir, 'log')
        self.log_dir = self.build_log_dir()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.log_dir = None

    def build_log_dir(self):
        t = time.localtime()
        log_dir = os.path.join(self.output_dir, f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir


    def build_logger(self):
        self.build_log_dir()
        return self.writer

if __name__ == '__main__':
    output_dir = f'D:\MySoftWare\Pycharm_20250301\PythonProject\CV_Project\projects\cv_classify\src/output'
    logger = Logger(output_dir)
    logger.build_log_dir()
