"""
日志工具类
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }

    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class TaskLogger:
    """任务日志管理器"""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        self.task_logs = {}
        os.makedirs(log_dir, exist_ok=True)

    def create_task_logger(self, task_id: str) -> logging.Logger:
        """创建任务特定的日志器"""
        logger = logging.getLogger(f'task_{task_id}')
        logger.setLevel(logging.INFO)

        # 避免重复添加处理器
        if not logger.handlers:
            # 文件处理器
            log_file = os.path.join(self.log_dir, f'task_{task_id}.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)

            # 控制台处理器（带颜色）
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            colored_formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(colored_formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        # 初始化任务日志列表
        self.task_logs[task_id] = []

        return logger

    def get_task_logs(self, task_id: str) -> list:
        """获取任务日志列表"""
        return self.task_logs.get(task_id, [])

    def log_task_progress(self, task_id: str, message: str, level: str = 'INFO'):
        """记录任务进度"""
        if task_id in self.task_logs:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] [{level}] {message}"
            self.task_logs[task_id].append(log_entry)

    def save_logs_to_file(self, task_id: str, file_path: str):
        """保存任务日志到文件"""
        if task_id in self.task_logs:
            with open(file_path, 'w', encoding='utf-8') as f:
                for log_entry in self.task_logs[task_id]:
                    f.write(log_entry + '\n')


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    获取日志器

    Args:
        name: 日志器名称
        log_file: 日志文件路径（可选）

    Returns:
        logging.Logger: 日志器实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加处理器
    if not logger.handlers:
        # 控制台处理器（带颜色）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

        # 如果指定了日志文件，添加文件处理器
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


class AnalysisLogger:
    """分析任务专用日志器"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.logger = get_logger(f'analysis_{task_id}')
        self.task_logger = TaskLogger()

    def log_start(self, config: dict):
        """记录分析开始"""
        self.logger.info("=" * 50)
        self.logger.info(f"分析任务开始: {self.task_id}")
        self.logger.info(f"目标矿种: {config.get('mineral_type', '未知')}")
        self.logger.info(f"启用探测器: {', '.join(config.get('detectors', []))}")
        self.logger.info(f"融合模式: {'开启' if config.get('fusion_mode', True) else '关闭'}")
        self.logger.info("=" * 50)

    def log_data_loaded(self, data_info: dict):
        """记录数据加载完成"""
        self.logger.info("数据加载完成")
        self.logger.info(f"数据目录: {data_info.get('data_dir', '未知')}")
        self.logger.info(f"ROI 文件: {data_info.get('roi_file', '未知')}")
        self.logger.info(f"ROI 点数: {data_info.get('roi_points', 0)}")

    def log_detector_progress(self, detector_name: str, progress: float):
        """记录探测器处理进度"""
        self.logger.info(f"{detector_name} 处理进度: {progress:.1f}%")

    def log_fusion_complete(self, detectors: list, fusion_time: float):
        """记录融合完成"""
        self.logger.info("多探测器融合完成")
        self.logger.info(f"融合探测器: {', '.join(detectors)}")
        self.logger.info(f"融合耗时: {fusion_time:.2f} 秒")

    def log_post_process_complete(self, result_stats: dict):
        """记录后处理完成"""
        self.logger.info("后处理完成")
        self.logger.info(f"最大值: {result_stats.get('max_value', 0):.4f}")
        self.logger.info(f"最小值: {result_stats.get('min_value', 0):.4f}")
        self.logger.info(f"均值: {result_stats.get('mean_value', 0):.4f}")
        self.logger.info(f"标准差: {result_stats.get('std_value', 0):.4f}")

    def log_visualization_complete(self, output_files: dict):
        """记录可视化完成"""
        self.logger.info("可视化生成完成")
        for file_type, file_name in output_files.items():
            self.logger.info(f"{file_type}: {file_name}")

    def log_complete(self, result_dir: str):
        """记录分析完成"""
        self.logger.info("=" * 50)
        self.logger.info(f"分析任务完成: {self.task_id}")
        self.logger.info(f"结果目录: {result_dir}")
        self.logger.info("=" * 50)

    def log_error(self, error: str):
        """记录错误"""
        self.logger.error(f"错误: {error}")

    def get_logs(self) -> list:
        """获取任务日志"""
        return self.task_logger.get_task_logs(self.task_id)