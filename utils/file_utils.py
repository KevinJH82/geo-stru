"""文件处理工具"""

import os


def get_file_size(file_path):
    """获取文件大小（格式化显示）"""
    if not os.path.exists(file_path):
        return "0 B"

    size = os.path.getsize(file_path) if os.path.isfile(file_path) else _dir_size(file_path)

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size:.2f} TB"


def _dir_size(directory):
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total
