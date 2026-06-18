"""遥感地质构造解译系统 - 配置文件"""

import os


class Config:
    HOST = '0.0.0.0'
    PORT = 8082
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY', 'structural-interpretation-secret-key-2024')

    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, 'temp')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(BASE_DIR, 'logs', 'app.log')

    STRUCTURAL_DEFAULT_DEM = os.path.join(
        os.path.dirname(BASE_DIR),
        '非洲布基纳法索地区油气-2920km2区块【油｜气】（20251025任务，20260407下载）',
        'data', 'DEM.tif',
    )
    STRUCTURAL_DEFAULT_LANDSAT = os.path.join(
        os.path.dirname(BASE_DIR),
        '非洲布基纳法索地区油气-2920km2区块【油｜气】（20251025任务，20260407下载）',
        'data', 'Landsat 8 L2',
    )

    @staticmethod
    def create_directories():
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
