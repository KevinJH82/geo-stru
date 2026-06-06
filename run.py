#!/usr/bin/env python3
"""遥感地质构造解译系统 - 启动脚本"""

import argparse
from config.config import Config
from app import app


def main():
    parser = argparse.ArgumentParser(description='遥感地质构造解译系统')
    parser.add_argument('--host', default=Config.HOST, help='服务器地址')
    parser.add_argument('--port', type=int, default=Config.PORT, help='服务器端口')
    parser.add_argument('--debug', action='store_true', default=Config.DEBUG, help='调试模式')
    args = parser.parse_args()

    Config.create_directories()

    print("=" * 60)
    print("遥感地质构造解译系统")
    print("=" * 60)
    print(f"服务器地址: http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
