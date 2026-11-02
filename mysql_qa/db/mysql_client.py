import os
import sys

import pymysql

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base import logger, Config


class MySQLClient:
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE
            )
            self.cursor = self.connection.cursor()
            self.logger.info('MysQL 连接成功')
        except pymysql.MySQLError as e:
            self.logger.error(f'MysQL 连接失败:{e}')
            raise

    def close(self):
        try:
            self.cursor.close()
            self.connection.close()
            self.logger.info("MysQL 连接已关闭")
        except pymysql.MySQLError as e:
            self.logger.error(f"关闭连接失败:{e}")
