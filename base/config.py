# 该脚本用于:配置文件管理.
import os                   # 导入路径操作模块
import configparser         # 导入配置文件解析库，解析INI格式的配置文件.

# todo 1.配置文件路径计算.
current_file_path = os.path.abspath(__file__)
# print(f'Current file path: {current_file_path}')

current_dir_path = os.path.dirname(current_file_path)
# print(f'Current directory path: {current_dir_path}')

project_root = os.path.dirname(current_dir_path)
# print(f'Project root path: {project_root}')

config_file_path = os.path.join(project_root, 'config.ini')
# print(f'Config file path: {config_file_path}')

# todo 2.配置解析类->封装配置文件读取逻辑，提供统一的配置参数访问接口.
class Config():
    def __init__(self, config_file=config_file_path):
        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 如果没有提供配置文件路径，则使用默认路径

        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(config_file,encoding='utf-8')

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        # MySQL 密码
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback=''))
        # MySQL 数据库名
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='subjects_kg'))

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        # Redis 密码
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback=''))
        # Redis 数据库编号
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='itcast'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name', fallback='devdocs_final'))

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key'))
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=10)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)
        # 相关性门控阈值(reranker 分数)：<=0 表示关闭门控(Phase5 验证实验确定)
        self.RELEVANCE_THRESHOLD = self.config.getfloat('retrieval', 'relevance_threshold', fallback=0.2)
        # 答案可支撑性判定(Phase6)：在相关性门控通过后，用 LLM 判定证据是否足够，三分类 sufficient/partial/insufficient
        self.SUPPORT_JUDGE = self.config.getboolean('retrieval', 'support_judge', fallback=True)
        # 可支撑性判定的上下文截断长度(字符)，控制判定调用成本
        self.SUPPORT_JUDGE_MAX_CHARS = self.config.getint('retrieval', 'support_judge_max_chars', fallback=4000)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["langchain", "milvus", "bge", "transformers"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')

        # 路径配置


if __name__ == '__main__':
    # 绝对路径.
    # config_file_path=r'E:\LLM_ubuntu\integrated_qa_system\config.ini'
    # 相对路径.
    config_file_path='../config.ini'
    conf = Config(config_file_path)
    print(f'mysqL: {conf.MYSQL_HOST}, {conf.MYSQL_USER}, {conf.MYSQL_DATABASE}')
    print(f'redis: {conf.REDIS_HOST}, {conf.REDIS_PORT}, {conf.REDIS_DB}')
    print(f'log_file: {conf.LOG_FILE}')


