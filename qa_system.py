"""
DevDocsRAG 集成问答系统：MySQL 会话历史 + Milvus 检索 + Qwen 流式生成。
"""
import warnings
warnings.filterwarnings("ignore")
# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa import MySQLClient
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import RAGSystem,VectorStore

# 导入配置和日志工具，用于系统配置和日志记录
from base.config import Config
from base.logger import logger
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time
import pymysql      # 这个库是Python操作MySQL的库，用于数据库操作的异常捕获，增删改查操作等
import uuid         # 生成唯一会话ID，标识:用户会话

# todo 2.集成问答系统的核心类
# 该类的作用:集成问答系统的核心功能(主类)，把 MySQL 会话历史、Milvus 检索和 Qwen 流式生成拼起来.
class IntegratedQASystem:
    def __init__(self):
        # 1.初始化日志工具，用于记录系统运行信息和配置对象
        self.logger = logger
        self.config = Config()
        # 2.初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 3.初始化大语言模型 LLM客户端
        try:
            self.client=OpenAI(api_key=self.config.DASHSCOPE_API_KEY,base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            # 记录 0penAI 初始化失败的错误日志
            self.logger.error(f"OpenAI 客户端初始化失败:{e}")
            # 抛出异常，终止初始化
            raise
        # 4.初始化向量数据库(用于RAG系统的知识库管理)和RAG系统
        self.vector_store = VectorStore()                   #创建向量数据库实例 ->包括:向量存储 和 检索
        self.rag_system = RAGSystem(self.vector_store,self.call_dashscope)          # 创建RAG系统实例包括:检索和生成
        # 记录最近一次查询的引用来源(供前端展示 citation)
        self.last_citations = []

        # 5.(优化1)初始化对话历史表:在MySQL中创建存储会话记录的表
        self.init_conversation_table()

    # todo 2.2 (新增)初始化对话历史表的方法:在MySQL中创建存储对话记录的表结构.
    def init_conversation_table(self):
        """
        函数功能:初始化MySQL中的Conversations表，用于存储对话历史.
        :return:
        """
        try:
            # 创建 conversations 表，包含会话 ID、问题、答案和时间戳
            self.mysql_client.cursor.execute("""
                       CREATE TABLE IF NOT EXISTS conversations (
                           id INT AUTO_INCREMENT PRIMARY KEY,       # 主键id
                           session_id VARCHAR(36) NOT NULL,         # 会话id
                           question TEXT NOT NULL,                  # 问题
                           answer TEXT NOT NULL,                    # 答案
                           timestamp DATETIME NOT NULL,             # 时间戳
                           # 提高了查询效率，但是修改删除操作会变得复杂，要多对这个索引表做操作。具体要不要这个看需求
                           INDEX idx_session_id (session_id)        # 创建索引列，目的:提高查询的效率.
                       )
                   """)
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录表初始化成功的日志
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            # 记录表初始化失败的错误日志
            self.logger.error(f"初始化对话历史表失败: {e}")
            # 抛出异常，终止初始化
            raise

     # todo 2.3 （有调整）调用大语言模型的方法:给RAG系统用的.
    def call_dashscope(self, prompt):
        """
        函数作用:调用DashScope的大模型生成文本-> 流式输出
        :param prompt:输入给模型的提示文本(字符串) 即:用户输入的查询问题
        :return:生成器，逐段返回大模型生成的答案内容(字符串)
        """
        # 1.定义调用 DashScope API的方法，生成自然语言答案
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是 DevDocsRAG 技术文档智能问答助手，面向开发者解答 LangChain、Milvus、BGE/FlagEmbedding、Hugging Face Transformers 的技术问题。请严格依据提供的上下文回答，不要编造 API、参数或版本行为。"},
                    {"role": "user", "content": prompt}
                ],
                timeout=30,     # 设置 30秒超时
                stream=True     # 启用流式输出
            )
            # 2.遍历流式输出的每个 chunk
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 获取当前 chunk的文本内容
                    content=chunk.choices[0].delta.content
                    # 逐 token 返回，供前端实时显示
                    yield content
        except Exception as e:
            # 记录 API 调用失败的错误日志
            self.logger.error(f"LLM 调用失败: {e}")
            # 返回错误信息，便于调试
            return f"错误：LLM 调用失败 - {e}"

    # todo 2.4 (新增)获取最近对话历史的内部方法:从MySQL中查询指定会话(ID)的最近5轮对话.
    def _fetch_recent_history(self,session_id):
        """
        函数功能:获取最近5轮对话历史
        :param session_id:会话唯一标识(字符串)，用于定位具体会话.
        :return:列表，包含字典形式的对话记录，例如:[{'question':问题，'answer':答案}]
        """
        try:
            # 执行 SQL 查询，获取最近 3 轮对话
            self.mysql_client.cursor.execute("""
                        SELECT question, answer FROM conversations WHERE session_id = %s
                        ORDER BY timestamp DESC LIMIT %s""", (session_id, 5))
            # 将查询结果转换为字典列表
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 反转结果，按时间正序返回
            return history[::-1]
        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            self.logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []

    # todo 2.5 (新增)获取会话历史的方法:对外提供获取指定会话历史的接口.
    # 即安全问题，对外看不到你内部是如何实现调用查询的
    def get_session_history(self, session_id: str) -> list:
        """
        函数功能:从MySQL获取会话历史.
        :param session_id:会话唯一标识(字符串)
        :return:列表，包含最近5轮对话记录-> 同_fetch_recent_history()的返回值
        """
        # 调用 _fetch_recent_history 获取对话历史
        return self._fetch_recent_history(session_id)

    # todo 2.6 (新增)更新会话历史的方法:将新对话记录存入MySQL，并保留最近5轮。
    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """
        函数功能:更新会话历史到MySQL，保留最近5轮对话
        :param session_id:会话唯一标识(字符串
        :param question:用户的问题(字符串)
        :param answer:系统生成的答案(字符串)
        :return:查询到的更新后的5条历史对话
        """
        try:
            # 插入新的对话记录
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())""", (session_id, question, answer))
            # 获取更新后的对话历史
            history = self._fetch_recent_history(session_id)
            # 删除超出 5 轮的旧记录
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM conversations WHERE session_id = %s ORDER BY timestamp DESC
                        LIMIT %s ) AS sub
                )""", (session_id, session_id, 5))
            # 提交事务
            self.mysql_client.connection.commit()
            # 记录更新成功的日志
            self.logger.info(f"会话 {session_id} 历史更新成功")
            # 返回更新后的历史
            return history
        except pymysql.MySQLError as e:
            # 记录数据库操作失败的错误日志
            self.logger.error(f"更新会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 抛出异常
            raise
        except Exception as e:
            # 记录意外错误的日志
            self.logger.error(f"更新会话历史意外错误: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 抛出异常
            raise

    # todo 2.7 (新增)清除会话历史:删除指定会话的所有记录.
    def clear_session_history(self, session_id: str) -> bool:
        """
        函数功能:清除指定会话历史
        :param session_id:会话唯一标识(字符串
        :return:True -> 清除成功，False -> 清除失败
        """
        try:
            # 删除指定 session_id 的所有对话记录
            self.mysql_client.cursor.execute("""
             DELETE FROM conversations WHERE session_id = %s""", (session_id,))
            # 提交事务
            self.mysql_client.connection.commit()
            # 记录清除成功的日志
            self.logger.info(f"会话 {session_id} 历史已清除")
            # 返回 True 表示成功
            return True
        except pymysql.MySQLError as e:
            # 记录清除失败的错误日志
            self.logger.error(f"清除会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 返回 False 表示失败
            return False

    # todo 2.8(优化)处理查询的核心方法:整合检索，大模型生成和对话历史，返回流式答案
    def query(self,query,source_filter=None,session_id=None) :
        """查询集成系统，支持对话历史和流式输出"""
        start_time = time.time()  # 记录查询开始时间
        # 记录查询信息到日志
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        # 获取对话历史，若无 session_id 则返回空列表
        history = self.get_session_history(session_id) if session_id else []
        self.logger.info("使用RAG回答")
        # 初始化收集完整答案的字符串
        collected_answer = ""
        # 从 RAG 系统获取流式输出
        for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
            # 累积答案
            collected_answer += token
            # 逐 token 返回，标记为部分答案
            yield token, False
        # 取出本次检索的引用来源，供前端展示
        self.last_citations = self.rag_system.last_citations
        if session_id:
            # 更新对话历史，存储完整答案
            self.update_session_history(session_id, query, collected_answer)
        # 计算处理时间
        processing_time = time.time() - start_time
        # 记录处理时间到日志
        self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
        # 返回空字符串，标记流结束
        yield "", True


# todo 3.定义main函数:提供命令行交互界面，测试 集成问答系统.
def main():
    # 定义主函数，提供命令行交互界面
    qa_system = IntegratedQASystem()  # 初始化问答系统
    # 生成唯一会话 ID
    session_id = str(uuid.uuid4())
    # 打印欢迎信息
    print("\n欢迎使用集成问答系统！")
    # 打印会话 ID
    print(f"会话ID: {session_id}")
    # 打印支持的技术来源
    print(f"支持的技术来源：{qa_system.config.VALID_SOURCES}")
    # 提示用户输入查询或退出
    print("输入查询进行问答，输入 'exit' 退出。")
    try:
        while True:
            # 获取用户输入的查询
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 如果用户输入 exit，记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见！欢迎下次使用！")
                # 退出循环
                break
            # 获取用户输入的技术来源过滤
            source_filter = input(
                f"请输入技术来源 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果来源过滤无效，记录警告日志
                logger.warning(f"无效的技术来源 '{source_filter}'，将不过滤")
                # 设置为空，忽略过滤
                source_filter = None
            # 打印答案提示
            print("\n答案: ", end="", flush=True)
            # 初始化累积答案的字符串
            answer = ""
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # 仅当 token 非空时打印
                    print(token, end="", flush=True)
                    # 累积答案
                    answer += token
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break
            # 打印对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        qa_system.mysql_client.close()


if __name__ == '__main__':
    main()
    # qa=IntegratedQASystem()
    # qa.clear_session_history('401d4d31-a4de-45c8-84db-2a0f163b5237')