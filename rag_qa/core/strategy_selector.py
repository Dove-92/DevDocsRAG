#该脚本用于:检索策略选择器

# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入日志和配置
from base.config import Config
from base.logger import logger
# 导入 OpenAI
from openai import OpenAI

# todo 1.定义StrategySelector类:用于根据用户查询选择最合适的检索增强策略
class StrategySelector:
    # todo 1.1初始化方法 -> 创建大模型客户端，加载策略选择的提示模板.
    def __init__(self):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=Config().DASHSCOPE_API_KEY,
                             base_url=Config().DASHSCOPE_BASE_URL)
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()

    # todo 1.2 调用大模型的API -> 向DashScope发送请求，获取模型返回结果.
    def call_dashscope(self, prompt):
        """
        调用DashScope大模型API，获取模型对输入提示的响应.
        :param prompt:发给大模型的提示文本(字符和窜)，包含:用户查询和任务要求.
        :return:大模型返回的文本结果(字符串)，这里是选择的策略名称，如果调用失败，直接返回:直接检索
        """
        # 调用 DashScope API
        try:
            # 1.创建聊天完成请求 -> 调用大模型，传入模型名称，对话内容，温度参数，
            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个检索策略选择器，严格按用户要求返回策略名称，不要输出多余内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1
            )
            # 返回完成结果
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"

    # todo 1.3 获取策略选择提示模板 -> 定义引导大模型选择策略的固定格式文本，
    def _get_strategy_prompt(self):

        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个检索策略选择器，负责分析用户查询 {query}，从以下四种检索增强策略中选择最合适的一个，直接返回策略名称，不要解释。

            1. 直接检索：查询意图明确、需检索特定信息。示例："Milvus 怎么创建 Collection？"、"BGE-M3 的 dense 向量维度是多少？"
            2. 假设问题检索（HyDE）：查询抽象、直接检索效果差。示例："向量检索在大模型应用里一般怎么用？"
            3. 子查询检索：查询涉及多个方面，需拆分检索。示例："比较 Milvus 的 dense 和 sparse 检索的适用场景与配置。"
            4. 回溯问题检索：查询复杂、需简化为更基础问题。示例："我有海量数据想用 Milvus 做语义检索，要怎么设计索引和参数？"

            根据查询 {query}，直接返回策略名称，例如 "直接检索"。不要输出其他内容。
            """
            ,
            input_variables=["query"],
        )

    # todo 1.4定义方法，选择检索策略选择检索策略的核心方法 整合模板和大模型调用，返回最终策略.
    # 定义方法，选择检索策略
    def select_strategy(self, query):
        """
        函数作用:根据用户查询，选择最合适的检索增强策略
        :param query:用户输入的查询文本(字符串)
        :return:字符串 -> 选中的检索策略名称 例如:直接检索，子查询检索...
        """
        # 1.格式化提示模板:将用户查询填充到提示模板的query为止，生成发给大模型的完整提示，调用大模型获取策略
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    ss = StrategySelector()
    # ss.select_strategy('你好吗')
    ss.select_strategy('Milvus 怎么创建 Collection？')