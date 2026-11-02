# 该脚本核心功能:统一管理RAG流程中所需的各类Prompt模板.
# 作用:通过LangChain#PromptTemplate创建 Prompt 模板，将不同场景的提示词(如:直接检索，子查询，回溯问题)，并返回最终的 Prompt 模板.
# 后续只需传入具体参数(例如:上下文，问题)即可快速生成符合需求的提示词，避免重复编写提示词.

from langchain_core.prompts import PromptTemplate


# todo 1.定义RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # 定义 RAG 提示模板
    # todo 1.1定义 RAG提示模板->根据上下文生成答案，无上下文则用自身知识，无法回答时返回:客服信息.
    # @staticmethod
    # # @staticmethod->表明是静态方法
    # def rag_prompt():
    #     # 创建并返回 PromptTemplate 对象
    #     return PromptTemplate(
    #         template="""
    #         你是一个智能助手，帮助用户回答问题。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         上下文: {context}
    #         问题: {question}
    #
    #         如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #         回答:
    #         """,
    #         #   定义输入变量
    #         input_variables=["context", "question", "phone"],
    #     )
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
        你是一名开发者技术文档智能问答助手（DevDocsRAG），面向开发者解答 LangChain、Milvus、BGE/FlagEmbedding、Hugging Face Transformers 的技术问题。

        回答规则：
        1. 优先严格依据下方「上下文」中的检索文档回答，不得编造 API 名称、参数、配置项、版本行为或文档结论。
        2. 涉及检索文档内容时，用 [来源n] 标注对应编号，编号必须与「上下文」中的编号一致，不得自行编造编号。
        3. 解释用中文；API 名称、类名、参数名、配置项和必要代码保留英文原文。
        4. 支持概念解释、代码示例（尽量完整可执行并说明关键步骤）、配置说明、报错排查和技术对比。
        5. 仅参考与当前问题相关的对话历史；无关历史忽略。
        6. 上下文与历史不足以回答时，明确说明「依据不足」，并给出有限、谨慎的通用建议，不要臆测。

        **上下文**: {context}
        **对话历史**:
        {history}
        **问题**: {question}

        **回答**:
        """,
            input_variables=["context", "history", "question"],
        )

    # todo 1.2定义假设问题生成的Prompt 模板 ->生成查询时的'假设性答案'，用于提升后续的检索精度.
    # 定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一名技术文档检索助手。为提升对以下技术问题的检索效果，请生成一个简短、具体、尽可能包含技术关键词的假设答案（用于构造检索向量）：
            问题: {query}
            假设答案:
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    # todo 1.3定义子查询生成的 Prompt 模板 ->将长/复杂查询拆分为多个简单子查询，便于分布检索.
    # 定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            将以下技术问题分解为多个更具体、可独立检索的子问题，每行一个，并保留 API/类名/参数名等关键词：
            查询: {query}
            子查询:
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    # todo 1.5定义答案可支撑性判定Prompt(Phase6)-> 在相关性门控之后，判定检索到的上下文是否真的
    #   包含足够回答问题的证据。输出严格 JSON，供程序解析为 sufficient/partial/insufficient 三分类。
    @staticmethod
    def support_judge_prompt():
        return PromptTemplate(
            template="""
你是检索证据充分性判定器。给定【问题】和【检索到的文档片段】，判断这些片段是否包含回答问题所问的关键信息。只依据片段内容判断，不要使用你自己的知识补充。

判定等级（三选一）：
- sufficient：片段包含回答该问题所需的核心事实/步骤/结论，可以据此完整回答。
- partial：片段与问题主题相关，但只包含部分信息，缺少回答所必需的关键细节，只能部分回答。
- insufficient：片段与问题无关，或问题所需信息完全不在片段中（例如询问作者、公司、营收、股价等片段未涉及的内容）。

只输出一个 JSON 对象，不要输出任何解释性文字或 Markdown 代码块：
{{"verdict":"sufficient|partial|insufficient","reason":"一句话理由","missing":"缺失的关键信息，可为空字符串"}}

【问题】: {query}
【检索到的文档片段】:
{context}
""",
            input_variables=["query", "context"],
        )

    # todo 1.4定义回溯问题生成的Prompt 模板 ->将复杂/冗长查询简化为简短问题，提升检索关键词集中度。
    # 定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            将以下技术问题简化为一个更基础、更聚焦核心意图的检索问题：
            查询: {query}
            简化问题:
            """,
            #   定义输入变量
            input_variables=["query"],
        )

if __name__ == '__main__':
    rag_prompt = RAGPrompts.rag_prompt()
    # format主要用于字符串格式化，这里关键字传参
    result = rag_prompt.format(context="[来源1] Milvus 是一款开源向量数据库，常用于 RAG 场景。", question="Milvus 是什么？")
    print(result)

    hyde_prompt = RAGPrompts.hyde_prompt()
    result1 = hyde_prompt.format(query='你认为什么是好人')
    print(result1)
