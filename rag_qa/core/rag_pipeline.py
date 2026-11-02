# 该脚本用于:新！！！
#    RAG系统的核心逻辑实现RAG(检索增强生成)系统的核心逻辑，整合查询分类(意图识别)，策略选择，文档检索和答案生成

#   导入 time 模块，用于计算时间
import time
import os,sys
import re, json

from rag_qa.core.prompts import RAGPrompts                  # 导入RAG相关的提示模板 -> 定义大模型的输入和输出格式
from rag_qa.core.query_classifier import QueryClassifier    # 导入查询分类器 -> 判断用户问题属于通用知识还是专业咨询，意图识别
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器 -> 专业咨询的情况下用哪种检索策略(例如:直接检索.
from rag_qa.core.vector_store import VectorStore            # 导入向量数据库对象 ->存储和检索文档向量，

# 优化1:新增导包内容
from transformers import BertTokenizer, BertModel
local_model_path=r'E:\LLM_ubuntu\rag_qa_system\rag_qa\models\bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model=BertModel.from_pretrained(local_model_path)
from openai import OpenAI

# todo 2.配置项目路径  确保程序能正确导入其它目录的模块.
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path=os.path.dirname(current_dir)
project_path=os.path.dirname(rag_qa_path)
sys.path.insert(0,rag_qa_path)

from base.config import Config
from base.logger import logger
# 加载项目配置
conf=Config()

# 无检索文档时注入的上下文占位说明：避免模型在空上下文下仍输出 [来源n] 引用标记
_NO_CONTEXT_NOTE = "（本次没有检索到任何知识库文档，上下文为空；请勿输出 [来源n] 引用标记，也不要编造引用来源。）"

# 检索到文档但经可支撑性判定证据不足时注入的占位说明(Phase6)
_INSUFFICIENT_NOTE = ("（检索到的文档与问题主题相关，但经证据充分性判定：缺少回答该问题所必需的关键信息，证据不足；"
                     "请明确说明「依据不足」，不要编造结论，也不要输出 [来源n] 引用标记。）")

# 匹配显式引用标记 [来源n] / 【来源n】(仅匹配显式"来源"形式，避免误伤代码中的下标如 arr[0])
_CITE_RE = re.compile(r'[\[【]\s*来源\s*(\d+)\s*[\]】]')


def sanitize_citations(text, n_citations):
    """移除越界/无效的显式引用标记 [来源k] / 【来源k】(k<1 或 k>n_citations)。
    只处理显式"来源"形式，避免破坏代码下标 arr[0] 等；n_citations=0 时移除全部显式引用。"""
    if not text:
        return text
    def _repl(m):
        k = int(m.group(1))
        return m.group(0) if 1 <= k <= n_citations else ''
    return _CITE_RE.sub(_repl, text)

# todo 3.定义RAGSystem类 封装RAG系统的完整流程，即:查询分类->策略选择 ->文档检索->答案生成
class RAGSystem():
    # todo 3.1初始化方法  创建RAG系统所需的核心组件 ->向量库，大模型，分类器...
    def __init__(self,vector_store,llm):
        """
        函数作用:初始化RAG系统
        :param vector_store:向量数据库对象 -> 用于存储和检索文档，提供相似性搜索功能.
        :param llm:大语言模型调用函数-> 接收提示文本，返回模型生成的回答.
        """
        # 1.保存向量数据库对象.
        self.vector_store=vector_store
        # 2.保存大模型调用函数 ->用于生成答案，子查询，假设答案等...
        self.llm=llm
        # 加载RAG提示模板 -> 定义生成答案时的固定格式.  这是其中一种
        self.rag_prompt=RAGPrompts.rag_prompt()
        # 4.初始化查询分类器 ->用于判断用户查询是'通用知识'还是'专业咨询
        # 4.1拼接分类模型的路径.
        classifier_path=os.path.join(rag_qa_path,'models','bert_query_classifier_devdocs')
        # 4.2创建查询分类器实例.
        self.query_classifier=QueryClassifier(classifier_path)
        # 5.初始化策略选择器 -> 用于专业咨询的检索策略选择.
        self.strategy_selector=StrategySelector()
        # 6.记录最近一次检索的引用来源(供前端展示 citation)
        self.last_citations=[]
        # 7.记录最近一次答案可支撑性判定结果(Phase6，供评测/调试)
        self.last_support_verdict=None
        # 8.记录最近一次最终上下文文档(Phase6，供评测/诊断)
        self.last_context_docs=[]
    # 兼容辅助:llm 可能返回字符串(非流式)或生成器(流式)，统一取完整文本，供检索策略内部使用.
    def _llm_text(self,prompt):
        """
        函数作用:调用 LLM 并返回完整文本。生产环境传入的 llm 为流式生成器(见 qa_system.call_dashscope)，
        若直接当作字符串使用会报 'generator' object has no attribute 'strip'，故在此统一拼接。
        :param prompt:提示文本(str)
        :return:模型生成的完整文本(str)
        """
        result = self.llm(prompt)
        # 非流式:已是字符串，直接返回
        if isinstance(result, str):
            return result
        # 流式:迭代生成器并拼接为完整文本
        return "".join(result)
    # 兼容辅助:策略执行失败时回退到直接检索，避免静默返回空上下文(空上下文会导致无依据回答)。
    def _fallback_direct(self,query,source_filter=None):
        """
        函数作用:HyDE/子查询/回溯任一策略失败时，回退到直接检索，保证仍能提供上下文。
        :param query:用户原始查询(str)
        :param source_filter:来源过滤条件
        :return:list[Document]，失败则返回空列表(打印错误，不再静默吞掉)
        """
        logger.warning(f"检索策略失败，回退到直接检索 (查询: '{query}')")
        try:
            return self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter)
        except Exception as e:
            logger.error(f"直接检索回退失败: {e}")
            return []

    # Phase6: 解析可支撑性判定器返回的 JSON，得到三分类 verdict。
    def _parse_verdict(self,text):
        """
        函数作用:从 LLM 输出中鲁棒地解析支撑性判定结果。
        :param text:判定器原始输出(str)
        :return:dict {'verdict': 'sufficient'|'partial'|'insufficient', 'reason':str, 'missing':str}
        """
        if text:
            m = re.search(r'\{.*?\}', text, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    v = str(obj.get('verdict', '')).strip().lower()
                    if v in ('sufficient', 'partial', 'insufficient'):
                        return {'verdict': v, 'reason': str(obj.get('reason', '')),
                                'missing': str(obj.get('missing', ''))}
                except Exception:
                    pass
            # 兜底:未解析出 JSON 时按关键词判定(insufficient 优先)
            low = text.lower()
            for v in ('insufficient', 'partial', 'sufficient'):
                if v in low:
                    return {'verdict': v, 'reason': 'keyword_fallback', 'missing': ''}
        return {'verdict': 'sufficient', 'reason': 'parse_failed', 'missing': ''}

    # Phase6: 答案可支撑性判定 -> 复用现有 LLM(不引入新模型)，判断上下文是否含足够证据。
    def assess_support(self,query,context):
        """
        函数作用:在相关性门控通过后，用 LLM 判定检索上下文对问题的证据充分性。
        :param query:用户查询(str)
        :param context:已构造的带编号上下文文本(str)
        :return:dict {'verdict','reason','missing'}；判定失败时按 sufficient 处理(避免误拒可回答问题)。
        """
        if not conf.SUPPORT_JUDGE:
            return {'verdict': 'sufficient', 'reason': 'disabled', 'missing': ''}
        try:
            ctx = context
            max_chars = getattr(conf, 'SUPPORT_JUDGE_MAX_CHARS', 4000)
            if max_chars and len(ctx) > max_chars:
                ctx = ctx[:max_chars]
            prompt = RAGPrompts.support_judge_prompt().format(query=query, context=ctx)
            raw = self._llm_text(prompt)
            verdict = self._parse_verdict(raw)
            logger.info(f"可支撑性判定完成: {verdict}")
            return verdict
        except Exception as e:
            # 判定失败时 fail-open(视为充分)，保证不因判定器故障而漏答
            logger.error(f"可支撑性判定失败，按 sufficient 处理: {e}")
            return {'verdict': 'sufficient', 'reason': f'judge_error:{e}', 'missing': ''}

    # Phase6: 流式安全的引用清理 -> 逐块输出时，仅回退可能构成 [来源n] 的尾部片段，保证跨块引用不被误删。
    def _sanitize_stream(self,gen,n_citations):
        """
        函数作用:对 LLM 流式输出做引用清理，移除越界 [来源k]。保留流式特性(仅回退极短的可能引用尾部)。
        :param gen:LLM 返回的生成器
        :param n_citations:有效引用数量
        :return:生成器(逐块 yield 清理后的文本)
        """
        buf = ""
        for chunk in gen:
            if not isinstance(chunk, str):
                chunk = str(chunk)
            buf += chunk
            cut = max(buf.rfind('['), buf.rfind('【'))
            if cut == -1:
                out, buf = buf, ""
            else:
                tail = buf[cut:]
                if ']' in tail or '】' in tail:
                    # 引用标记已闭合，可安全输出整段
                    out, buf = buf, ""
                elif re.fullmatch(r'[\[【]\s*[来源]{0,2}\s*\d*', tail):
                    # 可能是未完成的引用标记，回退到缓冲区，等待后续块
                    out, buf = buf[:cut], tail
                else:
                    out, buf = buf, ""
            if out:
                yield sanitize_citations(out, n_citations)
        if buf:
            yield sanitize_citations(buf, n_citations)

    # Phase6: 统一收尾 -> 无论 LLM 返回字符串还是生成器，都做引用清理。
    def _finalize_answer(self,answer,n_citations):
        if isinstance(answer, str):
            return sanitize_citations(answer, n_citations)
        return self._sanitize_stream(answer, n_citations)

    # todo 3.2定义私有方法，使用假设文档进行检索(HyDE)->生成假设答案，用假设答案检索相关文档.
    def _retrieve_with_hyde(self,query,source_filter=None):
        """
        函数作用:针对于抽象/开放性查询，先生成假设性答案，再用假设答案检索文档->解决 抽象查询 自接检索关键词 匹配度低 的问题.
        :param query:用户的原始查询文本(str).
        :param source_filter:检索来源过滤条件(str或者None)，例如:milvus表示只检索 Milvus 相关文档 None表示不过滤
        :return:list[Document]，每个Document对象的page_content属性为文档=文本，用于后续生成答案的上下文
        """
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = self._llm_text(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer,                        # 检索关键词:生成的假设答案.
                k=conf.RETRIEVAL_K,                 # 使用 K 而非  M
                source_filter=source_filter         # 应用来源过滤条件.
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return self._fallback_direct(query, source_filter)

    # todo 3.3定义私有方法，使用子查询进行检索 ->将复杂查询拆分为子查询，分别检索后合并结果.
    def _retrieve_with_subqueries(self,query,source_filter=None):
        """
        函数作用:针对于多实体/多方面的复杂查询，拆分为多个简单子查询，分别检索后合并去重解决多维度查询检索不全面的问题.
        :param query:用户的原始查询文本.
        :param source_filter:检索来源过滤条件(str或者None)，例如:milvus表示只检索 Milvus 相关文档，None表示不过滤
        :return:list[Document]对象
        """
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            subqueries_text = self._llm_text(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return self._fallback_direct(query, source_filter)

            #   初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            #   遍历每个子查询
            for sub_q in subqueries:
                #   使用子查询进行检索，并将结果添加到列表中
                #   这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K,  # 使用 K
                    source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            #   返回去重后的文档，限制数量 (是否需要在此处限制? retrieve_and_merge 末尾会限制)
            # return unique_docs[: Config.CANDIDATE_M]
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return self._fallback_direct(query, source_filter)

    # todo 3.4定义私有方法，使用回溯问题进行检索 -> 将复杂查询简化为基础问题后再检索.
    def _retrieve_with_backtracking(self,query,source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = self._llm_text(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K,  # 使用 K
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return self._fallback_direct(query, source_filter)

    # todo 3.5 核心方法:根据检索策略检索文档，用 合并/筛选最终上下文对象
    def retrieve_and_merge(self,query,source_filter=None,strategy=None):
        """
        函数作用:统一入口:根据指定策略(或白动选择策略)调用对应检索方法，筛选最终用于生成答案的上下文文档.
        :param query:用户的原始查询文本str，传递给策略选择器和检索方法
        :param source_filter:检索来源过滤条件
        :param strategy:指定检索策略，可选:直接检索，回溯问题检索...
        :return:list[Document]
        """
        # 1.如果未指定检索策略，自动选择检索策略.
        if not strategy:
            strategy =self.strategy_selector.select_strategy(query)
        # 2.根据策略调用对应的检索方法. ->获取候选文档列表.
        ranked_sub_chunks=[]
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query)  # 返回的是唯一文档列表
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
            # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")
        # 我觉得这步多余，因为ranked_sub_chunks返回的时候已经截取了前CANDIDATE_M了
        final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    # Phase6: 抽出"检索->去重->相关性门控->构造带编号上下文/引用"的公共流程，供生成与评测复用。
    def prepare_context(self,query,source_filter=None,strategy=None):
        """
        函数作用:执行检索并构造最终上下文与引用列表(不含 LLM 生成/可支撑性判定)。
        :param query:用户查询(str)
        :param source_filter:来源过滤条件
        :param strategy:检索策略；None 时自动选择
        :return:(context:str, citations:list[dict], context_docs:list[Document])
        """
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)
        #   检索相关文档
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )
        #   按 file_path 去重(保留首次出现顺序)，避免同一文件的多个父块产生重复 citation
        if context_docs:
            seen_files = set()
            unique_docs = []
            for doc in context_docs:
                fp = doc.metadata.get("file_path", "")
                if fp and fp in seen_files:
                    continue
                seen_files.add(fp)
                unique_docs.append(doc)
            context_docs = unique_docs

        #   相关性门控(Phase5): 最高 reranker 分数低于阈值 -> 判定无相关依据，清空上下文
        if context_docs and conf.RELEVANCE_THRESHOLD > 0:
            try:
                max_score = self.vector_store.max_relevance_score(query, context_docs)
                if max_score is not None and max_score < conf.RELEVANCE_THRESHOLD:
                    logger.info(
                        f"最高相关性分数 {max_score:.4f} < 阈值 {conf.RELEVANCE_THRESHOLD}，判定无相关依据，清空上下文"
                    )
                    context_docs = []
                else:
                    logger.info(f"相关性门控通过 (最高分 {max_score:.4f} >= {conf.RELEVANCE_THRESHOLD})")
            except Exception as e:
                logger.error(f"相关性门控计算失败，保持原检索结果: {e}")

        #   准备上下文(带来源编号，供模型引用 [n]) 并生成引用列表
        if context_docs:
            context_parts = []
            citations = []
            for i, doc in enumerate(context_docs):
                md = doc.metadata
                repo = md.get("repo", "")
                citations.append({
                    "source": md.get("source", ""),
                    "file_path": md.get("file_path", ""),
                    "repo": repo,
                    "version": md.get("version", ""),
                    "doc_type": md.get("doc_type", ""),
                    "url": f"https://github.com/{repo}" if repo else ""
                })
                fname = md.get("file_path", "") or md.get("source", "")
                context_parts.append(f"[来源{i+1}] {fname}\n{doc.page_content}")
            context = ("请基于以下编号文档片段回答，并在回答中用 [n] 标注引用来源编号。\n\n"
                       + "\n\n".join(context_parts))
            self.last_citations = citations
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
        else:
            context = _NO_CONTEXT_NOTE
            self.last_citations = []
            logger.info("未检索到相关文档，上下文为空")
        self.last_context_docs = context_docs
        return context, self.last_citations, context_docs

    # 优化2
    # todo 3.6 定义方法，生成答案
    def generate_answer(self,query,source_filter=None,history=None):
        """
        函数作用:根据查询类型选择直接用LLM或者执行RAG流程，支持流式输出.
        :param query:用户录入的原始查询文本(字符串形式)
        :param source_filter:检索来源过滤条件，仅对'专业咨询'生效.
        :param history:对话历史(列表，可选),包含字典:{'question':问题，'answer':答案}
        :return:生成器，逐段返回答案文本(字符串)
        """
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 来源过滤: {source_filter}")

        # 验证历史格式
        # [{}]
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]  # 限制最多5轮
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 构造历史上下文
        # TODO 这里构造了上下文
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        #   判断查询类型
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            self.last_citations = []
            prompt_input = self.rag_prompt.format(
                context=_NO_CONTEXT_NOTE, history=history_context, question=query
            )  # 不使用上下文，注入占位说明以避免空上下文下仍输出 [来源n]
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                answer = "抱歉，处理您的问题时出错，请稍后重试。"
            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            # 通用知识无检索上下文，引用数=0，统一清理可能出现的虚假引用
            return self._finalize_answer(answer, 0)

        #   否则，进行 RAG 检索并生成答案
        logger.info("查询为专业咨询，执行 RAG 流程")
        #   检索并构造上下文(含去重与相关性门控)
        context, _citations, context_docs = self.prepare_context(
            query, source_filter=source_filter
        )

        #   答案可支撑性判断(Phase6):分数门控通过后，再用 LLM 判断上下文是否真的包含足够证据，
        #   区分 sufficient(可回答)/partial(部分回答)/insufficient(拒答)。不通过则清空上下文。
        self.last_support_verdict = None
        if context_docs and conf.SUPPORT_JUDGE:
            verdict = self.assess_support(query, context)
            self.last_support_verdict = verdict
            v = verdict.get("verdict")
            if v == "insufficient":
                logger.info("可支撑性判定为 insufficient，清空上下文按无依据处理")
                context_docs = []
                context = _INSUFFICIENT_NOTE
                self.last_citations = []
            elif v == "partial":
                logger.info("可支撑性判定为 partial，要求仅回答有依据部分并说明缺失")
                context = ("注意：下方检索文档与问题主题相关，但只包含部分信息，可能不足以完整回答。"
                           "请仅回答文档确有依据的部分，并明确指出文档未涵盖的信息，不要编造。\n\n" + context)

        #   构造 Prompt，调用大语言模型生成答案

        # 构造提示
        # TODO generate阶段，放到了提示词里面，让大模型进行整合
        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query
        )
        # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = "抱歉，处理您的问题时出错，请稍后重试。"

        #   记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        #   统一清理越界引用后返回(引用数=当前有效引用数；insufficient 时已置 0)
        return self._finalize_answer(answer, len(self.last_citations))


if __name__=='__main__':
    # 1.实例化向量数据库.
    vector_store=VectorStore()
    # 2.定义大语言模型调用函数
    llm=StrategySelector().call_dashscope
    # 3.创建RAGSystem核心类的实例 -> 传入:向量数据库实例，大语言模型调用函数
    rag_system=RAGSystem(vector_store,llm)
    # 4.测试生成答案:查询'Milvus 怎么创建 Collection?’->只检索 Milvus 相关文档.
    # answer =rag_system.generate_answer('Milvus 怎么创建 Collection？', source_filter = 'milvus')
    # answer =rag_system.generate_answer('LangChain 怎么加载 PDF 文档？', source_filter = 'langchain')
    answer =rag_system.generate_answer( 'Milvus 怎么创建 Collection？', source_filter = 'milvus')
    # answer =rag_system.generate_answer('BGE-M3 的 dense 向量维度是多少？', source_filter = 'bge')
    # 5.打印模型生成的答案 ->实际部署时可改为:返回给前端或者存储，
    print(answer)



