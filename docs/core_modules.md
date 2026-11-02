# 核心模块说明

本文档按模块列出关键类与方法的职责、入参/返回值，便于二次开发。

## 1. `app.py` — FastAPI 服务入口

- 应用标题 `DevDocsRAG`，默认监听 `0.0.0.0:8081`（`HOST`/`PORT` 可覆盖）。
- 模块导入时创建全局 `IntegratedQASystem` 实例。
- `GREETING_PATTERNS`：问候/身份/在线/状态询问的正则与固定回复。
- `check_greeting(query) -> Optional[str]`：命中返回固定回复，否则 `None`。
- 路由：
  - `GET /`：返回 `static/index.html`
  - `GET /health`：`{"status":"healthy"}`
  - `GET /api/sources`：`{"sources": config.VALID_SOURCES}`
  - `POST /api/create_session`：UUID
  - `GET /api/history/{session_id}` / `DELETE /api/history/{session_id}`
  - `POST /api/query`：问候返回答案；非问候返回转流式提示
  - `WS /api/stream`：流式问答（`start/token/end/error`）

## 2. `qa_system.py` — `IntegratedQASystem`

集成层：会话历史 + RAG + LLM。

| 方法 | 说明 |
| --- | --- |
| `__init__` | 初始化日志、`Config`、`MySQLClient`、OpenAI 客户端、`VectorStore`、`RAGSystem`，并创建 `conversations` 表 |
| `init_conversation_table()` | `CREATE TABLE IF NOT EXISTS conversations(...)` |
| `call_dashscope(prompt)` | 流式生成器；`qwen-plus`，system prompt 限定 DevDocsRAG 领域 |
| `_fetch_recent_history(session_id)` | 读取最近 5 轮，按时间正序返回 |
| `get_session_history(session_id)` | 对外封装 |
| `update_session_history(session_id, question, answer)` | 插入新记录并裁剪至最近 5 轮 |
| `clear_session_history(session_id)` | 删除该会话全部记录，返回成功与否 |
| `query(query, source_filter=None, session_id=None)` | 生成器，`yield (token, False)` … `yield ("", True)`；结束后更新 `last_citations` 并落库 |

属性：`config`、`logger`、`mysql_client`、`client`、`vector_store`、`rag_system`、`last_citations`。

## 3. `rag_qa/core/rag_pipeline.py` — `RAGSystem`

RAG 主链路，`RAGSystem(vector_store, llm)`。

| 方法 | 说明 |
| --- | --- |
| `_llm_text(prompt)` | 兼容字符串/生成器，返回完整文本 |
| `_parse_verdict(text)` | 解析判定 JSON，兜底关键词，失败返回 `sufficient` |
| `assess_support(query, context)` | LLM 三分类证据判定；失败 fail-open |
| `sanitize_citations(text, n)` | 模块级函数，移除越界 `[来源k]` |
| `_sanitize_stream(gen, n)` | 流式安全的引用清理 |
| `_finalize_answer(answer, n)` | 统一收尾清理 |
| `_retrieve_with_hyde / _with_subqueries / _with_backtracking` | 三种增强检索，失败回退 |
| `_fallback_direct(query, source_filter)` | 直接检索回退 |
| `retrieve_and_merge(query, source_filter, strategy)` | 按策略检索并截取 `CANDIDATE_M` |
| `prepare_context(query, source_filter, strategy)` | 检索→去重→门控→构造上下文/引用，返回 `(context, citations, docs)` |
| `generate_answer(query, source_filter, history)` | 完整生成流程，返回流式生成器 |

模块常量：`_NO_CONTEXT_NOTE`、`_INSUFFICIENT_NOTE`、`_CITE_RE`。

## 4. `rag_qa/core/vector_store.py` — `VectorStore`

Milvus 读写与检索。

| 方法 | 说明 |
| --- | --- |
| `__init__(collection_name, host, port, database)` | 加载 Reranker 与 BGE-M3，连接 Milvus，建/载集合 |
| `_create_or_load_collection()` | 集合不存在时按 Schema 创建索引 |
| `add_documents(documents)` | 生成 dense/sparse 向量并分批 upsert（每批 1000） |
| `hybrid_search_with_rerank(query, k, source_filter)` | 混合检索 + 父文档去重 + 重排，返回 `CANDIDATE_M` 篇父文档 |
| `hybrid_search_with_rerank_scored(...)` | 同上，额外返回 reranker 分数 |
| `max_relevance_score(query, docs)` | 返回最高 reranker 分数（门控用） |
| `retrieve_top_k(query, k, source_filter)` | 只读返回融合检索的原始子块（评测用，含 `rank/distance`） |
| `_get_unique_parent_docs(sub_chunks)` | 按父块内容去重 |
| `_doc_from_hit(hit)` | Milvus 命中转 `Document` |

Schema 字段：`id/text/dense_vector/sparse_vector/parent_id/parent_content/source/timestamp/file_path/repo/version/language/doc_type/title/section`。

## 5. `rag_qa/core/query_classifier.py` — `QueryClassifier`

- 加载 `models/bert-base-chinese` 分词器；优先加载微调模型 `models/bert_query_classifier_devdocs`，否则用预训练 BERT 兜底。
- `predict_category(query) -> "通用知识" | "专业咨询"`（类别 0/1）。
- 另含 `train_model`、`preprocess_data`、`create_dataset`、`compute_metrics`、`evaluate_model`、`save_model` 等训练相关方法。

## 6. `rag_qa/core/strategy_selector.py` — `StrategySelector`

- `select_strategy(query) -> str`：调用 LLM（`temperature=0.1`），返回 `直接检索 / 假设问题检索 / 子查询检索 / 回溯问题检索`；调用失败默认 `直接检索`。
- `call_dashscope(prompt)`：可用于命令行独立测试 LLM。

## 7. `rag_qa/core/prompts.py` — `RAGPrompts`

静态方法返回 `PromptTemplate`：`rag_prompt()`（回答规则，含 `[来源n]` 约束）、`hyde_prompt()`、`subquery_prompt()`、`backtracking_prompt()`、`support_judge_prompt()`（严格 JSON 三分类）。

## 8. `base/config.py` — `Config`

- 解析 `config.ini`（路径默认项目根），环境变量优先级更高。
- 主要属性：MySQL/Redis/Milvus 连接参数、`LLM_MODEL`、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、检索参数（`PARENT_CHUNK_SIZE`、`CHILD_CHUNK_SIZE`、`CHUNK_OVERLAP`、`RETRIEVAL_K`、`CANDIDATE_M`、`RELEVANCE_THRESHOLD`、`SUPPORT_JUDGE`、`SUPPORT_JUDGE_MAX_CHARS`）、`VALID_SOURCES`、路径（`PROJECT_ROOT`、`DATA_DIR`、`MODELS_DIR`、`LOG_FILE`）。
- 无真实密钥兜底值；缺失时回退到空串或模板默认值。

## 9. `base/logger.py`

- `setup_logger()` 配置名为 `EDU_RAG` 的 logger，控制台 + `logs/app.log` 双输出，级别 `INFO`。
- 模块级 `logger = setup_logger()`，供全项目 `from base.logger import logger` 使用。

## 10. `mysql_qa/db/mysql_client.py` — `MySQLClient`

- `__init__`：用 `Config` 连接 MySQL，创建 `cursor`；失败抛异常。
- `close()`：关闭 `cursor` 与连接。
- 仅保留连接/游标/关闭职责，不含业务 SQL。

## 11. 离线模块

- `rag_qa/core/document_processor.py`：`load_document_from_directory`（多类型加载 + 溯源 metadata）、`process_documents`（父块 + 子块分层切分）。
- `rag_qa/core/build_knowledge_base.py`：独立示例脚本，构建 `edu_rag_exam_hybrid` 集合（含 `SimpleEmbedder/SimpleChunker/MilvusIndexer`），不属于在线服务链路。
- `rag_qa/classify_data/devdocs_classify/train_classifier.py`：查询分类器训练脚本。
