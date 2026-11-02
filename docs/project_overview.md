# 项目概览（DevDocsRAG）

## 1. 项目定位

DevDocsRAG 是一个**面向开发者的技术文档智能问答系统**，覆盖四类技术栈的官方文档：

- LangChain
- Milvus
- BGE / FlagEmbedding
- Hugging Face Transformers

系统把「查询分类 → 检索策略选择 → Milvus 混合检索 → 重排序 → 证据可支撑性判定 → 大模型流式生成」串成一条 RAG 主链路，并用 MySQL 保存多轮会话历史。

## 2. 核心能力

| 能力 | 说明 |
| --- | --- |
| 领域限定问答 | 只回答上述四类技术栈的问题，域外问题明确拒答并说明依据不足 |
| 查询分类 | 本地 BERT 二分类：`通用知识`（直接由 LLM 回答）/ `专业咨询`（走检索） |
| 检索策略选择 | LLM 在「直接检索 / HyDE / 子查询 / 回溯问题」四种策略中选择 |
| 混合检索 | BGE-M3 稠密 + 稀疏向量，Milvus `WeightedRanker(1.0, 0.7)` 融合 |
| 重排序与门控 | BGE-Reranker 对父文档重排，低于阈值判定为「无相关依据」 |
| 证据可支撑性判定 | LLM 三分类 `sufficient / partial / insufficient`，抑制无依据作答 |
| 引用溯源 | 回答使用 `[来源n]` 标注，`citations` 返回 `source/file_path/repo/version/doc_type/url` |
| 流式输出 | WebSocket 逐 token 推送，协议 `start → token* → end` |
| 会话历史 | MySQL `conversations` 表，按 `session_id` 保留最近 5 轮 |
| 问候短路 | 中英文问候/身份询问走固定文案，不触发检索与 LLM |

## 3. 技术栈

- Web 服务：FastAPI + Uvicorn（HTTP + WebSocket）
- 向量库：Milvus 2.5.x（`devdocs_final`，数据库 `itcast`）
- 向量模型：BGE-M3（dense 1024 维 + sparse）
- 重排序：BGE-Reranker-Large（CrossEncoder）
- 查询分类：`bert-base-chinese` 微调的 `bert_query_classifier_devdocs`
- 生成模型：DashScope `qwen-plus`（OpenAI 兼容接口，流式）
- 会话存储：MySQL 8.x（`conversations` 表）
- 前端：单页 `static/index.html`

## 4. 高层架构

```text
static/index.html
        │  HTTP / WebSocket
        ▼
app.py (FastAPI)  ── 问候短路 / 健康检查 / 会话接口
        │
        ▼
qa_system.py (IntegratedQASystem)
        ├── MySQL conversations       会话历史（最近 5 轮）
        └── rag_qa/core/rag_pipeline.py (RAGSystem)
                ├── QueryClassifier     通用知识 / 专业咨询
                ├── StrategySelector    直接检索 / HyDE / 子查询 / 回溯
                ├── VectorStore         Milvus devdocs_final + BGE-M3 + Reranker
                └── call_dashscope      Qwen 流式生成
```

## 5. 运行入口

| 入口 | 用途 |
| --- | --- |
| `app.py` | 生产 Web 服务（HTTP + WebSocket），监听 `0.0.0.0:8081` |
| `qa_system.py` | 集成问答核心，可作为命令行交互（`python qa_system.py`） |
| `rag_qa/core/build_knowledge_base.py` | 独立的知识库构建示例脚本（离线，写入另一个集合） |

## 6. 关键设计决策

- **仅 RAG，无 FAQ 快通道**：Phase 9 已移除旧教育 FAQ/BM25/Redis 快速通道，所有非问候问题统一走 RAG 主链路。
- **失败即回退**：任一检索策略异常时回退到直接检索，避免静默空上下文。
- **判定失败 fail-open**：可支撑性判定器故障时按 `sufficient` 处理，优先不漏答。
- **引用清理流式安全**：`_sanitize_stream` 只回退「可能构成引用标记」的尾部片段，保证跨 chunk 引用不被误删。
- **不覆盖历史证据**：所有评测脚本支持 `RAG_RESULT_OUT`，新阶段结果使用独立文件名，历史基线不可变。

## 7. 文档索引

- [代码与目录导读](codebase_guide.md)
- [架构与工作流](architecture_and_workflow.md)
- [核心模块说明](core_modules.md)
- [部署与故障排查](setup_and_troubleshooting.md)
- [测试与评测](testing_and_evaluation.md)
