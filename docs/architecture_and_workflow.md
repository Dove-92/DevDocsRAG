# 架构与工作流

本文档描述一次问答请求从进入服务到返回答案的完整链路，以及各层的职责边界。

## 1. 分层架构

```text
┌───────────────────────────────────────────────────────────┐
│ 前端 static/index.html                                      │
│   HTTP: /api/query /api/history /api/sources ...           │
│   WebSocket: /api/stream                                   │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│ app.py (FastAPI)                                           │
│   - 问候短路 GREETING_PATTERNS                             │
│   - 会话创建/历史查询/清除                                 │
│   - WebSocket 流式转发 start/token/end/error              │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│ qa_system.py (IntegratedQASystem)                          │
│   - 会话历史读写（MySQL conversations，最近 5 轮）         │
│   - call_dashscope：流式 LLM 调用                          │
│   - last_citations：最近一次引用                           │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│ rag_qa/core/rag_pipeline.py (RAGSystem)                    │
│   分类 → 策略选择 → 检索 → 去重 → 相关性门控 → 证据判定   │
│   → 构造带编号上下文 → 生成 → 引用清理                     │
└───────┬───────────────────────────────┬───────────────────┘
        │                               │
┌───────▼───────────────┐   ┌───────────▼───────────────────┐
│ QueryClassifier (BERT)│   │ VectorStore (Milvus + BGE)    │
│ 通用知识 / 专业咨询    │   │ 混合检索 + Reranker + 相关性分 │
└───────────────────────┘   └───────────────────────────────┘
```

## 2. HTTP 请求流程

### 2.1 非流式 `/api/query`

1. `check_greeting` 命中问候 → 直接返回固定文案，`is_streaming=false`。
2. 非问候问题 → 返回 `is_streaming=true`，提示改用 WebSocket（真正答案走流式）。

### 2.2 会话接口

- `POST /api/create_session`：返回 UUID `session_id`。
- `GET /api/history/{session_id}`：从 MySQL 读取最近 5 轮。
- `DELETE /api/history/{session_id}`：删除该会话全部记录。

### 2.3 WebSocket `/api/stream`

客户端发送：

```json
{"query": "Milvus 如何创建 Collection？", "source_filter": "milvus", "session_id": "..."}
```

服务端消息序列：

```text
{"type":"start","session_id":...}
{"type":"token","token":"...","session_id":...}   # 0..n 次
{"type":"end","session_id":...,"is_complete":true,"processing_time":...,"citations":[...]}
```

异常时发送 `{"type":"error","error":"..."}`。

## 3. RAG 主链路（RAGSystem.generate_answer）

```
query, source_filter, history
  │
  ├─ 1. 校验 history（最多 5 轮，格式 {question, answer}），拼成 history_context
  │
  ├─ 2. query_classifier.predict_category(query)
  │        ├─ 通用知识 → rag_prompt(context=_NO_CONTEXT_NOTE) → LLM 流式 → 引用清理(n=0)
  │        └─ 专业咨询 → 继续
  │
  ├─ 3. prepare_context(query, source_filter)
  │        ├─ strategy_selector.select_strategy(query)（或使用指定策略）
  │        ├─ retrieve_and_merge → 各策略检索
  │        ├─ 按 file_path 去重
  │        ├─ 相关性门控：max_relevance_score < relevance_threshold → 清空上下文
  │        └─ 构造带 [来源n] 的上下文 + citations 列表
  │
  ├─ 4. assess_support(query, context)（support_judge=true 且上下文非空）
  │        ├─ sufficient  → 正常生成
  │        ├─ partial     → 追加「仅回答有依据部分」指令
  │        └─ insufficient→ 清空上下文与引用，改用 _INSUFFICIENT_NOTE
  │
  ├─ 5. rag_prompt.format(context, history, question) → llm(prompt) 流式
  │
  └─ 6. _finalize_answer：sanitize_citations 移除越界 [来源k]
```

## 4. 检索策略

`StrategySelector` 让 LLM 在四种策略中选择（返回中文策略名）：

| 策略 | 触发场景 | 实现 |
| --- | --- | --- |
| 直接检索 | 意图明确、需检索特定信息 | `hybrid_search_with_rerank(query)` |
| 假设问题检索（HyDE） | 抽象、开放性查询 | 生成假设答案 → 用假设答案检索 |
| 子查询检索 | 多实体/多方面复杂查询 | 拆分子问题 → 分别检索 → 按内容去重 |
| 回溯问题检索 | 复杂查询需简化 | 简化为基础问题后检索 |

任一策略抛异常时调用 `_fallback_direct` 回退到直接检索，保证仍有上下文。

> 注意：当前实现中 `source_filter` 仅在 `直接检索` 分支透传；HyDE / 子查询 / 回溯分支只传 `query`。如需对全部策略生效过滤，可在 `retrieve_and_merge` 中统一传入。

## 5. 混合检索与重排序（VectorStore）

1. BGE-M3 生成查询的 dense(1024) + sparse 向量。
2. 分别构造 `AnnSearchRequest`（`metric_type=IP`，dense `nprobe=10`）。
3. `WeightedRanker(1.0, 0.7)` 融合 dense 与 sparse 结果。
4. 命中子块 → `_get_unique_parent_docs` 按父块内容去重。
5. 候选父文档 ≥ 2 时用 BGE-Reranker 打分降序排序，取 `CANDIDATE_M` 篇。
6. 相关性打分由 `max_relevance_score` 提供，供门控使用。

## 6. 引用与清理

- 上下文以 `[来源i] file_path\n内容` 编号拼接，要求模型用 `[来源n]` 标注。
- `citations` 元素：`source`、`file_path`、`repo`、`version`、`doc_type`、`url`（`https://github.com/{repo}`）。
- `sanitize_citations(text, n)`：删除 `k<1` 或 `k>n` 的显式引用；`n=0` 时清除全部。
- 流式场景用 `_sanitize_stream`：仅回退可能构成「未闭合引用标记」的尾部，避免跨 chunk 误删。

## 7. 失败与降级策略

| 场景 | 行为 |
| --- | --- |
| 检索策略异常 | 回退直接检索 |
| 直接检索回退失败 | 返回空列表 → 空上下文 |
| 相关性分数低于阈值 | 清空上下文，走「依据不足」 |
| 可支撑性判定失败 | fail-open，按 `sufficient` 处理 |
| 判定为 insufficient | 清空上下文与引用 |
| LLM 调用异常 | 返回错误提示文本，不中断服务 |
| 无检索结果 | 注入 `_NO_CONTEXT_NOTE`，禁止输出引用标记 |

## 8. 数据与状态

- MySQL `conversations(id, session_id, question, answer, timestamp)`，`session_id` 建索引；每次写入后仅保留最近 5 轮。
- Milvus 集合 `devdocs_final`（约 32,854 条），字段含 `dense_vector`、`sparse_vector` 及溯源字段。
- `RAGSystem` 每次调用后更新 `last_citations`、`last_support_verdict`、`last_context_docs`，供接口与评测复用。
