# DevDocsRAG

面向开发者的技术文档智能问答系统，覆盖 **LangChain、Milvus、BGE/FlagEmbedding、Hugging Face Transformers**。系统使用 Milvus 向量检索 + BGE-M3 混合检索 + 查询分类/策略选择 + 可支撑性判定，通过 DashScope Qwen 生成流式答案，并使用 MySQL 保存会话历史。

## 架构

```text
static/index.html
        │  HTTP / WebSocket
        ▼
app.py (FastAPI)
        │
        ▼
qa_system.py (IntegratedQASystem)
        ├── MySQL conversations    会话历史（最近 5 轮）
        └── rag_qa/core/rag_pipeline.py (RAGSystem)
                ├── QueryClassifier     通用知识 / 专业咨询
                ├── StrategySelector    直接检索 / HyDE / 子查询 / 回溯
                ├── VectorStore         Milvus devdocs_final + BGE-M3 + reranker
                └── call_dashscope      Qwen 流式生成
```

## 目录

```text
app.py                     FastAPI 入口（HTTP + WebSocket）
qa_system.py               集成问答核心：会话历史 + RAG 流式输出
config.ini                 本地配置（含密钥，不应提交）
config.example.ini         配置模板（占位值）
requirements.txt           Python 依赖
base/                      配置与日志
mysql_qa/                  MySQL 连接（仅保留连接/游标/关闭）
rag_qa/core/               RAG 主链路与提示词
rag_qa/models/             DevDocs 查询分类模型
rag_qa/data/               文档数据
eval/                      评测脚本、回归脚本与 results/ 结果
static/                    前端静态页面
docs/                      项目文档（概览、代码导读、架构、模块、部署、评测）
logs/                      运行日志
_backup_* / *.phase5bak    历史备份（保留）
```

## 文档

- [项目概览](docs/project_overview.md)
- [代码与目录导读](docs/codebase_guide.md)
- [架构与工作流](docs/architecture_and_workflow.md)
- [核心模块说明](docs/core_modules.md)
- [部署与故障排查](docs/setup_and_troubleshooting.md)
- [测试与评测](docs/testing_and_evaluation.md)

## 环境要求

- Python 3.10（建议使用独立 conda 环境）
- MySQL 8.x，数据库默认 `subjects_kg`
- Milvus 2.5.x，默认 `localhost:19530`，数据库 `itcast`，集合 `devdocs_final`
- DashScope API Key（Qwen）
- 可选：Redis。Phase 9 后 FAQ 缓存已移除，运行时不依赖 Redis

安装依赖：

```powershell
conda activate edu_rag
pip install -r requirements.txt
```

`torch==2.10.0` 需按平台选择 wheel。GPU 环境建议从 PyTorch 官方源安装对应 CUDA 版本（本项目验证环境为 CUDA 12.8）。

## 配置

```powershell
Copy-Item config.example.ini config.ini
```

编辑 `config.ini`，至少修改：

- `[mysql] password`
- `[llm] dashscope_api_key`
- 如 Milvus 不在本机，修改 `[milvus] host/port`

密钥也可以使用环境变量覆盖：

```text
MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE
MILVUS_HOST / MILVUS_PORT / MILVUS_DATABASE_NAME / MILVUS_COLLECTION_NAME
DASHSCOPE_API_KEY
```

不要把真实 `config.ini`、日志或评测输出中的敏感信息提交到公开仓库。

## 启动

```powershell
python app.py
# 或使用验证环境
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" app.py
```

默认监听 `0.0.0.0:8081`（可用 `HOST`、`PORT` 环境变量覆盖）。启动后访问：

- 首页：`http://127.0.0.1:8081/`
- 健康检查：`http://127.0.0.1:8081/health`
- Swagger：`http://127.0.0.1:8081/docs`

## HTTP / WebSocket 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 前端首页 |
| GET | `/health` | 健康检查 |
| GET | `/api/sources` | 支持的技术来源 |
| POST | `/api/create_session` | 创建会话 ID |
| GET | `/api/history/{session_id}` | 获取最近 5 轮历史 |
| DELETE | `/api/history/{session_id}` | 清除会话历史 |
| POST | `/api/query` | 非流式入口；非问候问题返回转流式提示 |
| WS | `/api/stream` | 流式问答 |

WebSocket 请求：

```json
{"query": "Milvus 如何创建 Collection？", "source_filter": "milvus", "session_id": "..."}
```

服务端消息类型：

```text
start -> token* -> end
error
```

`end` 帧包含 `is_complete`、`processing_time` 和 `citations`。`citations` 元素包含 `source`、`file_path`、`repo`、`version`、`doc_type`、`url`。

## 检索参数

`config.ini` 的 `[retrieval]`：

- `retrieval_k = 10`
- `candidate_m = 2`
- `relevance_threshold = 0.2`
- `support_judge = true`
- `support_judge_max_chars = 4000`

## 测试与回归

先启动服务（默认 8081），再运行运行期脚本：

```powershell
python eval/test_phase9_cleanup.py
python eval/verify_runtime_phase8.py
python eval/verify_runtime_phase7.py
python eval/test_domain_consistency.py
python eval/test_phase6.py
python eval/test_fixes.py
python eval/test_phase51_regression.py
```

结果写入 `eval/results/`。可用 `RAG_RESULT_OUT` 指定隔离的输出路径，避免覆盖历史基线：

```powershell
$env:RAG_RESULT_OUT="eval/results/phase9_runtime.json"
python eval/verify_runtime_phase8.py
```

## Phase 10 命名整理

- `new_main.py` → `qa_system.py`
- `rag_qa/core/new_rag_system.py` → `rag_qa/core/rag_pipeline.py`
- `mysql_qa/db/mysql_client.py` 类 `mysql_client` → `MySQLClient`
- `rag_qa/rag_assessment/rag_as.py` → `ragas_eval.py`
- `rag_qa/classify_data/提示词模块.txt` → `prompt_template.txt`

`base/`、`rag_qa/edu_document_loaders/`、`rag_qa/edu_text_spliter/` 等历史包名暂保留，避免影响既有导入与运行链路。

## Phase 9 清理记录

已移除：

- `mysql_qa/data/JP学科知识问答.csv`
- MySQL `subjects_kg.jpkb` 表（467 行）
- Redis `qa_original_questions`、`qa_tokenized_questions`、`answer:*`（14 个键）
- `mysql_qa/cache/`、`mysql_qa/retrieval/`、`mysql_qa/utils/`、`mysql_qa/main.py`
- `eval/audit_faq.py`、`eval/test_phase8_faq_off.py`
- 旧教育领域 CLI/主链路：`rag_qa/main.py`、`rag_qa/core/rag_system.py`、`rag_qa/core/rag_query.py`
- `faq_enabled` / `FAQ_ENABLED` 配置项与运行时 FAQ 分支

保留：

- MySQL `conversations` 表与会话历史功能
- Milvus `devdocs_final`（约 32,854 条）与 `edurag_final`
- 模型、文档数据、历史评测结果、回归基线、备份目录
- Redis 中来源不明的 `test_key`（未删除）

Phase 8 的历史结果 `eval/results/phase8_runtime.json` 与 `phase8_faq_audit.json` 保留作为证据；Phase 9 结果使用 `phase9_*` 文件名。

## 交付与安全

- 交付包不得包含 `config.ini`、`config.ini*` 备份、`*.phase6bak`、`logs/`、`_backup_*/`；这些路径已加入 `.gitignore`。
- `base/config.py` 不保存真实密码兜底值；密码和 API Key 仅从 `config.ini` 或环境变量读取。
- 共享或发布前应轮换 `config.ini` 中的 MySQL 密码与 DashScope Key。

## 已知限制

- 无检索结果时，模型可能给出有限的通用建议，并在回答中说明“依据不足”；此时 `citations` 为 0。运行期脚本接受“拒答/说明无依据”语义，不依赖固定句式；若验收要求严格拒答（不允许任何通用建议），需收紧提示词或无答案门控后再验收。
- 旧教育 FAQ/BM25 快速通道已停用并清理；所有非问候问题统一走 RAG。
- 本项目按技术文档领域配置，`valid_sources` 仅包含 `langchain`、`milvus`、`bge`、`transformers`。
