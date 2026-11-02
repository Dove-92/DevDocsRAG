# 代码与目录导读

本文档说明 DevDocsRAG 的目录结构、命名约定与常见改动位置，帮助快速定位代码。

## 1. 顶层目录

```text
rag_qa_system/
├── app.py                    FastAPI 入口（HTTP + WebSocket）
├── qa_system.py              集成问答核心 IntegratedQASystem（原 new_main.py）
├── config.ini                本地真实配置（含密钥，禁止提交/交付）
├── config.example.ini        配置模板（占位值）
├── requirements.txt          Python 依赖
├── README.md                 项目说明与快速上手
├── .gitignore                忽略 config.ini*、logs/、备份等
├── base/                     配置与日志基础设施
├── mysql_qa/                 仅 MySQL 连接封装
├── rag_qa/                   RAG 主链路、模型与文档数据
├── static/                   前端单页
├── eval/                     评测、回归、运行期验证脚本与 results/
├── docs/                     本目录：项目文档
├── logs/                     运行日志（app.log）
├── _backup_domain_migration/ 历史备份（交付排除）
└── _backup_phase3_prompt/    历史备份（交付排除）
```

## 2. 业务包结构

```text
base/
├── __init__.py       将 base/ 与项目根加入 sys.path，并导出 Config、logger
├── config.py         Config：解析 config.ini + 环境变量覆盖
└── logger.py         logger：控制台 + logs/app.log 双输出

mysql_qa/
├── __init__.py       导出 MySQLClient
└── db/mysql_client.py  MySQLClient：连接 / cursor / close

rag_qa/
├── __init__.py       将 rag_qa/ 加入 sys.path，导出 RAGPrompts、VectorStore、RAGSystem
├── core/
│   ├── rag_pipeline.py       RAGSystem：分类 → 策略 → 检索 → 门控 → 判定 → 生成
│   ├── vector_store.py       VectorStore：Milvus 混合检索 + rerank + 打分
│   ├── query_classifier.py   QueryClassifier：BERT 二分类
│   ├── strategy_selector.py  StrategySelector：LLM 选择检索策略
│   ├── prompts.py            RAGPrompts：全部 Prompt 模板
│   ├── document_processor.py 离线：文档加载与父子分块
│   └── build_knowledge_base.py 独立知识库构建示例（离线）
├── edu_document_loaders/     历史加载器包（OCR PDF/Word/PPT/图片）
├── edu_text_spliter/         历史中文分割器包
├── models/                   本地模型权重（bge-m3、bge-reranker-large、bert-*）
├── data/                     文档数据与 _removed 归档
└── classify_data/            分类器训练数据、Prompt 资料、训练脚本
```

## 3. 命名约定

- 入口与核心类语义化命名，避免 `new_`、`main` 等含糊前缀：
  - `new_main.py` → `qa_system.py`
  - `rag_qa/core/new_rag_system.py` → `rag_qa/core/rag_pipeline.py`
- 类名使用大驼峰，模块名使用小写加下划线：
  - `mysql_qa/db/mysql_client.py` 的类 `mysql_client` → `MySQLClient`
- 其他整理：
  - `rag_qa/rag_assessment/rag_as.py` → `ragas_eval.py`
  - `rag_qa/classify_data/提示词模块.txt` → `prompt_template.txt`
- **保留的历史名称**（改动影响面大，暂不调整）：`base/`、`rag_qa/edu_document_loaders/`、`rag_qa/edu_text_spliter/`、评测脚本中的 `phaseN_*` 文件命名（作为历史证据标识）。

## 4. 导入约定

- 包内模块使用显式路径导入，如 `from rag_qa.core.vector_store import VectorStore`、`from base.config import Config`。
- `base/__init__.py` 会把 `base/` 与项目根目录加入 `sys.path`，因此 `base` 包内可扁平导入 `from config import Config`、`from logger import logger`；其他目录请使用 `base.config` / `base.logger` 完整路径。
- `rag_qa/__init__.py` 会把 `rag_qa/` 加入 `sys.path`，使 `rag_qa.core.*` 内部可用 `from core.xxx import ...`。
- 由于上述 `sys.path` 注入，**从项目根目录运行**是最稳妥的方式（`python app.py`、`python eval/xxx.py`）。

## 5. 常见改动位置

| 需求 | 修改位置 |
| --- | --- |
| 调整回答风格 / 引用规则 | `rag_qa/core/prompts.py` 的 `RAGPrompts.rag_prompt()` |
| 新增/调整检索策略 | `rag_qa/core/strategy_selector.py` + `rag_pipeline.py` 的 `retrieve_and_merge` |
| 调整混合检索权重 / TopK | `rag_qa/core/vector_store.py` 的 `WeightedRanker(1.0, 0.7)` 与 `RETRIEVAL_K` |
| 调整相关性门控阈值 | `config.ini` 的 `[retrieval] relevance_threshold` |
| 开关证据可支撑性判定 | `config.ini` 的 `[retrieval] support_judge` |
| 新增受支持的技术来源 | `config.ini` 的 `[app] valid_sources`，并同步前端文案 |
| 问候语 / 身份文案 | `app.py` 的 `GREETING_PATTERNS` |
| 会话历史轮数 | `qa_system.py` 的 `_fetch_recent_history`（`LIMIT 5`）与 `rag_pipeline.py` 历史的 `history[-5:]` |
| 新增接口 | `app.py` 的 FastAPI 路由 |
| 数据入库 | `rag_qa/core/document_processor.py` + `vector_store.add_documents` |

## 6. 依赖与数据

- `rag_qa/models/` 存放运行必需模型：`bge-m3`、`bge-reranker-large`、`bert-base-chinese`、`bert_query_classifier_devdocs`。请勿删除。
- `rag_qa/data/` 为文档语料；`_removed` 为历史归档。
- `eval/results/` 保存所有评测 JSON/日志，历史文件不应被覆盖。
