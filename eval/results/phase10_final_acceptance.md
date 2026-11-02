# Phase 10 最终验收报告（DevDocsRAG）

- 范围：整体代码/目录整理、命名优化、安全与数据保护、技术文档编写、最终验收。
- 原则：保持功能、API、WebSocket 协议与核心架构稳定；不做大规模重构；新结果写独立 `phase10_*` 文件；不覆盖历史基线；不删除/重建 Milvus 集合，不清空 DB/Redis 与历史数据。
- 验收结论：**通过**。全部回归与运行期验证通过，历史基线未变，命名整理后服务可正常启动并对外提供 HTTP/WebSocket。

## 1. 命名与结构整理

| 原名称 | 新名称 | 说明 |
| --- | --- | --- |
| `new_main.py` | `qa_system.py` | 集成问答核心，去除 `new_` 前缀 |
| `rag_qa/core/new_rag_system.py` | `rag_qa/core/rag_pipeline.py` | RAG 主链路；避免与 Phase 9 已删除的旧 `rag_system.py` 重名 |
| `mysql_qa/db/mysql_client.py` 类 `mysql_client` | `MySQLClient` | 类名改为大驼峰 |
| `rag_qa/rag_assessment/rag_as.py` | `ragas_eval.py` | 见名知义 |
| `rag_qa/classify_data/提示词模块.txt` | `prompt_template.txt` | 去除非 ASCII 文件名 |

同步更新了所有引用：`app.py`、`rag_qa/__init__.py`、`mysql_qa/__init__.py`、`eval/*.py`（评测/回归/验证脚本）、`README.md`。

保留（影响面大，未改动）：`base/`、`rag_qa/edu_document_loaders/`、`rag_qa/edu_text_spliter/`、评测脚本 `phaseN_*` 历史命名。

## 2. 顶层目录

```text
rag_qa_system/
├── app.py                     FastAPI 入口（HTTP + WebSocket）
├── qa_system.py               集成问答核心 IntegratedQASystem
├── base/                      配置与日志
├── mysql_qa/                  MySQL 连接封装（MySQLClient）
├── rag_qa/
│   ├── core/                  RAG 主链路与提示词
│   │   ├── rag_pipeline.py     RAGSystem
│   │   ├── vector_store.py     VectorStore
│   │   ├── query_classifier.py QueryClassifier
│   │   ├── strategy_selector.py StrategySelector
│   │   ├── prompts.py          RAGPrompts
│   │   ├── document_processor.py
│   │   └── build_knowledge_base.py
│   ├── models/                本地模型
│   ├── data/                  文档语料
│   └── classify_data/         分类器数据与脚本
├── static/index.html          前端单页
├── eval/                      评测/回归脚本与 results/
├── docs/                      项目文档（新增）
├── logs/                      运行日志
├── config.example.ini         配置模板
├── config.ini                 本地真实配置（禁止交付）
├── README.md / requirements.txt / .gitignore
└── _backup_*/                 历史备份（交付排除）
```

## 3. 新增文档

| 文件 | 内容 |
| --- | --- |
| `docs/project_overview.md` | 项目定位、能力、技术栈、高层架构 |
| `docs/codebase_guide.md` | 目录、命名约定、导入约定、常见改动位置 |
| `docs/architecture_and_workflow.md` | 分层架构、HTTP/WS 流程、RAG 主链路、降级策略 |
| `docs/core_modules.md` | 各核心模块类与方法说明 |
| `docs/setup_and_troubleshooting.md` | 环境、配置、启动（含 Windows 后台）、故障排查 |
| `docs/testing_and_evaluation.md` | 脚本清单、运行方式、指标、基线、Phase10 结果 |

README 已增加 `docs/` 文档入口与「Phase 10 命名整理」说明。

## 4. 测试结果（均写入 `eval/results/phase10_*`）

| 脚本 | 结果 | 输出文件 |
| --- | --- | --- |
| `eval/test_phase9_cleanup.py` | 49/49 | `phase10_cleanup.json` |
| `eval/verify_runtime_phase8.py` | 59/59 | `phase10_runtime_phase8.json` |
| `eval/verify_runtime_phase7.py` | 19/19 | `phase10_runtime_phase7.json` |
| `eval/test_domain_consistency.py` | 25/25 | `phase10_domain_consistency.json` |
| `eval/test_phase6.py` | 21/21 | `phase10_phase6.json` |
| `eval/test_phase51_regression.py` | 15/15 | `phase10_phase51_regression.json` |
| `eval/test_fixes.py` | 14/14 | `phase10_fixes.log` |

运行期验证覆盖：`/health`、`/api/sources`、`/api/create_session`、`/api/query`、WebSocket `start → token → end`（含 `is_complete`、`processing_time`、`citations`）、引用编号范围、不可回答问题拒答、非法输入 error 分支。

## 5. 历史基线校验（SHA256 未变化）

| 文件 | SHA256 |
| --- | --- |
| `phase6_tests.json` | `bf368b0426761243e52e13a7d385efcfec09167bb38c4d38eea956575c61a3a2` |
| `phase51_regression.json` | `f286c0564ac0556bbd20af2b27d9376e2b67c4808f027865affa6477e59a7dfe` |
| `phase7_domain_consistency.json` | `a12d8f5024a3dcc5b29a5ce48c1343e54ce2e74069842e28b32e62a0f954f2ff` |
| `phase7_runtime.json` | `cadeee332eda54d5eb3b55ac70055e590f7c98f5e51fffb19bee456a44885854` |
| `phase8_runtime.json` | `588d1d0a1107792d5a6e576757d197d28b0b6f29c0b609693c4617ba41a6626c` |

## 6. 数据与安全

- 数据保留：Milvus `devdocs_final`（32,854 条）/`edurag_final` 未删除或重建；MySQL `conversations` 保留；Redis 仍仅保留来源不明的 `test_key`。
- 凭据：`base/config.py` 无真实密码兜底；`config.ini` 及含真实密码的历史备份（`config.ini.phase6bak`、`base/config.py.phase6bak`、`_backup_domain_migration/base__config.py`）**不得纳入交付**。
- `.gitignore` 已排除 `config.ini*`、`logs/`、`__pycache__/`、`*.bak`、`*.phase6bak`、`_backup_*/`、`.idea/`、`*.pid`。

## 7. 启动方式

```powershell
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" app.py
```

Windows 后台运行需 `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`，并以分离进程方式启动（详见 `docs/setup_and_troubleshooting.md`）。默认监听 `0.0.0.0:8081`。

## 8. 遗留问题

- 旧课程类陷阱问题存在**间歇性离域通用建议**的残余风险（多数为明确拒答，个别会给出谨慎的通用说明），已如实记录，未使用弱化断言掩盖。
- 三处历史包名 `base/`、`edu_document_loaders/`、`edu_text_spliter/` 及评测 `phaseN_*` 命名保留，未做进一步重命名。
- `source_filter` 目前仅在「直接检索」策略透传，HyDE/子查询/回溯分支未带过滤，已在 `docs/architecture_and_workflow.md` 标注。
