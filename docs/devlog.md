# 开发过程记录（Dev Log）

> 本文件保存开发过程中的清理与命名整理记录，供追溯参考。原位于 README，为保持 README 聚焦项目本身而迁移至此。

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
