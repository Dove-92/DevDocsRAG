# 第三方内容与数据来源声明（NOTICE）

本仓库根目录的 [MIT License](LICENSE) **仅适用于本项目自研代码**（`app.py`、`qa_system.py`、`base/`、`mysql_qa/`、`rag_qa/core/`、`eval/`、`static/` 及文档中由本项目撰写的部分）。仓库内包含的第三方文档语料、以及运行时下载的第三方模型，**各自遵循其原始许可证，不因本仓库的 MIT 声明而改变授权**。

## 1. 仓库内第三方文档语料（rag_qa/data/）

| 目录 | 内容 | 文件数 | 上游来源 | 上游许可证 |
| --- | --- | --- | --- | --- |
| `rag_qa/data/langchain_data/` | LangChain 官方文档（Markdown/Notebook 抓取副本） | 113 | https://github.com/langchain-ai/docs | MIT（Copyright (c) 2025 LangChain） |
| `rag_qa/data/transformers_data/` | Hugging Face Transformers 官方文档抓取副本 | 198 | https://github.com/huggingface/transformers（docs 目录） | Apache License 2.0（Copyright 2018- The Hugging Face team） |
| `rag_qa/data/knowledge.txt` | 课程示例文本（少量） | 1 | 教学示例 | 按教学用途使用 |

说明：

- 上游许可证核验方式：直接读取上游仓库 LICENSE 文件（langchain-ai/docs 为 MIT；huggingface/transformers 为 Apache-2.0）。两份上游 LICENSE 全文已随附于本仓库 `licenses/` 目录：[licenses/langchain-docs-MIT.txt](licenses/langchain-docs-MIT.txt)、[licenses/transformers-Apache-2.0.txt](licenses/transformers-Apache-2.0.txt)。
- 合规依据与现状：MIT 要求在所有副本或实质性部分中保留版权声明与许可文本，随附 LICENSE 全文即满足该要求；Apache-2.0 第 4 条要求再分发时附带许可证副本、保留源形式中的版权/专利/商标/归属声明，修改过的文件需显著标注变更，上游若含 NOTICE 文件需随附其归属声明（经核验 huggingface/transformers 仓库根目录无 NOTICE 文件）。本仓库语料为未修改的抓取副本，未逐文件保留上游版权头；随附 LICENSE 全文后满足许可文本保留要求。若需更严格的逐文件合规，可改为运行时自行从上游抓取语料（仓库内副本可删除），由使用者直接遵循上游许可。
- 授权范围澄清：本仓库根目录 MIT License 仅覆盖本项目自研代码；`rag_qa/data/` 下第三方语料仍分别遵循 MIT（LangChain）与 Apache-2.0（Transformers），`licenses/` 内文本为上游许可证的忠实副本，不构成对本仓库 MIT 授权范围的任何扩大或变更。
- Milvus 与 BGE/FlagEmbedding 文档语料**不在本仓库中**（`rag_qa/data/bge_data/` 已被 `.gitignore` 排除，Milvus 语料未入库）。知识库中这两类来源的数据需使用者自行从上游获取：Milvus 文档（https://github.com/milvus-io/milvus-docs ，Apache-2.0）、FlagEmbedding 文档（https://github.com/FlagOpen/FlagEmbedding ，MIT）。
- 文档语料仅用于检索问答演示与研究学习，不代表本项目对上游内容主张任何权利。

## 2. 运行时下载的第三方模型（不随仓库分发）

模型权重已被 `.gitignore` 排除（`rag_qa/models/`），需按下表自行下载：

| 模型 | 用途 | 下载来源 | 许可证 |
| --- | --- | --- | --- |
| BGE-M3 | 稠密+稀疏混合检索向量 | https://huggingface.co/BAAI/bge-m3 | MIT |
| BGE-Reranker-Large | 检索结果精排 | https://huggingface.co/BAAI/bge-reranker-large | MIT |
| bert-base-chinese | 查询分类基座模型 | https://huggingface.co/google-bert/bert-base-chinese | Apache-2.0 |
| bert_query_classifier_devdocs | 本项目微调的查询分类模型（基座为 bert-base-chinese） | 由 `rag_qa/classify_data/devdocs_classify/train_classifier.py` 训练产出 | 基座 Apache-2.0；微调权重为本项目产出 |

## 3. 其他依赖

Python 依赖见 `requirements.txt`，各自遵循其开源许可证（FastAPI: MIT；PyMilvus/milvus-model: Apache-2.0；FlagEmbedding: MIT；Transformers: Apache-2.0；PyTorch: BSD 风格）。生成模型通过 DashScope API 调用（Qwen），遵循阿里云服务条款，不在本仓库分发。
