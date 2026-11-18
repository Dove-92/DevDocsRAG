# 部署与故障排查

## 1. 环境要求

- Windows / Linux，Python 3.10（验证环境为 conda 环境 `edu_rag`）
- MySQL 8.x，库默认 `subjects_kg`
- Milvus 2.5.x，默认 `localhost:19530`，数据库 `itcast`，集合 `devdocs_final`（约 32,854 条）
- DashScope API Key（Qwen）
- 本地模型：`rag_qa/models/bge-m3`、`rag_qa/models/bge-reranker-large`、`rag_qa/models/bert-base-chinese`、`rag_qa/models/bert_query_classifier_devdocs`

> Redis 在 Phase 9 后已不参与运行（旧 FAQ 缓存移除），无需安装。

## 2. 安装依赖

```bash
# 建议独立虚拟环境（conda 或 venv 均可）
pip install -r requirements.txt
```

`torch==2.10.0` 需按平台选择 wheel；GPU 环境建议从 PyTorch 官方源安装对应 CUDA 版本（验证环境 CUDA 12.8）。

## 3. 配置

```bash
cp config.example.ini config.ini        # Windows PowerShell: Copy-Item config.example.ini config.ini
```

编辑 `config.ini`，至少修改：

- `[mysql] password`
- `[llm] dashscope_api_key`

可用环境变量覆盖（优先级高于文件）：

```text
MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE
MILVUS_HOST / MILVUS_PORT / MILVUS_DATABASE_NAME / MILVUS_COLLECTION_NAME
DASHSCOPE_API_KEY
REDIS_HOST / REDIS_PORT / REDIS_PASSWORD / REDIS_DB
```

`config.ini` 含真实密钥，**不得提交或打包交付**；`.gitignore` 已排除 `config.ini*`。

## 4. 启动

```bash
python app.py
```

默认监听 `0.0.0.0:8081`（`HOST`/`PORT` 可覆盖）。访问：

- 首页：`http://127.0.0.1:8081/`
- 健康检查：`http://127.0.0.1:8081/health`
- Swagger：`http://127.0.0.1:8081/docs`

### Windows 后台启动注意事项

- 需设置 `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`，并用「分离进程」方式启动，避免控制台关闭导致子进程被终止。
- 推荐用一个批处理包装环境变量与重定向，再通过 `Win32_Process Create` 启动：

将下面内容保存为仓库根目录下的 `start_service.ps1`，在 PowerShell 中执行 `.\start_service.ps1` 即可（脚本用 `$PSScriptRoot` 自动定位仓库目录，无需手工填写绝对路径）：

```powershell
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$PSScriptRoot\run.bat`""; CurrentDirectory = $PSScriptRoot
}
```

将下面内容保存为仓库根目录下的 `run.bat`（`%~dp0` 自动展开为脚本所在目录，放在仓库根目录即无需修改任何路径；logs 目录不存在时自动创建）：

```bat
@echo off
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist logs mkdir logs
python app.py > logs\run.log 2>&1
```

- 不要用 `Start-Process -RedirectStandardOutput` 直接托管长驻服务，可能出现句柄占用导致进程被拖住/终止。

## 5. 模型与知识库复现（从零恢复运行环境）

模型权重与 Milvus 集合数据不随仓库分发（`.gitignore` 排除 `rag_qa/models/`），按以下步骤恢复。

### 5.1 模型下载与目录结构

| 本地目录（相对仓库根） | 模型 | 用途 | 下载来源 |
| --- | --- | --- | --- |
| `rag_qa/models/bge-m3` | BAAI/bge-m3 | 稠密+稀疏混合检索向量（`milvus_model.hybrid.BGEM3EmbeddingFunction` 加载） | https://huggingface.co/BAAI/bge-m3 |
| `rag_qa/models/bge-reranker-large` | BAAI/bge-reranker-large | 检索结果精排（`sentence_transformers.CrossEncoder` 加载） | https://huggingface.co/BAAI/bge-reranker-large |
| `rag_qa/models/bert-base-chinese` | google-bert/bert-base-chinese | 查询分类基座模型 | https://huggingface.co/google-bert/bert-base-chinese |
| `rag_qa/models/bert_query_classifier_devdocs` | 本项目微调分类模型 | 区分「通用知识 / 专业咨询」查询（`QueryClassifier` 加载） | 无公开下载；用下述训练脚本产出 |

下载命令（在仓库根目录执行，可直接复制；需先 `pip install -U "huggingface_hub[cli]"`）：

```bash
huggingface-cli download BAAI/bge-m3 --local-dir rag_qa/models/bge-m3
huggingface-cli download BAAI/bge-reranker-large --local-dir rag_qa/models/bge-reranker-large
huggingface-cli download google-bert/bert-base-chinese --local-dir rag_qa/models/bert-base-chinese
```

第四个目录 `rag_qa/models/bert_query_classifier_devdocs` 为本项目微调产物，无公开下载，由下方训练脚本产出。国内网络无法访问 Hugging Face 时，可设置镜像环境变量 `HF_ENDPOINT=https://hf-mirror.com` 后重试，或从 ModelScope 下载同名模型放入对应目录。四个目录需齐全，缺任一项启动即报错。

微调分类模型复现：

```bash
python rag_qa/classify_data/devdocs_classify/train_classifier.py
```

- 训练数据已入库：`train.jsonl` 272 条、`val.jsonl` 58 条、`test.jsonl` 59 条（另有 `train_aug*.jsonl` 增广数据与 `generate_data.py` 生成脚本）。
- 脚本以仓库根为基准定位数据与输出目录（可移植路径），基座为 `bert-base-chinese`，产出写入 `rag_qa/models/bert_query_classifier_devdocs`。
- 训练需 GPU 环境（CPU 可运行但很慢）；仓库内未保存历史训练准确率记录，复现时以脚本运行输出的验证集指标为准。

### 5.2 服务配置与初始化

- **MySQL**：创建数据库 `subjects_kg`（或修改 `[mysql] database`）。`conversations` 会话历史表由 `mysql_qa/db/mysql_client.py` 在首次连接时自动创建，无需手工建表。
- **Milvus**：启动 Milvus 2.5.x，创建数据库 `itcast`（或修改 `[milvus] database_name`）。集合 `devdocs_final` 在知识库构建时由 `VectorStore` 自动创建（含 dense/sparse 双向量字段与 IP 度量）。
- **Redis**：Phase 9 后已无运行时使用，`[redis]` 配置段仅为历史兼容保留，无需安装 Redis。
- **DashScope**：在 `[llm]` 填入 API Key（默认模型 `qwen-plus`）。

### 5.3 知识库构建

在线服务使用的生产集合为 `devdocs_final`（约 32,854 条子块记录），入库链路为：

1. `rag_qa/edu_document_loaders/`：解析 Markdown/Notebook 文档（历史命名模块，保留以兼容导入）。
2. `rag_qa/edu_text_spliter/` + `rag_qa/core/document_processor.py`：父子块切分（`parent_chunk_size=1200`、`child_chunk_size=300`、`chunk_overlap=50`，见 `config.example.ini`）。
3. `VectorStore.add_documents(documents)`（`rag_qa/core/vector_store.py`）：BGE-M3 生成 dense+sparse 向量并写入 Milvus。

语料来源与授权见 [NOTICE.md](../NOTICE.md)：仓库内含 LangChain（113 个文件）与 Transformers（198 个文件）文档抓取副本；Milvus 与 BGE/FlagEmbedding 语料未入库，需自行从上游仓库获取后按同一链路入库。

> 注意：`rag_qa/core/build_knowledge_base.py` 是独立的教学示例脚本，写入的是 `edu_rag_exam_hybrid` 集合且含 `drop_collection` 逻辑，**不是生产集合 `devdocs_final` 的构建器**，请勿对生产库运行。

**生产集合重建的评估结论与限制**（本轮审查结论：暂不提供一键重建脚本，原因如下）：

- 技术路径本身可行且对在线服务无破坏：`VectorStore.add_documents` 以子块文本的 MD5 为主键做 upsert（幂等，相同文本覆盖更新），不删除集合、不清空数据；可用独立集合名（`VectorStore(collection_name=...)` 或环境变量 `MILVUS_COLLECTION_NAME`）先构建、验证后再切换，全程不影响现有 `devdocs_final`。
- 但仓库未留存生产集合的原始入库脚本与语料清单/快照版本，约 32,854 条记录无法确认能被现有语料精确复现：`parent_id` 按 `doc_{加载序号}_parent_{父块序号}` 生成，依赖目录遍历与加载顺序；语料文件增删或顺序变化都会改变编号；Milvus/BGE 语料未入库，需自行获取且版本未知。
- 新增独立构建脚本属于较大改动，且当前环境（无 Milvus/GPU/模型权重）无法端到端验证其与生产集合的等价性，故本轮不实施，仅在此说明限制。
- 如需自行重建：按上述链路对每个 `rag_qa/data/<来源>_data/` 目录调用 `process_documents(directory_path)`（目录名去掉 `_data` 后缀即 `source` 字段），再调用 `VectorStore.add_documents(documents)`；建议先写入独立集合验证检索效果，再决定是否切换。此流程未在本环境实际执行验证。

### 5.4 启动验证

完成上述步骤后 `python app.py`，访问 `/health` 应返回正常；再用 `eval/` 下运行期脚本（见 [测试与评测](testing_and_evaluation.md)）做回归验证。

## 6. 接口速查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 前端首页 |
| GET | `/health` | 健康检查 |
| GET | `/api/sources` | 支持的技术来源 |
| POST | `/api/create_session` | 创建会话 ID |
| GET | `/api/history/{session_id}` | 最近 5 轮历史 |
| DELETE | `/api/history/{session_id}` | 清除会话历史 |
| POST | `/api/query` | 非流式入口；非问候返回转流式提示 |
| WS | `/api/stream` | 流式问答 |

## 7. 常见问题

| 症状 | 排查 |
| --- | --- |
| 启动即报模型/路径错误 | 确认 `rag_qa/models/` 下四个模型目录齐全 |
| Milvus 连接失败 | 检查 `MILVUS_HOST/PORT/DATABASE_NAME` 与 Milvus 服务状态、集合是否存在 |
| MySQL 连接失败 | 检查 `config.ini` 的 `[mysql]` 与 MySQL 服务；`conversations` 表会自动创建 |
| 回答总说「依据不足」 | 确认检索命中；必要时下调 `relevance_threshold` 或关闭 `support_judge` 复测 |
| DuckDB/稀疏向量报错 | 确认 `milvus_model`、`FlagEmbedding` 依赖已安装 |
| 中文/emoji 控制台乱码 | 以 UTF-8 JSON/日志为准；脚本已对 `sys.stdout` 做 `errors="replace"` |
| 端口被占用 | 使用 `PORT` 环境变量换端口；`Get-NetTCPConnection -LocalPort 8081` 查占用进程 |
| 集成知识库脚本报错 | `build_knowledge_base.py` 为独立示例，会写入 `edu_rag_exam_hybrid`，与在线服务无关 |

## 8. 停止服务

```powershell
$pids = (Get-NetTCPConnection -LocalPort 8081 -State Listen).OwningProcess
Stop-Process -Id $pids -Force
```

## 9. 交付清单排除项

- `config.ini`、`config.ini*` 备份（含 `config.ini.phase6bak`）
- `*.phase6bak`、`*.bak`、`_backup_*/`、`logs/`、`__pycache__/`、`.idea/`
- 含真实密码的历史备份文件（如 `base/config.py.phase6bak`、`_backup_domain_migration/base__config.py`）
