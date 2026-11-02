# 部署与故障排查

## 1. 环境要求

- Windows / Linux，Python 3.10（验证环境为 conda 环境 `edu_rag`）
- MySQL 8.x，库默认 `subjects_kg`
- Milvus 2.5.x，默认 `localhost:19530`，数据库 `itcast`，集合 `devdocs_final`（约 32,854 条）
- DashScope API Key（Qwen）
- 本地模型：`rag_qa/models/bge-m3`、`rag_qa/models/bge-reranker-large`、`rag_qa/models/bert-base-chinese`、`rag_qa/models/bert_query_classifier_devdocs`

> Redis 在 Phase 9 后已不参与运行（旧 FAQ 缓存移除），无需安装。

## 2. 安装依赖

```powershell
conda activate edu_rag
pip install -r requirements.txt
```

`torch==2.10.0` 需按平台选择 wheel；GPU 环境建议从 PyTorch 官方源安装对应 CUDA 版本（验证环境 CUDA 12.8）。

## 3. 配置

```powershell
Copy-Item config.example.ini config.ini
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

```powershell
# 验证环境
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" app.py
```

默认监听 `0.0.0.0:8081`（`HOST`/`PORT` 可覆盖）。访问：

- 首页：`http://127.0.0.1:8081/`
- 健康检查：`http://127.0.0.1:8081/health`
- Swagger：`http://127.0.0.1:8081/docs`

### Windows 后台启动注意事项

- 需设置 `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`，并用「分离进程」方式启动，避免控制台关闭导致子进程被终止。
- 推荐用一个批处理包装环境变量与重定向，再通过 `Win32_Process Create` 启动：

```powershell
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = '"C:\path\run.bat"'; CurrentDirectory = 'E:\LLM_ubuntu\rag_qa_system'
}
```

`run.bat` 内容示例：

```bat
@echo off
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set PYTHONIOENCODING=utf-8
cd /d E:\LLM_ubuntu\rag_qa_system
"C:\ProgramData\anaconda3\envs\edu_rag\python.exe" app.py > logs\run.log 2>&1
```

- 不要用 `Start-Process -RedirectStandardOutput` 直接托管长驻服务，可能出现句柄占用导致进程被拖住/终止。

## 5. 接口速查

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

## 6. 常见问题

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

## 7. 停止服务

```powershell
$pids = (Get-NetTCPConnection -LocalPort 8081 -State Listen).OwningProcess
Stop-Process -Id $pids -Force
```

## 8. 交付清单排除项

- `config.ini`、`config.ini*` 备份（含 `config.ini.phase6bak`）
- `*.phase6bak`、`*.bak`、`_backup_*/`、`logs/`、`__pycache__/`、`.idea/`
- 含真实密码的历史备份文件（如 `base/config.py.phase6bak`、`_backup_domain_migration/base__config.py`）
