# 测试与评测

本项目包含**离线回归测试**、**运行期端到端验证**与**真实 API 评测**三类脚本，全部放在 `eval/`。

## 1. 脚本清单

| 脚本 | 类型 | 前置 | 说明 | 默认输出 |
| --- | --- | --- | --- | --- |
| `test_phase9_cleanup.py` | 离线 | MySQL/Milvus 可连 | 校验旧 FAQ 数据/代码/配置已清理，公共能力与向量库保留 | `phase9_cleanup.json` |
| `test_phase6.py` | 离线 | 无（桩） | 可支撑性判定解析、引用清理、端到端集成（桩 VectorStore/LLM） | `phase6_tests.json` |
| `test_fixes.py` | 离线 | 无（桩） | 验证最小修复：生成器/字符串兼容、三策略、空上下文等 | 仅打印（重定向为日志） |
| `test_phase51_regression.py` | 离线 | Milvus 可连 | 真实 VectorStore（只读）+ 桩 LLM 的策略与门控回归 | `phase51_regression.json` |
| `test_domain_consistency.py` | 离线 | 无 | 前端文案、来源配置、接口协议与 DevDocsRAG 定位一致 | `phase7_domain_consistency.json` |
| `verify_runtime_phase7.py` | 运行期 | 服务已启动 | HTTP + WebSocket 端到端（首页、接口、WS 流、拒答、错误分支） | `phase7_runtime.json` |
| `verify_runtime_phase8.py` | 运行期 | 服务已启动 | 旧 FAQ 关闭后 HTTP/WS 均进入 RAG，含陷阱问题与引用校验 | `phase8_runtime.json` |
| `eval_retrieval.py` | 真实只读 | Milvus 可连 | Recall@5 / MRR@10 | `retrieval_*.json` |
| `eval_answer.py` | 真实 API | Milvus + DashScope | 回答/引用/性能记录 | `answer_*.json` |
| `eval_supportability.py` | 真实 API | Milvus + DashScope | 三分类可支撑性评测 | `phase6_supportability.json` |
| `eval_phase6_e2e.py` | 真实 API | Milvus + DashScope | CANDIDATE_M(2 vs 3) 对比与端到端性能 | `phase6_e2e.json` |
| `analyze_phase6.py` | 离线 | 已有结果 | 汇总分析 Phase6 结果 | `phase6_analysis.json` |
| `generate_eval_set.py` | 离线 | 无 | 生成评测集 | `devdocs_eval_set_v1.jsonl` |

评测数据：`devdocs_eval_set_v1.jsonl`（人工标注，`needs_review=false` 且 `expected_docs/answer_points` 非空的样本参与指标计算），说明见 `ANNOTATION_GUIDE.md`；`faq_routing_cases.jsonl` 为历史 FAQ 路由用例（保留为证据）。

## 2. 运行方式

先启动服务（默认 8081）。

运行期脚本（需要服务）：

```powershell
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/verify_runtime_phase8.py
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/verify_runtime_phase7.py
```

离线/只读脚本：

```powershell
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/test_phase9_cleanup.py
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/test_domain_consistency.py
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/test_phase6.py
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/test_phase51_regression.py
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/test_fixes.py
```

## 3. 隔离输出（不覆盖历史基线）

支持 `RAG_RESULT_OUT` 的脚本会写到指定路径：

```powershell
$env:RAG_RESULT_OUT="eval/results/phase10_cleanup.json"
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" eval/test_phase9_cleanup.py
```

运行期脚本还支持 `RAG_HOST`、`RAG_PORT` 指定服务地址。

## 4. 历史基线（不可修改）

以下文件是历史验收基线，SHA256 固定，新阶段一律写独立文件名：

| 文件 | SHA256 |
| --- | --- |
| `phase6_tests.json` | `bf368b0426761243e52e13a7d385efcfec09167bb38c4d38eea956575c61a3a2` |
| `phase51_regression.json` | `f286c0564ac0556bbd20af2b27d9376e2b67c4808f027865affa6477e59a7dfe` |
| `phase7_domain_consistency.json` | `a12d8f5024a3dcc5b29a5ce48c1343e54ce2e74069842e28b32e62a0f954f2ff` |
| `phase7_runtime.json` | `cadeee332eda54d5eb3b55ac70055e590f7c98f5e51fffb19bee456a44885854` |
| `phase8_runtime.json` | `588d1d0a1107792d5a6e576757d197d28b0b6f29c0b609693c4617ba41a6626c` |

校验：

```powershell
& "C:\ProgramData\anaconda3\envs\edu_rag\python.exe" -c "import hashlib; print(hashlib.sha256(open('eval/results/phase6_tests.json','rb').read()).hexdigest())"
```

## 5. 评测指标

- **Recall@5**：top-5 检索结果是否命中任一 `expected_docs`（检索单元为 `source:file_path`）。
- **MRR@10**：top-10 中首个命中文档排名的倒数。
- **可支撑性**：`sufficient/partial/insufficient` 三分类；正例期望非 `insufficient`，负例期望被拒答。
- **引用有效性**：回答中 `[来源k]` 必须落在 `citations` 数量范围内，越界引用会被 `sanitize_citations` 移除。

## 6. Phase 10 验收结果

命名整理后重跑全部脚本，结果写入 `eval/results/phase10_*`：

| 脚本 | 结果 |
| --- | --- |
| `test_phase9_cleanup.py` | 49/49 |
| `verify_runtime_phase8.py` | 59/59 |
| `verify_runtime_phase7.py` | 19/19 |
| `test_domain_consistency.py` | 25/25 |
| `test_phase6.py` | 21/21 |
| `test_phase51_regression.py` | 15/15 |
| `test_fixes.py` | 14/14 |

历史 5 个基线文件 SHA256 未变化。汇总报告见 `eval/results/phase10_final_acceptance.md`。

## 7. 已知限制

- 无检索结果时，模型可能给出有限通用建议并说明「依据不足」，此时 `citations` 为 0；运行期脚本接受「拒答/说明无依据」语义，不依赖固定句式。
- 旧课程类陷阱问题存在间歇性离域通用建议的残余风险（非硬拒答），已在验收报告中如实记录。
- 评测中真实 API 的指标受模型随机性与网络影响，建议多次运行取稳定区间。
