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

评测数据：`devdocs_eval_set_v1.jsonl` 共 **69 题**（人工标注；题型分布：api_usage 14、concept 10、general 10、error_diag 7、config 7、concept_compare 6、version_diff 5、boundary 5、no_answer 5）。其中 `needs_review=false` 且 `expected_docs` 非空的**检索参评样本当前为 21 题**；README 引用的 Recall@5/MRR@10 为历史评测时点 19 题子集的结果。标注规范见 `ANNOTATION_GUIDE.md`；`faq_routing_cases.jsonl` 为历史 FAQ 路由用例（保留为证据）。

查询分类模型数据：`rag_qa/classify_data/devdocs_classify/` 下 train 272 / val 58 / test 59 条（二分类：通用知识 / 专业咨询）。仓库内未保存历史训练准确率数值记录，复现训练时以脚本输出的验证/测试指标为准，不得引用未经记录的数字。

## 2. 运行方式

先启动服务（默认 8081）。

运行期脚本（需要服务）：

```bash
python eval/verify_runtime_phase8.py
python eval/verify_runtime_phase7.py
```

离线/只读脚本：

```bash
python eval/test_phase9_cleanup.py
python eval/test_domain_consistency.py
python eval/test_phase6.py
python eval/test_phase51_regression.py
python eval/test_fixes.py
```

## 3. 隔离输出（不覆盖历史基线）

支持 `RAG_RESULT_OUT` 的脚本会写到指定路径：

```bash
RAG_RESULT_OUT=eval/results/phase10_cleanup.json python eval/test_phase9_cleanup.py
# Windows PowerShell: $env:RAG_RESULT_OUT="eval/results/phase10_cleanup.json"; python eval/test_phase9_cleanup.py
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

```bash
python -c "import hashlib; print(hashlib.sha256(open('eval/results/phase6_tests.json','rb').read()).hexdigest())"
```

> 注：`eval/results/` 的原始 JSON 输出按 `.gitignore` 约定不入库，基线文件保存在作者本地环境；仓库内保留的验收证据为 `eval/results/phase10_final_acceptance.md`（该文件被显式加入版本控制）。

## 5. 评测指标计算口径

三类测试的边界：**离线测试**（桩或只读，不调用生成 API：`test_phase6/test_fixes/test_domain_consistency/test_phase9_cleanup/test_phase51_regression`）；**运行期测试**（对已启动服务发 HTTP/WS 请求：`verify_runtime_phase7/8`）；**真实 API 评测**（直连 Milvus + DashScope：`eval_retrieval/eval_answer/eval_supportability/eval_phase6_e2e`）。

- **Recall@5**（`eval_retrieval.py`，离线只读检索，不调用 LLM）：对每个参评题取 `retrieve_top_k(k=10)` 的混合检索融合排名（dense IP + sparse IP，WeightedRanker 权重 1.0/0.7，distance 降序）；top-5 内命中任一 `expected_docs`（检索单元 `source:file_path`）记 1，否则 0，按题平均。历史结果 84.2% = 16/19。
- **MRR@10**（同上）：top-10 内首个命中文档排名 r 的倒数 1/r（未命中记 0），按题平均。历史结果 63.2%。
- **负例误答率**（`eval_supportability.py`，真实 API）：负例共 **11 个** = 评测集 `no_answer` 5 题（q055–q058 为「主题相关但库缺失」，q059 为域外）+ 脚本内置 6 条补充域外问句（`NEG_EXTRA`）。每题走完整 RAG 链路后由判定器给出 `sufficient/partial/insufficient/gate_rejected`；**误答 = 判定为 `sufficient`**（即未被门控拒绝且被认为依据充分）。历史结果 36.4%（4/11）→ 优化后 9.1%（1/11）。
- **域外拒答率**（同上）：负例中的域外子集（q059 + 6 条补充问句）被拒绝（`insufficient` 或 `gate_rejected`）的比例，历史结果 100%。
- **p50 延迟**（`eval_phase6_e2e.py`，真实 API）：固定 7 题子集（q022/q035/q039/q006/q008/q011/q057）× `CANDIDATE_M ∈ {2,3}` 共 **14 次正式计时调用**（另有冷启动预热调用不计入统计）；计时区间为发起 `generate_answer` 到流式输出结束的端到端墙钟时间；p50 用最近秩分位数（`round(p/100*(n-1))` 取序）计算。历史 p50 约 11s，属小样本估计，受网络与模型随机性影响。
- **可支撑性**：`sufficient/partial/insufficient` 三分类；正例期望非 `insufficient`，负例期望被拒答。判定器为 LLM 判定（`support_judge=true`），存在判定噪声。
- **引用有效性**：回答中 `[来源k]` 必须落在 `citations` 数量范围内，越界引用会被 `sanitize_citations` 移除。

实验环境（历史评测时点）：Windows + conda `edu_rag`（Python 3.10，CUDA 12.8 GPU），Milvus 2.5.x 集合 `devdocs_final`（约 32,854 条），生成模型 DashScope `qwen-plus`。检索参数：`retrieval_k=10`、`candidate_m=2`、`relevance_threshold=0.2`。原始逐题结果 JSON 不入库（见第 4 节注），仓库内可核对的汇总证据为 `phase10_final_acceptance.md`。

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
