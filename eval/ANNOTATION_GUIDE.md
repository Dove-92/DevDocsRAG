# DevDocsRAG 评估集标注说明

评估集文件：`eval/devdocs_eval_set_v1.jsonl`（JSONL，每行一条样本）。

## 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 样本唯一编号（q001…q069） |
| `query` | string | 用户问题 |
| `type` | string | 问题类型：`api_usage / concept / concept_compare / error_diag / config / version_diff / boundary / no_answer / general` |
| `docset` | string | 所属文档集：`langchain / milvus / bge / transformers / cross_lib / general` |
| `expected_route` | string | 预期路由：`专业咨询`（走 RAG）或 `通用知识`（直接 LLM） |
| `expected_docs` | list[string] | 预期相关文档路径，格式 `"<docset>:<相对路径>"`，例如 `"milvus:site/en/userGuide/xxx.md"` |
| `answer_points` | list[string] | 参考答案要点（判断 Answer Accuracy 用） |
| `citation_criteria` | string | 引用判断依据（判断 Citation Accuracy 用） |
| `needs_review` | bool | `true`=待人工确认；`false`=已标注确认 |

## 标注状态区分

- **`needs_review: true`**：尚未完成标注（`expected_docs/answer_points/citation_criteria` 为空或不确定）。
- **`needs_review: false`**：已人工确认（含 `type: no_answer` 类——已确认知识库中无相关文档，此时 `expected_docs=[]` 且 `answer_points` 标注为“知识库无相关答案”）。

## expected_docs 填写规则

1. 格式统一为 `"<docset>:<相对路径>"`，路径分隔符用正斜杠 `/`。
2. `<相对路径>` 是相对 `rag_qa/data/<docset>_data/` 的路径，例如：
   - `milvus:site/en/userGuide/search-query-get/full-text-search.md`
   - `langchain:src/oss/langchain/knowledge-base.mdx`
   - `bge:Tutorials/5_Reranking/5.2_BGE_Reranker.ipynb`
   - `transformers:en/pipeline_tutorial.md`
3. **必须确认路径真实存在**。校验方式（相对项目根目录）：
   ```python
   import os
   src, fp = "milvus:site/en/userGuide/xxx.md".split(":", 1)
   full = os.path.join("rag_qa/data", f"{src}_data", fp.replace("/", os.sep))
   print(os.path.exists(full))  # 应为 True
   ```
4. 只填“确实能回答该问题的相关文档”，不要把检索到的无关文档填进去。
5. 无法确定相关性时，保持 `needs_review: true`，不要猜测。

## answer_points 填写规则

1. 列出“参考答案必须覆盖的核心要点”（1～3 条短句）。
2. 用于规则化 Answer Accuracy（检查答案是否覆盖要点），不要求逐字匹配。
3. `type: no_answer` 样本填 `["知识库中无相关答案"]`。

## citation_criteria 填写规则

1. 描述“正确的引用应满足的条件”，例如：`引用应指向 pipeline_tutorial.md`。
2. 用于 Citation Accuracy 规则检查：判断 `last_citations` 的 `file_path` 是否满足该条件、是否真实支撑答案。

## 标注流程建议

1. 对每条 `needs_review: true` 样本，先跑真实检索（`vector_store.hybrid_search_with_rerank` 或后续的 `retrieve_top_k`）拿候选 `file_path`。
2. 打开候选文件核对内容是否确实能回答该问题。
3. 相关则填入 `expected_docs`，并补 `answer_points` / `citation_criteria`，把 `needs_review` 改为 `false`。
4. 检索不到相关文档的，若确认知识库确实无答案，标为 `type: no_answer`、`expected_docs: []`、`needs_review: false`；否则保持 `needs_review: true`。
