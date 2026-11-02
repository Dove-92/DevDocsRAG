# -*- coding: utf-8 -*-
"""生成 DevDocsRAG 固定评估集 eval/devdocs_eval_set_v1.jsonl(69 条骨架 + 8 条已标注示例)。"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devdocs_eval_set_v1.jsonl")

# (id, query, type, docset, expected_route)
SAMPLES = [
    # ===== langchain (12) =====
    ("q003", "LangChain 怎么加载 PDF 文档？", "api_usage", "langchain", "专业咨询"),
    ("q009", "LangChain 的文本分割器怎么设置 chunk_size？", "api_usage", "langchain", "专业咨询"),
    ("q010", "LangChain 怎么集成向量数据库做检索？", "api_usage", "langchain", "专业咨询"),
    ("q011", "LangChain 的 ChatOpenAI 怎么开启流式输出？", "api_usage", "langchain", "专业咨询"),
    ("q004", "LangChain 的 LCEL 是什么？", "concept", "langchain", "专业咨询"),
    ("q012", "LangChain 的 Chain 和 Agent 有什么区别？", "concept", "langchain", "专业咨询"),
    ("q013", "LangChain 的 Memory 组件有什么作用？", "concept", "langchain", "专业咨询"),
    ("q014", "LangChain 报 ModuleNotFoundError 怎么办？", "error_diag", "langchain", "专业咨询"),
    ("q015", "LangChain 的 API key 报错怎么解决？", "error_diag", "langchain", "专业咨询"),
    ("q016", "LangChain 0.2 和 0.3 有什么区别？", "version_diff", "langchain", "专业咨询"),
    ("q017", "LangChain 的 langchain_core 和 langchain_community 有什么区别？", "version_diff", "langchain", "专业咨询"),
    ("q018", "LangChain 怎么控制模型温度 temperature？", "config", "langchain", "专业咨询"),

    # ===== milvus (12) =====
    ("q001", "Milvus 怎么创建 Collection？", "api_usage", "milvus", "专业咨询"),
    ("q019", "Milvus 怎么插入向量数据？", "api_usage", "milvus", "专业咨询"),
    ("q020", "Milvus 怎么给向量字段建索引？", "api_usage", "milvus", "专业咨询"),
    ("q021", "MilvusClient 的 hybrid_search 参数怎么传？", "api_usage", "milvus", "专业咨询"),
    ("q002", "WeightedRanker 和 RRF 怎么选？", "concept_compare", "milvus", "专业咨询"),
    ("q022", "Milvus 的 dense 和 sparse 检索有什么区别？", "concept", "milvus", "专业咨询"),
    ("q023", "Milvus 的 Collection 和 Partition 是什么关系？", "concept", "milvus", "专业咨询"),
    ("q024", "Milvus 报 collection not found 怎么排查？", "error_diag", "milvus", "专业咨询"),
    ("q025", "Milvus 连接超时怎么解决？", "error_diag", "milvus", "专业咨询"),
    ("q026", "Milvus 的 IVF_FLAT 和 HNSW 索引怎么选？", "config", "milvus", "专业咨询"),
    ("q027", "Milvus 怎么用 Docker 部署？", "config", "milvus", "专业咨询"),
    ("q028", "Milvus 2.4 和 2.5 有什么区别？", "version_diff", "milvus", "专业咨询"),

    # ===== bge (10) =====
    ("q006", "bge-reranker-large 怎么用来做精排？", "api_usage", "bge", "专业咨询"),
    ("q029", "FlagEmbedding 怎么生成稀疏向量？", "api_usage", "bge", "专业咨询"),
    ("q030", "BGE-M3 怎么生成 dense 和 sparse 向量？", "api_usage", "bge", "专业咨询"),
    ("q005", "BGE-M3 的 dense 向量维度是多少？", "concept", "bge", "专业咨询"),
    ("q031", "BGE-M3 的 dense 和 sparse 检索各适合什么场景？", "concept", "bge", "专业咨询"),
    ("q032", "BGE-M3 支持多语言吗？", "concept", "bge", "专业咨询"),
    ("q033", "BGE-M3 的 use_fp16 参数有什么用？", "config", "bge", "专业咨询"),
    ("q034", "BGE-M3 怎么加载到 CUDA？", "config", "bge", "专业咨询"),
    ("q035", "BGE-M3 和 bge-base 有什么区别？", "version_diff", "bge", "专业咨询"),
    ("q036", "BGE-M3 加载报显存不足怎么办？", "error_diag", "bge", "专业咨询"),

    # ===== transformers (10) =====
    ("q007", "Transformers 的 pipeline 支持哪些任务？", "api_usage", "transformers", "专业咨询"),
    ("q037", "AutoModel.from_pretrained 怎么加载本地模型？", "api_usage", "transformers", "专业咨询"),
    ("q038", "Transformers 的 tokenizer 怎么用？", "api_usage", "transformers", "专业咨询"),
    ("q039", "什么是 attention mask？", "concept", "transformers", "专业咨询"),
    ("q040", "Transformers 的 attention 机制是什么？", "concept", "transformers", "专业咨询"),
    ("q008", "用 Transformers 训练时报 OOM 怎么解决？", "error_diag", "transformers", "专业咨询"),
    ("q041", "from_pretrained 报连接错误怎么办？", "error_diag", "transformers", "专业咨询"),
    ("q042", "Transformers 怎么设置 max_length？", "config", "transformers", "专业咨询"),
    ("q043", "Transformers 怎么做模型量化？", "config", "transformers", "专业咨询"),
    ("q044", "Transformers 的 safetensors 和 pytorch_model.bin 有什么区别？", "version_diff", "transformers", "专业咨询"),

    # ===== 跨库 (5) =====
    ("q045", "LangChain 怎么集成 Milvus 做 RAG？", "concept_compare", "cross_lib", "专业咨询"),
    ("q046", "BGE-M3 和 Milvus 的嵌入函数怎么配合使用？", "concept_compare", "cross_lib", "专业咨询"),
    ("q047", "Transformers 的 tokenizer 和 BGE-M3 的 tokenizer 有什么区别？", "concept_compare", "cross_lib", "专业咨询"),
    ("q048", "用 LangChain + BGE-M3 + Milvus 搭建 RAG 的完整流程是什么？", "concept_compare", "cross_lib", "专业咨询"),
    ("q049", "Milvus 的 reranker 和 BGE 的 bge-reranker 是什么关系？", "concept_compare", "cross_lib", "专业咨询"),

    # ===== 边界 (5) =====
    ("q050", "怎么给文档做 embedding？", "boundary", "cross_lib", "专业咨询"),
    ("q051", "向量检索怎么加速？", "boundary", "cross_lib", "专业咨询"),
    ("q052", "怎么搭一个 RAG 系统？", "boundary", "cross_lib", "专业咨询"),
    ("q053", "它和另一个检索方案有什么区别？", "boundary", "cross_lib", "专业咨询"),
    ("q054", "有更简单的做法吗？", "boundary", "cross_lib", "专业咨询"),

    # ===== 无答案 (5) =====
    ("q055", "LangChain 怎么和 ChatGPT 的 o3 模型集成？", "no_answer", "langchain", "专业咨询"),
    ("q056", "Milvus 支持哪些 GPU 型号？", "no_answer", "milvus", "专业咨询"),
    ("q057", "BGE-M3 的作者是谁？", "no_answer", "bge", "专业咨询"),
    ("q058", "Transformers 的 GPT-5 模型怎么用？", "no_answer", "transformers", "专业咨询"),
    ("q059", "DevDocsRAG 的作者是谁？", "no_answer", "cross_lib", "专业咨询"),

    # ===== 通用 (10) =====
    ("q060", "用 Python 写一个冒泡排序", "general", "general", "通用知识"),
    ("q061", "5 的阶乘是多少？", "general", "general", "通用知识"),
    ("q062", "什么是 HTTPS？", "general", "general", "通用知识"),
    ("q063", "什么是 Docker？", "general", "general", "通用知识"),
    ("q064", "数据库三大范式是什么？", "general", "general", "通用知识"),
    ("q065", "TCP 和 UDP 有什么区别？", "general", "general", "通用知识"),
    ("q066", "你好", "general", "general", "通用知识"),
    ("q067", "今天天气怎么样？", "general", "general", "通用知识"),
    ("q068", "推荐一部好看的电影", "general", "general", "通用知识"),
    ("q069", "怎么用 Python 处理 JSON？", "general", "general", "通用知识"),
]

# 已标注的 8 条示例(id -> annotation)
ANNOTATED = {
    "q001": {
        "expected_docs": [
            "milvus:site/en/userGuide/search-query-get/full-text-search.md",
            "milvus:site/en/userGuide/tools/cli_commands.md",
        ],
        "answer_points": ["用 MilvusClient 创建 schema 与索引参数，再调用 create_collection 创建集合"],
        "citation_criteria": "引用应指向创建 Collection 的文档（full-text-search.md 或 cli_commands.md）",
        "needs_review": False,
    },
    "q002": {
        "expected_docs": [
            "milvus:site/en/userGuide/embeddings-reranking/reranking/rrf-ranker.md",
        ],
        "answer_points": ["RRF 按排名倒数融合、无需权重；WeightedRanker 按权重线性融合；多模态等权重场景适合 RRF"],
        "citation_criteria": "引用应指向 rrf-ranker.md",
        "needs_review": False,
    },
    "q003": {
        "expected_docs": [
            "langchain:src/oss/langchain/knowledge-base.mdx",
        ],
        "answer_points": ["用 LangChain 的 DocumentLoader（如 PyPDFLoader）加载 PDF 页面"],
        "citation_criteria": "引用应指向 knowledge-base.mdx（含 load_pdf_pages 示例）",
        "needs_review": False,
    },
    "q004": {
        "expected_docs": [
            "milvus:site/en/tutorials/full_text_search_with_milvus.md",
        ],
        "answer_points": [],
        "citation_criteria": "",
        "needs_review": True,
    },
    "q005": {
        "expected_docs": [
            "bge:Tutorials/1_Embedding/1.2.1_BGE_Series.ipynb",
        ],
        "answer_points": ["BGE-M3 的 dense 向量维度为 1024"],
        "citation_criteria": "引用应指向 BGE_Series.ipynb 或含 BGE-M3 维度说明的文档",
        "needs_review": True,
    },
    "q006": {
        "expected_docs": [
            "bge:Tutorials/5_Reranking/5.2_BGE_Reranker.ipynb",
            "milvus:site/en/rerankers/rerankers-bge.md",
        ],
        "answer_points": ["用 FlagReranker 或 sentence_transformers 的 CrossEncoder 加载 bge-reranker-large，对 query-doc pair 打分并排序"],
        "citation_criteria": "引用应指向 BGE_Reranker 教程或 rerankers-bge.md",
        "needs_review": False,
    },
    "q007": {
        "expected_docs": [
            "transformers:en/pipeline_tutorial.md",
        ],
        "answer_points": ["pipeline 支持文本分类、NER、问答、翻译、摘要、图像分类等多种任务"],
        "citation_criteria": "引用应指向 pipeline_tutorial.md",
        "needs_review": False,
    },
    "q008": {
        "expected_docs": [
            "transformers:en/troubleshooting.md",
            "transformers:en/deepspeed.md",
        ],
        "answer_points": ["减小 batch_size、梯度累积、启用混合精度、使用 DeepSpeed/量化等降低显存"],
        "citation_criteria": "引用应指向 troubleshooting.md 或 deepspeed.md",
        "needs_review": False,
    },
}

if __name__ == "__main__":
    out = []
    for sid, query, typ, docset, route in SAMPLES:
        ann = ANNOTATED.get(sid, {})
        out.append({
            "id": sid,
            "query": query,
            "type": typ,
            "docset": docset,
            "expected_route": route,
            "expected_docs": ann.get("expected_docs", []),
            "answer_points": ann.get("answer_points", []),
            "citation_criteria": ann.get("citation_criteria", ""),
            "needs_review": ann.get("needs_review", True),
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    annotated = sum(1 for r in out if not r["needs_review"])
    print(f"生成 {len(out)} 条 | 已标注 {annotated} 条 | 待标注 {len(out) - annotated} 条")
    print("输出:", OUT)
