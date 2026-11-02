# -*- coding: utf-8 -*-
"""Phase5.1 离线回归测试（项目内，可重复运行）。
真实 VectorStore（只读 Milvus）+ 本地 embedding/reranker + 打桩 LLM（不消耗 Qwen API）。

覆盖：直接检索 / HyDE / 子查询 / 回溯检索 / 策略失败回退 / 无答案门控拒答 /
      正常专业问答 / 通用问题 / 引用生成 / 分类路由。

运行：<python> eval/test_phase51_regression.py
退出码：0=全部通过，1=存在失败。
"""
import os, sys, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_pipeline import RAGSystem
from rag_qa.core import rag_pipeline as nrs

RES = []


def check(name, cond, extra=""):
    RES.append({"name": name, "pass": bool(cond), "extra": extra})
    print("[%s] %s %s" % ("PASS" if cond else "FAIL", name, extra))


class GenLLM:
    """流式生成器 LLM（模拟生产 call_dashscope），记录 prompt。"""
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)

        def gen():
            for t in self.tokens:
                yield t
        return gen()


class RaiseLLM:
    def __call__(self, prompt):
        raise RuntimeError("simulated LLM failure")


def main():
    vs = VectorStore()
    rag = RAGSystem(vs, GenLLM(["ok"]))
    rag.strategy_selector.select_strategy = lambda q: "直接检索"  # 避免策略选择走真实 API

    # 1. 直接检索 + 引用生成（正常专业问答）
    rag.llm = GenLLM(["答案"])
    out = "".join(rag.generate_answer("Milvus 的 dense 和 sparse 检索有什么区别？"))
    c = rag.last_citations
    check("1 direct professional: citations non-empty", len(c) > 0, "n=%d" % len(c))
    check("1b citation fields ok", all(x.get("source") and x.get("file_path") for x in c))

    # 2-4. 三种检索策略
    rag.llm = GenLLM(["Milvus dense vector semantic search hybrid retrieval"])
    check("2 hyde returns docs", len(rag._retrieve_with_hyde("Milvus 的 dense 和 sparse 检索有什么区别？")) > 0)
    rag.llm = GenLLM(["Milvus dense vector\nMilvus sparse vector\n"])
    check("3 subquery returns deduped docs", len(rag._retrieve_with_subqueries("Milvus 的 dense 和 sparse 检索有什么区别？")) > 0)
    rag.llm = GenLLM(["Milvus dense sparse difference"])
    check("4 backtracking returns docs", len(rag._retrieve_with_backtracking("Milvus 的 dense 和 sparse 检索有什么区别？")) > 0)

    # 5. 策略失败 -> 回退直接检索（非空，不崩溃）
    rag.llm = RaiseLLM()
    rag.vector_store = vs
    check("5 subquery failure -> fallback direct non-empty", len(rag._retrieve_with_subqueries("Milvus 怎么创建 Collection？")) > 0)
    check("5b hyde failure -> fallback direct non-empty", len(rag._retrieve_with_hyde("Milvus 怎么创建 Collection？")) > 0)

    # 6. 无答案门控拒答（q059）
    rag.llm = GenLLM(["依据不足。"])
    "".join(rag.generate_answer("DevDocsRAG 的作者是谁？"))
    check("6 no-answer gate: citations empty", rag.last_citations == [], "n=%d" % len(rag.last_citations))

    # 7. 可回答的专业问题通过门控
    rag.llm = GenLLM(["答案"])
    "".join(rag.generate_answer("Milvus 的 dense 和 sparse 检索有什么区别？"))
    check("7 answerable passes gate", len(rag.last_citations) > 0, "n=%d" % len(rag.last_citations))

    # 8. 通用问题 -> 无引用、答案无 [来源n]
    rag.llm = GenLLM(["120"])
    out = "".join(rag.generate_answer("5 的阶乘是多少？"))
    check("8 general: citations empty", rag.last_citations == [], "n=%d" % len(rag.last_citations))
    check("8b general: prompt injects no-context note", nrs._NO_CONTEXT_NOTE in rag.llm.prompts[-1])
    check("8c general: answer has no [来源 markers", "[来源" not in out)

    # 9. 分类路由
    check("9 q039 -> 专业咨询", rag.query_classifier.predict_category("什么是 attention mask？") == "专业咨询")
    check("9b q061 -> 通用知识", rag.query_classifier.predict_category("5 的阶乘是多少？") == "通用知识")
    check("9c q059 -> 专业咨询", rag.query_classifier.predict_category("DevDocsRAG 的作者是谁？") == "专业咨询")

    ok = sum(1 for r in RES if r["pass"])
    print("\nRESULT: %d/%d" % (ok, len(RES)))
    out_path = os.environ.get("RAG_RESULT_OUT") or os.path.join(ROOT, "eval", "results", "phase51_regression.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"passed": ok, "total": len(RES), "checks": RES}, f, ensure_ascii=False, indent=2)
    return 0 if ok == len(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
