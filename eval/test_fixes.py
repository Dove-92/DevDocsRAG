# -*- coding: utf-8 -*-
"""离线验证 rag_pipeline 的最小修复（不连 Milvus、不调用 Qwen）。

覆盖：
A. _llm_text 对 生成器/字符串 均返回 str；
B. 三个检索策略(子查询/HyDE/回溯)在流式生成器 llm 下不再报 'generator ... strip'；
C. 空上下文(专业咨询无文档 / 通用知识)注入 _NO_CONTEXT_NOTE 且 citations 为空；
D. 有文档时正常构建上下文与 citations（回归）。
"""
import os, sys, json

PROJECT_ROOT = r"E:\LLM_ubuntu\rag_qa_system"
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document
from rag_qa.core import rag_pipeline as nrs
from rag_qa.core.rag_pipeline import RAGSystem, _NO_CONTEXT_NOTE

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond), extra))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


class GenLLM:
    """流式生成器 llm（模拟 qa_system.call_dashscope）。"""
    def __init__(self, tokens):
        self.tokens = tokens
        self.prompts = []
    def __call__(self, prompt):
        self.prompts.append(prompt)
        def gen():
            for t in self.tokens:
                yield t
        return gen()


class StrLLM:
    def __init__(self, text):
        self.text = text
        self.prompts = []
    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.text


class StubVS:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []
    def hybrid_search_with_rerank(self, query, k=None, source_filter=None):
        self.calls.append(query)
        return list(self.docs)
    def retrieve_top_k(self, query, k=10, source_filter=None):
        return []


def make_doc(path, content="doc内容"):
    return Document(page_content=content, metadata={"file_path": path, "source": "langchain",
                                                    "repo": "r", "version": "v", "doc_type": "t"})


def main():
    # 用空 vector_store 构造，行为在测试中通过替换属性控制
    vs = StubVS([])
    rag = RAGSystem(vs, GenLLM(["x"]))

    # A. _llm_text
    rag.llm = GenLLM(["a", "b", "c"])
    check("A1 _llm_text(generator)=='abc'", rag._llm_text("p") == "abc", repr(rag._llm_text("p")))
    rag.llm = StrLLM("hello")
    check("A2 _llm_text(str)=='hello'", rag._llm_text("p") == "hello")

    # B. 三个检索策略在流式生成器下不报错
    rag.llm = GenLLM(["子问题A\n", "子问题B\n"])
    rag.vector_store = StubVS([])
    try:
        out = rag._retrieve_with_subqueries("q")
        check("B1 _retrieve_with_subqueries 流式不报错", out == [], f"out={out}")
    except Exception as e:
        check("B1 _retrieve_with_subqueries 流式不报错", False, f"{type(e).__name__}: {e}")

    rag.llm = GenLLM(["假设答案"])
    try:
        out = rag._retrieve_with_hyde("q")
        check("B2 _retrieve_with_hyde 流式不报错", out == [])
    except Exception as e:
        check("B2 _retrieve_with_hyde 流式不报错", False, f"{type(e).__name__}: {e}")

    rag.llm = GenLLM(["简化问题"])
    try:
        out = rag._retrieve_with_backtracking("q")
        check("B3 _retrieve_with_backtracking 流式不报错", out == [])
    except Exception as e:
        check("B3 _retrieve_with_backtracking 流式不报错", False, f"{type(e).__name__}: {e}")

    # B4. 有文档时子查询检索能返回去重文档
    rag.llm = GenLLM(["子A\n子B\n"])
    rag.vector_store = StubVS([make_doc("p1", "c1"), make_doc("p2", "c2")])
    out = rag._retrieve_with_subqueries("q")
    check("B4 子查询有文档时返回去重结果", len(out) == 2, f"len={len(out)}")

    # C1. 专业咨询 + 空上下文
    rag.llm = GenLLM(["答", "案"])
    rag.vector_store = StubVS([])
    rag.query_classifier.predict_category = lambda q: "专业咨询"
    rag.strategy_selector.select_strategy = lambda q: "直接检索"
    gen = rag.generate_answer("Milvus 怎么插入向量数据？")
    ans = "".join(gen)
    p = rag.llm.prompts[-1]
    check("C1 专业空上下文: 注入无上下文说明", _NO_CONTEXT_NOTE in p)
    check("C1 专业空上下文: 提示勿输出[来源n]", "请勿输出 [来源n]" in p)
    check("C1 专业空上下文: citations 为空", rag.last_citations == [], f"{rag.last_citations}")

    # C2. 通用知识
    rag.llm = GenLLM(["120"])
    rag.query_classifier.predict_category = lambda q: "通用知识"
    gen = rag.generate_answer("5 的阶乘是多少？")
    ans = "".join(gen)
    p = rag.llm.prompts[-1]
    check("C2 通用知识: 注入无上下文说明", _NO_CONTEXT_NOTE in p)
    check("C2 通用知识: citations 为空", rag.last_citations == [])

    # D. 有文档回归: 上下文含编号，citations 非空
    rag.llm = GenLLM(["ok"])
    rag.vector_store = StubVS([make_doc("a.md", "AAA"), make_doc("b.md", "BBB")])
    rag.query_classifier.predict_category = lambda q: "专业咨询"
    rag.strategy_selector.select_strategy = lambda q: "直接检索"
    gen = rag.generate_answer("q")
    ans = "".join(gen)
    p = rag.llm.prompts[-1]
    check("D1 有文档: 上下文含[来源1]", "[来源1]" in p)
    check("D2 有文档: citations==2", len(rag.last_citations) == 2, f"{len(rag.last_citations)}")
    check("D3 有文档: 不含无上下文说明", _NO_CONTEXT_NOTE not in p)

    ok = sum(1 for _, c, _ in RESULTS if c)
    print(f"\n结果: {ok}/{len(RESULTS)} 通过")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
