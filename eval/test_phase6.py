# -*- coding: utf-8 -*-
"""Phase6 离线回归测试：答案可支撑性判定(三分类) + 引用清理 + 端到端集成。

完全离线：使用桩 VectorStore / 桩 LLM / 桩分类器，不连接 Milvus、不调用真实 API。
运行：python eval/test_phase6.py
输出：eval/results/phase6_tests.json
"""
import os, sys, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import rag_qa.core.rag_pipeline as nrs
from rag_qa.core.rag_pipeline import RAGSystem, sanitize_citations

OUT = os.environ.get("RAG_RESULT_OUT") or os.path.join(SCRIPT_DIR, "results", "phase6_tests.json")

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"name": name, "pass": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name + (("  | " + detail) if detail and not cond else ""))


class Doc:
    def __init__(self, content, md=None):
        self.page_content = content
        self.metadata = md or {}


class StubVS:
    def __init__(self, docs, score=0.9):
        self.docs = docs
        self.score = score

    def hybrid_search_with_rerank(self, query, k=10, source_filter=None):
        return list(self.docs)

    def max_relevance_score(self, query, docs):
        return self.score


class StubClassifier:
    def __init__(self, cat):
        self.cat = cat

    def predict_category(self, q):
        return self.cat


class StubStrategy:
    def select_strategy(self, q):
        return "直接检索"


def build_rag(docs, judge_raw, answer, classifier="专业咨询", score=0.9):
    rag = RAGSystem.__new__(RAGSystem)
    from rag_qa.core.prompts import RAGPrompts
    rag.rag_prompt = RAGPrompts.rag_prompt()
    rag.query_classifier = StubClassifier(classifier)
    rag.strategy_selector = StubStrategy()
    rag.vector_store = StubVS(docs, score=score)
    rag.last_citations = []
    rag.last_support_verdict = None
    counter = {"judge": 0, "gen": 0}

    def llm(prompt):
        if "检索证据充分性判定器" in prompt:
            counter["judge"] += 1
            return judge_raw
        counter["gen"] += 1
        if isinstance(answer, str):
            return answer
        def gen():
            for c in answer:
                yield c
        return gen()
    rag.llm = llm
    return rag, counter


def docs_two():
    return [
        Doc("Milvus 是向量数据库。", {"file_path": "a.md", "source": "milvus", "repo": "milvus-io/milvus"}),
        Doc("BGE 是嵌入模型。", {"file_path": "b.md", "source": "bge", "repo": "FlagOpen/FlagEmbedding"}),
    ]


CONF_DEFAULTS = None


def set_conf(**kw):
    global CONF_DEFAULTS
    if CONF_DEFAULTS is None:
        CONF_DEFAULTS = {}
        for k in ("SUPPORT_JUDGE", "RELEVANCE_THRESHOLD", "RETRIEVAL_K", "CANDIDATE_M", "SUPPORT_JUDGE_MAX_CHARS"):
            CONF_DEFAULTS[k] = getattr(nrs.conf, k, None)
    nrs.conf.SUPPORT_JUDGE = kw.get("SUPPORT_JUDGE", True)
    nrs.conf.RELEVANCE_THRESHOLD = kw.get("RELEVANCE_THRESHOLD", 0.2)
    nrs.conf.RETRIEVAL_K = kw.get("RETRIEVAL_K", 10)
    nrs.conf.CANDIDATE_M = kw.get("CANDIDATE_M", 2)
    nrs.conf.SUPPORT_JUDGE_MAX_CHARS = kw.get("SUPPORT_JUDGE_MAX_CHARS", 4000)


def test_sanitize():
    a = sanitize_citations("[来源1] 好 [来源3] 坏 [来源0] 零 arr[0] 代码", 2)
    check("sanitize: 保留范围内/删除越界与0/不伤代码下标",
          a == "[来源1] 好  坏  零 arr[0] 代码", repr(a))
    check("sanitize: n=0 删除全部显式引用", sanitize_citations("[来源1][来源2]", 0) == "")
    check("sanitize: 中文括号形式", sanitize_citations("【来源1】内容【来源5】", 1) == "【来源1】内容")
    check("sanitize: 无引用不变", sanitize_citations("普通文本 [注意] 无引用", 0) == "普通文本 [注意] 无引用")


def test_parse_verdict():
    rag = RAGSystem.__new__(RAGSystem)
    check("parse: 纯JSON", rag._parse_verdict('{"verdict":"insufficient","reason":"r","missing":"m"}')["verdict"] == "insufficient")
    check("parse: 嵌入JSON带解释", rag._parse_verdict('判定如下：{"verdict":"partial"} 以上。')["verdict"] == "partial")
    check("parse: 关键词兜底", rag._parse_verdict("结论是 insufficient，因为没有证据")["verdict"] == "insufficient")
    check("parse: 无法解析->sufficient", rag._parse_verdict("完全看不懂的输出")["verdict"] == "sufficient")


def test_stream():
    rag = RAGSystem.__new__(RAGSystem)
    out = "".join(rag._sanitize_stream(iter(["答案", "[来", "源1]", " ok", "[来源9]"]), 1))
    check("stream: 跨块引用保留范围内/删除越界", out == "答案[来源1] ok", repr(out))
    out0 = "".join(rag._sanitize_stream(iter(["[来", "源1]", " 无依据"]), 0))
    check("stream: n=0 跨块引用被清除", "来源" not in out0.replace("无依据", ""), repr(out0))


def test_integration_sufficient():
    set_conf(SUPPORT_JUDGE=True)
    rag, c = build_rag(docs_two(), '{"verdict":"sufficient","reason":"ok","missing":""}',
                       ["答案", "[来", "源1]", " 依据充分 [来源9]"])
    ans = "".join(rag.generate_answer("Milvus 是什么？"))
    check("集成-sufficient: 判定被调用且保留有效引用",
          c["judge"] == 1 and "[来源1]" in ans and "[来源9]" not in ans, repr(ans))
    check("集成-sufficient: 引用数=2", len(rag.last_citations) == 2)
    check("集成-sufficient: verdict 记录", rag.last_support_verdict["verdict"] == "sufficient")


def test_integration_insufficient():
    set_conf(SUPPORT_JUDGE=True)
    rag, c = build_rag(docs_two(), '{"verdict":"insufficient","reason":"只谈概念","missing":"作者"}',
                       "[来源1] 这是编造的作者结论。")
    ans = "".join(rag.generate_answer("Milvus 的作者是谁？"))
    check("集成-insufficient: 上下文清空/引用清空", rag.last_citations == [] and rag.last_support_verdict["verdict"] == "insufficient")
    check("集成-insufficient: 虚假引用被清除", "[来源1]" not in ans and "[来源" not in ans, repr(ans))


def test_integration_partial():
    set_conf(SUPPORT_JUDGE=True)
    captured = {}

    def llm(prompt):
        if "检索证据充分性判定器" in prompt:
            return '{"verdict":"partial","reason":"仅部分","missing":"版本细节"}'
        captured["prompt"] = prompt
        return iter(["部分回答", "[来源1]"])

    rag = RAGSystem.__new__(RAGSystem)
    from rag_qa.core.prompts import RAGPrompts
    rag.rag_prompt = RAGPrompts.rag_prompt()
    rag.query_classifier = StubClassifier("专业咨询")
    rag.strategy_selector = StubStrategy()
    rag.vector_store = StubVS(docs_two())
    rag.last_citations = []
    rag.last_support_verdict = None
    rag.llm = llm
    ans = "".join(rag.generate_answer("Milvus 版本行为？"))
    check("集成-partial: 上下文含部分回答指令", "只包含部分信息" in captured.get("prompt", ""))
    check("集成-partial: 引用保留", len(rag.last_citations) == 2 and "[来源1]" in ans)


def test_general():
    set_conf(SUPPORT_JUDGE=True)
    rag, c = build_rag([], '{"verdict":"sufficient"}', "通用回答 [来源1] 结束", classifier="通用知识")
    ans = "".join(rag.generate_answer("你好"))
    check("集成-通用: 不调用判定/无检索引用", c["judge"] == 0 and rag.last_citations == [] and "[来源" not in ans, repr(ans))


def test_empty_retrieval():
    set_conf(SUPPORT_JUDGE=True)
    rag, c = build_rag([], '{"verdict":"sufficient"}', "依据不足 [来源1]", classifier="专业咨询")
    ans = "".join(rag.generate_answer("库外问题"))
    check("集成-空检索: 不调用判定/引用清空", c["judge"] == 0 and rag.last_citations == [] and "[来源" not in ans, repr(ans))


def test_gate_blocks():
    set_conf(SUPPORT_JUDGE=True, RELEVANCE_THRESHOLD=0.2)
    rag, c = build_rag(docs_two(), '{"verdict":"sufficient"}', "回答 [来源1]", score=0.05)
    ans = "".join(rag.generate_answer("无关问题"))
    check("集成-分数门控: 低分拦截且不调用判定", c["judge"] == 0 and rag.last_citations == [])


def test_disabled():
    set_conf(SUPPORT_JUDGE=False)
    rag, c = build_rag(docs_two(), '{"verdict":"insufficient"}', "回答 [来源1]")
    ans = "".join(rag.generate_answer("Milvus 是什么？"))
    check("集成-判定关闭: 回退到 Phase5 基线(不调用判定/引用保留)", c["judge"] == 0 and len(rag.last_citations) == 2)


def main():
    set_conf()
    test_sanitize()
    test_parse_verdict()
    test_stream()
    test_integration_sufficient()
    test_integration_insufficient()
    test_integration_partial()
    test_general()
    test_empty_retrieval()
    test_gate_blocks()
    test_disabled()
    passed = sum(1 for r in RESULTS if r["pass"])
    print(f"\n总计: {passed}/{len(RESULTS)} 通过")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"total": len(RESULTS), "passed": passed, "results": RESULTS}, f, ensure_ascii=False, indent=2)
    print("saved ->", OUT)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
