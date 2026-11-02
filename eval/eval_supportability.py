# -*- coding: utf-8 -*-
"""Phase6 Task1：答案可支撑性判定(三分类)真实 API 离线评测。

对每个探针问题：只读检索 + 相关性门控 -> 构造上下文 -> 调用 LLM 判定(sufficient/partial/insufficient)。
- 正例(21)：期望 NOT insufficient(可回答/部分回答)；统计可回答问题误拒率。
- 负例(主题相关但库缺失 q055-q058 + 域外 q059/neg00-05)：期望 gate 拒绝 或 insufficient(不产生有依据完整回答)。
不改写任何历史结果。输出：eval/results/phase6_supportability.json
"""
import os, sys, json, time, io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from base.config import Config
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_pipeline import RAGSystem
import rag_qa.core.rag_pipeline as nrs

EVAL_PATH = os.path.join(SCRIPT_DIR, "devdocs_eval_set_v1.jsonl")
OUT = os.path.join(SCRIPT_DIR, "results", "phase6_supportability.json")

TOPIC_MISSING_IDS = {"q055", "q056", "q057", "q058"}       # 主题相关但库缺失
OUT_OF_DOMAIN_IDS = {"q059"}                                # 域外
NEG_EXTRA = [
    "公司今年的营收是多少？",
    "DevDocsRAG 支持哪些支付方式？",
    "Milvus 的创始人是谁？",
    "LangChain 的公司总部在哪里？",
    "BGE-M3 的训练成本是多少？",
    "Transformers 背后的公司股价如何？",
]

cfg = Config()
client = OpenAI(api_key=cfg.DASHSCOPE_API_KEY, base_url=cfg.DASHSCOPE_BASE_URL)
USAGE = {"last": None, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def call_dashscope(prompt):
    USAGE["calls"] += 1
    completion = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是 DevDocsRAG 检索证据充分性判定器，严格按要求输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        timeout=30,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in completion:
        u = getattr(chunk, "usage", None)
        if u is not None:
            USAGE["last"] = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                             "completion_tokens": getattr(u, "completion_tokens", None)}
            if getattr(u, "prompt_tokens", None):
                USAGE["prompt_tokens"] += u.prompt_tokens
            if getattr(u, "completion_tokens", None):
                USAGE["completion_tokens"] += u.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def load_eval(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def main():
    rows = load_eval(EVAL_PATH)
    positives = [r for r in rows if (r.get("expected_docs") and not r.get("needs_review"))]
    neg_rows = [r for r in rows if r.get("type") == "no_answer"]

    vs = VectorStore()
    rag = RAGSystem(vs, call_dashscope)
    vs.retrieve_top_k("warmup", k=3)  # 预热检索/重排

    # 固定为"直接检索"以消除策略选择器(LLM, 非确定性)带来的噪声，保证评测可复现；
    # 与 Phase5 gate_scores 的直接检索口径一致。管线默认仍为自动策略。
    EVAL_STRATEGY = "直接检索"

    def judge_one(qid, query, category):
        rec = {"id": qid, "query": query, "category": category}
        t0 = time.time()
        context, citations, docs = rag.prepare_context(query, strategy=EVAL_STRATEGY)
        rec["gate_rejected"] = (len(docs) == 0)
        rec["n_context_docs"] = len(docs)
        if not context or not docs:
            rec["verdict"] = "gate_rejected"
            rec["judge_ms"] = None
            rec["tokens"] = None
        else:
            t1 = time.time()
            v = rag.assess_support(query, context)
            rec["judge_ms"] = round((time.time() - t1) * 1000, 1)
            rec["verdict"] = v.get("verdict")
            rec["reason"] = v.get("reason")
            rec["missing"] = v.get("missing")
            rec["tokens"] = USAGE.get("last")
        rec["total_ms"] = round((time.time() - t0) * 1000, 1)
        return rec

    pos_recs = [judge_one(r["id"], r["query"], "positive") for r in positives]
    neg_recs = []
    for r in neg_rows:
        cat = "topic_missing" if r["id"] in TOPIC_MISSING_IDS else "out_of_domain"
        neg_recs.append(judge_one(r["id"], r["query"], cat))
    for i, q in enumerate(NEG_EXTRA):
        neg_recs.append(judge_one("neg%02d" % i, q, "out_of_domain"))

    # ---- 指标 ----
    def frac(items, pred):
        return round(sum(1 for x in items if pred(x)) / len(items), 4) if items else None

    pos_false_reject = [r for r in pos_recs if r["verdict"] == "insufficient"]
    neg_sufficient = [r for r in neg_recs if r["verdict"] == "sufficient"]
    neg_rejected = [r for r in neg_recs if r["verdict"] in ("insufficient", "gate_rejected")]
    neg_partial = [r for r in neg_recs if r["verdict"] == "partial"]

    summary = {
        "model": cfg.LLM_MODEL,
        "support_judge_enabled": bool(cfg.SUPPORT_JUDGE),
        "n_positives": len(pos_recs),
        "n_negatives": len(neg_recs),
        "positive_not_refused_rate": frac(pos_recs, lambda r: r["verdict"] != "insufficient"),
        "positive_false_reject_rate": frac(pos_recs, lambda r: r["verdict"] == "insufficient"),
        "negative_reject_rate_insufficient_or_gate": frac(neg_recs, lambda r: r["verdict"] in ("insufficient", "gate_rejected")),
        "negative_false_answer_rate_sufficient": frac(neg_recs, lambda r: r["verdict"] == "sufficient"),
        "negative_partial_rate": frac(neg_recs, lambda r: r["verdict"] == "partial"),
        "topic_missing_detail": {
            "n": len([r for r in neg_recs if r["category"] == "topic_missing"]),
            "verdicts": [r["verdict"] for r in neg_recs if r["category"] == "topic_missing"],
        },
        "out_of_domain_detail": {
            "n": len([r for r in neg_recs if r["category"] == "out_of_domain"]),
            "verdicts": [r["verdict"] for r in neg_recs if r["category"] == "out_of_domain"],
        },
        "positive_verdict_counts": {v: sum(1 for r in pos_recs if r["verdict"] == v)
                                    for v in ("sufficient", "partial", "insufficient", "gate_rejected")},
        "positive_false_reject_ids": [r["id"] for r in pos_false_reject],
        "negative_false_answer_ids": [r["id"] for r in neg_sufficient],
        "api_calls": USAGE["calls"],
        "tokens_total": {"prompt": USAGE["prompt_tokens"], "completion": USAGE["completion_tokens"]},
    }

    out = {"summary": summary, "positives": pos_recs, "negatives": neg_recs}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: summary[k] for k in (
        "positive_not_refused_rate", "positive_false_reject_rate",
        "negative_reject_rate_insufficient_or_gate", "negative_false_answer_rate_sufficient",
        "negative_partial_rate", "positive_verdict_counts", "positive_false_reject_ids",
        "negative_false_answer_ids", "api_calls", "tokens_total")}, ensure_ascii=False, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
