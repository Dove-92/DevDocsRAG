# -*- coding: utf-8 -*-
"""Phase6 Task3+Task4：CANDIDATE_M(2 vs 3) 对比 + 端到端性能(延迟 P50/P95、LLM 调用、token、引用有效性)。

真实 API 主链路；每个查询在 M=2 与 M=3 下各跑一次 generate_answer。
- 预热冷启动 -> 正式测量。
- 记录：e2e/检索/生成耗时、判定/生成调用数、token、引用数量与有效性、上下文文件与冗余。
输出：eval/results/phase6_e2e.json
"""
import os, sys, json, time, io, re, statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from base.config import Config
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_pipeline import RAGSystem, sanitize_citations
import rag_qa.core.rag_pipeline as nrs

EVAL_PATH = os.path.join(SCRIPT_DIR, "devdocs_eval_set_v1.jsonl")
OUT = os.path.join(SCRIPT_DIR, "results", "phase6_e2e.json")
OUT_M = os.path.join(SCRIPT_DIR, "results", "phase6_m_compare.json")

SUBSET = ["q022", "q035", "q039", "q006", "q008", "q011", "q057"]
JUDGE_MARK = "检索证据充分性判定器"

cfg = Config()
client = OpenAI(api_key=cfg.DASHSCOPE_API_KEY, base_url=cfg.DASHSCOPE_BASE_URL)
STATE = {"calls": 0, "judge": 0, "gen": 0, "pt": 0, "ct": 0}


def call_dashscope(prompt):
    is_judge = JUDGE_MARK in prompt
    STATE["calls"] += 1
    STATE["judge" if is_judge else "gen"] += 1
    completion = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是 DevDocsRAG 技术文档智能问答助手。"},
            {"role": "user", "content": prompt},
        ],
        timeout=60,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in completion:
        u = getattr(chunk, "usage", None)
        if u is not None:
            STATE["pt"] += getattr(u, "prompt_tokens", 0) or 0
            STATE["ct"] += getattr(u, "completion_tokens", 0) or 0
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def load_eval(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def citation_validity(answer, citations, retrieved):
    m = [int(x) for x in re.findall(r'[\[【]\s*来源\s*(\d+)\s*[\]】]', answer or "")]
    in_range = all(1 <= k <= len(citations) for k in m)
    cites_in_retrieved = all(c["file_path"] in set(retrieved) for c in citations)
    return {
        "n_cite_marks": len(m),
        "marks_max": max(m) if m else 0,
        "n_citations": len(citations),
        "cite_in_range": in_range,
        "cite_in_retrieved": cites_in_retrieved,
        "endpoint_in_context": None,
    }


def main():
    rows = load_eval(EVAL_PATH)
    by_id = {r["id"]: r for r in rows}
    subset = [by_id[i] for i in SUBSET if i in by_id]

    vs = VectorStore()
    rag = RAGSystem(vs, call_dashscope)
    vs.retrieve_top_k("warmup", k=3)

    results = []
    # 预热(冷启动)：对同一查询跑一次不入统计
    try:
        vs.retrieve_top_k(subset[0]["query"], k=10)
        list(rag.generate_answer(subset[0]["query"]))
    except Exception as e:
        print("warmup warn:", e)

    for M in (2, 3):
        nrs.conf.CANDIDATE_M = M
        for s in subset:
            q = s["query"]
            expected = set(s.get("expected_docs") or [])
            snap = dict(STATE)
            t0 = time.time()
            gen = rag.generate_answer(q)
            answer = "".join(gen)
            e2e_ms = round((time.time() - t0) * 1000, 1)
            d = {k: STATE[k] - snap[k] for k in STATE}
            cits = rag.last_citations or []
            ctx_files = [c["file_path"] for c in cits]
            ctx_docs = [{"file_path": d.metadata.get("file_path", ""),
                         "source": d.metadata.get("source", ""),
                         "chars": len(d.page_content or ""),
                         "head": (d.page_content or "")[:400]}
                        for d in (rag.last_context_docs or [])]
            rec = {
                "id": s["id"], "query": q, "M": M, "e2e_ms": e2e_ms,
                "verdict": (rag.last_support_verdict or {}).get("verdict"),
                "n_citations": len(cits), "ctx_files": ctx_files,
                "ctx_dup_files": len(ctx_files) - len(set(ctx_files)),
                "expected_docs": sorted(expected),
                "expected_hit_in_ctx": len(set(ctx_files) & set(f.split(":", 1)[1] for f in expected)),
                "llm": d, "tokens": {"prompt": d["pt"], "completion": d["ct"]},
                "citation_validity": citation_validity(answer, cits, ctx_files),
                "ctx_docs": ctx_docs,
                "judge_reason": (rag.last_support_verdict or {}).get("reason"),
                "judge_missing": (rag.last_support_verdict or {}).get("missing"),
                "answer_chars": len(answer),
                "answer": answer,
            }
            results.append(rec)
            print("M=%d %-6s %6.1fms judge=%d gen=%d cites=%d %s" % (
                M, s["id"], e2e_ms, d["judge"], d["gen"], len(cits), rec["verdict"]), flush=True)

    def agg(M):
        rs = [r for r in results if r["M"] == M]
        e2 = [r["e2e_ms"] for r in rs]
        return {
            "M": M, "n": len(rs),
            "e2e_ms_avg": round(statistics.mean(e2), 1),
            "e2e_ms_p50": pctl(e2, 50), "e2e_ms_p95": pctl(e2, 95),
            "citations_total": sum(r["n_citations"] for r in rs),
            "ctx_dup_files_total": sum(r["ctx_dup_files"] for r in rs),
            "expected_hit_in_ctx_total": sum(r["expected_hit_in_ctx"] for r in rs),
            "cite_in_range_all": all(r["citation_validity"]["cite_in_range"] for r in rs),
            "cite_in_retrieved_all": all(r["citation_validity"]["cite_in_retrieved"] for r in rs),
            "llm_calls": sum(r["llm"]["calls"] for r in rs),
            "tokens_prompt": sum(r["tokens"]["prompt"] for r in rs),
            "tokens_completion": sum(r["tokens"]["completion"] for r in rs),
            "verdicts": {v: sum(1 for r in rs if r["verdict"] == v) for v in
                         ("sufficient", "partial", "insufficient", None)},
        }

    summary = {
        "model": cfg.LLM_MODEL, "support_judge": bool(cfg.SUPPORT_JUDGE),
        "relevance_threshold": cfg.RELEVANCE_THRESHOLD,
        "subset": SUBSET, "note": "n 较小，P95 为小样本估计；冷启动已由 warmup 排除",
        "M2": agg(2), "M3": agg(3),
        "total_api_calls": STATE["calls"],
        "total_tokens": {"prompt": STATE["pt"], "completion": STATE["ct"]},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    with io.open(OUT_M, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps({"M2": summary["M2"], "M3": summary["M3"],
                             "total_api_calls": summary["total_api_calls"],
                             "total_tokens": summary["total_tokens"]}, ensure_ascii=False, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
