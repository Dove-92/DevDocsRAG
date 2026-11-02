# -*- coding: utf-8 -*-
"""DevDocsRAG 回答/引用/性能/成本评估（真实 DashScope 主链路，非 Mock）。

- 仅评估已确认(needs_review=false)且 answer_points/expected_docs 非空的样本。
- 记录：问题、回答、实际引用、检索来源、端到端/检索/生成耗时、token usage(当前接口无法获取则标记)。
- 自动规则与人工判断分开记录；不用 LLM 自评当客观指标。
"""
import os, sys, json, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from base.config import Config
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_pipeline import RAGSystem

EVAL_PATH = os.path.join(SCRIPT_DIR, "devdocs_eval_set_v1.jsonl")
DATA_DIR = os.path.join(PROJECT_ROOT, "rag_qa", "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "results")

cfg = Config()
client = OpenAI(api_key=cfg.DASHSCOPE_API_KEY, base_url=cfg.DASHSCOPE_BASE_URL)


def call_dashscope(prompt):
    """与 qa_system.call_dashscope 相同的流式 LLM 调用(相同 system prompt / 参数)。"""
    completion = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是 DevDocsRAG 技术文档智能问答助手，面向开发者解答 LangChain、Milvus、BGE/FlagEmbedding、Hugging Face Transformers 的技术问题。请严格依据提供的上下文回答，不要编造 API、参数或版本行为。"},
            {"role": "user", "content": prompt},
        ],
        timeout=30,
        stream=True,
    )
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def load_eval(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def path_exists(source, file_path):
    return os.path.exists(os.path.join(DATA_DIR, f"{source}_data", file_path.replace("/", os.sep)))


def auto_rules(sample, answer, citations, top_sources):
    r = {}
    r["answer_nonempty"] = bool(answer and answer.strip())
    r["citations_nonempty"] = bool(citations)
    r["citation_fields_ok"] = all(c.get("source") and c.get("file_path") for c in citations)
    r["citation_paths_exist"] = all(path_exists(c["source"], c["file_path"]) for c in citations) if citations else None
    r["citation_in_retrieved_or_expected"] = None
    if citations:
        expected = set(sample["expected_docs"])
        ok = all(f"{c['source']}:{c['file_path']}" in expected or f"{c['source']}:{c['file_path']}" in set(top_sources)
                 for c in citations)
        r["citation_in_retrieved_or_expected"] = ok
    return r


def main():
    samples = load_eval(EVAL_PATH)
    eval_samples = [s for s in samples if not s["needs_review"] and s["expected_docs"] and s["answer_points"]]
    skipped = len(samples) - len(eval_samples)
    print(f"评估集={len(samples)} | 参与回答评估={len(eval_samples)} | 跳过={skipped}")

    vs = VectorStore()
    rag = RAGSystem(vs, call_dashscope)
    vs.retrieve_top_k("warmup", k=3)  # 预热检索
    rag.query_classifier.predict_category("warmup")  # 预热分类器

    api_calls = 0  # 估算: 每样本 1 次策略选择 + 1 次生成

    results = []
    failures = []
    for s in eval_samples:
        q = s["query"]
        rec = {"id": s["id"], "query": q, "type": s["type"], "docset": s["docset"],
               "expected_docs": s["expected_docs"], "answer_points": s["answer_points"],
               "citation_criteria": s["citation_criteria"]}

        # 检索耗时(单独只读, 不调 LLM)
        t0 = time.time()
        top = vs.retrieve_top_k(q, k=10)
        rec["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        rec["retrieval_sources"] = [f"{x['source']}:{x['file_path']}" for x in top[:10]]

        # 主链路
        try:
            t0 = time.time()
            gen = rag.generate_answer(q, source_filter=None)
            rec["pre_generation_ms"] = round((time.time() - t0) * 1000, 1)
            t1 = time.time()
            answer = "".join(gen)
            rec["generation_ms"] = round((time.time() - t1) * 1000, 1)
            rec["e2e_ms"] = round(rec["pre_generation_ms"] + rec["generation_ms"], 1)
            rec["answer"] = answer
            rec["citations"] = rag.last_citations
            rec["token_usage"] = None  # 当前流式接口未启用 usage 统计
            rec["auto_rules"] = auto_rules(s, answer, rag.last_citations, rec["retrieval_sources"])
            api_calls += 2  # 策略选择 + 生成
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            failures.append(rec)
            api_calls += 1  # 策略选择可能已发生
        results.append(rec)

    # 汇总自动规则
    def stat(key):
        vals = [r["auto_rules"].get(key) for r in results if "auto_rules" in r and r["auto_rules"].get(key) is not None]
        ok = sum(1 for v in vals if v)
        return f"{ok}/{len(vals)}" if vals else "N/A"

    summary = {
        "model": cfg.LLM_MODEL,
        "collection": "devdocs_final",
        "n": len(eval_samples),
        "skipped": skipped,
        "api_calls_estimate": api_calls,
        "auto_metrics": {
            "answer_nonempty": stat("answer_nonempty"),
            "citations_nonempty": stat("citations_nonempty"),
            "citation_fields_ok": stat("citation_fields_ok"),
            "citation_paths_exist": stat("citation_paths_exist"),
            "citation_in_retrieved_or_expected": stat("citation_in_retrieved_or_expected"),
        },
        "failures": len(failures),
    }

    print(f"\n自动规则汇总: {json.dumps(summary['auto_metrics'], ensure_ascii=False)}")
    print(f"失败样本: {len(failures)}")
    print(f"\n[注意] 参与样本仅 {len(eval_samples)} 条，结果仅供调试参考，不代表整体性能。token usage 无法获取(流式接口未启用 include_usage)。")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "answer_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"原始结果已保存: {out}")


if __name__ == "__main__":
    main()
