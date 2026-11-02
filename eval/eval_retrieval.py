# -*- coding: utf-8 -*-
"""DevDocsRAG 检索评估：Recall@5 / MRR@10（只处理已确认且 expected_docs 非空的样本）。

- 检索单元：file_path（即 "source:file_path"）。
- Recall@5：top-5 检索结果是否命中任一 expected_docs。
- MRR@10：top-10 中首个命中 expected_docs 的排名的倒数。
- 排名来源：retrieve_top_k 返回的混合检索融合分数（dense IP + sparse IP，WeightedRanker 1.0/0.7），distance 降序，rank 从 1 开始。
"""
import os, sys, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from rag_qa.core.vector_store import VectorStore

EVAL_PATH = os.path.join(SCRIPT_DIR, "devdocs_eval_set_v1.jsonl")
K = 10
OUT_DIR = os.path.join(SCRIPT_DIR, "results")


def load_eval(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    samples = load_eval(EVAL_PATH)
    eval_samples = [s for s in samples if not s["needs_review"] and s["expected_docs"]]
    skipped = len(samples) - len(eval_samples)

    print(f"评估集总数={len(samples)} | 参与评估(已确认且有expected_docs)={len(eval_samples)} | 跳过={skipped}")

    vs = VectorStore()
    vs.retrieve_top_k("warmup", k=3)  # 预热，避免冷启动影响

    recall_hits = 0
    mrr_sum = 0.0
    results = []

    for s in eval_samples:
        top = vs.retrieve_top_k(s["query"], k=K)
        expected = set(s["expected_docs"])

        hit_ranks = []
        for x in top:
            key = f"{x['source']}:{x['file_path']}"
            if key in expected:
                hit_ranks.append(x['rank'])

        top5_keys = [f"{x['source']}:{x['file_path']}" for x in top[:5]]
        rec5 = any(k in expected for k in top5_keys)
        first_rank = hit_ranks[0] if hit_ranks else None
        rr = 1.0 / first_rank if first_rank else 0.0

        if rec5:
            recall_hits += 1
        mrr_sum += rr

        results.append({
            "id": s["id"],
            "query": s["query"],
            "expected_docs": s["expected_docs"],
            "hit_ranks": hit_ranks,
            "recall5": rec5,
            "rr": round(rr, 4),
            "top10": [f"{x['source']}:{x['file_path']}" for x in top[:10]],
        })

    n = len(eval_samples)
    recall5 = recall_hits / n if n else 0.0
    mrr10 = mrr_sum / n if n else 0.0

    print(f"\nRecall@5 = {recall5:.4f} ({recall_hits}/{n})")
    print(f"MRR@10   = {mrr10:.4f}")
    print(f"\n[注意] 参与评估样本仅 {n} 条，结果仅供调试参考，不代表系统整体检索性能。")

    print("\n逐题结果:")
    for r in results:
        print(f"  [{r['id']}] {r['query']}")
        print(f"    命中排名={r['hit_ranks']} | Recall@5={r['recall5']} | RR={r['rr']}")
        print(f"    expected={r['expected_docs']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "retrieval_debug.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"k": K, "ranker": "WeightedRanker(1.0,0.7)", "metric": "IP",
                       "collection": "devdocs_final", "eval_set": "devdocs_eval_set_v1.jsonl"},
            "metrics": {"recall5": recall5, "mrr10": mrr10, "n": n, "skipped": skipped},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已保存: {out}")


if __name__ == "__main__":
    main()
