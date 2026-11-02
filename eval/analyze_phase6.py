# -*- coding: utf-8 -*-
"""Phase6 离线分析(无 API)：可支撑性判定收益 + 引用可靠性。

输入：results/phase6_supportability.json（判定结果）、results/phase6_e2e.json（端到端含引用/上下文）。
输出：results/phase6_analysis.json
- baseline(Phase5 仅分数门控) vs Phase6(门控+可支撑性判定) 对负例/正例的对照。
- 引用：越界/空上下文虚假引用检测；基于词元重叠的"结论是否有对应文档支持"启发式(受限，见 note)。
"""
import os, sys, json, io, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "results", "phase6_analysis.json")
SUP = os.path.join(SCRIPT_DIR, "results", "phase6_supportability.json")
E2E = os.path.join(SCRIPT_DIR, "results", "phase6_e2e.json")

CITE_RE = re.compile(r'[\[【]\s*来源\s*(\d+)\s*[\]】]')
TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9_.\-]{2,}')


def split_sentences(text):
    parts = re.split(r'(?<=[。！？!?；;\n])', text or "")
    return [p.strip() for p in parts if p and p.strip()]


def lex_support_answer(answer, ctx_docs):
    """对每个含引用的句子，取其引用文档 head 内容做技术词元重叠；无重叠则标注为"待人工复核"。"""
    doc_heads = [d.get("head", "") for d in ctx_docs]
    sentences = []
    unsupported = 0
    cited_sentences = 0
    for s in split_sentences(answer):
        marks = [int(x) for x in CITE_RE.findall(s)]
        if not marks:
            continue
        cited_sentences += 1
        toks = set(t.lower() for t in TOKEN_RE.findall(s))
        if not toks:
            overlap = None
        else:
            overlap = 0
            for k in marks:
                if 1 <= k <= len(doc_heads):
                    doc = set(t.lower() for t in TOKEN_RE.findall(doc_heads[k - 1]))
                    overlap += len(toks & doc)
        flagged = bool(toks) and overlap == 0
        if flagged:
            unsupported += 1
        sentences.append({"sentence": s[:200], "marks": marks, "tech_tokens": len(toks),
                          "overlap_with_cited_doc": overlap, "flagged_unsupported": flagged})
    return {"sentences": sentences, "cited_sentences": cited_sentences,
            "flagged_unsupported": unsupported}


def main():
    sup = json.load(io.open(SUP, encoding="utf-8"))
    e2e = json.load(io.open(E2E, encoding="utf-8"))

    pos = sup["positives"]
    neg = sup["negatives"]

    def cnt(items, pred):
        return sum(1 for x in items if pred(x))

    # 负例：baseline(仅门控) vs Phase6(门控+判定)
    base_false_answer = cnt(neg, lambda r: r["verdict"] != "gate_rejected")  # 仅门控：非 gate 拒绝即会被作答
    p6_rejected = cnt(neg, lambda r: r["verdict"] in ("insufficient", "gate_rejected"))
    p6_partial = cnt(neg, lambda r: r["verdict"] == "partial")
    p6_false_answer = cnt(neg, lambda r: r["verdict"] == "sufficient")

    # 正例：误拒
    base_false_reject = cnt(pos, lambda r: r["verdict"] == "gate_rejected")
    p6_false_reject = cnt(pos, lambda r: r["verdict"] == "insufficient")

    # 引用分析(基于 M=2 记录，避免与 M=3 重复)
    cite_agg = {"answers": 0, "marks_out_of_range": 0, "answers_marks_but_no_cites": 0,
                "answers_cites_but_no_marks": 0, "cited_sentences": 0, "flagged_unsupported": 0,
                "cite_in_range_all": True, "per_answer": []}
    for r in e2e["results"]:
        if r["M"] != 2:
            continue
        ans = r.get("answer", "")
        cits = r.get("ctx_files", [])  # = 有效引用 file_path 列表
        marks = [int(x) for x in CITE_RE.findall(ans)]
        oob = [k for k in marks if not (1 <= k <= len(cits))]
        cite_agg["answers"] += 1
        cite_agg["marks_out_of_range"] += len(oob)
        if oob:
            cite_agg["cite_in_range_all"] = False
        if marks and not cits:
            cite_agg["answers_marks_but_no_cites"] += 1
        if cits and not marks:
            cite_agg["answers_cites_but_no_marks"] += 1
        lex = lex_support_answer(ans, r.get("ctx_docs", []))
        cite_agg["cited_sentences"] += lex["cited_sentences"]
        cite_agg["flagged_unsupported"] += lex["flagged_unsupported"]
        cite_agg["per_answer"].append({"id": r["id"], "M": r["M"], "n_citations": len(cits),
                                       "marks": marks, "oob_marks": oob,
                                       "cited_sentences": lex["cited_sentences"],
                                       "flagged_unsupported": lex["flagged_unsupported"],
                                       "verdict": r.get("verdict")})

    analysis = {
        "negative_comparison": {
            "n": len(neg),
            "baseline_gate_only_false_answer": base_false_answer,
            "phase6_rejected": p6_rejected,
            "phase6_partial": p6_partial,
            "phase6_false_answer": p6_false_answer,
            "by_category": {
                cat: {
                    "n": cnt([r for r in neg if r["category"] == cat], lambda r: True),
                    "baseline_false_answer": cnt([r for r in neg if r["category"] == cat], lambda r: r["verdict"] != "gate_rejected"),
                    "phase6_rejected": cnt([r for r in neg if r["category"] == cat], lambda r: r["verdict"] in ("insufficient", "gate_rejected")),
                    "phase6_partial": cnt([r for r in neg if r["category"] == cat], lambda r: r["verdict"] == "partial"),
                    "phase6_false_answer": cnt([r for r in neg if r["category"] == cat], lambda r: r["verdict"] == "sufficient"),
                } for cat in ("topic_missing", "out_of_domain")
            },
        },
        "positive_comparison": {
            "n": len(pos),
            "baseline_gate_only_false_reject": base_false_reject,
            "phase6_false_reject": p6_false_reject,
            "phase6_false_reject_ids": [r["id"] for r in pos if r["verdict"] == "insufficient"],
        },
        "citation": cite_agg,
        "note": ("引用'结论支持'采用技术词元与引用文档 head(前400字符)的重叠启发式，"
                 "仅用于发现明显的引用-内容不匹配；head 截断会造成漏检(召回偏低)，命中重叠不代表语义支持。"),
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: analysis[k] for k in ("negative_comparison", "positive_comparison")},
                     ensure_ascii=False, indent=2))
    print("citation:", json.dumps({k: v for k, v in cite_agg.items() if k != "per_answer"}, ensure_ascii=False))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
