# -*- coding: utf-8 -*-
"""Phase8 运行期验证：faq_enabled=false 时旧 FAQ 被跳过，HTTP/WS 均进入 RAG。

覆盖：中文技术问题、英文技术问题、旧课程陷阱(含旧答案签名)、词面陷阱、
      无可靠答案、WS start/token/end+citations、错误输入、问候与来源接口。
输出: eval/results/phase8_runtime.json
"""
import os, sys, json, io, re, asyncio, time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RAG_RESULT_OUT") or os.path.join(SCRIPT_DIR, "results", "phase8_runtime.json")

import requests
import websockets

HOST = os.environ.get("RAG_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAG_PORT", "8081"))
BASE = f"http://{HOST}:{PORT}"
WS = f"ws://{HOST}:{PORT}/api/stream"
CITE_RE = re.compile(r"[\[【]\s*来源\s*(\d+)\s*[\]】]")
OLD_SIGNATURE = "idea.imsxm.com"

R = []


def check(name, cond, detail=""):
    R.append({"name": name, "pass": bool(cond), "detail": str(detail)[:500]})
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else ("  | " + str(detail)[:300])))


async def ws_query(payload, timeout=180):
    msgs = []
    async with websockets.connect(WS, max_size=None, open_timeout=20, ping_interval=None) as ws:
        await ws.send(json.dumps(payload))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            m = json.loads(raw)
            msgs.append(m)
            if m.get("type") in ("end", "error"):
                break
    return msgs


def parse(msgs):
    types = [m.get("type") for m in msgs]
    answer = "".join(m.get("token", "") for m in msgs if m.get("type") == "token")
    end = next((m for m in msgs if m.get("type") == "end"), {})
    return types, answer, end


def protocol_ok(name, msgs):
    types, answer, end = parse(msgs)
    check(f"{name}: 首帧 start", types[:1] == ["start"], types[:3])
    check(f"{name}: 末帧 end", types[-1:] == ["end"], types[-3:])
    check(f"{name}: 中间仅 token", all(t == "token" for t in types[1:-1]), types[:10])
    check(f"{name}: end.is_complete", end.get("is_complete") is True, end)
    check(f"{name}: end.processing_time", isinstance(end.get("processing_time"), (int, float)), end)
    check(f"{name}: end.citations 字段", isinstance(end.get("citations"), list), end)
    return types, answer, end


def refs_in_range(answer, citations):
    refs = [int(x) for x in CITE_RE.findall(answer)]
    if not refs:
        return True, []
    return all(1 <= r <= len(citations) for r in refs), refs


def main():
    # 0. health
    try:
        h = requests.get(f"{BASE}/health", timeout=10)
        check("GET /health healthy", h.status_code == 200 and h.json().get("status") == "healthy", h.text[:120])
    except Exception as e:
        check("GET /health healthy", False, f"服务不可用: {e}")
        _save(); return 1

    # 1. 来源
    src = requests.get(f"{BASE}/api/sources", timeout=10).json()
    check("GET /api/sources 不变", src.get("sources") == ["langchain", "milvus", "bge", "transformers"], src)

    # 2. 问候（非 FAQ、非 RAG）
    g = requests.post(f"{BASE}/api/query", json={"query": "你好"}, timeout=20).json()
    check("HTTP 问候: 非流式", g.get("is_streaming") is False, g)
    check("HTTP 问候: 含 DevDocsRAG", "DevDocsRAG" in g.get("answer", ""), g.get("answer", "")[:120])

    # 3. HTTP /api/query: FAQ 关闭后不返回旧答案，而是转 WS/RAG
    for name, q in [("中文技术问题", "Milvus 如何创建 Collection？"),
                    ("旧课程陷阱", "pycharm专业版如何激活")]:
        try:
            r = requests.post(f"{BASE}/api/query", json={"query": q}, timeout=20).json()
            check(f"HTTP {name}: is_streaming=true", r.get("is_streaming") is True, r)
            check(f"HTTP {name}: 返回转流式提示", r.get("answer") == "请使用WebSocket接口获取流式响应", r)
            check(f"HTTP {name}: 未返回旧 FAQ 答案", OLD_SIGNATURE not in str(r), r.get("answer", "")[:120])
        except Exception as e:
            check(f"HTTP {name}: 请求成功", False, e)

    # 4. WS 中文技术问题
    msgs = asyncio.run(ws_query({"query": "Milvus 如何创建 Collection？", "source_filter": "milvus"}))
    types, ans, end = protocol_ok("WS 中文技术问题", msgs)
    cites = end.get("citations", [])
    check("WS 中文技术问题: 有 token", bool(ans.strip()), ans[:120])
    check("WS 中文技术问题: 非旧答案", OLD_SIGNATURE not in ans, ans[:120])
    check("WS 中文技术问题: 有引用", len(cites) > 0, cites)
    check("WS 中文技术问题: 引用字段完整",
          all(("source" in c and "file_path" in c) for c in cites), cites[:2])
    ok_refs, refs = refs_in_range(ans, cites)
    check("WS 中文技术问题: [来源k] 在范围内", ok_refs, {"refs": refs, "citations": len(cites)})

    # 5. WS 英文技术问题（旧 FAQ 低分分支应不再提前拦截）
    msgs = asyncio.run(ws_query({"query": "LangChain document loaders", "source_filter": "langchain"}))
    types, ans, end = protocol_ok("WS 英文技术问题", msgs)
    check("WS 英文技术问题: 非 '未找到答案'", ans.strip() != "未找到答案", ans[:120])
    check("WS 英文技术问题: 走 RAG(耗时>3s)", float(end.get("processing_time", 0)) > 3.0,
          end.get("processing_time"))
    check("WS 英文技术问题: 有 token", bool(ans.strip()), ans[:120])

    # 6. WS 旧课程陷阱
    msgs = asyncio.run(ws_query({"query": "pycharm专业版如何激活"}))
    types, ans, end = protocol_ok("WS 旧课程陷阱", msgs)
    check("WS 旧课程陷阱: 未返回旧 FAQ 答案", OLD_SIGNATURE not in ans, ans[:200])
    check("WS 旧课程陷阱: 走 RAG(耗时>3s)", float(end.get("processing_time", 0)) > 3.0,
          end.get("processing_time"))
    refusal = (
        "依据不足" in ans
        or "上下文为空" in ans
        or ("无相关" in ans and ("文档" in ans or "依据" in ans))
        or (("不属于" in ans or "超出" in ans or "不在" in ans)
            and ("范围" in ans or "领域" in ans))
    )
    check("WS 旧课程陷阱: 拒答/说明无依据", refusal, ans[:200])
    check("WS 旧课程陷阱: 无引用", len(end.get("citations", [])) == 0, end.get("citations"))
    check("WS 旧课程陷阱: 无伪造引用标记", not CITE_RE.search(ans), CITE_RE.findall(ans))

    # 7. WS 词面相似但语义不同
    msgs = asyncio.run(ws_query({"query": "Milvus 专业版如何激活？"}))
    types, ans, end = protocol_ok("WS 词面陷阱", msgs)
    check("WS 词面陷阱: 未返回旧 FAQ 答案", OLD_SIGNATURE not in ans, ans[:200])
    check("WS 词面陷阱: 走 RAG(耗时>3s)", float(end.get("processing_time", 0)) > 3.0,
          end.get("processing_time"))

    # 8. WS 无可靠答案
    msgs = asyncio.run(ws_query({"query": "DevDocsRAG 的创始人是谁？"}))
    types, ans, end = protocol_ok("WS 无可靠答案", msgs)
    check("WS 无可靠答案: 依据不足", "依据不足" in ans, ans[:200])
    check("WS 无可靠答案: 0 引用", len(end.get("citations", [])) == 0, end.get("citations"))
    check("WS 无可靠答案: 非 '未找到答案'", ans.strip() != "未找到答案", ans[:120])

    # 9. WS 错误输入
    try:
        async def raw():
            async with websockets.connect(WS, max_size=None, open_timeout=15, ping_interval=None) as ws:
                await ws.send("这不是JSON")
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=30)
                return json.loads(raw_msg)
        err = asyncio.run(raw())
        check("WS 非法输入返回 error", err.get("type") == "error", err)
    except Exception as e:
        check("WS 非法输入返回 error", False, e)

    passed = sum(1 for r in R if r["pass"])
    print(f"\n总计: {passed}/{len(R)} 通过")
    _save()
    return 0 if passed == len(R) else 1


def _save():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"host": BASE, "total": len(R), "passed": sum(1 for r in R if r["pass"]), "results": R},
                  f, ensure_ascii=False, indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
