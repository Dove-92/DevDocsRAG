# -*- coding: utf-8 -*-
"""Phase7 运行期验证：对已启动的服务做真实 HTTP + WebSocket 端到端检查(非浏览器)。

前置: 在另一进程启动 `python app.py` (默认 127.0.0.1:8081)。
检查: 首页标题/定位、api/sources、api/create_session、api/query(问候/普通)、
      WS 流式(start/token/end+citations)、不可回答问题拒答、错误分支。
输出: eval/results/phase7_runtime.json
"""
import os, sys, json, re, io, time, asyncio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT = os.environ.get("RAG_RESULT_OUT") or os.path.join(SCRIPT_DIR, "results", "phase7_runtime.json")

import requests
import websockets

HOST = os.environ.get("RAG_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAG_PORT", "8081"))
BASE = f"http://{HOST}:{PORT}"
WS = f"ws://{HOST}:{PORT}/api/stream"
CITE_RE = re.compile(r"[\[【]\s*来源\s*(\d+)\s*[\]】]")

R = []


def check(name, cond, detail=""):
    R.append({"name": name, "pass": bool(cond), "detail": str(detail)[:300]})
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else ("  | " + str(detail)[:200])))


async def ws_query(payload, timeout=180):
    """发送一条消息，收集所有 WS 消息直到收到 end 或 error。"""
    msgs = []
    async with websockets.connect(WS, max_size=None, open_timeout=15, ping_interval=None) as ws:
        await ws.send(json.dumps(payload))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            m = json.loads(raw)
            msgs.append(m)
            if m.get("type") in ("end", "error"):
                break
    return msgs


async def ws_raw(raw_text, timeout=30):
    async with websockets.connect(WS, max_size=None, open_timeout=15, ping_interval=None) as ws:
        await ws.send(raw_text)
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            return json.loads(raw)
        except asyncio.TimeoutError:
            return {"type": "timeout"}


def main():
    # 0. 健康
    try:
        h = requests.get(f"{BASE}/health", timeout=10)
        check("GET /health -> 200 healthy", h.status_code == 200 and h.json().get("status") == "healthy", h.text[:120])
    except Exception as e:
        check("GET /health -> 200 healthy", False, f"服务不可用: {e}")
        _save(); return 1

    # 1. 首页
    home = requests.get(f"{BASE}/", timeout=10).text
    check("首页 <title> 为 DevDocsRAG", "<title>DevDocsRAG" in home)
    check("首页包含技术栈文案", all(k in home for k in ("LangChain", "Milvus", "BGE", "Transformers")))
    check("首页不再出现教育标识", "传智" not in home and "EduRAG" not in home)

    # 2. 来源
    src = requests.get(f"{BASE}/api/sources", timeout=10).json()
    check("GET /api/sources 返回技术栈", src.get("sources") == ["langchain", "milvus", "bge", "transformers"], src)

    # 3. 会话
    s = requests.post(f"{BASE}/api/create_session", timeout=10).json()
    sid = s.get("session_id")
    check("POST /api/create_session 返回 session_id", bool(sid), s)

    # 4. 非流式：问候
    g = requests.post(f"{BASE}/api/query", json={"query": "你好", "session_id": sid}, timeout=15).json()
    check("问候走非流式且含 DevDocsRAG", g.get("is_streaming") is False and "DevDocsRAG" in (g.get("answer") or ""), g)

    # 5. 非流式：普通问题 -> 应提示使用 WebSocket
    q = requests.post(f"{BASE}/api/query", json={"query": "Milvus 怎么创建 Collection？", "session_id": sid}, timeout=20).json()
    check("新问题 is_streaming=True(转 WS)", q.get("is_streaming") is True, q)

    # 6. WS 流式：可回答的技术问题
    t0 = time.time()
    try:
        msgs = asyncio.run(ws_query({"query": "Milvus 怎么创建 Collection？", "source_filter": None, "session_id": sid}))
    except Exception as e:
        msgs = []
        check("WS 可回答问题返回 end", False, e)
    if msgs:
        types = [m.get("type") for m in msgs]
        check("WS 序列含 start", "start" in types, types[:5])
        check("WS 含 token 帧", "token" in types)
        end = next((m for m in msgs if m.get("type") == "end"), None)
        check("WS 以 end 结束", end is not None)
        answer = "".join(m.get("token", "") for m in msgs if m.get("type") == "token")
        cites = (end or {}).get("citations") or []
        check("可回答问题产生引用(n>0)", len(cites) > 0, f"citations={len(cites)}")
        check("引用字段含 source/file_path", all(c.get("source") and c.get("file_path") for c in cites), cites[:2])
        marks = [int(x) for x in CITE_RE.findall(answer)]
        check("回答引用编号均在有效范围", all(1 <= k <= len(cites) for k in marks), f"marks={marks} n={len(cites)}")
        check("end 含 processing_time", "processing_time" in (end or {}))
        print(f"    [info] 可回答问题 answer_chars={len(answer)} citations={len(cites)} ws_elapsed={time.time()-t0:.1f}s")
    else:
        for n in ["WS 序列含 start", "WS 含 token 帧", "WS 以 end 结束", "可回答问题产生引用(n>0)"]:
            check(n, False, "no msgs")

    # 7. WS 流式：知识库无法充分回答 -> 拒答/无引用
    try:
        msgs2 = asyncio.run(ws_query({"query": "DevDocsRAG 的创始人是谁？", "session_id": sid}))
    except Exception as e:
        msgs2 = []
        check("WS 不可回答问题正常结束", False, e)
    if msgs2:
        end2 = next((m for m in msgs2 if m.get("type") == "end"), None)
        ans2 = "".join(m.get("token", "") for m in msgs2 if m.get("type") == "token")
        cites2 = (end2 or {}).get("citations") or []
        marks2 = CITE_RE.findall(ans2)
        check("不可回答: 无引用", len(cites2) == 0, f"citations={len(cites2)}")
        check("不可回答: 无虚假引用标记", len(marks2) == 0, f"marks={marks2}")
        refusal = (
            "依据不足" in ans2
            or "上下文为空" in ans2
            or ("无相关" in ans2 and ("文档" in ans2 or "依据" in ans2))
            or (("不属于" in ans2 or "超出" in ans2 or "不在" in ans2)
                and ("范围" in ans2 or "领域" in ans2))
        )
        check("不可回答: 拒答/说明无依据", refusal, ans2[:80])
        print(f"    [info] 不可回答 answer_head={ans2[:80]!r}")

    # 8. 错误分支: 非法 JSON
    err = asyncio.run(ws_raw("这不是JSON"))
    check("WS 非法输入返回 error", err.get("type") == "error", err)

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
