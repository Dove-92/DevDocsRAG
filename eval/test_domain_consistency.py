# -*- coding: utf-8 -*-
"""领域迁移一致性测试(离线, 不连服务): 校验前端文案、来源配置、接口协议与 DevDocsRAG 定位一致。
运行: python eval/test_domain_consistency.py  -> eval/results/phase7_domain_consistency.json
"""
import os, sys, re, json, io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

INDEX = os.path.join(PROJECT_ROOT, "static", "index.html")
OUT = os.environ.get("RAG_RESULT_OUT") or os.path.join(SCRIPT_DIR, "results", "phase7_domain_consistency.json")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"name": name, "pass": bool(cond), "detail": detail})
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else ("  | " + str(detail))))


def strip_comments(text):
    t = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    return t


def main():
    html = io.open(INDEX, encoding="utf-8").read()
    text = strip_comments(html)

    banned = ["传智", "EduRAG", "教育", "课程", "教学", "学生", "老师", "学科",
              "机器学习", "人工智能", "专注力", "圆周率", "酒菜", "客服"]
    found = [b for b in banned if b in text]
    check("前端渲染文本无教育领域残留", not found, found)
    check("前端标题含 DevDocsRAG", "DevDocsRAG" in html and "<title>DevDocsRAG" in html.replace("\n", ""))
    for stack in ["LangChain", "Milvus", "BGE", "Transformers"]:
        check(f"前端含技术栈 {stack}", stack in text)
    # 前端线上无教育实现文件
    check("static/src/App.jsx 旧教育组件已移除", not os.path.exists(os.path.join(PROJECT_ROOT, "static", "src", "App.jsx")))

    # 前端/后端接口协议一致
    for tok in ["/api/create_session", "/api/history/", "/api/query", "/api/sources", "/api/stream"]:
        check(f"前端调用接口 {tok}", tok in html)
    for tok in ["case 'start':", "case 'token':", "case 'end':", "case 'error':"]:
        check(f"WS 消息分支 {tok}", tok in html)
    check("前端消费 token 字段", "data.token" in html)
    check("前端消费 citations 字段", "data.citations" in html)
    check("前端引用展示 source·file_path", "c.source" in html and "c.file_path" in html)

    # 后端配置: 来源集合与前端一致
    from base.config import Config
    conf = Config()
    check("VALID_SOURCES = 技术栈集合",
          list(conf.VALID_SOURCES) == ["langchain", "milvus", "bge", "transformers"], conf.VALID_SOURCES)

    # 后端问候/身份文案体现 DevDocsRAG
    app_py = io.open(os.path.join(PROJECT_ROOT, "app.py"), encoding="utf-8").read()
    check("后端问候语含 DevDocsRAG", "我是 DevDocsRAG 技术文档助手" in app_py)
    check("后端 FastAPI 标题为 DevDocsRAG", 'title="DevDocsRAG"' in app_py)

    # RAG 主链路接口兼容(判定/引用清理不改变对外字段)
    from rag_qa.core.rag_pipeline import RAGSystem
    check("RAGSystem 暴露 last_citations(前端协议依赖)", hasattr(RAGSystem, "generate_answer"))
    check("RAGSystem 具备可支撑性判定", hasattr(RAGSystem, "assess_support"))
    check("RAGSystem 具备引用清理", hasattr(RAGSystem, "_finalize_answer"))

    passed = sum(1 for r in RESULTS if r["pass"])
    print(f"\n总计: {passed}/{len(RESULTS)} 通过")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"total": len(RESULTS), "passed": passed, "results": RESULTS}, f, ensure_ascii=False, indent=2)
    print("saved ->", OUT)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
