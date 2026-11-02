# -*- coding: utf-8 -*-
"""Phase9 清理验收：旧 FAQ 数据/代码/配置已移除，公共能力与向量库保留。"""
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("RAG_RESULT_OUT") or os.path.join(ROOT, "eval", "results", "phase9_cleanup.json")
sys.path.insert(0, ROOT)

R = []


def check(name, cond, detail=""):
    R.append({"name": name, "pass": bool(cond), "detail": str(detail)[:500]})
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else ("  | " + str(detail)[:300])))


def read_text(rel):
    with io.open(os.path.join(ROOT, rel), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def py_files(rel_dir):
    out = []
    base = os.path.join(ROOT, rel_dir)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/"))
    return sorted(out)


def main():
    csv_path = os.path.join(ROOT, "mysql_qa", "data", "JP学科知识问答.csv")
    check("旧 FAQ CSV 已删除", not os.path.exists(csv_path), csv_path)
    for rel in ["mysql_qa/cache", "mysql_qa/retrieval", "mysql_qa/utils", "mysql_qa/main.py"]:
        check(f"{rel} 已删除", not os.path.exists(os.path.join(ROOT, rel)), rel)
    for rel in ["eval/audit_faq.py", "eval/test_phase8_faq_off.py"]:
        check(f"{rel} 已删除", not os.path.exists(os.path.join(ROOT, rel)), rel)
    for rel in ["rag_qa/main.py", "rag_qa/core/rag_system.py", "rag_qa/core/rag_query.py"]:
        check(f"{rel} 已删除", not os.path.exists(os.path.join(ROOT, rel)), rel)

    core_expected = sorted([
        "rag_qa/core/build_knowledge_base.py",
        "rag_qa/core/document_processor.py",
        "rag_qa/core/rag_pipeline.py",
        "rag_qa/core/prompts.py",
        "rag_qa/core/query_classifier.py",
        "rag_qa/core/strategy_selector.py",
        "rag_qa/core/vector_store.py",
    ])
    core_actual = sorted(p for p in py_files("rag_qa/core") if ".bak" not in p and ".phase" not in p)
    check("rag_qa/core 仅保留主链路文件", core_actual == core_expected, core_actual)

    for rel in ["README.md", "requirements.txt", "config.example.ini"]:
        check(f"{rel} 已提供", os.path.exists(os.path.join(ROOT, rel)), rel)

    remaining = py_files("mysql_qa")
    check("mysql_qa 仅保留 __init__ 与 db/mysql_client",
          remaining == ["mysql_qa/__init__.py", "mysql_qa/db/mysql_client.py"], remaining)

    mc = read_text("mysql_qa/db/mysql_client.py")
    for token in ["jpkb", "create_table", "insert_data", "fetch_questions", "fetch_answers", "pandas"]:
        check(f"mysql_client 不再包含 {token}", token not in mc, token)

    for rel in ["qa_system.py", "app.py", "base/config.py"]:
        text = read_text(rel)
        for token in ["FAQ_ENABLED", "try_faq_answer", "BM25Search", "RedisClient", "jpkb"]:
            check(f"{rel} 不再引用 {token}", token not in text, token)

    check("config.ini 不再包含 faq_enabled", "faq_enabled" not in read_text("config.ini"), "faq_enabled")

    import pymysql
    from base.config import Config
    cfg = Config()
    check("Config 不再暴露 FAQ_ENABLED", not hasattr(cfg, "FAQ_ENABLED"), "FAQ_ENABLED")

    mysql_ok = False
    try:
        conn = pymysql.connect(host=cfg.MYSQL_HOST, user=cfg.MYSQL_USER,
                               password=cfg.MYSQL_PASSWORD, database=cfg.MYSQL_DATABASE)
        cur = conn.cursor()
        cur.execute("SHOW TABLES")
        tables = [r[0] for r in cur.fetchall()]
        check("MySQL jpkb 表已删除", "jpkb" not in tables, tables)
        check("MySQL conversations 表保留", "conversations" in tables, tables)
        if "conversations" in tables:
            cur.execute("SELECT COUNT(*) FROM conversations")
            check("MySQL conversations 可查询", True, cur.fetchone()[0])
        cur.close(); conn.close()
        mysql_ok = True
    except Exception as e:
        check("MySQL 清理校验", False, repr(e))

    try:
        import redis as redislib
        r = redislib.StrictRedis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT,
                                 password=cfg.REDIS_PASSWORD, db=cfg.REDIS_DB,
                                 decode_responses=True, socket_connect_timeout=3)
        check("Redis qa_original_questions 已删除", r.exists("qa_original_questions") == 0)
        check("Redis qa_tokenized_questions 已删除", r.exists("qa_tokenized_questions") == 0)
        answer_keys = list(r.scan_iter("answer:*"))
        check("Redis answer:* 已清空", len(answer_keys) == 0, answer_keys[:5])
        r.close()
    except Exception as e:
        check("Redis 清理校验", False, repr(e))

    try:
        from pymilvus import connections, utility
        connections.connect(alias="p9check", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT,
                            db_name=cfg.MILVUS_DATABASE_NAME)
        cols = sorted(utility.list_collections(using="p9check"))
        check("Milvus devdocs_final 保留", "devdocs_final" in cols, cols)
        check("Milvus edurag_final 保留", "edurag_final" in cols, cols)
        if "devdocs_final" in cols:
            from pymilvus import Collection
            n = Collection("devdocs_final", using="p9check").num_entities
            check("Milvus devdocs_final 非空", n > 0, n)
        connections.disconnect("p9check")
    except Exception as e:
        check("Milvus 保留校验", False, repr(e))

    check("Phase8 审计结果保留", os.path.exists(os.path.join(ROOT, "eval", "results", "phase8_faq_audit.json")))
    check("FAQ 路由用例保留", os.path.exists(os.path.join(ROOT, "eval", "faq_routing_cases.jsonl")))

    passed = sum(1 for r in R if r["pass"])
    print(f"\n总计: {passed}/{len(R)} 通过")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"total": len(R), "passed": passed, "results": R}, f, ensure_ascii=False, indent=2)
    print("saved ->", OUT)
    return 0 if passed == len(R) else 1


if __name__ == "__main__":
    sys.exit(main())
