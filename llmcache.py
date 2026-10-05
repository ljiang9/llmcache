"""llmcache — SQLite-backed cache for OpenAI-compatible chat completions.

同样的 prompt 进，缓存的回答出，本次调用 $0 花费。

Library usage::

    import llmcache
    ans, meta = llmcache.chat(
        [{"role": "user", "content": "你好"}],
        model="gpt-4o-mini",
        api_key="sk-...",
    )
    print(ans, meta["cached"])  # False on first call, True on repeats

CLI usage::

    llmcache ask "天空为什么是蓝色的？" --model gpt-4o-mini

只用 Python 标准库。缓存 key = {model, messages, temperature, ...}
的规范 JSON 的 sha256。
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error

__version__ = "0.1.0"

DEFAULT_DB = os.path.expanduser("~/.cache/llmcache/cache.db")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60

# 每 1M token 的美元价格 (input, output)。粗略公开价，仅用于估算。
PRICE_TABLE = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    "glm-4-flash": (0.10, 0.10),
}


def _estimate_tokens(text):
    """粗略估算 token 数：字符数 / 4（英文偏准，中文偏少）。"""
    return max(1, len(text) // 4)


def _estimate_cost_usd(model, prompt_text, completion_text):
    in_price, out_price = PRICE_TABLE.get(model, (1.00, 3.00))
    in_tok = _estimate_tokens(prompt_text)
    out_tok = _estimate_tokens(completion_text)
    return (in_tok / 1e6) * in_price + (out_tok / 1e6) * out_price


def _canonical_key(model, messages, temperature, max_tokens, extra):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": extra or {},
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _connect(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cache (
               key TEXT PRIMARY KEY,
               model TEXT NOT NULL,
               prompt TEXT NOT NULL,
               response TEXT NOT NULL,
               created_at REAL NOT NULL,
               last_used_at REAL NOT NULL,
               hit_count INTEGER NOT NULL DEFAULT 0
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS stats (
               id INTEGER PRIMARY KEY CHECK (id = 1),
               hits INTEGER NOT NULL DEFAULT 0,
               misses INTEGER NOT NULL DEFAULT 0,
               est_saved_usd REAL NOT NULL DEFAULT 0.0
           )"""
    )
    conn.execute(
        "INSERT OR IGNORE INTO stats (id, hits, misses, est_saved_usd)"
        " VALUES (1, 0, 0, 0.0)"
    )
    conn.commit()
    return conn


def _post_chat(base_url, api_key, payload, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError("API 返回 HTTP %d：%s" % (e.code, body))
    except urllib.error.URLError as e:
        raise RuntimeError("网络请求失败：%s" % e.reason)


def chat(messages, model=DEFAULT_MODEL, temperature=0.0, max_tokens=None,
         base_url=None, api_key=None, db_path=None, ttl=None,
         no_cache=False, timeout=DEFAULT_TIMEOUT, extra=None):
    """发一次 chat 请求，带 SQLite 缓存。

    返回 (answer_text, meta)，meta 包含：
      cached: 是否命中缓存
      key: 缓存 key（sha256）
      est_cost_usd: 本次调用的估算花费（命中时为 0）
      model / latency_s 等
    """
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    db_path = db_path or os.environ.get("LLMCACHE_DB", DEFAULT_DB)

    key = _canonical_key(model, messages, temperature, max_tokens, extra)
    prompt_text = "\n".join(
        m.get("content", "") if isinstance(m.get("content"), str)
        else json.dumps(m.get("content"), ensure_ascii=False)
        for m in messages
    )

    conn = _connect(db_path)
    try:
        if not no_cache:
            row = conn.execute(
                "SELECT response, created_at FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row:
                response_text, created_at = row
                if ttl is None or (time.time() - created_at) <= ttl:
                    conn.execute(
                        "UPDATE cache SET last_used_at = ?, hit_count = hit_count + 1"
                        " WHERE key = ?",
                        (time.time(), key),
                    )
                    saved = _estimate_cost_usd(model, prompt_text, response_text)
                    conn.execute(
                        "UPDATE stats SET hits = hits + 1,"
                        " est_saved_usd = est_saved_usd + ? WHERE id = 1",
                        (saved,),
                    )
                    conn.commit()
                    return response_text, {
                        "cached": True, "key": key, "model": model,
                        "est_cost_usd": 0.0, "est_saved_usd": saved,
                        "latency_s": 0.0,
                    }
                else:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()

        if not api_key:
            raise RuntimeError(
                "未找到 API key。请设置环境变量 OPENAI_API_KEY，"
                "或在调用时传入 api_key=。"
            )
        payload = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)

        t0 = time.time()
        data = _post_chat(base_url, api_key, payload, timeout)
        latency = time.time() - t0

        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("API 返回结构异常：%s" % json.dumps(data)[:300])

        cost = _estimate_cost_usd(model, prompt_text, answer or "")
        if not no_cache:
            now = time.time()
            conn.execute(
                "INSERT OR REPLACE INTO cache"
                " (key, model, prompt, response, created_at, last_used_at, hit_count)"
                " VALUES (?, ?, ?, ?, ?, ?, 0)",
                (key, model, prompt_text, answer or "", now, now),
            )
            conn.execute(
                "UPDATE stats SET misses = misses + 1 WHERE id = 1"
            )
            conn.commit()
        return answer, {
            "cached": False, "key": key, "model": model,
            "est_cost_usd": cost, "latency_s": latency,
        }
    finally:
        conn.close()


def get_stats(db_path=None):
    db_path = db_path or os.environ.get("LLMCACHE_DB", DEFAULT_DB)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT hits, misses, est_saved_usd FROM stats WHERE id = 1"
        ).fetchone()
        n_rows = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        return {
            "hits": row[0], "misses": row[1],
            "est_saved_usd": row[2], "entries": n_rows,
            "db_bytes": size, "db_path": db_path,
        }
    finally:
        conn.close()


def clear(db_path=None):
    db_path = db_path or os.environ.get("LLMCACHE_DB", DEFAULT_DB)
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM cache")
        conn.execute(
            "UPDATE stats SET hits = 0, misses = 0, est_saved_usd = 0.0 WHERE id = 1"
        )
        conn.commit()
    finally:
        conn.close()


def evict(max_mb, db_path=None):
    """按 last_used_at 从旧到新删除，直到库文件 <= max_mb。返回删除条数。"""
    db_path = db_path or os.environ.get("LLMCACHE_DB", DEFAULT_DB)
    conn = _connect(db_path)
    try:
        removed = 0
        while True:
            try:
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
            except OSError:
                break
            if size_mb <= max_mb:
                break
            row = conn.execute(
                "SELECT key FROM cache ORDER BY last_used_at ASC LIMIT 1"
            ).fetchone()
            if not row:
                break
            conn.execute("DELETE FROM cache WHERE key = ?", (row[0],))
            conn.commit()
            removed += 1
        try:
            conn.execute("VACUUM")
            conn.commit()
        except sqlite3.Error:
            pass
        return removed
    finally:
        conn.close()


# ---------------- CLI ----------------

def _build_parser():
    ap = argparse.ArgumentParser(
        prog="llmcache",
        description="LLM 调用缓存：同样的 prompt 直接读 SQLite 缓存，不再花钱。",
    )
    ap.add_argument("--version", action="version", version="llmcache " + __version__)
    ap.add_argument("--db", default=None, help="缓存库路径（默认 ~/.cache/llmcache/cache.db）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="模型名（默认 %s）" % DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None, help="API base URL（默认 OPENAI_BASE_URL 或官方）")
    ap.add_argument("--api-key", default=None, help="API key（默认 OPENAI_API_KEY）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)

    sub = ap.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="问一个问题（走缓存）")
    ask.add_argument("question", help="问题文本")
    ask.add_argument("--system", default=None, help="system prompt")
    ask.add_argument("--no-cache", action="store_true", help="跳过缓存，直接请求")
    ask.add_argument("--ttl", type=int, default=None, help="缓存有效期（秒）")

    sub.add_parser("stats", help="缓存命中统计")
    sub.add_parser("clear", help="清空缓存与统计")

    ev = sub.add_parser("evict", help="按 LRU 删除旧缓存，直到库 <= 上限")
    ev.add_argument("--max-mb", type=float, default=100, help="库大小上限 MB（默认 100）")

    bg = sub.add_parser("budget", help="估算花费：已花（估算）/ 缓存省下（估算）")
    return ap


def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)
    db = args.db or os.environ.get("LLMCACHE_DB", DEFAULT_DB)

    if args.cmd == "ask":
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.question})
        try:
            answer, meta = chat(
                messages, model=args.model, temperature=args.temperature,
                max_tokens=args.max_tokens, base_url=args.base_url,
                api_key=args.api_key, db_path=db, ttl=args.ttl,
                no_cache=args.no_cache, timeout=args.timeout,
            )
        except RuntimeError as e:
            sys.stderr.write("error: %s\n" % e)
            return 1
        tag = "[缓存命中]" if meta["cached"] else "[实时调用]"
        print("%s（模型 %s）" % (tag, meta["model"]))
        print(answer)
        if meta["cached"]:
            print("\n本次花费 $0（估算省下 $%.6f）" % meta["est_saved_usd"])
        else:
            print("\n本次估算花费 $%.6f" % meta["est_cost_usd"])
        return 0

    if args.cmd == "stats":
        s = get_stats(db)
        total = s["hits"] + s["misses"]
        rate = (s["hits"] / total * 100) if total else 0.0
        print("缓存库：%s" % s["db_path"])
        print("条目数：%d，库大小：%.1f KB" % (s["entries"], s["db_bytes"] / 1024))
        print("命中：%d，未命中：%d，命中率：%.1f%%"
              % (s["hits"], s["misses"], rate))
        print("估算累计省下：$%.6f（估算值，仅供参考）" % s["est_saved_usd"])
        return 0

    if args.cmd == "clear":
        clear(db)
        print("已清空缓存与统计：%s" % db)
        return 0

    if args.cmd == "evict":
        removed = evict(args.max_mb, db)
        print("已按 LRU 删除 %d 条，库上限 %.1f MB" % (removed, args.max_mb))
        return 0

    if args.cmd == "budget":
        s = get_stats(db)
        print("估算累计省下：$%.6f" % s["est_saved_usd"])
        print("（按 字符数/4 估算 token × 公开价目表计算，为估算值，非账单）")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
