# llmcache

给 OpenAI-compatible API 加一层 SQLite 缓存：**同样的 prompt，直接读本地缓存，不再花钱**。

做 prompt 调试、跑回归测试、反复问同一个问题时特别省钱。纯标准库，零依赖。

## 安装

```bash
# 直接用（需要 Python 3.10+）
python -m llmcache ask "天空为什么是蓝色的？"
```

API key 从环境变量 `OPENAI_API_KEY` 读取（也支持 `--api-key`），
base URL 默认官方，也可用 `OPENAI_BASE_URL` / `--base-url` 指向任何
OpenAI-compatible 服务。

## 用法

```bash
# 问一个问题（第一次走网络，之后同样问题走缓存）
llmcache ask "用一句话解释相对论" --model gpt-4o-mini

# 跳过缓存，强制实时调用
llmcache ask "现在几点了？" --no-cache

# 缓存只保留 1 小时
llmcache ask "今天的新闻" --ttl 3600

# 看命中统计：命中/未命中/命中率/估算省了多少钱
llmcache stats

# 估算口径说明
llmcache budget

# 库太大时按 LRU 删除旧条目，直到 <= 100MB
llmcache evict --max-mb 100

# 清空
llmcache clear

# 换缓存位置
llmcache --db /tmp/my.db ask "你好"
```

输出示例：

```
[缓存命中]（模型 gpt-4o-mini）
……
本次花费 $0（估算省下 $0.000023）
```

## 当库用

```python
import llmcache

answer, meta = llmcache.chat(
    [{"role": "user", "content": "你好"}],
    model="gpt-4o-mini",
    api_key="sk-...",
)
print(answer, meta["cached"])  # 第一次 False，之后 True
```

## 缓存规则

- key = `{model, messages, temperature, max_tokens, extra}` 规范 JSON 的 sha256。
  换模型、换 temperature、改一个标点都是不同的 key。
- `temperature > 0` 时，同 prompt 也会返回第一次缓存的答案——
  这是故意的（省钱优先），要随机性请用 `--no-cache`。
- `--ttl` 过期后自动删除重取。
- 非流式接口：`chat/completions` 的普通（非 stream）响应才会被缓存。

## 花费估算（诚实说明）

`stats` / `budget` 里的美元数是**估算值**，不是账单：

- token 按「字符数 ÷ 4」估算（英文大致准，中文不准）；
- 价格用内置的公开价目表（`PRICE_TABLE`），可能过期；
- 只用于「缓存大概帮我省了多少」的体感，不要拿去报销。

## 已知局限

- 缓存 key 包含 temperature：`temperature=0.7` 的同样问题会命中缓存，
  返回上次的答案而不是重新采样。
- 流式（stream）响应不缓存。
- 缓存存在本地 SQLite，`--clear` 可清空；敏感问答请注意本地文件权限。
- 真实 API 未做扣费实测（请求构造经本地 stub server 完整验证）。

## License

MIT，Copyright (c) 2026 ljiang9。
