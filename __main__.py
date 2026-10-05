"""Enables `python -m llmcache`."""
import sys

if __package__:
    from .llmcache import main
else:  # 直接运行 llmcache/__main__.py 时的兜底
    from llmcache import main

if __name__ == "__main__":
    sys.exit(main())
