"""CLI entrypoint: python -m reward_serve --port 8000 ..."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402

import uvicorn  # noqa: E402

from reward_serve.model import DEFAULT_MODEL_ID  # noqa: E402
from reward_serve.server import Settings, create_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """CLI flags mirror the env vars; a flag wins over its env var."""
    env = Settings.from_env()
    p = argparse.ArgumentParser(prog="reward_serve", description=__doc__)
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--max-batch-size", type=int, default=env.max_batch_size)
    p.add_argument("--max-wait-ms", type=float, default=env.max_wait_ms)
    p.add_argument("--queue-depth", type=int, default=env.queue_depth)
    p.add_argument("--request-timeout-ms", type=float, default=env.request_timeout_ms)
    p.add_argument("--model-id", default=env.model_id or DEFAULT_MODEL_ID)
    p.add_argument("--log-level", default="info")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings(
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        queue_depth=args.queue_depth,
        request_timeout_ms=args.request_timeout_ms,
        model_id=args.model_id,
    )
    # One worker process on purpose: the batcher owns a single in-process queue.
    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, log_level=args.log_level
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
