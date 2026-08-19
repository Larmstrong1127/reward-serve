"""Minimal end-to-end check: start nothing, just drive a running server.

Sends 10 real requests concurrently and prints the batch metadata that comes
back, so batching is visible rather than asserted. Charter rule 10.

    python scripts/dry_run.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402

import httpx  # noqa: E402

PROMPTS: list[tuple[str, str]] = [
    ("How do I reverse a list?", "Use the reversed() builtin."),
    ("How do I reverse a list?", "idk"),
    ("What is a bounded queue?", "A queue with a fixed capacity that rejects when full."),
    ("What is 2 + 2?", "4"),
    ("What is 2 + 2?", "Seventeen, probably."),
    ("Name a colour.", "Blue."),
    ("Define latency.", "Time between a request and its response."),
    ("Define throughput.", "Completed requests per unit of time."),
    ("Explain padding in a batch.", "Shorter sequences are filled to the longest length."),
    ("Explain a deadline.", "The time after which a queued request is no longer useful."),
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=60.0) as c:
        ready = await c.get("/readyz")
        print(f"readyz -> {ready.status_code} {ready.text}")
        if ready.status_code != 200:
            print("server is not ready; dry run stops here")
            return 1

        async def one(i: int, prompt: str, response: str) -> dict[str, object]:
            r = await c.post("/v1/score", json={"prompt": prompt, "response": response})
            r.raise_for_status()
            body = r.json()
            item = body["results"][0]
            return {"i": i, **item}

        results = await asyncio.gather(
            *(one(i, p, r) for i, (p, r) in enumerate(PROMPTS))
        )

    print(f"\n{'req':>3} {'score':>9} {'batch_size':>11} {'queue_wait_ms':>14} {'forward_ms':>11}")
    for row in results:
        print(
            f"{row['i']:>3} {row['score']:>9.4f} {row['batch_size']:>11} "
            f"{row['queue_wait_ms']:>14.2f} {row['forward_ms']:>11.2f}"
        )

    sizes = [int(r["batch_size"]) for r in results]
    batched = [s for s in sizes if s > 1]
    print(f"\nmax batch_size observed: {max(sizes)}")
    print(f"requests that rode in a batch larger than 1: {len(batched)}/{len(sizes)}")
    print(json.dumps({"max_batch_size_observed": max(sizes), "n": len(sizes)}))
    return 0 if max(sizes) > 1 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
