"""Print one line for the server's readiness, whatever state it is in.

    python scripts/readyz.py http://127.0.0.1:8000/readyz

Exit code 0 when the server answered 200, 1 otherwise. A server that is not
listening yet is a normal answer here, not a crash, so it prints one line
rather than a traceback -- this is what `scripts/record_demo.py` polls with
while the model is still loading.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402

import httpx  # noqa: E402


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else "http://127.0.0.1:8000/readyz"
    try:
        r = httpx.get(url, timeout=5.0)
    except httpx.HTTPError as exc:
        print(f"not listening yet ({type(exc).__name__})")
        return 1
    print(f"{r.status_code} {r.text}")
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
