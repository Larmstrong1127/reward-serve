"""Record a real terminal session and render it as an animated SVG.

Runs the sequence a reader would run: start the server, ask ``/readyz`` while
the model is still loading and again once it is up, then fire
``scripts/dry_run.py`` (10 concurrent requests) and let the batch metadata come
back. Every line in the produced SVG was captured from a real process on this
machine. Nothing is typed in by hand and nothing is edited out.

No external recorder is needed. The renderer writes plain SVG with CSS
keyframes, which GitHub animates inside an ``<img>``.

    set CUDA_VISIBLE_DEVICES=
    python scripts/record_demo.py --out docs/demo.svg

The timings visible in the recording are whatever the run produced. This is a
demonstration, not a benchmark: the box is not gated quiet by
``scripts/box_quiet.py``, so no number in it is a published figure.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import IO, Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# --- terminal geometry --------------------------------------------------------
COLS = 132
CHAR_W = 8.4
LINE_H = 20.0
PAD_X = 18.0
PAD_Y = 16.0
FONT = 14.0
# Share of the animation loop spent revealing lines; the remainder is a hold.
SPREAD = 62.0

BG = "#12151b"
FG = "#d7dce3"
DIM = "#8b95a6"
PROMPT = "#7fd1a3"
CMD = "#e6dfa8"



class Transcript:
    """Time-stamped lines, in the order a viewer of the terminal would see them."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.lines: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def add(self, text: str, kind: str = "out") -> None:
        for chunk in _wrap(text.rstrip("\r\n")):
            with self.lock:
                entry = {
                    "t": round(time.monotonic() - self.t0, 3),
                    "kind": kind,
                    "text": chunk,
                }
                self.lines.append(entry)
            print(f"[{entry['t']:>7.3f}] {chunk}", flush=True)


def _wrap(text: str) -> list[str]:
    text = text.replace("\t", "    ")
    if text == "":
        return [""]
    return [text[i : i + COLS] for i in range(0, len(text), COLS)]


def _pump(stream: IO[str], tr: Transcript) -> None:
    try:
        for raw in iter(stream.readline, ""):
            tr.add(raw)
    except Exception as exc:  # a dead reader thread would silently lose lines
        print(f"[pump] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        stream.close()


def _probe(tr: Transcript, python: str, url: str, env: dict[str, str]) -> int:
    """Run the readyz check as its own process so the printed line is its output.

    Returns the HTTP status it printed, or 0 if it could not reach the server.
    """
    target = f"{url}/readyz"
    tr.add(f"$ python scripts/readyz.py {target}", "cmd")
    run = subprocess.run(
        [python, "scripts/readyz.py", target],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = (run.stdout + run.stderr).splitlines() or ["<no output>"]
    for line in lines:
        tr.add(line)
    head = lines[0].split(" ", 1)[0]
    return int(head) if head.isdigit() else 0


def record(port: int, python: str) -> Transcript:
    tr = Transcript()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    tr.add(f"$ python -m reward_serve --port {port}", "cmd")
    proc = subprocess.Popen(
        [python, "-m", "reward_serve", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    threading.Thread(target=_pump, args=(proc.stdout, tr), daemon=True).start()

    url = f"http://127.0.0.1:{port}"
    # Ask from the moment the process exists, so the recording shows /readyz
    # answering before the model is loaded as well as after it is.
    time.sleep(0.3)
    code = 0
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        code = _probe(tr, python, url, env)
        if code == 200:
            break
        time.sleep(1.2)

    try:
        if code != 200:
            tr.add("server never became ready; recording stops here")
        else:
            tr.add(f"$ python scripts/dry_run.py --url {url}", "cmd")
            # Streamed, not captured in one lump, so every line keeps the
            # moment it was actually printed.
            dry = subprocess.Popen(
                [python, "scripts/dry_run.py", "--url", url],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert dry.stdout is not None
            _pump(dry.stdout, tr)
            tr.add(f"exit code {dry.wait(timeout=60)}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(0.7)
    return tr


# --- rendering ----------------------------------------------------------------
def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(lines: list[dict[str, Any]], playback: float, real: float, note: str) -> str:
    n = len(lines)
    width = PAD_X * 2 + CHAR_W * COLS
    height = PAD_Y * 2 + LINE_H * (n + 2)
    css = [
        f"text{{font-family:'DejaVu Sans Mono','Cascadia Mono',Consolas,'Courier New',"
        f"monospace;font-size:{FONT}px;fill:{FG};white-space:pre}}",
        f".p{{fill:{PROMPT}}}",
        f".c{{fill:{CMD}}}",
        f".d{{fill:{DIM}}}",
        ".ln{opacity:0}",
    ]
    body: list[str] = []
    for i, ln in enumerate(lines):
        t = float(ln["t"])
        # Each line appears at its own share of the real session, replayed over
        # `playback` seconds. Only the first SPREAD% of the loop is used for
        # that; the rest holds the finished screen so a reader can read it.
        pct = 0.0 if real <= 0 else min(SPREAD, (t / real) * SPREAD)
        css.append(f"@keyframes k{i}{{0%,{pct:.3f}%{{opacity:0}}{pct:.3f}%,100%{{opacity:1}}}}")
        css.append(f".n{i}{{animation:k{i} {playback:.1f}s linear infinite}}")
        y = PAD_Y + LINE_H * (i + 1)
        text = str(ln["text"])
        if ln["kind"] == "cmd" and text.startswith("$ "):
            inner = (
                f'<tspan class="p">$ </tspan><tspan class="c">{_esc(text[2:])}</tspan>'
            )
        else:
            inner = _esc(text)
        body.append(f'<g class="ln n{i}"><text x="{PAD_X}" y="{y}">{inner}</text></g>')
    body.append(
        f'<text class="d" x="{PAD_X}" y="{PAD_Y + LINE_H * (n + 2)}">{_esc(note)}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="Terminal recording: reward-serve starts, /readyz flips to ready, '
        f'and ten concurrent requests come back scored with batch metadata.">\n'
        f'<rect width="100%" height="100%" fill="{BG}" rx="8"/>\n'
        f"<style>{''.join(css)}</style>\n" + "\n".join(body) + "\n</svg>\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="docs/demo.svg")
    ap.add_argument("--playback-seconds", type=float, default=20.0)
    ap.add_argument(
        "--from-transcript",
        default=None,
        help="re-render an SVG from a transcript this script wrote earlier",
    )
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    if args.from_transcript:
        saved = json.loads(Path(args.from_transcript).read_text(encoding="utf-8"))
        lines, real = saved["lines"], float(saved["real_seconds"])
    else:
        tr = record(args.port, args.python)
        lines = tr.lines
        real = max((float(ln["t"]) for ln in lines), default=1.0)
    note = (
        f"real session, {real:.1f}s wall clock, replayed over "
        f"{args.playback_seconds:.0f}s | CPU only | demonstration, not a benchmark: "
        f"this box was not gated quiet"
    )
    out = ROOT / args.out
    out.write_text(render_svg(lines, args.playback_seconds, real, note), encoding="utf-8")
    out.with_suffix(".transcript.json").write_text(
        json.dumps({"real_seconds": real, "lines": lines}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out} ({out.stat().st_size} bytes), {len(lines)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
