"""Checks on the written artefacts: the SLO doc, the alert rules, the dashboard.

A dashboard nobody can load and a PromQL expression that does not parse are the
two ways an observability setup silently becomes decoration. These tests fail
before either reaches a reader.

``promtool`` is used when it is on PATH; when it is not, the PromQL falls back
to the structural checker below. That checker is not a PromQL parser and does
not pretend to be one -- it catches unbalanced delimiters, malformed range
selectors, rate() without a range, unknown functions, and references to metric
names this server does not emit, which is the class of mistake that actually
gets written into an SLO doc.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from reward_serve import obs

ROOT = Path(__file__).resolve().parents[1]
SLO_DOC = ROOT / "docs" / "slo.md"
RULES = ROOT / "docker" / "rules" / "slo_burn.yml"
COMPOSE = ROOT / "docker" / "docker-compose.yml"
DASHBOARD = ROOT / "docker" / "grafana" / "dashboards" / "reward-serve-overview.json"

PROMQL_FUNCTIONS = {
    "rate", "irate", "increase", "sum", "avg", "min", "max", "count",
    "histogram_quantile", "clamp_min", "clamp_max", "vector", "scalar",
    "absent", "delta", "topk", "quantile", "label_replace", "time",
}
KNOWN_METRIC_PREFIXES = ("rs_", "rs:", "up", "otelcol_")
KEYWORDS = {
    "and", "or", "unless", "by", "without", "on", "ignoring", "offset",
    "group_left", "group_right", "bool",
}

RANGE_RE = re.compile(r"\[(\d+)(ms|s|m|h|d|w|y)\]")
IDENT_RE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")


def promql_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.findall(r"```promql\n(.*?)```", text, re.S)]


def structural_errors(expr: str) -> list[str]:
    """Return every structural complaint about one PromQL expression."""
    errors: list[str] = []

    if expr.count('"') % 2:
        errors.append("unbalanced double quotes")
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if expr.count(opening) != expr.count(closing):
            errors.append(f"unbalanced {opening}{closing}")

    depth = 0
    for char in expr:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                errors.append("closing paren before its opener")
                break

    for raw_range in re.findall(r"\[[^\]]*\]", expr):
        if not RANGE_RE.fullmatch(raw_range):
            errors.append(f"malformed range selector {raw_range}")

    # rate() and increase() need a range vector; without one Prometheus errors
    # at query time, which on a dashboard means an empty panel, not a message.
    for func in ("rate", "increase"):
        for match in re.finditer(rf"\b{func}\(", expr):
            tail = expr[match.end() :]
            close = tail.find(")")
            if close == -1 or "[" not in tail[:close]:
                errors.append(f"{func}() without a range selector")

    if re.search(r"[-+*/]\s*$", expr.strip()):
        errors.append("expression ends on an operator")

    # Strip the parts that legitimately contain non-identifier text before
    # deciding what is a function, a metric, or a keyword: string literals,
    # range selectors, and label matchers.
    cleaned = re.sub(r'"[^"]*"', '""', expr)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
    # by (label, label) / without (label) hold label names, not metrics.
    cleaned = re.sub(r"\b(by|without|on|ignoring)\s*\([^)]*\)", " ", cleaned)

    for match in IDENT_RE.finditer(cleaned):
        word = match.group()
        is_call = cleaned[match.end() :].lstrip().startswith("(")
        if is_call and word not in KEYWORDS:
            if word not in PROMQL_FUNCTIONS and not word.startswith("rs"):
                errors.append(f"unknown function {word}()")
            continue
        if word in KEYWORDS or word in PROMQL_FUNCTIONS:
            continue
        if word.startswith(KNOWN_METRIC_PREFIXES):
            continue
        errors.append(f"unknown identifier {word}")

    return errors


# -- the checker itself -----------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        'sum(rate(rs_requests_total{status="ok"}[5m]))) / sum(rate(rs_requests_total[5m]))',
        'sum(rate(rs_requests_total{status="ok}[5m]))',
        "sum(rate(rs_requests_total[5]))",
        "sum(rate(rs_requests_total))",
        "sum(rate(rs_requests_total[5m])) /",
        "sum(rate(http_requests_total[5m]))",
        "histgram_quantile(0.99, rs_request_latency_seconds_bucket)",
    ],
)
def test_the_checker_rejects_broken_promql(bad: str) -> None:
    """A validator nobody has seen fail is not a validator.

    Unbalanced parens, an unterminated string, a range without a unit, rate()
    with no range, a dangling operator, a metric this server never emits, and a
    misspelt function -- each must produce at least one complaint.
    """
    assert structural_errors(bad), f"checker accepted {bad!r}"


def test_the_checker_accepts_the_expressions_it_should() -> None:
    for good in (
        'sum by (status) (rate(rs_requests_total{status="ok"}[1m]))',
        "histogram_quantile(0.95, sum by (le) (rate(rs_request_latency_seconds_bucket[5m])))",
        "rs_queue_depth",
        'up{job="reward-serve"} == 0',
    ):
        assert not structural_errors(good), good


# -- the SLO document -------------------------------------------------------
def test_slo_doc_exists_and_states_a_window_and_targets() -> None:
    text = SLO_DOC.read_text(encoding="utf-8")
    assert "30 days" in text
    assert "99.5%" in text          # availability target
    assert "14.4x" in text          # fast burn multiplier, shown
    assert "6x" in text             # slow burn multiplier, shown
    assert "216 minutes" in text    # error budget in time
    assert "1,000,000" in text      # error budget in requests
    assert "does not cover" in text


def test_every_promql_block_in_the_slo_doc_is_structurally_valid() -> None:
    blocks = promql_blocks(SLO_DOC.read_text(encoding="utf-8"))
    assert len(blocks) >= 6, "the SLO doc should carry PromQL for both SLIs and both burns"
    for block in blocks:
        # The latency threshold is a documented placeholder until it is measured.
        expr = block.replace("<THRESHOLD>", "0.35")
        errors = structural_errors(expr)
        assert not errors, f"{errors} in:\n{expr}"


def test_slo_doc_only_names_metrics_the_server_emits() -> None:
    metrics = obs.setup_metrics()
    emitted = {s.name for m in metrics.registry.collect() for s in m.samples}
    emitted |= {m.name for m in metrics.registry.collect()}
    text = SLO_DOC.read_text(encoding="utf-8")
    for name in re.findall(r"\brs_[a-z_]+\b", text):
        base = re.sub(r"_(bucket|count|sum|total)$", "", name)
        assert any(e.startswith(base) for e in emitted), f"{name} is not emitted"


# -- the alert rules --------------------------------------------------------
def test_rules_file_parses_and_carries_both_burn_alerts() -> None:
    doc = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    groups = doc["groups"]
    rules = [r for g in groups for r in g["rules"]]
    alerts = {r["alert"] for r in rules if "alert" in r}
    records = {r["record"] for r in rules if "record" in r}

    assert "RewardServeAvailabilityFastBurn" in alerts
    assert "RewardServeAvailabilitySlowBurn" in alerts
    assert "rs:availability_budget_remaining:ratio30d" in records
    # An unscrapable server must not read as healthy.
    assert "RewardServeNoMetrics" in alerts

    for rule in rules:
        errors = structural_errors(rule["expr"].strip())
        assert not errors, f"{errors} in {rule.get('alert') or rule.get('record')}"


def test_rules_pass_promtool_when_it_is_available() -> None:
    promtool = shutil.which("promtool")
    if promtool is None:
        pytest.skip("promtool is not installed; the structural checker ran instead")
    done = subprocess.run(
        [promtool, "check", "rules", str(RULES)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr


# -- the compose stack ------------------------------------------------------
def test_compose_file_parses_and_declares_the_five_services() -> None:
    doc = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = doc["services"]
    assert set(services) == {"server", "otel-collector", "jaeger", "prometheus", "grafana"}
    assert "healthcheck" in services["server"]
    # /readyz, not /healthz: readiness is what gates traffic.
    assert "readyz" in json.dumps(services["server"]["healthcheck"])
    assert services["server"]["environment"]["CUDA_VISIBLE_DEVICES"] == ""


def test_dockerfile_runs_as_a_non_root_user_on_cpu_torch() -> None:
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "USER serve" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert 'CUDA_VISIBLE_DEVICES=""' in text
    assert "HEALTHCHECK" in text and "readyz" in text


def test_prometheus_scrapes_the_server_and_loads_the_rules() -> None:
    doc = yaml.safe_load((ROOT / "docker" / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {j["job_name"]: j for j in doc["scrape_configs"]}
    assert "reward-serve" in jobs
    assert jobs["reward-serve"]["metrics_path"] == "/metrics"
    assert jobs["reward-serve"]["static_configs"][0]["targets"] == ["server:8000"]
    assert doc["rule_files"] == ["/etc/prometheus/rules/*.yml"]


def test_collector_receives_otlp_grpc_and_exports_traces_and_prometheus() -> None:
    doc = yaml.safe_load(
        (ROOT / "docker" / "otel-collector-config.yaml").read_text(encoding="utf-8")
    )
    assert "grpc" in doc["receivers"]["otlp"]["protocols"]
    assert doc["receivers"]["otlp"]["protocols"]["grpc"]["endpoint"].endswith(":4317")
    assert "otlp/jaeger" in doc["exporters"]
    assert "prometheus" in doc["exporters"]
    traces = doc["service"]["pipelines"]["traces"]
    assert "otlp/jaeger" in traces["exporters"]


# -- the dashboard ----------------------------------------------------------
def test_dashboard_json_is_valid_and_has_every_required_panel() -> None:
    doc = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    titles = " | ".join(p["title"] for p in doc["panels"]).lower()
    for required in (
        "request rate",
        "error rate",
        "latency p50 / p95 / p99",
        "queue depth",
        "batch size distribution",
        "fast burn",
        "slow burn",
    ):
        assert required in titles, f"no panel for {required}"

    exprs = [t["expr"] for p in doc["panels"] for t in p.get("targets", [])]
    assert any("histogram_quantile(0.99" in e for e in exprs)
    assert any("rs_batch_size_bucket" in e for e in exprs)
    assert any("rs_queue_depth" in e for e in exprs)
    assert any("14.4" in e for e in exprs)

    for expr in exprs:
        substituted = re.sub(r"\$slo_[a-z_]+", "0.35", expr)
        errors = structural_errors(substituted)
        assert not errors, f"{errors} in dashboard expr:\n{expr}"


def test_dashboard_datasource_uid_matches_the_provisioned_one() -> None:
    provisioned = yaml.safe_load(
        (ROOT / "docker" / "grafana" / "provisioning" / "datasources" / "datasources.yml")
        .read_text(encoding="utf-8")
    )
    uids = {d["uid"] for d in provisioned["datasources"]}
    doc = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    used = {p["datasource"]["uid"] for p in doc["panels"] if "datasource" in p}
    assert used <= uids, f"dashboard points at datasources that are not provisioned: {used - uids}"


def test_the_unmeasured_latency_threshold_is_flagged_in_both_places() -> None:
    """The doc and the dashboard must agree about whether the threshold is real.

    Grafana cannot hold an empty constant, so the dashboard ships a placeholder.
    The failure mode that matters is the placeholder outliving the measurement,
    or the doc being updated while the dashboard keeps the placeholder: either
    way a burn panel would plot against a threshold nobody chose. This test ties
    them together so the pair moves at once.
    """
    doc_says_unset = "the threshold is not set" in SLO_DOC.read_text(encoding="utf-8").lower()
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    var = next(v for v in dashboard["templating"]["list"] if v["name"] == "slo_latency_le")
    dashboard_says_provisional = "provisional" in var["label"].lower()

    assert doc_says_unset == dashboard_says_provisional, (
        "docs/slo.md and the dashboard disagree about whether the latency "
        "threshold has been measured"
    )
    if doc_says_unset:
        assert "obs_dry_run.json" in var["description"]


def test_the_measured_threshold_would_be_a_real_bucket_edge() -> None:
    """Whatever the dry run yields, the threshold has to land on a bucket edge.

    The latency SLI is a bucket count. A threshold between edges cannot be read
    without interpolating, which puts an estimate at the centre of the error
    budget. This checks the placeholder already obeys that rule, so the habit is
    in place before the real number arrives.
    """
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    var = next(v for v in dashboard["templating"]["list"] if v["name"] == "slo_latency_le")
    assert float(var["query"]) in obs.LATENCY_BUCKETS
