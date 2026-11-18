"""Validation and deterministic rendering for cost-aware research reports."""

from __future__ import annotations

from html import escape
import json
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence


NO_PASS_STATEMENT = "No configuration passed all research gates."

_METRIC_FIELDS = (
    "return",
    "sharpe",
    "max_drawdown",
    "trades",
    "win_rate",
)
_CANDIDATE_FIELDS = (
    "rank",
    "strategy",
    "family",
    "timeframe",
    "passed",
    "adaptive_params",
    "latest_params",
    "oos",
    "neighbor",
    "cost_off",
    "reference",
    "calendar_returns",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _percent(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _number(value: object) -> str:
    return f"{float(value):.3f}"


def _integer(value: object) -> str:
    return str(int(value))


def _markdown(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _require_fields(
    value: Mapping[str, object],
    fields: Sequence[str],
    path: str,
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{path} missing required fields: {', '.join(missing)}")


def _finite(value: object, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    if not isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _validate_metrics(value: object, path: str) -> None:
    metrics = _require_mapping(value, path)
    _require_fields(metrics, _METRIC_FIELDS, path)
    for field in ("return", "sharpe", "max_drawdown", "win_rate"):
        _finite(metrics[field], f"{path}.{field}")
    trades = _finite(metrics["trades"], f"{path}.trades")
    if trades < 0 or not trades.is_integer():
        raise ValueError(f"{path}.trades must be a nonnegative integer")


def _validate_calendar_returns(value: object, path: str) -> None:
    returns = _require_mapping(value, path)
    _require_fields(returns, ("7d", "30d", "90d"), path)
    for window in ("7d", "30d", "90d"):
        _finite(returns[window], f"{path}.{window}")


def validate_summary(summary: Mapping[str, object]) -> None:
    """Reject incomplete or internally inconsistent machine summaries."""
    root = _require_mapping(summary, "summary")
    _require_fields(
        root,
        (
            "schema_version",
            "status",
            "generated_at",
            "data",
            "research",
            "candidates",
            "passing_count",
            "calibration",
            "warnings",
        ),
        "summary",
    )
    data = _require_mapping(root["data"], "data")
    _require_fields(
        data,
        (
            "metadata_generated_at",
            "source",
            "requested_start",
            "requested_end",
            "universe",
            "timeframes",
            "coverage",
        ),
        "data",
    )
    research = _require_mapping(root["research"], "research")
    _require_fields(
        research,
        (
            "cost_model",
            "primary_sizing",
            "reference_sizing",
            "folds",
            "calendar_window_end",
            "calendar_window_definition",
            "neighbor_rule",
            "passing_gates",
            "search_counts",
        ),
        "research",
    )

    candidates = root["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    passing = 0
    ranks: list[int] = []
    for index, raw_candidate in enumerate(candidates):
        path = f"candidates[{index}]"
        candidate = _require_mapping(raw_candidate, path)
        _require_fields(candidate, _CANDIDATE_FIELDS, path)
        rank = int(_finite(candidate["rank"], f"{path}.rank"))
        ranks.append(rank)
        if not isinstance(candidate["passed"], bool):
            raise ValueError(f"{path}.passed must be boolean")
        passing += int(candidate["passed"])

        adaptive = candidate["adaptive_params"]
        if not isinstance(adaptive, list) or len(adaptive) != 3:
            raise ValueError(f"{path}.adaptive_params must contain three folds")
        for fold_index, raw_fold in enumerate(adaptive):
            fold_path = f"{path}.adaptive_params[{fold_index}]"
            fold = _require_mapping(raw_fold, fold_path)
            _require_fields(fold, ("fold", "params"), fold_path)
            if int(_finite(fold["fold"], f"{fold_path}.fold")) != fold_index:
                raise ValueError(f"{fold_path}.fold must match its ordered index")
            _require_mapping(fold["params"], f"{fold_path}.params")
        latest = _require_mapping(candidate["latest_params"], f"{path}.latest_params")
        if dict(latest) != dict(adaptive[-1]["params"]):
            raise ValueError(f"{path}.latest_params must equal the final fold params")

        _validate_metrics(candidate["oos"], f"{path}.oos")
        oos = _require_mapping(candidate["oos"], f"{path}.oos")
        fold_returns = oos.get("fold_returns")
        if not isinstance(fold_returns, list) or len(fold_returns) != 3:
            raise ValueError(f"{path}.oos.fold_returns must contain three values")
        for fold_index, value in enumerate(fold_returns):
            _finite(value, f"{path}.oos.fold_returns[{fold_index}]")

        neighbor = _require_mapping(candidate["neighbor"], f"{path}.neighbor")
        _require_fields(
            neighbor,
            ("positive", "total", "robustness"),
            f"{path}.neighbor",
        )
        positive = int(_finite(neighbor["positive"], f"{path}.neighbor.positive"))
        total = int(_finite(neighbor["total"], f"{path}.neighbor.total"))
        robustness = _finite(
            neighbor["robustness"],
            f"{path}.neighbor.robustness",
        )
        if positive < 0 or total < 0 or positive > total:
            raise ValueError(f"{path}.neighbor counts are inconsistent")
        expected_robustness = positive / total if total else 0.0
        if abs(robustness - expected_robustness) > 1e-12:
            raise ValueError(f"{path}.neighbor robustness is inconsistent")

        _validate_metrics(candidate["cost_off"], f"{path}.cost_off")
        _validate_calendar_returns(
            candidate["calendar_returns"],
            f"{path}.calendar_returns",
        )
        reference = _require_mapping(candidate["reference"], f"{path}.reference")
        _require_fields(reference, ("oos", "calendar_returns"), f"{path}.reference")
        _validate_metrics(reference["oos"], f"{path}.reference.oos")
        _validate_calendar_returns(
            reference["calendar_returns"],
            f"{path}.reference.calendar_returns",
        )

    if ranks != list(range(1, len(candidates) + 1)):
        raise ValueError("candidate ranks must be contiguous and ordered")
    if int(root["passing_count"]) != passing:
        raise ValueError("passing_count does not match candidate pass flags")
    search_counts = _require_mapping(research["search_counts"], "research.search_counts")
    expected_candidates = int(
        _finite(
            search_counts.get("family_timeframe_candidates"),
            "research.search_counts.family_timeframe_candidates",
        )
    )
    if expected_candidates != len(candidates):
        raise ValueError(
            "research.search_counts.family_timeframe_candidates "
            "does not match candidates"
        )


def _candidate_cells(candidate: Mapping[str, object]) -> list[str]:
    oos = _require_mapping(candidate["oos"], "candidate.oos")
    cost_off = _require_mapping(candidate["cost_off"], "candidate.cost_off")
    reference = _require_mapping(candidate["reference"], "candidate.reference")
    reference_oos = _require_mapping(reference["oos"], "candidate.reference.oos")
    reference_calendar = _require_mapping(
        reference["calendar_returns"],
        "candidate.reference.calendar_returns",
    )
    calendar = _require_mapping(
        candidate["calendar_returns"],
        "candidate.calendar_returns",
    )
    return [
        _integer(candidate["rank"]),
        "PASS" if candidate["passed"] else "NO",
        str(candidate["strategy"]),
        str(candidate["family"]),
        str(candidate["timeframe"]),
        _json(candidate["adaptive_params"]),
        _json(candidate["latest_params"]),
        _percent(oos["return"]),
        _percent(calendar["7d"]),
        _percent(calendar["30d"]),
        _percent(calendar["90d"]),
        _number(oos["sharpe"]),
        _percent(oos["max_drawdown"]),
        _integer(oos["trades"]),
        _percent(oos["win_rate"]),
        _percent(cost_off["return"]),
        _percent(reference_oos["return"]),
        _percent(reference_calendar["7d"]),
        _percent(reference_calendar["30d"]),
        _percent(reference_calendar["90d"]),
        _number(reference_oos["sharpe"]),
        _percent(reference_oos["max_drawdown"]),
        _integer(reference_oos["trades"]),
        _percent(reference_oos["win_rate"]),
    ]


_CANDIDATE_HEADERS = (
    "Rank",
    "Pass",
    "Strategy",
    "Family",
    "Timeframe",
    "Adaptive OOS exact params",
    "Latest exact params",
    "OOS return",
    "Last 7d",
    "Last 30d",
    "Last 90d",
    "Sharpe",
    "Max drawdown",
    "Trades",
    "Win rate",
    "Cost-off return",
    "5% x 5x OOS return",
    "5% x 5x Last 7d",
    "5% x 5x Last 30d",
    "5% x 5x Last 90d",
    "5% x 5x Sharpe",
    "5% x 5x Max drawdown",
    "5% x 5x Trades",
    "5% x 5x Win rate",
)


def render_markdown(summary: Mapping[str, object]) -> str:
    """Render one complete Markdown report from a validated summary."""
    validate_summary(summary)
    data = _require_mapping(summary["data"], "data")
    research = _require_mapping(summary["research"], "research")
    candidates = summary["candidates"]
    calibration = _require_mapping(summary["calibration"], "calibration")
    counts = _require_mapping(research["search_counts"], "search_counts")

    lines = [
        "# Cost-aware OKX strategy research",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Outcome",
        "",
    ]
    if int(summary["passing_count"]) == 0:
        lines.extend([NO_PASS_STATEMENT, ""])
    else:
        lines.extend(
            [
                f"{summary['passing_count']} configuration(s) passed all research gates.",
                "",
            ]
        )
    lines.extend(
        [
            "All family/timeframe candidates remain listed below, including failures.",
            "",
            "## Data and execution contract",
            "",
            f"- Data snapshot generated: `{data['metadata_generated_at']}`.",
            f"- Source: {_markdown(data['source'])}.",
            f"- Requested complete-calendar coverage: `{data['requested_start']}` through `{data['requested_end']}`.",
            f"- Universe ({len(data['universe'])}): {_markdown(', '.join(data['universe']))}.",
            f"- Coverage: {_markdown(data['coverage'])}.",
            f"- Timeframes: {_markdown(', '.join(data['timeframes']))}.",
            f"- Cost model: `{_markdown(_json(research['cost_model']))}`.",
            f"- Primary sizing: `{_markdown(_json(research['primary_sizing']))}`.",
            f"- 5% x 5x reference sizing: `{_markdown(_json(research['reference_sizing']))}`.",
            (
                f"- Calendar return windows end at `{research['calendar_window_end']}`: "
                f"{_markdown(research['calendar_window_definition'])}."
            ),
            "- Signals use the preceding fully closed bar and enter at the next open; each independent window starts flat.",
            "",
            "## Walk-forward folds",
            "",
            "| Fold | IS start | IS end | OOS start | OOS end |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for fold in research["folds"]:
        lines.append(
            "| {fold} | {is_start} | {is_end} | {oos_start} | {oos_end} |".format(
                **{key: _markdown(value) for key, value in fold.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Full bounded search",
            "",
            (
                f"Searched {counts['strategies']} strategies × "
                f"{counts['timeframes']} timeframes; "
                f"{counts['grid_configurations']} complete grid configurations "
                f"and {counts['is_fold_evaluations']} IS fold evaluations."
            ),
            "",
            f"Neighbor rule: {_markdown(research['neighbor_rule'])}.",
            f"Passing gates: `{_markdown(_json(research['passing_gates']))}`.",
            "",
            "| " + " | ".join(_CANDIDATE_HEADERS) + " |",
            "| " + " | ".join("---" for _ in _CANDIDATE_HEADERS) + " |",
        ]
    )
    for candidate in candidates:
        lines.append(
            "| "
            + " | ".join(_markdown(cell) for cell in _candidate_cells(candidate))
            + " |"
        )

    primary = _require_mapping(calibration["primary_returns"], "calibration.primary")
    legacy = calibration.get("legacy_equal_weight_returns")
    reference = _require_mapping(
        calibration["user_reference_returns"],
        "calibration.user_reference_returns",
    )
    lines.extend(
        [
            "",
            "## Supplied 15m Donchian calibration",
            "",
            f"- Exact params: `{_markdown(_json(calibration['params']))}`.",
            "- Direction: both.",
            "",
            "| Sizing / source | 7d | 30d | 56d |",
            "| --- | ---: | ---: | ---: |",
            (
                "| Primary 10% x 8x, max 3 | "
                f"{_percent(primary['7d'])} | {_percent(primary['30d'])} | "
                f"{_percent(primary['56d'])} |"
            ),
        ]
    )
    if isinstance(legacy, Mapping):
        lines.append(
            "| Legacy equal-weight 1/N x 1x | "
            f"{_percent(legacy['7d'])} | {_percent(legacy['30d'])} | "
            f"{_percent(legacy['56d'])} |"
        )
    lines.append(
        "| User comparison | "
        f"{_percent(reference['7d'])} | {_percent(reference['30d'])} | "
        f"{_percent(reference['56d'])} |"
    )
    lines.extend(
        [
            "",
            (
                "Calibration differences are reported as observed; the engine "
                "was not tuned to match the user comparison. Universe selection, "
                "the exact complete-calendar end, and sizing/accounting semantics "
                "can change the figures."
            ),
            "",
            "Cost-off values are exact reruns with fee=0, slippage=0, and no funding; they are not gross-PnL approximations.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {_markdown(warning)}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    heading = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<div class=\"table-wrap\"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"


def render_html(summary: Mapping[str, object]) -> str:
    """Render a standalone, escaped HTML report from the same summary."""
    validate_summary(summary)
    data = _require_mapping(summary["data"], "data")
    research = _require_mapping(summary["research"], "research")
    calibration = _require_mapping(summary["calibration"], "calibration")
    counts = _require_mapping(research["search_counts"], "search_counts")
    outcome = (
        NO_PASS_STATEMENT
        if int(summary["passing_count"]) == 0
        else f"{summary['passing_count']} configuration(s) passed all research gates."
    )
    fold_rows = [
        [
            str(fold["fold"]),
            str(fold["is_start"]),
            str(fold["is_end"]),
            str(fold["oos_start"]),
            str(fold["oos_end"]),
        ]
        for fold in research["folds"]
    ]
    candidate_rows = [_candidate_cells(candidate) for candidate in summary["candidates"]]
    primary = _require_mapping(calibration["primary_returns"], "calibration.primary")
    reference = _require_mapping(
        calibration["user_reference_returns"],
        "calibration.user_reference_returns",
    )
    calibration_rows = [
        [
            "Primary 10% x 8x, max 3",
            _percent(primary["7d"]),
            _percent(primary["30d"]),
            _percent(primary["56d"]),
        ]
    ]
    legacy = calibration.get("legacy_equal_weight_returns")
    if isinstance(legacy, Mapping):
        calibration_rows.append(
            [
                "Legacy equal-weight 1/N x 1x",
                _percent(legacy["7d"]),
                _percent(legacy["30d"]),
                _percent(legacy["56d"]),
            ]
        )
    calibration_rows.append(
        [
            "User comparison",
            _percent(reference["7d"]),
            _percent(reference["30d"]),
            _percent(reference["56d"]),
        ]
    )
    warning_items = "".join(
        f"<li>{escape(str(warning))}</li>" for warning in summary["warnings"]
    )
    universe = ", ".join(str(item) for item in data["universe"])
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Cost-aware OKX strategy research</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:1600px;margin:0 auto;"
        "padding:24px;color:#17202a;background:#f7f9fb}h1,h2{color:#102a43}"
        ".card{background:white;border:1px solid #d9e2ec;border-radius:8px;"
        "padding:16px;margin:16px 0}.table-wrap{overflow:auto}table{border-collapse:"
        "collapse;width:100%;font-size:13px}th,td{border:1px solid #bcccdc;"
        "padding:7px;vertical-align:top;text-align:left}th{background:#eaf2f8;"
        "position:sticky;top:0}td:nth-child(n+8){white-space:nowrap}pre{white-space:"
        "pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:10px}"
        ".outcome{font-weight:700}.warning{border-left:4px solid #d64545}"
        "</style></head><body>"
        "<h1>Cost-aware OKX strategy research</h1>"
        f"<p>Generated: <code>{escape(str(summary['generated_at']))}</code></p>"
        f'<section class="card"><h2>Outcome</h2><p class="outcome">{escape(outcome)}</p>'
        "<p>All family/timeframe candidates remain listed below, including failures.</p></section>"
        '<section class="card"><h2>Data and execution contract</h2>'
        f"<p>Data snapshot generated: <code>{escape(str(data['metadata_generated_at']))}</code>. "
        f"Source: {escape(str(data['source']))}.</p>"
        f"<p>Requested complete-calendar coverage: <code>{escape(str(data['requested_start']))}</code> "
        f"through <code>{escape(str(data['requested_end']))}</code>.</p>"
        f"<p>Universe ({len(data['universe'])}): {escape(universe)}.</p>"
        f"<p>Coverage: {escape(str(data['coverage']))}. "
        f"Timeframes: {escape(', '.join(data['timeframes']))}.</p>"
        f"<p>Cost model:</p><pre>{escape(_json(research['cost_model']))}</pre>"
        f"<p>Primary sizing:</p><pre>{escape(_json(research['primary_sizing']))}</pre>"
        f"<p>5% x 5x reference sizing:</p><pre>{escape(_json(research['reference_sizing']))}</pre>"
        f"<p>Calendar return windows end at <code>{escape(str(research['calendar_window_end']))}</code>: "
        f"{escape(str(research['calendar_window_definition']))}.</p>"
        "<p>Signals use the preceding fully closed bar and enter at the next open; "
        "each independent window starts flat.</p></section>"
        '<section class="card"><h2>Walk-forward folds</h2>'
        + _html_table(
            ("Fold", "IS start", "IS end", "OOS start", "OOS end"),
            fold_rows,
        )
        + "</section>"
        '<section class="card"><h2>Full bounded search</h2>'
        f"<p>Searched {escape(str(counts['strategies']))} strategies × "
        f"{escape(str(counts['timeframes']))} timeframes; "
        f"{escape(str(counts['grid_configurations']))} complete grid configurations "
        f"and {escape(str(counts['is_fold_evaluations']))} IS fold evaluations.</p>"
        f"<p>Neighbor rule: {escape(str(research['neighbor_rule']))}.</p>"
        f"<p>Passing gates:</p><pre>{escape(_json(research['passing_gates']))}</pre>"
        + _html_table(_CANDIDATE_HEADERS, candidate_rows)
        + "</section>"
        '<section class="card"><h2>Supplied 15m Donchian calibration</h2>'
        f"<p>Exact params (both directions):</p><pre>{escape(_json(calibration['params']))}</pre>"
        + _html_table(("Sizing / source", "7d", "30d", "56d"), calibration_rows)
        + (
            "<p>Calibration differences are reported as observed; the engine "
            "was not tuned to match the user comparison. Universe selection, "
            "the exact complete-calendar end, and sizing/accounting semantics "
            "can change the figures.</p>"
        )
        + "</section>"
        '<section class="card warning"><h2>Limitations</h2>'
        "<p>Cost-off values are exact reruns with fee=0, slippage=0, and no "
        "funding; they are not gross-PnL approximations.</p>"
        f"<ul>{warning_items}</ul></section>"
        "</body></html>\n"
    )


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_reports(
    summary: Mapping[str, object],
    markdown_path: Path | str,
    html_path: Path | str,
) -> None:
    """Validate once, then atomically write Markdown and standalone HTML."""
    validate_summary(summary)
    _write_text_atomic(Path(markdown_path), render_markdown(summary))
    _write_text_atomic(Path(html_path), render_html(summary))
