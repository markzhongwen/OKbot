# Cost-aware OKX perpetual research implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a tested, reproducible OKX USDT-SWAP walk-forward research pipeline and cost-aware ranked Markdown/HTML reports.

**Architecture:** A small `cost_aware/` package separates immutable research
configuration, public data acquisition, signal generation, event-driven
execution, walk-forward selection, and reporting. A command-line runner joins
the components and persists data/results without coupling them to the legacy
Windows-only scripts.

**Tech Stack:** Python 3.12, NumPy, pandas, Numba where profiling justifies it,
PyArrow, pytest, OKX public HTTP endpoints.

## Global constraints

- Research only: no credentials and no order endpoints.
- Closed-candle signals; next-bar-open entries; no lookahead.
- Fee 0.05% and slippage 0.02% per side, plus realized funding.
- Evaluate 3m, 15m, 1h, and 4h over at least 90 complete days.
- OOS majority-positive folds, neighbor robustness >=70%, OOS trades >=30,
  and maximum drawdown <=30%.
- Report 10% margin x8 leverage and 5% margin x5 leverage.

---

### Task 1: Execution semantics and accounting

**Files:**
- Create: `cost_aware/config.py`
- Create: `cost_aware/engine.py`
- Test: `tests/test_cost_aware_engine.py`

**Interfaces:**
- Produces `ExecutionConfig`, `SizingConfig`, `Trade`, and
  `run_backtest(candles, signals, funding, execution, sizing)`.

- [ ] Write tests asserting next-open entry, adverse entry/exit slippage,
  two-sided fees, funding signs and settlement boundary, stop-first ambiguity,
  and concurrent-position limits.
- [ ] Run `pytest tests/test_cost_aware_engine.py -q` and confirm behavior tests
  fail because the package does not exist.
- [ ] Implement only the execution and accounting required by those tests.
- [ ] Re-run the test module and confirm it passes.
- [ ] Commit the execution unit.

### Task 2: Public OKX data acquisition

**Files:**
- Create: `cost_aware/data.py`
- Create: `scripts/fetch_okx_history.py`
- Test: `tests/test_cost_aware_data.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces `select_universe`, `fetch_daily_archives`,
  `fetch_funding_history`, `validate_ohlcv`, and `resample_ohlcv`.

- [ ] Write synthetic tests for selection, UTC alignment, missing-bar handling,
  resampling, duplicate rejection, and OHLC validation.
- [ ] Run the data tests and observe the expected missing-function failures.
- [ ] Implement public HTTP pagination, retries, cache files, validation, and
  deterministic resampling.
- [ ] Run data tests, then download at least 90 complete days for the selected
  universe and inspect coverage.
- [ ] Commit the data unit without cached market data.

### Task 3: Strategy families without lookahead

**Files:**
- Create: `cost_aware/strategies.py`
- Test: `tests/test_cost_aware_strategies.py`

**Interfaces:**
- Produces `StrategySpec`, `parameter_grid`, `neighbor_params`, and
  `generate_signals`.

- [ ] Write tests proving current-bar data cannot influence prior signals and
  covering one representative configuration per requested family.
- [ ] Run tests and observe expected failures.
- [ ] Implement bounded grids and closed-candle signals for trend/breakout,
  mean reversion, extreme reversal, cross-sectional rotation, and regime gates.
- [ ] Run tests and commit the strategy unit.

### Task 4: Walk-forward selection and metrics

**Files:**
- Create: `cost_aware/walkforward.py`
- Test: `tests/test_cost_aware_walkforward.py`

**Interfaces:**
- Produces `Fold`, `build_folds`, `compute_metrics`, `evaluate_candidate`, and
  `passes_research_gates`.

- [ ] Write tests for non-overlapping OOS folds, IS-only selection,
  majority-positive folds, >=30 trades, <=30% drawdown, and >=70% positive
  neighboring parameters.
- [ ] Run tests and observe expected failures.
- [ ] Implement deterministic fold selection, concatenated OOS accounting, and
  calendar-window metrics.
- [ ] Run tests and commit the validation unit.

### Task 5: Reproducible search and reports

**Files:**
- Create: `cost_aware/run_research.py`
- Create: `cost_aware/report.py`
- Test: `tests/test_cost_aware_report.py`
- Create: `reports/cost_aware_results.md`
- Create: `reports/cost_aware_results.html`

**Interfaces:**
- Consumes all preceding package interfaces.
- Produces machine-readable intermediate results plus Markdown and standalone
  HTML reports.

- [ ] Write report tests requiring exact parameters, OOS and 7/30/90-day
  cost-on returns, Sharpe, drawdown, trades, win rate, cost-off comparison,
  both sizing references, and an explicit no-pass statement.
- [ ] Run report tests and observe expected failures.
- [ ] Implement checkpointed coarse search, walk-forward validation, neighbor
  evaluation, calibration, and report rendering.
- [ ] Execute the complete search on downloaded data and generate both reports.
- [ ] Run report tests and commit the research/report unit.

### Task 6: Final verification and delivery

**Files:**
- Modify only files required by verification findings.

- [ ] Run `pytest -q`.
- [ ] Re-run the calibration and compare 7/30/56-day values with the supplied
  benchmark, documenting universe/window differences.
- [ ] Check report metrics against machine-readable results and verify no
  credential/order endpoint appears in the code.
- [ ] Run the complete pipeline from cache once more for reproducibility.
- [ ] Commit fixes, push the branch, and create a PR against `main`.
