# Cost-aware OKX perpetual strategy research design

## Objective

Build a reproducible, research-only pipeline that finds OKX USDT-margined
perpetual strategies whose out-of-sample returns remain positive after the live
engine's costs and timing are reproduced. The pipeline must state plainly when
no configuration passes.

## Non-negotiable execution model

- Signals use only a fully closed candle.
- Market entry occurs at the next candle open (`signal_lag_bars=1`).
- Entry and exit each pay `fee_per_side=0.0005` and
  `slippage_per_side=0.0002`.
- Slippage worsens both fills: buys fill above the reference price and sells
  fill below it.
- Funding is applied at the published funding timestamp only when a position is
  already open; positive funding is paid by longs and received by shorts.
- Same-candle stop and take-profit ambiguity is resolved conservatively in
  favor of the stop.
- All cost and timing values are explicit configuration fields.

## Data

Select approximately 20 active USDT-SWAP instruments by current 24-hour quote
volume while forcing BTC, ETH, and SOL into the universe. Download at least 90
complete UTC calendar days from OKX's public daily candle archive and realized
funding history from the public API. Build 3m bars from 1m source data, then
derive 15m, 1h, and 4h bars from the same aligned source. Cache data under
`data/okx_cost_aware/`, which is gitignored.

Reject incomplete bars, duplicate timestamps, non-positive OHLC values, and
bars violating OHLC bounds. Do not forward-fill prices across missing source
minutes.

## Research pipeline

Implement focused representatives of the requested families: Donchian and
Keltner trend/breakout, RSI and rolling-VWAP mean reversion, extreme
return/volume reversal including short-only variants, cross-sectional momentum
winner/loser rotation, and regime-gated long/short signals.

Use deterministic bounded grids. For each family/timeframe:

1. Rolling walk-forward folds optimize on the preceding in-sample window.
2. Evaluate only the selected parameters on the immediately following OOS
   window.
3. Concatenate non-overlapping OOS trades/equity across folds.
4. Require positive OOS return in a strict majority of usable folds.
5. Require at least 30 concatenated OOS trades.
6. Require maximum drawdown no worse than -30%.
7. Require at least 70% of one-step neighboring parameter configurations to
   have positive OOS return.

Rank passing configurations by OOS Sharpe, then OOS return. Report the full
search even if the passing set is empty. The primary sizing is 10% current
equity margin at 8x notional with at most three simultaneous positions; the
reference comparison is 5% margin at 5x with the same signal stream and at most
three positions.

## Reports and calibration

Generate `reports/cost_aware_results.md` and
`reports/cost_aware_results.html`. Every passing row contains exact parameters,
OOS return, last 7/30/90 complete-calendar-day returns, Sharpe, maximum
drawdown, trades, win rate, and cost-disabled return, plus the 5%x5 reference
result.

Also run the supplied 15m Donchian calibration
(`n=20, sl=3.5, tp=4, ts=24, cool=6`, both directions). Differences from the
given 7/30/56-day figures are documented rather than silently tuning the engine
to match, because the selected universe and exact end timestamp may differ.

## Verification

Unit tests cover next-open fills, adverse slippage, round-trip fees, long and
short funding signs, funding boundary timing, conservative intrabar exits, no
lookahead in indicators, portfolio position caps, calendar-window slicing, and
walk-forward/neighbor pass gates. An integration smoke test uses synthetic
multi-instrument candles. The final report records the data timestamp,
instrument universe, search count, fold definitions, cost model, and warnings
against immediate live deployment.
