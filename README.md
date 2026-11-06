# OKbot — OKX strategy screening (HK Windows)

Backtest / search only. **No live trading keys.** Live stays on Tokyo VPS.

## Layout
- `intraday/` — search3m / engine / evolve (VPS-aligned)
- `make_report_html.py` — detailed HTML report
- `run_report_and_sync.ps1` — every 4h report + sync to VPS `:8089/hk_screen_latest.html`

## Report
http://45.76.106.159:8089/hk_screen_latest.html
