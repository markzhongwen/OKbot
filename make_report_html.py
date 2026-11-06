#!/usr/bin/env python3
"""Build a detailed HTML strategy-screening report from SEARCH_3m.md / STATUS / champion.json."""
from __future__ import annotations
import argparse, html, json, pathlib, datetime as dt, shutil, subprocess, os

ROOT = pathlib.Path(__file__).resolve().parent


def md_to_html_pre(text: str) -> str:
    return "<pre class='md'>" + html.escape(text) + "</pre>"


def load_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else f"(missing {p.name})"


def pick_status(intra: pathlib.Path) -> str:
    for name in ("SEARCH_3m_STATUS.md", "STATUS.md", "SEARCH_STATUS.md"):
        p = intra / name
        if p.exists():
            return load_text(p)
    return "(no STATUS file yet)"


def load_jsonl_tail(p: pathlib.Path, n: int = 12) -> list:
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            out.append({"raw": line})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    intra = root / "intraday"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(args.out) if args.out else reports / f"screen_{stamp}.html"

    search_md = load_text(intra / "SEARCH_3m.md")
    status_md = pick_status(intra)
    champ = {}
    for cj_name in ("champion.json", "champ.json"):
        cj = intra / cj_name
        if cj.exists():
            try:
                champ = json.loads(cj.read_text(encoding="utf-8"))
            except Exception as e:
                champ = {"error": str(e)}
            break

    recent = load_jsonl_tail(intra / "search_3m.jsonl", 15)
    recent_html = html.escape(json.dumps(recent, ensure_ascii=False, indent=2))

    body = f"""<!doctype html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OKX Strategy Screen {html.escape(now.strftime('%Y-%m-%d %H:%M %Z'))}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#0b1020;color:#e8eefc;line-height:1.45}}
h1,h2{{color:#9ec1ff}}
.card{{background:#141b2f;border:1px solid #243056;border-radius:12px;padding:16px 18px;margin:16px 0}}
.md{{white-space:pre-wrap;font-size:13px;overflow:auto;max-height:70vh}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.badge{{display:inline-block;background:#234;color:#9cf;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px}}
a{{color:#8ec7ff}}
</style>
</head>
<body>
<h1>OKX 策略筛选报告</h1>
<p>
<span class="badge">HK Windows · backtest only</span>
<span class="badge">live 仅东京 VPS</span>
更新时间：{html.escape(now.strftime('%Y-%m-%d %H:%M:%S %Z'))}
</p>
<div class="card">
<h2>Champion</h2>
<pre class="md">{html.escape(json.dumps(champ, ensure_ascii=False, indent=2))}</pre>
</div>
<div class="card">
<h2>SEARCH_3m（周/30日稳健榜）</h2>
{md_to_html_pre(search_md)}
</div>
<div class="card">
<h2>STATUS（当前搜索）</h2>
{md_to_html_pre(status_md)}
</div>
<div class="card">
<h2>最近候选（search_3m.jsonl 尾部）</h2>
<pre class="md">{recent_html}</pre>
</div>
<p style="opacity:.7;font-size:12px">本机只做筛选，不装交易密钥。报告文件：{html.escape(str(out))}</p>
</body></html>"""
    out.write_text(body, encoding="utf-8")
    latest = reports / "latest.html"
    latest.write_text(body, encoding="utf-8")
    # Best-effort push to VPS public report path
    key = pathlib.Path.home() / ".ssh" / "vultr_okx"
    box_key = pathlib.Path(r"C:\Users\matth\.ssh\vultr_okx")
    for k in (key, box_key, pathlib.Path(r"C:\Users\matth\proxy-node\vultr_okx")):
        if k.exists():
            try:
                subprocess.run([
                    "scp", "-i", str(k), "-o", "StrictHostKeyChecking=accept-new",
                    str(latest), "root@45.76.106.159:/var/www/html/hk_screen_latest.html"
                ], check=False, timeout=60)
                subprocess.run([
                    "scp", "-i", str(k), "-o", "StrictHostKeyChecking=accept-new",
                    str(latest), "root@45.76.106.159:/opt/okx-bt/reports/hk_screen_latest.html"
                ], check=False, timeout=60)
            except Exception as e:
                print("scp_fail", e)
            break
    print(out)
    print(latest)


if __name__ == "__main__":
    main()