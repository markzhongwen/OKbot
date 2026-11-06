#!/usr/bin/env python3
"""Rewrite hardcoded /opt/okx-bt paths to local Windows root."""
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
old = "/opt/okx-bt"
new = str(root).replace("\\", "/")
# also keep a windows-native form for some strings
count = 0
for p in list((root / "intraday").glob("*.py")) + [root / "search.py", root / "config.json"]:
    if not p.exists():
        continue
    t = p.read_text(encoding="utf-8", errors="replace")
    if old not in t:
        continue
    p.write_text(t.replace(old, new), encoding="utf-8")
    count += 1
    print("patched", p)
print("files", count)
