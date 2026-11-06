$ErrorActionPreference = "Continue"
$root = "C:\Users\matth\okx-bt"
& "$root\venv\Scripts\python.exe" "$root\make_report_html.py" --root $root
$key = "$env:USERPROFILE\.ssh\vultr_okx_hk"
$latest = "$root\reports\latest.html"
if (Test-Path $key) {
  scp -i $key -o StrictHostKeyChecking=accept-new $latest "root@45.76.106.159:/tmp/hk_screen_latest.html"
  ssh -i $key root@45.76.106.159 "cp /tmp/hk_screen_latest.html /opt/okx-bt/reports/hk_screen_latest.html; cp /tmp/hk_screen_latest.html /hk_screen_latest.html"
}