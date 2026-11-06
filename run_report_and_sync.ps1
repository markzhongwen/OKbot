$ErrorActionPreference = "Continue"
$root = "C:\Users\matth\okx-bt"
& "$root\venv\Scripts\python.exe" "$root\make_report_html.py" --root $root
$key = "$env:USERPROFILE\.ssh\vultr_okx_hk"
$latest = "$root\reports\latest.html"
if (Test-Path $key) {
  scp -i $key -o StrictHostKeyChecking=accept-new $latest "root@45.76.106.159:/var/www/rpt/hk_screen_latest.html"
  scp -i $key -o StrictHostKeyChecking=accept-new $latest "root@45.76.106.159:/opt/okx-bt/reports/hk_screen_latest.html"
}