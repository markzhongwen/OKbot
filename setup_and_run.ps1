$ErrorActionPreference = 'Stop'
$root = 'C:\Users\matth\okx-bt'
Set-Location $root
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
if (Test-Path 'C:\Program Files\Git\cmd\git.exe') { $env:Path = 'C:\Program Files\Git\cmd;' + $env:Path }

# extract archives if needed
if (-not (Test-Path "$root\intraday\search3m.py")) {
  if (Test-Path "$root\okx-bt-code.tgz") {
    tar -xzf "$root\okx-bt-code.tgz" -C $root
  }
}
if (-not (Test-Path "$root\ud\data\okx\futures")) {
  if (Test-Path "$root\okx-bt-data.tgz") {
    tar -xzf "$root\okx-bt-data.tgz" -C $root
  }
}

python -m venv "$root\venv"
& "$root\venv\Scripts\python.exe" -m pip install -U pip
& "$root\venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt"
& "$root\venv\Scripts\python.exe" "$root\patch_paths.py" $root
& "$root\venv\Scripts\python.exe" "$root\make_report_html.py" --root $root

# start search3m in background if not running
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'search3m.py' }
if (-not $running) {
  Start-Process -FilePath "$root\venv\Scripts\python.exe" -ArgumentList @('-u',"$root\intraday\search3m.py") -WorkingDirectory "$root\intraday" -WindowStyle Hidden -RedirectStandardOutput "$root\reports\search3m.out.log" -RedirectStandardError "$root\reports\search3m.err.log"
  'search3m started'
} else { 'search3m already running' }

# scheduled task every 4 hours for HTML report
$action = New-ScheduledTaskAction -Execute "$root\venv\Scripts\python.exe" -Argument "`"$root\make_report_html.py`" --root `"$root`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddHours([math]::Ceiling((Get-Date).Hour/4)*4) -RepetitionInterval (New-TimeSpan -Hours 4) -RepetitionDuration ([TimeSpan]::MaxValue)
Register-ScheduledTask -TaskName 'OKX-BT-Screen-Report' -Action $action -Trigger $trigger -Description 'OKX strategy screen HTML every 4h' -Force | Out-Null
'report task registered'
Get-ChildItem $root\intraday\search3m.py, $root\ud\data\okx\futures -ErrorAction SilentlyContinue | Select-Object -First 5 FullName
