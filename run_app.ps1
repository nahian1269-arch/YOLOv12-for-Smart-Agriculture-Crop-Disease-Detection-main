$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:MPLCONFIGDIR = Join-Path $Root '.matplotlib'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
Set-Location $Root

$Python = Join-Path $Root '.venv\Scripts\python.exe'
$App = Join-Path $Root 'app.py'
$Log = Join-Path $Root 'flask-live.log'

& cmd.exe /c "`"$Python`" -u `"$App`" > `"$Log`" 2>&1"
