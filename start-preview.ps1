param([string]$PythonPath = "", [int]$Port = 8501, [switch]$ListenAll)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:MPLCONFIGDIR) {
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'work/matplotlib-cache'
    New-Item -ItemType Directory -Path $env:MPLCONFIGDIR -Force | Out-Null
}
if (-not $PythonPath) {
    $candidates = @('.venv/Scripts/python.exe', 'work/t03-device-a/.venv/Scripts/python.exe')
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c 'import av, streamlit, boxing_coach' 2>$null
            if ($LASTEXITCODE -eq 0) { $PythonPath = $candidate; break }
        }
    }
}
if (-not $PythonPath) { throw 'Install requirements.lock and this project first; see docs/ENGINEERING.md.' }
$bindAddr = if ($ListenAll) { '0.0.0.0' } else { '127.0.0.1' }
if ($ListenAll) {
    $lanIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1 -ExpandProperty IPAddress)
    if ($lanIp) {
        Write-Host "Phone / LAN URL: http://${lanIp}:${Port}/?mobile=1"
    }
}
Write-Host "Open http://${bindAddr}:${Port} in a browser. Press Ctrl+C to stop."
& $PythonPath -m streamlit run app/main.py --server.address=$bindAddr --server.port=$Port --server.headless=true --browser.gatherUsageStats=false
exit $LASTEXITCODE
