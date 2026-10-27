param([string]$PythonPath = "", [int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
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
Write-Host "Open http://127.0.0.1:$Port in a browser. Press Ctrl+C to stop."
& $PythonPath -m streamlit run app/main.py --server.address=127.0.0.1 --server.port=$Port --server.headless=true --browser.gatherUsageStats=false
exit $LASTEXITCODE
