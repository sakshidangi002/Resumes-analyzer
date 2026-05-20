$ports = @(8000, 8001, 8501, 5001)
foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($conn in $connections) {
            $targetPid = $conn.OwningProcess
            if ($targetPid -ne 0 -and $targetPid -ne $PID) {
                Write-Host "Killing Process $targetPid listening on Port $port"
                Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
Start-Sleep -Seconds 2
Write-Host "Starting run_app.py..."
Start-Process python -ArgumentList "run_app.py" -WorkingDirectory "c:\sakshi folder\application\Resume analyzer"
Write-Host "Restart Complete!"
