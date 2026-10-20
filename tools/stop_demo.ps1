$ErrorActionPreference = "Stop"
Write-Host "== myimg 一键停止 =="
$cf = Get-Process cloudflared -ErrorAction SilentlyContinue
if ($cf) { Stop-Process -Id $cf.Id -Force; Write-Host ("[tunnel] 已停 pid " + ($cf.Id -join ", ")) } else { Write-Host "[tunnel] 没在跑" }
$conn = Get-NetTCPConnection -State Listen -LocalPort 8501 -ErrorAction SilentlyContinue
if ($conn) {
    $pids = $conn.OwningProcess | Sort-Object -Unique
    foreach ($p in $pids) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
    Write-Host ("[streamlit] 已停 pid " + ($pids -join ", "))
} else { Write-Host "[streamlit] 8501 没在监听" }
Write-Host "done"