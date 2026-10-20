$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$py    = "W:\Anaconda\python.exe"
$cf    = Join-Path $env:LOCALAPPDATA "Programs\cloudflared\cloudflared.exe"
$tmp   = $env:TEMP
$cfErr = Join-Path $tmp "myimg-cloudflared.log"
$cfOut = Join-Path $tmp "myimg-cloudflared.out.log"
$stOut = Join-Path $tmp "myimg-streamlit.log"
$stErr = Join-Path $tmp "myimg-streamlit.err.log"

Write-Host "== myimg 一键启动 =="
Write-Host ("repo: " + $root)

# ---- 1) streamlit：8501 已被占用就复用，不重复起 ----
if (Get-NetTCPConnection -State Listen -LocalPort 8501 -ErrorAction SilentlyContinue) {
    Write-Host "[streamlit] 8501 已在监听，复用现有进程"
} else {
    if (-not (Test-Path $py)) { throw "找不到 Python: $py" }
    Start-Process -NoNewWindow -FilePath $py -WorkingDirectory $root `
        -ArgumentList "-X","utf8","-m","streamlit","run","web/app.py","--server.headless","true","--server.port","8501" `
        -RedirectStandardOutput $stOut -RedirectStandardError $stErr
    Write-Host "[streamlit] 启动中…"
}
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop http://localhost:8501/_st/health
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) { throw "streamlit 60 秒没起来，看 $stErr" }
Write-Host "[streamlit] OK  http://localhost:8501"

# ---- 2) 公网隧道：重启 cloudflared，保证拿到的是本次的新链接 ----
if (-not (Test-Path $cf)) { throw "找不到 cloudflared: $cf" }
$old = Get-Process cloudflared -ErrorAction SilentlyContinue
if ($old) {
    Write-Host ("[tunnel] 停掉旧隧道 pid: " + ($old.Id -join ", "))
    Stop-Process -Id $old.Id -Force
    Start-Sleep -Seconds 1
}
Remove-Item $cfErr -ErrorAction SilentlyContinue
Start-Process -NoNewWindow -FilePath $cf `
    -ArgumentList "tunnel","--url","http://localhost:8501","--no-autoupdate" `
    -RedirectStandardOutput $cfOut -RedirectStandardError $cfErr
$url = $null
for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $cfErr) {
        $m = Select-String -Path $cfErr -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value; break }
    }
}
if ($url) {
    Write-Host ("[tunnel] OK  " + $url)
    try { Set-Clipboard -Value $url; Write-Host "[tunnel] 公网链接已复制到剪贴板" } catch { }
} else {
    Write-Host "[tunnel] 45 秒没拿到链接，看 $cfErr"
}

# ---- 3) 汇总三条链接 ----
$lan = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
        Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "本机       http://localhost:8501"
if ($lan) { Write-Host ("同局域网   http://" + $lan + ":8501   （防火墙规则 myimg-streamlit-8501 已放行）") }
if ($url) { Write-Host ("公网       " + $url + "   （隧道进程/休眠即失效，重启换链接）") }
Write-Host "停止：双击 tools\stop_demo.bat"