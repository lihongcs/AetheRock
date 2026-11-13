@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "IPAD_HOST=%~1"
set "LAN_PORT=8766"
if not "%~2"=="" set "LAN_PORT=%~2"
if "%IPAD_HOST%"=="" set "IPAD_HOST=auto"

set "LOG_DIR=%CD%\remote_control_logs\launcher"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_ID=%%I"
set "LOG_FILE=%LOG_DIR%\lan_bridge_%RUN_ID%.log"

call :log HTC LAN bridge launcher
call :log Script dir: %CD%
call :log LAN port: %LAN_PORT%
call :log Launcher log: %LOG_FILE%
call :log

call :log Before this connects:
call :log   1. Connect iPad and Windows to the same Ethernet/LAN.
call :log   2. Open instaumi on the iPad: Data Collection - Windows HTC Control - enable.
call :log   3. Set USB/LAN port in the app to %LAN_PORT%.
call :log   4. Allow Local Network permission if iPad asks.
call :log

call :diagnostics
call :cleanup_stale_processes

if /I "%IPAD_HOST%"=="auto" goto :auto_discover

:continue_with_ip
call :log iPad IP: %IPAD_HOST%
call :log LAN port: %LAN_PORT%
call :log

call :log Testing IP reachability to iPad %IPAD_HOST% ...
ping -n 1 -w 1000 %IPAD_HOST% 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
call :log

call :log Testing TCP reachability to %IPAD_HOST%:%LAN_PORT% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Test-NetConnection -ComputerName '%IPAD_HOST%' -Port %LAN_PORT%" 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
call :log

call :run_bridge htc_usb_bridge.py --host %IPAD_HOST% --port %LAN_PORT%
goto :done

:auto_discover
call :log Mode: auto discovery
call :log LAN port fallback: %LAN_PORT%
call :log Searching for iPad via Bonjour/mDNS...
call :log If discovery fails, rerun with the iPad IP address.
call :log

call :run_bridge htc_usb_bridge.py --discover --port %LAN_PORT%
goto :done

:diagnostics
call :log === Diagnostics ===
call :log Python:
where python 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
python --version 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
call :log
call :log Network adapters:
ipconfig 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
call :log === End diagnostics ===
call :log
exit /b 0

:cleanup_stale_processes
call :log Cleaning stale HTC bridge/collector processes from this folder...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Resolve-Path '.').Path; $targets=@('htc_usb_bridge.py','htc_two_trackers_bridge.py','htc_only.py','htc_two_trackers.py'); Get-CimInstance Win32_Process | Where-Object { $cmd=$_.CommandLine; if(-not $cmd){ return $false }; foreach($target in $targets){ if($cmd -like ('*' + $target + '*') -and $cmd -like ('*' + $root + '*')){ return $true } }; return $false } | ForEach-Object { Write-Host ('[cleanup] stopping pid={0} name={1}' -f $_.ProcessId,$_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
call :log Cleanup finished.
call :log
exit /b 0

:run_bridge
call :log Starting Python HTC LAN bridge...
call :log Command: python %*
call :log
powershell -NoProfile -ExecutionPolicy Bypass -Command "& python %* 2>&1 | Tee-Object -FilePath '%LOG_FILE%' -Append"
set "BRIDGE_EXIT=%ERRORLEVEL%"
call :log
call :log Bridge exited with code %BRIDGE_EXIT%.
call :log Full launcher log: %LOG_FILE%
exit /b %BRIDGE_EXIT%

:done
call :log
call :log If data was not collected, open the latest files in:
call :log   %CD%\remote_control_logs
call :log Look especially at:
call :log   remote_control_events.jsonl
call :log   launcher\lan_bridge_%RUN_ID%.log
pause
exit /b 0

:log
if "%~1"=="" (
    echo.
    >> "%LOG_FILE%" echo.
) else (
    echo %*
    >> "%LOG_FILE%" echo %*
)
exit /b 0
