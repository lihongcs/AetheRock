@echo off
setlocal
cd /d "%~dp0"

set "IPAD_HOST=%~1"
set "LAN_PORT=8766"
if not "%~2"=="" set "LAN_PORT=%~2"

echo HTC two-trackers LAN bridge launcher
echo Script dir: %CD%

if "%IPAD_HOST%"=="" set "IPAD_HOST=auto"
if /I "%IPAD_HOST%"=="auto" goto :auto_discover

:continue_with_ip
echo iPad IP: %IPAD_HOST%
echo LAN port: %LAN_PORT%
echo.

echo Before this connects:
echo   1. Connect iPad and Windows to the same Ethernet/LAN.
echo   2. Open instaumi on the iPad: Data Collection - Windows HTC Control - enable.
echo   3. Set USB/LAN port in the app to %LAN_PORT%.
echo   4. Allow Local Network permission if iPad asks.
echo.

echo Testing IP reachability to iPad %IPAD_HOST% ...
ping -n 1 -w 1000 %IPAD_HOST%
echo.

echo Testing TCP reachability to %IPAD_HOST%:%LAN_PORT% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Test-NetConnection -ComputerName '%IPAD_HOST%' -Port %LAN_PORT%" 2>nul
echo.

echo Starting Python HTC two-trackers LAN bridge...
python htc_two_trackers_bridge.py --host %IPAD_HOST% --port %LAN_PORT%

pause
exit /b 0

:auto_discover
echo Mode: auto discovery
echo LAN port fallback: %LAN_PORT%
echo.
echo Before this connects:
echo   1. Connect iPad and Windows with Ethernet, or to the same LAN.
echo   2. Open instaumi on the iPad: Data Collection - Windows HTC Control - enable.
echo   3. Set USB/LAN port in the app to %LAN_PORT%.
echo   4. Allow Local Network permission if iPad asks.
echo.
echo Searching for iPad via Bonjour/mDNS...
echo If discovery fails, rerun with the iPad IP address.
echo.
python htc_two_trackers_bridge.py --discover --port %LAN_PORT%

pause
exit /b 0
