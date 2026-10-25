@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title Clean ASMR Model Cache
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\bootstrap.ps1" -Mode clean
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%ASMR_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
