@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title Offline Japanese ASMR Subtitles
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\bootstrap.ps1" -Mode run %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%ASMR_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
