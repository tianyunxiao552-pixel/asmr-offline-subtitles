@echo off
setlocal DisableDelayedExpansion
start "" powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0app\view_progress.ps1"
