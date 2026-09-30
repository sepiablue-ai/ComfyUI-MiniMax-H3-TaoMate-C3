@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "C3_EXIT=%ERRORLEVEL%"
if not "%C3_EXIT%"=="0" echo TaoMate C3 setup stopped. Read the message above.
if "%~1"=="" pause
exit /b %C3_EXIT%
