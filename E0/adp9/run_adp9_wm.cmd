@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_adp9_wm.cmd STAGE [--force]
  exit /b 2
)
"E:\conda\Scripts\conda.exe" run -n wm --no-capture-output python -X faulthandler "%~dp0run_adp9.py" --stage "%~1" --seed 0 %2
exit /b %ERRORLEVEL%
