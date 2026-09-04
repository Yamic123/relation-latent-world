@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_adp7_wm.cmd STAGE [SEED] [--force]
  exit /b 2
)
set "ADP7_SEED=%~2"
if "%ADP7_SEED%"=="" set "ADP7_SEED=0"
"E:\conda\Scripts\conda.exe" run -n wm --no-capture-output python -X faulthandler "%~dp0run_adp7.py" --stage "%~1" --seed "%ADP7_SEED%" %3
exit /b %ERRORLEVEL%
