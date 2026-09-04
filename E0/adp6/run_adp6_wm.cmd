@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_adp6_wm.cmd STAGE [SEED] [--force]
  exit /b 2
)
set "ADP6_STAGE=%~1"
set "ADP6_SEED=%~2"
if "%ADP6_SEED%"=="" set "ADP6_SEED=0"
"E:\conda\Scripts\conda.exe" run -n wm --no-capture-output python -X faulthandler "%~dp0run_adp6.py" --stage "%ADP6_STAGE%" --seed "%ADP6_SEED%" %3
exit /b %ERRORLEVEL%
