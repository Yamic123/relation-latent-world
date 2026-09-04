@echo off
setlocal
"E:\conda\Scripts\conda.exe" run -n wm --no-capture-output python -X faulthandler "%~dp0run_adp8.py" --seed 0 %*
exit /b %ERRORLEVEL%
