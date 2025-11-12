@echo off
cd /d "c:\Users\Div\Downloads\Projects\Healwell\ml-api"
python -m py_compile app.py
if %ERRORLEVEL% EQU 0 (
    echo ✅ Syntax check passed
) else (
    echo ❌ Syntax errors found
)
