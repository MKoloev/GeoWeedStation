@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "VENV_PYW=%SCRIPT_DIR%.venv\Scripts\pythonw.exe"
set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment not found at %SCRIPT_DIR%.venv
    echo Please run "setup.bat" first to configure the environment.
    echo.
    pause
    exit /b 1
)

set "QT_PLUGIN_PATH="
set "QML2_IMPORT_PATH="
set "QT_STYLE_OVERRIDE=Fusion"
set "PYTHONPATH=%SCRIPT_DIR%src;%PYTHONPATH%"

if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" "%SCRIPT_DIR%main.py" %*
) else (
    "%VENV_PY%" "%SCRIPT_DIR%main.py" %*
)
