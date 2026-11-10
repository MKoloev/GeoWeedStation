@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
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

"%VENV_PY%" "%SCRIPT_DIR%main.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo GeoWeedStation finished with exit code %EXIT_CODE%.
echo.
echo Press any key to close this console window...
pause >nul
exit /b %EXIT_CODE%
