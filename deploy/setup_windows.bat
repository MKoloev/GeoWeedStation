@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ===================================================
echo   GeoWeedStation Windows Environment Setup
echo ===================================================

:: Determine repository root directory regardless of current working directory
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

:: 1. Check Python installation and version
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo.
    echo Please download and install Python 3.10+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Make sure to check the box:
    echo   "Add python.exe to PATH"
    echo during the installation process.
    echo.
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.10 or newer is required!
    for /f "tokens=*" %%V in ('python --version 2^>^&1') do echo Current version detected: %%V
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%V in ('python --version 2^>^&1') do echo [OK] Python found: %%V

:: 2. Setup Virtual Environment (.venv)
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYW=%VENV_DIR%\Scripts\pythonw.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

if not exist "%VENV_PY%" (
    echo [INFO] Creating virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
) else (
    echo [INFO] Using existing virtual environment at %VENV_DIR%.
)

:: 3. Upgrade packaging tools
echo [INFO] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Could not upgrade pip toolchain. Continuing with existing version...
)

:: 4. Install dependencies in editable mode
echo [INFO] Installing project dependencies...
if exist "%PROJECT_ROOT%\pyproject.toml" (
    "%VENV_PY%" -m pip install -e "%PROJECT_ROOT%[dev]"
) else (
    if exist "%PROJECT_ROOT%\requirements.txt" (
        "%VENV_PY%" -m pip install -r "%PROJECT_ROOT%\requirements.txt"
        "%VENV_PY%" -m pip install -e "%PROJECT_ROOT%"
    )
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation failed! Check network connection or compiler tools.
    pause
    exit /b 1
)

:: 5. Guarantee opencv-python-headless (prevent Qt plugin conflicts)
echo [INFO] Checking Qt / OpenCV compatibility...
"%VENV_PIP%" show opencv-python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [WARNING] Conflicting opencv-python detected! Replacing with opencv-python-headless...
    "%VENV_PIP%" uninstall -y opencv-python
    "%VENV_PIP%" install opencv-python-headless
)

"%VENV_PIP%" show opencv-contrib-python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [WARNING] Conflicting opencv-contrib-python detected! Replacing with opencv-python-headless...
    "%VENV_PIP%" uninstall -y opencv-contrib-python
    "%VENV_PIP%" install opencv-python-headless
)

:: 6. Verify native dependencies (PySide6, onnxruntime) & warn about VC++ Redist
echo [INFO] Verifying native runtime libraries (PySide6, onnxruntime)...
"%VENV_PY%" -c "import PySide6; import onnxruntime" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo =========================================================================
    echo [WARNING] Ошибка загрузки PySide6 или onnxruntime!
    echo.
    echo В системе отсутствуют библиотеки среды выполнения Microsoft Visual C++.
    echo Пожалуйста, скачайте и установите:
    echo   Visual C++ 2015-2022 Redistributable (x64):
    echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo =========================================================================
    echo.
) else (
    echo [OK] Native runtime libraries (PySide6, onnxruntime) loaded successfully.
)

:: 7. Generate launcher scripts in project root
echo [INFO] Generating launchers in project root...

:: Launcher 1: Standard windowless launcher (run_geoweed.bat)
(
echo @echo off
echo setlocal
echo set "SCRIPT_DIR=%%~dp0"
echo set "QT_PLUGIN_PATH="
echo set "QML2_IMPORT_PATH="
echo set "QT_STYLE_OVERRIDE=Fusion"
echo set "PYTHONPATH=%%SCRIPT_DIR%%src;%%PYTHONPATH%%"
echo start "" "%%SCRIPT_DIR%%.venv\Scripts\pythonw.exe" "%%SCRIPT_DIR%%main.py" %%*
) > "%PROJECT_ROOT%\run_geoweed.bat"

:: Launcher 2: Console / Debug launcher (run_geoweed_debug.bat)
(
echo @echo off
echo setlocal
echo set "SCRIPT_DIR=%%~dp0"
echo set "QT_PLUGIN_PATH="
echo set "QML2_IMPORT_PATH="
echo set "QT_STYLE_OVERRIDE=Fusion"
echo set "PYTHONPATH=%%SCRIPT_DIR%%src;%%PYTHONPATH%%"
echo "%%SCRIPT_DIR%%.venv\Scripts\python.exe" "%%SCRIPT_DIR%%main.py" %%*
echo.
echo GeoWeedStation finished with exit code %%ERRORLEVEL%%.
echo.
echo Press any key to close this console window...
echo pause ^>nul
) > "%PROJECT_ROOT%\run_geoweed_debug.bat"

:: Compatibility launcher: geoweedstation.bat
(
echo @echo off
echo call "%%~dp0run_geoweed.bat" %%*
) > "%PROJECT_ROOT%\geoweedstation.bat"

echo ===================================================
echo   GeoWeedStation setup completed successfully!
echo ===================================================
echo Standard launcher (no console):  run_geoweed.bat
echo Debug launcher (with console):   run_geoweed_debug.bat
echo Self-test:                       run_geoweed_debug.bat --selftest --offscreen
echo.
