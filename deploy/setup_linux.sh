#!/usr/bin/env bash
set -euo pipefail

echo "==================================================="
echo "   GeoWeedStation Linux Environment Setup"
echo "==================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 1. Check and install critical system packages (Ubuntu/Debian)
CRITICAL_PACKAGES=(
    "python3-venv"
    "libxcb-cursor0"
    "libxcb-xinerama0"
    "libxcb-icccm4"
    "libxcb-image0"
    "libxcb-keysyms1"
    "libxcb-render-util0"
    "libxkbcommon-x11-0"
    "libegl1"
    "libgl1"
)

MISSING_PACKAGES=()
if command -v dpkg-query &>/dev/null; then
    for pkg in "${CRITICAL_PACKAGES[@]}"; do
        if ! dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
            MISSING_PACKAGES+=("${pkg}")
        fi
    done
fi

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "[!] Missing critical system packages: ${MISSING_PACKAGES[*]}"
    if [ "$(id -u)" -eq 0 ]; then
        echo "[INFO] Running apt-get install as root..."
        apt-get update -y && apt-get install -y --no-install-recommends "${MISSING_PACKAGES[@]}"
    elif command -v sudo &>/dev/null; then
        echo "[INFO] Attempting to install missing packages via sudo..."
        if sudo -n true 2>/dev/null || [ -t 0 ]; then
            sudo apt-get update -y && sudo apt-get install -y --no-install-recommends "${MISSING_PACKAGES[@]}"
        else
            echo "[WARNING] Sudo is non-interactive or requires password."
            echo "Please run manually: sudo apt-get update && sudo apt-get install -y ${MISSING_PACKAGES[*]}"
        fi
    else
        echo "[WARNING] Neither root nor sudo available to install system packages."
        echo "Please install them manually: sudo apt-get update && sudo apt-get install -y ${MISSING_PACKAGES[*]}"
    fi
else
    echo "[OK] All required system libraries are installed."
fi

# 2. Python version check
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    echo "[ERROR] ${PYTHON_BIN} is not installed or not in PATH."
    exit 1
fi

PY_OK="$("${PYTHON_BIN}" -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')"
if [ "${PY_OK}" != "1" ]; then
    PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "[ERROR] Python 3.10+ is required. Found Python ${PY_VER}."
    exit 1
fi
echo "[OK] Detected Python version: $("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"

# 3. Virtual Environment Setup
VENV_DIR="${PROJECT_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ] || [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "[INFO] Creating virtual environment at ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "[INFO] Using existing virtual environment at ${VENV_DIR}."
fi

VENV_PIP="${VENV_DIR}/bin/pip"
VENV_PY="${VENV_DIR}/bin/python"

# 4. Upgrade packaging toolchain
echo "[INFO] Upgrading pip, setuptools, wheel..."
"${VENV_PIP}" install --upgrade pip setuptools wheel

# 5. Install project dependencies in editable mode
echo "[INFO] Installing GeoWeedStation dependencies..."
if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    "${VENV_PIP}" install -e "${PROJECT_ROOT}[dev]"
elif [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    "${VENV_PIP}" install -r "${PROJECT_ROOT}/requirements.txt"
    "${VENV_PIP}" install -e "${PROJECT_ROOT}"
fi

# 6. Ensure Qt / OpenCV compatibility (avoid Qt plugin conflicts)
echo "[INFO] Checking Qt / OpenCV compatibility..."
if "${VENV_PIP}" show opencv-python &>/dev/null; then
    echo "[WARNING] Conflicting opencv-python detected! Replacing with opencv-python-headless..."
    "${VENV_PIP}" uninstall -y opencv-python
    "${VENV_PIP}" install opencv-python-headless
fi
if "${VENV_PIP}" show opencv-contrib-python &>/dev/null; then
    echo "[WARNING] Conflicting opencv-contrib-python detected! Replacing with opencv-python-headless..."
    "${VENV_PIP}" uninstall -y opencv-contrib-python
    "${VENV_PIP}" install opencv-python-headless
fi

# 7. Generate launch scripts in root
LAUNCHER="${PROJECT_ROOT}/run_geoweed.sh"
echo "[INFO] Generating launcher at ${LAUNCHER}..."
cat << 'EOF' > "${LAUNCHER}"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ] || [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found at ${VENV_DIR}."
    echo "Please run './setup.sh' or './deploy/setup_linux.sh' first."
    exit 1
fi

# Activate virtual environment
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Isolate Qt from GNOME/GTK/XFCE theme conflicts and Wayland issues
unset QT_PLUGIN_PATH
unset QML2_IMPORT_PATH
export QT_PLATFORM_PLUGIN=""
export QT_STYLE_OVERRIDE="Fusion"
export QT_QPA_PLATFORMTHEME=""

# Detect offscreen mode or headless environment
IS_OFFSCREEN=0
for arg in "$@"; do
    if [ "$arg" = "--offscreen" ]; then
        IS_OFFSCREEN=1
        break
    fi
done

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    IS_OFFSCREEN=1
fi

if [ "$IS_OFFSCREEN" -eq 1 ]; then
    export QT_QPA_PLATFORM="offscreen"
else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/main.py" "$@"
EOF
chmod +x "${LAUNCHER}"

# Also generate or update geoweedstation.sh for backward compatibility
ln -sf "run_geoweed.sh" "${PROJECT_ROOT}/geoweedstation.sh" 2>/dev/null || cp "${LAUNCHER}" "${PROJECT_ROOT}/geoweedstation.sh"
chmod +x "${PROJECT_ROOT}/geoweedstation.sh"

echo "==================================================="
echo "   GeoWeedStation setup completed successfully!    "
echo "==================================================="
echo "Launch application:         ./run_geoweed.sh"
echo "Run self-diagnostics:       ./run_geoweed.sh --selftest --offscreen"
