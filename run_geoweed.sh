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
