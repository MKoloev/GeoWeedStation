#!/usr/bin/env bash
set -euo pipefail

echo "=== GeoWeedStation Linux Environment Installer ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 1. Python version check
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    echo "ERROR: ${PYTHON_BIN} is not installed or not in PATH."
    exit 1
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Detected Python version: ${PY_VER}"

# 2. Virtual Environment Setup
VENV_DIR="${PROJECT_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PIP="${VENV_DIR}/bin/pip"
VENV_PY="${VENV_DIR}/bin/python"

# 3. Upgrade pip and install dependencies
echo "Upgrading pip, setuptools, wheel..."
"${VENV_PIP}" install --quiet --upgrade pip setuptools wheel

if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    "${VENV_PIP}" install --quiet -r "${PROJECT_ROOT}/requirements.txt"
fi

# Install local package in editable mode
if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    echo "Installing GeoWeedStation package in editable mode..."
    "${VENV_PIP}" install --quiet -e "${PROJECT_ROOT}"
fi

# 4. Create local runner script
LAUNCHER="${PROJECT_ROOT}/geoweedstation.sh"
cat << 'EOF' > "${LAUNCHER}"
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
exec "${SCRIPT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/main.py" "$@"
EOF
chmod +x "${LAUNCHER}"

echo "=== GeoWeedStation installation completed successfully! ==="
echo "You can run the application with: ${LAUNCHER}"
