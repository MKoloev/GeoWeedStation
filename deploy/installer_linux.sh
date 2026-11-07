#!/usr/bin/env bash
# Backward compatibility wrapper for deploy/setup_linux.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/setup_linux.sh" "$@"
