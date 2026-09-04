#!/usr/bin/env bash
# AegisBank — Loan Default Prediction System Launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🏦 Starting AegisBank — AI-Based Loan Default Prediction"
echo "============================================================"

# Detect Python interpreter (priority: .conda_env -> .venv -> python3)
if [ -f "$SCRIPT_DIR/.conda_env/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.conda_env/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "Using Python: $PYTHON_EXEC"
"$PYTHON_EXEC" --version

# Run sanity smoke tests
echo ""
echo "🔍 Running sanity & model checks..."
"$PYTHON_EXEC" test_predict.py

echo ""
echo "🚀 Launching Web Application..."
echo "📍 Target URL: http://127.0.0.1:5001"
echo ""
echo "Demo Accounts:"
echo "  • Admin:              admin@aegisbank.com      / Admin@1234"
echo "  • Risk Manager:       risk@aegisbank.com       / Risk@1234"
echo "  • Credit Analyst:     analyst@aegisbank.com    / Analyst@1234"
echo "  • Compliance Officer: compliance@aegisbank.com / Comply@1234"
echo "============================================================"
echo ""

exec "$PYTHON_EXEC" webapp/app.py
