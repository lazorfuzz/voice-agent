#!/usr/bin/env bash
#
# setup_robinhood.sh — Sets up the Robinhood MCP Agent
#
# Usage:
#   chmod +x setup_robinhood.sh
#   ./setup_robinhood.sh
#
# This script:
#   1. Creates a virtual environment (if not exists)
#   2. Installs required packages
#   3. Creates a .env.robinhood file with your credentials
#   4. Verifies the setup by running a test command

set -e

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$AGENT_DIR/.venv_robinhood"
PYTHON="${VENV_DIR}/bin/python"

echo "=== Robinhood MCP Agent Setup ==="
echo ""

# Detect Python
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "ERROR: Python not found. Please install Python 3.9+."
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version 2>&1)"
echo ""

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment in $VENV_DIR ..."
    $PYTHON_BIN -m venv "$VENV_DIR"
    echo "Done."
else
    echo "[1/4] Virtual environment already exists."
fi
echo ""

# Install packages
echo "[2/4] Installing packages ..."
"$VENV_DIR/bin/pip" install --upgrade pip > /dev/null 2>&1
"$VENV_DIR/bin/pip" install requests > /dev/null 2>&1
echo "Done."
echo ""

# Configure credentials
echo "[3/4] Configuring credentials ..."
ENV_FILE="$AGENT_DIR/.env.robinhood"

if [ -f "$ENV_FILE" ]; then
    echo "Existing .env.robinhood found. Skipping credential setup."
    echo "To update, edit: $ENV_FILE"
else
    echo "No .env.robinhood found. Creating one..."
    echo "# Robinhood MCP Agent credentials" > "$ENV_FILE"
    echo "# Get these from your Robinhood account" >> "$ENV_FILE"
    echo 'ROBINHOOD_USERNAME="your_robinhood_username"' >> "$ENV_FILE"
    echo 'ROBINHOOD_PASSWORD="your_robinhood_password"' >> "$ENV_FILE"
    echo '# ROBINHOOD_MCP_SERVER_URL="http://localhost:8081"  # for MCP mode' >> "$ENV_FILE"
    echo ""
    echo "Created $ENV_FILE"
    echo ""
    echo "IMPORTANT: Edit $ENV_FILE with your actual Robinhood credentials."
    echo "Your username and password are required for direct API mode."
fi
echo ""

# Verify setup
echo "[4/4] Verifying setup ..."
"$VENV_DIR/bin/python" -c "
import sys
sys.path.insert(0, '$AGENT_DIR')
from robinhood_mcp import RobinhoodMCPClient, RobinhoodDirectAPI
from robinhood_agent import RobinhoodTools
print('  [OK] robinhood_mcp imports successfully')
print('  [OK] robinhood_agent imports successfully')
print('  [OK] All modules loaded.')
"
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo ""
echo "  1. Edit $ENV_FILE with your Robinhood credentials"
echo ""
echo "  2. Test direct API mode:"
echo "     $PYTHON_BIN $AGENT_DIR/robinhood_agent.py price AAPL"
echo ""
echo "  3. Run interactive mode:"
echo "     $PYTHON_BIN $AGENT_DIR/robinhood_agent.py --interactive"
echo ""
echo "  4. Integrate with voice agent (see README_robinhood.md)"
echo ""
echo "  5. Run as MCP server (requires FastMCP framework):"
echo "     $PYTHON_BIN $AGENT_DIR/robinhood_mcp.py --server --port 8081"
echo ""
