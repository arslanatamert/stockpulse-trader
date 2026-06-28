#!/bin/bash
set -e

echo "=== StockPulse Trader Setup ==="

# Install Python 3.11 via Homebrew if not present
if ! command -v python3.11 &>/dev/null; then
  echo "Installing Python 3.11 via Homebrew..."
  brew install python@3.11
fi

PYTHON=$(command -v python3.11 || command -v python3)

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv .venv

# Activate and install
echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env"
echo "  2. Edit .env and add your ANTHROPIC_API_KEY"
echo "  3. source .venv/bin/activate"
echo "  4. streamlit run app.py"
