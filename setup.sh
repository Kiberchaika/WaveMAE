#!/bin/bash
set -e

echo "Creating virtual environment in ./.venv ..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "To activate the environment in your shell, run:"
echo "source .venv/bin/activate" 