#!/bin/bash
set -e

# Define the venv directory
VENV_DIR=".venv"

echo "Creating virtual environment in ./$VENV_DIR ..."
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
pip install auraloss

echo ""
echo "Setup complete."
echo "To activate the environment in your shell, run:"
echo "source $VENV_DIR/bin/activate" 