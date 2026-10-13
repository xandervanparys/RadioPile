#!/usr/bin/env bash
set -e

ENV_NAME=".venv_pipeline"

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "Using Python: $PYTHON_BIN"

"$PYTHON_BIN" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))' || {
  echo "Python 3.11 to 3.13 is required (the pinned wheels do not exist for other versions)."
  exit 1
}

# 1. Create venv if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
  echo "Creating virtual environment in $ENV_NAME..."
  $PYTHON_BIN -m venv "$ENV_NAME"
else
  echo "Virtual environment '$ENV_NAME' already exists, skipping creation."
fi

# 2. Activate venv
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"

# 3. Upgrade pip & install requirements
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Registering Jupyter kernel..."

python -m ipykernel install --user \
  --name venv_pipeline \
  --display-name "venv_pipeline"

echo "Jupyter kernel registered as: venv_pipeline"

echo
echo "Environment setup complete."
echo "To activate it next time, run:"
echo "  source $ENV_NAME/bin/activate"
