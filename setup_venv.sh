#!/bin/bash
# Setup script for Differentiable SFC project
# Creates a Python virtual environment and installs dependencies

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Differentiable SFC: Virtual Environment Setup${NC}"
echo "=================================================="
echo

# Check if Python 3.11+ is available
echo -e "${BLUE}Checking Python version...${NC}"
python3 --version

# Create virtual environment
echo
echo -e "${BLUE}Creating virtual environment 'venv'...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install requirements
echo
echo -e "${BLUE}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Verify installation
echo
echo -e "${BLUE}Verifying installation...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"

echo
echo -e "${GREEN}✓ Virtual environment setup complete!${NC}"
echo
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo
echo "To deactivate, run:"
echo "  deactivate"
echo
echo "To start Jupyter, run:"
echo "  jupyter notebook experiments_notebook.ipynb"
echo
