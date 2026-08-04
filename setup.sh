#!/bin/bash
# Setup script for UniMol-GP

set -e

echo "=============================================="
echo "UniMol-GP Setup"
echo "=============================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p experiments
mkdir -p logs

# unimol_source/ is a vendored, PATCHED fork of Uni-Mol tools -- do not replace it
# with upstream: MolTrain.fit() here honours the external VALID column (so the
# scaffold split is respected) and k-fold has been stripped out.
if [ ! -d "unimol_source/unimol_tools" ]; then
    echo ""
    echo "ERROR: unimol_source/unimol_tools not found -- re-clone the repository."
    exit 1
fi

# Check if we're in a conda environment
echo ""
if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda environment detected: $CONDA_PREFIX"
else
    echo "Warning: No conda environment detected."
    echo "Please activate your conda environment (e.g., conda activate conan_es)"
fi

# Install the patched unimol_tools from source
echo ""
echo "Installing UniMol-tools (patched fork) from source..."
cd unimol_source
pip install -e . --quiet
cd ..

# Install project requirements
echo ""
echo "Installing project requirements..."
pip install -r requirements.txt --quiet

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Put your raw data files in data/raw/"
echo "   - refined_ESOL.csv"
echo "   - refined_FreeSolv.csv"
echo "   - refined_Lipophilicity.csv"
echo "   - refined_BACE.csv"
echo ""
echo "2. Scaffold-split the data (81/9/10):"
echo "   python scripts/preprocess_data.py --dataset all --split-seed 0 1 2 3 4"
echo ""
echo "3. Train the UniMol v1 baseline:"
echo "   python scripts/run_step1.py --dataset esol --split-seed 0"
echo ""
