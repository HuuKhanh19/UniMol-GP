#!/bin/bash
# Setup script for CONAN Project

set -e

echo "=============================================="
echo "CONAN Project Setup"
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
mkdir -p experiments/step1_baseline
mkdir -p experiments/step2_eggroll
mkdir -p experiments/step3_gp
mkdir -p logs

# Check if unimol_tools exists
if [ ! -d "unimol_tools" ]; then
    echo ""
    echo "Cloning UniMol-tools..."
    git clone https://github.com/dptech-corp/Uni-Mol.git temp_unimol
    mv temp_unimol/unimol_tools ./unimol_tools
    rm -rf temp_unimol
else
    echo "UniMol-tools already exists, skipping clone."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."

# Check if we're in a conda environment
if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda environment detected: $CONDA_PREFIX"
else
    echo "Warning: No conda environment detected."
    echo "Please activate your conda environment (e.g., conda activate conan_es)"
fi

# Install unimol_tools from source
echo ""
echo "Installing UniMol-tools from source..."
cd unimol_tools
pip install -e . --quiet
cd ..

# Install additional requirements
echo ""
echo "Installing additional requirements..."
pip install pandas scikit-learn pyyaml rdkit --quiet

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Upload your data files to data/raw/"
echo "   - refined_ESOL.csv"
echo "   - refined_FreeSolv.csv"
echo "   - refined_Lipophilicity.csv"
echo "   - refined_BACE.csv"
echo ""
echo "2. Preprocess data:"
echo "   python scripts/preprocess_data.py --dataset all"
echo ""
echo "3. Run Step 1 training:"
echo "   python scripts/run_step1.py --dataset all"
echo ""
