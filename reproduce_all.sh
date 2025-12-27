#!/bin/bash
# ============================================================================
# SR-Adam Full Reproducibility Pipeline (Linux/Mac)
# ============================================================================
# Automated shell script to:
# 1. Run all experiments (CIFAR10/100, noise 0.0/0.05/0.1, 5 seeds each)
# 2. Aggregate results
# 3. Generate tables and figures
# 4. Compile final paper
#
# Prerequisites:
#   - conda environment 'pth' with PyTorch installed
#   - pdflatex in PATH (for paper compilation)
#
# Expected runtime: 2-4 hours depending on GPU
# Usage: bash reproduce_all.sh
# ============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================================================"
echo "SR-Adam: Full Reproducibility Pipeline"
echo "============================================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda not found in PATH${NC}"
    echo "Please install Anaconda/Miniconda first"
    exit 1
fi

# Check if pth environment exists
if ! conda env list | grep -q "pth"; then
    echo -e "${RED}ERROR: 'pth' conda environment not found${NC}"
    echo "Please create it:"
    echo "  conda create -n pth python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"
echo ""

# Activate pth environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pth

echo ""
echo "============================================================================"
echo "Step 1: Running Experiments"
echo "============================================================================"
echo "This will run 30 configurations:"
echo "   - 2 datasets (CIFAR10, CIFAR100)"
echo "   - 3 noise levels (0.0, 0.05, 0.1)"
echo "   - 5 seeds per config"
echo "Expected time: ~2-3 hours on GPU"
echo ""
echo "Press ENTER to start, or Ctrl+C to cancel"
read

echo -e "${YELLOW}[1/6] CIFAR10, noise=0.0${NC}"
python main.py --dataset CIFAR10 --noise 0.0 --num_epochs 20 --batch_size 2048

echo -e "${YELLOW}[2/6] CIFAR10, noise=0.05${NC}"
python main.py --dataset CIFAR10 --noise 0.05 --num_epochs 20 --batch_size 2048

echo -e "${YELLOW}[3/6] CIFAR10, noise=0.1${NC}"
python main.py --dataset CIFAR10 --noise 0.1 --num_epochs 20 --batch_size 2048

echo -e "${YELLOW}[4/6] CIFAR100, noise=0.0${NC}"
python main.py --dataset CIFAR100 --noise 0.0 --num_epochs 20 --batch_size 2048

echo -e "${YELLOW}[5/6] CIFAR100, noise=0.05${NC}"
python main.py --dataset CIFAR100 --noise 0.05 --num_epochs 20 --batch_size 2048

echo -e "${YELLOW}[6/6] CIFAR100, noise=0.1${NC}"
python main.py --dataset CIFAR100 --noise 0.1 --num_epochs 20 --batch_size 2048

echo ""
echo -e "${GREEN}✓ Experiments completed!${NC}"
echo ""

# ============================================================================
# Step 2: Aggregate Results
# ============================================================================
echo "============================================================================"
echo "Step 2: Aggregating Results"
echo "============================================================================"
echo ""

echo -e "${YELLOW}Running regenerate_aggregates.py${NC}"
python regenerate_aggregates.py

echo -e "${YELLOW}Running regenerate_summary_statistics.py${NC}"
python regenerate_summary_statistics.py

echo -e "${YELLOW}Running regenerate_epoch_pngs.py${NC}"
python regenerate_epoch_pngs.py

echo ""
echo -e "${GREEN}✓ Results aggregated!${NC}"
echo ""

# ============================================================================
# Step 3: Generate Tables and Figures
# ============================================================================
echo "============================================================================"
echo "Step 3: Generating Publication-Ready Tables and Figures"
echo "============================================================================"
echo ""

echo -e "${YELLOW}Generating architecture table${NC}"
python generate_architecture_table.py

echo -e "${YELLOW}Generating method comparison tables${NC}"
python make_minimal_tables.py

echo -e "${YELLOW}Generating accuracy figures${NC}"
python make_testacc_plots_simplecnn.py

echo -e "${YELLOW}Generating loss figures${NC}"
python make_loss_plots_simplecnn.py

echo -e "${YELLOW}Generating comparison figures${NC}"
python make_figures.py

echo ""
echo -e "${GREEN}✓ All tables and figures generated!${NC}"
echo ""

# ============================================================================
# Step 4: Compile Paper
# ============================================================================
echo "============================================================================"
echo "Step 4: Compiling Final Paper"
echo "============================================================================"
echo ""

cd paper_figures

echo -e "${YELLOW}First LaTeX pass (cross-references)${NC}"
pdflatex -interaction=nonstopmode paper-draft.tex > /dev/null 2>&1 || true

echo -e "${YELLOW}Second LaTeX pass (resolve references)${NC}"
pdflatex -interaction=nonstopmode paper-draft.tex > /dev/null 2>&1

if [ -f paper-draft.pdf ]; then
    echo ""
    echo -e "${GREEN}✓ Paper compiled successfully!${NC}"
    echo "Output: paper_figures/paper-draft.pdf"
else
    echo -e "${RED}ERROR: paper-draft.pdf not found after compilation${NC}"
    echo "Check: paper_figures/paper-draft.log"
    exit 1
fi

cd ..

echo ""
echo "============================================================================"
echo -e "${GREEN}✓ FULL PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo "============================================================================"
echo ""
echo "Output locations:"
echo "   - Results: results/"
echo "   - Tables: paper_figures/minimal-tables.tex"
echo "   - Figures: paper_figures/*.pdf"
echo "   - Paper: paper_figures/paper-draft.pdf"
echo ""
echo "Next steps:"
echo "   1. Review paper_figures/paper-draft.pdf"
echo "   2. Check results in results/summary_statistics.csv"
echo "   3. Commit all outputs to version control"
echo ""
