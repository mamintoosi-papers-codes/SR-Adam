@echo off
REM ============================================================================
REM SR-Adam Full Reproducibility Pipeline
REM ============================================================================
REM Automated batch script for full SR-Adam research pipeline:
REM 1. Run experiments (30 configurations × 5 runs)
REM 2. Aggregate results
REM 3. Generate tables and figures
REM 4. Compile paper with LaTeX
REM
REM Prerequisites:
REM   - conda environment 'pth' with PyTorch installed
REM   - pdflatex in PATH (for paper compilation)
REM
REM Usage:
REM   reproduce_all.bat
REM
REM For argument details, see: REPRODUCE.md
REM Expected runtime: 2-4 hours on GPU
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo SR-Adam: Full Reproducibility Pipeline
echo ============================================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda not found in PATH
    echo Please install Anaconda/Miniconda first
    exit /b 1
)

REM Activate pth environment
call conda activate pth
if errorlevel 1 (
    echo ERROR: Failed to activate 'pth' environment
    exit /b 1
)

echo [OK] Environment activated
echo.
echo ============================================================================
echo Step 1: Running Experiments
echo ============================================================================
echo Note: All noise levels [0.0, 0.05, 0.1] are processed automatically
echo.

python main.py --dataset ALL --model simplecnn --optimizers ALL --num_epochs 20 --num_runs 5 --batch_size 512
if errorlevel 1 goto error

echo.
echo [OK] Experiments completed!
echo.

REM ============================================================================
REM Step 2: Aggregate Results
REM ============================================================================
echo ============================================================================
echo Step 2: Aggregating Results
echo ============================================================================
echo.

python regenerate_aggregates.py
if errorlevel 1 goto error

python regenerate_summary_statistics.py
if errorlevel 1 goto error

echo [OK] Results aggregated!
echo.

REM ============================================================================
REM Step 3: Generate Publication Tables and Figures
REM ============================================================================
echo ============================================================================
echo Step 3: Generating Publication-Ready Tables and Figures
echo ============================================================================
echo.

python generate_architecture_table.py
if errorlevel 1 goto error

python make_minimal_tables.py
if errorlevel 1 goto error

python make_testacc_plots_simplecnn.py
if errorlevel 1 goto error

python make_loss_plots_simplecnn.py
if errorlevel 1 goto error

echo [OK] All tables and figures generated!
echo.

REM ============================================================================
REM Step 4: Compile Paper
REM ============================================================================
echo ============================================================================
echo Step 4: Compiling Final Paper
echo ============================================================================
echo.

cd paper_figures

pdflatex -interaction=nonstopmode paper-draft.tex >nul 2>&1
pdflatex -interaction=nonstopmode paper-draft.tex >nul 2>&1

if exist paper-draft.pdf (
    echo [OK] Paper compiled: paper_figures\paper-draft.pdf
) else (
    echo ERROR: paper-draft.pdf not created
    goto error_paper
)

cd ..

echo.
echo ============================================================================
echo [SUCCESS] FULL PIPELINE COMPLETED!
echo ============================================================================
echo.
echo Output:
echo   - Paper:     paper_figures\paper-draft.pdf
echo   - Tables:    paper_figures\*.tex
echo   - Figures:   paper_figures\*.pdf
echo   - Results:   results\summary_statistics.csv
echo.
pause
exit /b 0

:error
echo.
echo [ERROR] Pipeline failed. Check output above.
pause
exit /b 1

:error_paper
echo.
echo [ERROR] Paper compilation failed. Check paper_figures\paper-draft.log
pause
exit /b 1

