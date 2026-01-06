@echo off
REM ============================================================================
REM SR-Adam Full Reproducibility Pipeline
REM ============================================================================
REM Automated batch script for full SR-Adam research pipeline:
REM 1. Run experiments (multiple datasets × noise levels × optimizers × runs)
REM 2. Aggregate results with optional filtering
REM 3. Generate tables and figures
REM 4. Compile paper with LaTeX
REM
REM Prerequisites:
REM   - conda environment 'pth' with PyTorch installed
REM   - pdflatex in PATH (for paper compilation)
REM
REM Usage:
REM   reproduce_all.bat                    (default: all results, batch_size=512)
REM   reproduce_all.bat --ablation         (include ablation study results)
REM
REM Advanced filtering (after experiments):
REM   python tools\regenerate_aggregates.py --batch_size 512
REM   python tools\make_minimal_tables.py --batch_size "256|512|2048"
REM   python tools\make_figures.py --dataset CIFAR10 --noise 0.05
REM
REM For detailed argument options, see: QUICK_START.md
REM Expected runtime: 2-4 hours on GPU (full pipeline)
REM ============================================================================

setlocal enabledelayedexpansion

set "ABLATION_MODE=0"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--ablation" set "ABLATION_MODE=1"
shift
goto parse_args
:args_done

echo.
echo ============================================================================
echo SR-Adam: Full Reproducibility Pipeline
echo ============================================================================
if "%ABLATION_MODE%"=="1" (
    echo Mode: WITH ABLATION STUDY ^(multiple batch sizes^)
) else (
    echo Mode: STANDARD ^(batch_size=512 only^)
)
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

REM ============================================================================
REM Step 1: Running Experiments
REM ============================================================================
echo ============================================================================
echo Step 1: Running Experiments
echo ============================================================================
echo.

if "%ABLATION_MODE%"=="1" (
    echo Running with ablation study: batch_sizes = 256^|512^|2048
    echo This will take longer...
    python main.py ^
      --dataset ALL ^
      --model simplecnn ^
      --optimizers ALL ^
      --num_epochs 20 ^
      --num_runs 5 ^
      --noise ALL ^
      --batch_size "256|512|2048"
) else (
    echo Running standard experiments: batch_size = 512
    echo Note: All noise levels [0.0, 0.05, 0.1] are processed automatically
    python main.py ^
      --dataset ALL ^
      --model simplecnn ^
      --optimizers ALL ^
      --num_epochs 20 ^
      --num_runs 5 ^
      --noise ALL ^
      --batch_size 512
)

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

if "%ABLATION_MODE%"=="1" (
    echo Aggregating all batch sizes: 256, 512, 2048
    python tools\regenerate_aggregates.py --batch_size "256|512|2048"
    if errorlevel 1 goto error
    
    python tools\regenerate_summary_statistics.py --batch_size "256|512|2048"
    if errorlevel 1 goto error
) else (
    echo Aggregating batch_size=512 only
    python tools\regenerate_aggregates.py --batch_size 512
    if errorlevel 1 goto error
    
    python tools\regenerate_summary_statistics.py --batch_size 512
    if errorlevel 1 goto error
)

echo [OK] Results aggregated!
echo.

REM ============================================================================
REM Step 3: Generate Publication Tables and Figures
REM ============================================================================
echo ============================================================================
echo Step 3: Generating Publication-Ready Tables and Figures
echo ============================================================================
echo.

python tools\generate_architecture_table.py
if errorlevel 1 goto error

if "%ABLATION_MODE%"=="1" (
    echo Generating tables for all batch sizes
    python tools\make_minimal_tables.py --batch_size "256|512|2048" --output paper_figures\minimal-tables-content.tex
    if errorlevel 1 goto error
    
    python tools\make_minimal_tables.py --batch_size "256|512|2048" --output paper_figures\ablation-batch-size.tex
    if errorlevel 1 goto error
    
    python tools\make_figures.py --batch_size "256|512|2048"
    if errorlevel 1 goto error
) else (
    echo Generating tables for batch_size=512
    python tools\make_minimal_tables.py --batch_size 512
    if errorlevel 1 goto error
    
    python tools\make_figures.py --batch_size 512
    if errorlevel 1 goto error
)

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
if "%ABLATION_MODE%"=="1" (
    echo Ablation study outputs:
    echo   - Ablation table: paper_figures\ablation-batch-size.tex
    echo   - Ablation figures: paper_figures\ablation\*.pdf
    echo.
)
echo ============================================================================
echo Advanced Usage - Custom Filtering:
echo ============================================================================
echo.
echo After running experiments, you can regenerate outputs with filters:
echo.
echo   # Aggregate specific batch sizes
echo   python tools\regenerate_aggregates.py --batch_size "256|512"
echo.
echo   # Generate tables for specific dataset/noise
echo   python tools\make_minimal_tables.py --dataset CIFAR10 --noise 0.05
echo.
echo   # Compare only Adam and SR-Adam
echo   python tools\make_minimal_tables.py --optimizers "Adam|SR-Adam"
echo.
echo For more examples, see: QUICK_START.md
echo ============================================================================
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

