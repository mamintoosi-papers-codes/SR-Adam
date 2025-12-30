@echo off
REM Batch script to test multiple random seeds for qualitative comparison
REM Find the best visualization by trying different sample selections

setlocal enabledelayedexpansion

set DATASET=%1
set NOISE=%2

if "%DATASET%"=="" set DATASET=CIFAR100
if "%NOISE%"=="" set NOISE=0.05

echo ================================================================
echo Testing multiple random seeds for qualitative comparison
echo Dataset: %DATASET%
echo Noise: %NOISE%
echo ================================================================
echo.

REM Common seed values to try 789 999 1000 
set SEEDS= 42 123 256 512 2021 2022 2025 2026 3030

echo Trying seeds: %SEEDS%
echo.
echo TIP: After each seed, check the generated PDF in paper_figures/
echo      Keep the seed that produces the most informative samples!
echo.

for %%s in (%SEEDS%) do (
    echo ----------------------------------------
    echo Testing seed: %%s
    echo ----------------------------------------
    python generate_qualitative_comparison.py --dataset %DATASET% --noise %NOISE% --seed %%s
    echo.
    echo Generated: paper_figures\qualitative_%DATASET%_noise%NOISE%.pdf
    echo.
    
    REM Pause to allow user to review (comment out for batch mode)
    REM pause
)

echo.
echo ================================================================
echo Completed testing all seeds!
echo ================================================================
echo.
echo NEXT STEPS:
echo 1. Review all generated PDFs in paper_figures/ folder
echo 2. Choose the seed that gives the most informative visualization
echo 3. Document the chosen seed for reproducibility
echo 4. Re-run with chosen seed: python generate_qualitative_comparison.py --dataset %DATASET% --noise %NOISE% --seed YOUR_SEED
echo.

pause
