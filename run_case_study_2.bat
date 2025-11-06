@echo off
REM Batch script to run Case Study 2 reproduction for Windows

echo ================================================================================
echo Reproducing Case Study 2: Tables 3-5 and Figures 9-10
echo ================================================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Available datasets:
echo   0  = tic-tac-toe
echo   1  = wisconsin
echo   2  = coupon_full
echo   3  = compas_complete
echo   4  = wine_quality
echo   5  = broward_general_2y
echo   6  = fico_complete
echo   7  = iris_setosa
echo   8  = iris_versicolor
echo   9  = iris_virginica
echo   10 = spiral
echo.

set /p CHOICE="Run all datasets (y/n) or enter specific indices (e.g., 0 3 4): "

if /i "%CHOICE%"=="y" goto runall
if /i "%CHOICE%"=="n" goto skip

REM Run specific datasets
echo.
echo Running specific datasets: %CHOICE%
for %%i in (%CHOICE%) do (
    echo.
    echo ================================================================================
    echo Running dataset %%i
    echo ================================================================================
    python -m experiments.run_rid_comparison %%i
)
goto collect

:runall
echo.
echo Running all datasets (this will take 1-2 hours)...
for /L %%i in (0,1,10) do (
    echo.
    echo ================================================================================
    echo Running dataset %%i
    echo ================================================================================
    python -m experiments.run_rid_comparison %%i
)
goto collect

:skip
echo Skipping dataset processing...
goto collect

:collect
echo.
echo ================================================================================
echo Collecting results...
echo ================================================================================

REM Create a simple Python script to collect results
python -c "import pandas as pd; from pathlib import Path; results_dir = Path('rid_plots_1_22'); summaries = []; [summaries.append(pd.read_csv(f / 'summary.csv').assign(dataset=f.name)) for f in results_dir.iterdir() if f.is_dir() and (f / 'summary.csv').exists()]; combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(); combined.to_csv('case_study_2_combined_results.csv', index=False) if not combined.empty else None; print(f'\nCollected results from {len(summaries)} datasets'); print(f'Saved to case_study_2_combined_results.csv')"

echo.
echo ================================================================================
echo Reproduction Complete!
echo ================================================================================
echo.
echo Results are saved in:
echo   - rid_plots_1_22\                     (individual dataset results)
echo   - case_study_2_combined_results.csv    (all results combined)
echo.
echo Check CASE_STUDY_2_INSTRUCTIONS.md for details on interpreting the results.
echo.

pause
