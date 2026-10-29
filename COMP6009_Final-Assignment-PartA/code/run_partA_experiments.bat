@echo off
REM ======================================================================
REM  COMP6009 Final Assignment Part A - run all experiments
REM  Author:           Pavithra Solomons
REM  Last modified by: Pavithra Solomons
REM  Last modified:    29 October 2026
REM  Group:            Colombo-group-08
REM  greedy, A* and GA agents x levels 1-24 x 5 runs  (about 60-75 min)
REM  Results: partA_results.csv, partA_summary.csv, partA_per_level.csv,
REM           chart_*.png  (all in this TutorialProject folder)
REM  Safe to stop and restart: finished episodes are skipped.
REM  Keep the game window visible and do not click inside it.
REM ======================================================================
cd /d "%~dp0"
call "%~dp0..\venv\Scripts\activate.bat"
python -m pip install -q matplotlib

set LEVELS=1-24
set RUNS=5
set LOG=partA_results.csv
set AUTO=1

for %%A in (greedy astar ga) do (
    echo.
    echo ===== Running agent %%A =====
    set AGENT=%%A
    python main_partA.py
)

echo.
echo ===== Analysing results =====
python analyse_partA.py partA_results.csv
echo.
echo Done. Results are in %~dp0
pause
