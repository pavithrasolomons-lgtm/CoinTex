# COMP6009 Final Assignment Part A: CoinTex Agents

**Colombo-group-08:** Pavithra Solomons (24079123), Ravindu Chandrasekara (24100201), Rashmika Dilshan (20543909)

Author: Pavithra Solomons  
Last modified by: Pavithra Solomons  
Last modified: 29 October 2026

This folder has only the code we wrote for Part A. The CoinTex game files (cointex.kv, images and music) and main_work_greedy.py are not included, as they come from the CoinTex repository and Blackboard.

Results (CSV files and charts): https://github.com/pavithrasolomons-lgtm/CoinTex/tree/master/COMP6009_Final-Assignment-PartA/results

Full project folder (report, slides, code and results): https://github.com/pavithrasolomons-lgtm/CoinTex/tree/master/COMP6009_Final-Assignment-PartA

## Files

| File | What it does |
|:---|:---|
| `agent_astar.py` | Our danger-aware A* agent (the improved algorithm) |
| `agent_greedy.py` | The original greedy logic from main_work_greedy.py, run through our test program |
| `agent_ga.py` | The CoinTex tutorial GA (main.py), updated to run on PyGAD 3.9.2 |
| `partA_common.py` | Shared code: the position snapshot and the read, move and wait calls |
| `main_partA.py` | The game plus our test program. It plays levels by itself, times them and saves the results |
| `analyse_partA.py` | Makes the summary tables and charts from the results file |
| `sim_partA.py` | A copy of the game rules without graphics, used only for tuning |
| `run_partA_experiments.bat` | Runs all three agents on all 24 levels (5 runs each), then the analysis |

## Setup (Windows)

1. Clone CoinTex and install the requirements, as in Part 1 of the brief:

   ```
   git clone https://github.com/ahmedfgad/CoinTex.git
   cd CoinTex\PlayerGA
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   pip install matplotlib
   ```

2. Copy all the files from this folder into `CoinTex\PlayerGA\TutorialProject`.
3. Put `main_work_greedy.py` from Blackboard in the same folder. `sim_partA.py` reads the level settings from it.

We used Python 3.12.10, Kivy 2.3.1, PyGAD 3.9.2, NumPy 2.5.4 and Matplotlib 3.11.2.

## Run one agent

In Command Prompt (not PowerShell), inside `TutorialProject` with the venv active:

```
set AGENT=astar
set LEVELS=18
set RUNS=1
set LOG=demo.csv
python main_partA.py
```

| Setting | Values | Default |
|:---|:---|:---|
| `AGENT` | greedy, ga or astar | astar |
| `LEVELS` | for example 1-24 or 1,5,18 | 1-24 |
| `RUNS` | games per level | 5 |
| `LOG` | results file | partA_results.csv |
| `TIMEOUT` | seconds before a game counts as failed | 60 |
| `AUTO` | 1 plays by itself, 0 lets you click a level | 1 |

Games already in the LOG file are skipped, so use a new LOG name for a fresh run. Keep the game window visible and don't click inside it.

## Run all experiments

Double click `run_partA_experiments.bat`. It runs greedy, A* and the GA on levels 1 to 24 with 5 runs each (360 games, about 60 to 75 minutes), then makes the tables and charts. If it stops, run it again and it carries on from where it stopped.

## Make the tables and charts

```
python analyse_partA.py partA_results.csv
```

This writes `partA_summary.csv`, `partA_per_level.csv` and three `chart_*.png` files.

## Simulator (tuning only)

```
python sim_partA.py --agents greedy astar --levels 1-24 --runs 5
```

The results in our report come from the real game, not the simulator.

## Our results

| | Greedy | GA | A* (ours) |
|:---|---:|---:|---:|
| Games won (of 120) | 96 | 99 | 120 |
| Success rate | 80.0% | 82.5% | 100.0% |
| Levels completed (of 24) | 21 | 21 | 24 |
| Deaths | 24 | 21 | 0 |
| RACR (our new metric) | 1.48 | 0.81 | 3.10 |

RACR = coins collected per second × fraction of life left. The full results are in the report and on GitHub (links above).
