# Scatter Search: Foundations and Implementations <a href="https://doi.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/1/11/DOI_logo.svg" alt="DOI" width="20"/></a> <a href="https://doi.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/e/e8/Zenodo-gradient-square.svg" alt="Zenodo" width="60"/></a>

<!-- Load Material Symbols Outlined for the mail icon -->
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&icon_names=mail" />

Authors

- Manuel Laguna<sup>1</sup>   <a href="mailto:laguna@colorado.edu" aria-label="Email Manuel"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b1/Email_Shiny_Icon.svg" alt="email" width="20" style="vertical-align:middle;"/></a>
    <a href="https://orcid.org/0000-0002-8759-5523"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" alt="ORCID" width="20" style="vertical-align:middle;"/></a>

- Sergio Cavero<sup>2</sup>   <a href="mailto:sergio.cavero@urjc.es" aria-label="Email Sergio"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b1/Email_Shiny_Icon.svg" alt="email" width="20" style="vertical-align:middle;"/></a>
    <a href="https://orcid.org/0000-0002-5258-5915"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" alt="ORCID" width="20" style="vertical-align:middle;"/></a>

- Rafael Martí<sup>1,3</sup>  <a href="mailto:rafael.marti@uv.es" aria-label="Email Rafael"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b1/Email_Shiny_Icon.svg" alt="email" width="20" style="vertical-align:middle;"/></a>
    <a href="https://orcid.org/0000-0002-5258-5915"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" alt="ORCID" width="20" style="vertical-align:middle;"/></a>

Affiliations (short)

1. University of Colorado Boulder (Manuel Laguna & Rafael Martí)

2. Universidad Rey Juan Carlos (Sergio Cavero)

3. Universitat de València (Rafael Martí)

## Abstract

This repository contains the implementation and tutorial material associated with the article "Scatter Search: Foundations and Implementations". It includes a didactic Python implementation of Scatter Search applied to the 0-1 Knapsack problem, a Jupyter notebook with explanatory material and visualizations, and a collection of test instances.


## Files and folders

- `main_tutorial.py` — Entry script that loads an instance, sets seeds, and runs the tutorial Scatter Search.
- `scatter_search_tutorial.py` — Modular implementation of the method: problem definition, diversification generation, improvement (repair + greedy), RefSet management, combination and main loop.
- `instance_reader.py` — Utilities to read and validate instance files.
- `ScatterSearch_Knapsack_Tutorial.ipynb` — Notebook with step-by-step explanation, plots and interactive examples.
- `instances_01_KP/` — Folder containing instances (subfolders: `large_scale/`, `low-dimensional/`, and their _-optimum_ variants).


## Instance file format

Each instance file follows this simple plain-text format:

- First line: `n capacity` (number of items and knapsack capacity).
- Next `n` lines: `profit weight` for each item (integers).

Minimal example:

```
20 150
24 12
18 25
15 18
... (until 20 item lines)
```


## How to run `main_tutorial.py` (quick)

Requirements: Python 3.8+ is recommended. Use a virtual environment (venv or conda).

1) Create and activate a virtualenv (macOS / Linux example):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install the runtime requirements (see Requirements section below for details):

```bash
pip install -r requirements_py.txt
```

3) Run the tutorial script (the script uses a default instance path inside the file):

```bash
python3 main_tutorial.py
```

4) To run a different instance, either edit the `instance_path` value inside `main_tutorial.py` or create a small wrapper that calls the function with another path. If you prefer, I can add a CLI (`argparse`) to pass the instance path and algorithm parameters from the command line.

What `main_tutorial.py` does internally

- Sets reproducible seeds: `random.seed(42)` and `np.random.seed(42)`.
- Loads an instance using `load_instance_from_file` from `instance_reader.py`.
- Constructs a `KnapsackProblem` and calls `run_scatter_search(pb, max_iter=1000, population_size=30, refset_size=10)` from `scatter_search_tutorial.py`.
- Prints final results (best solution, objective value, total weight, selected items and feasibility).


## Notebook: run locally or on Google Colab

You can open `ScatterSearch_Knapsack_Tutorial.ipynb` locally or on Google Colab. Two convenient options:

- Google Colab: upload the notebook or open it from the GitHub repo (File > Open notebook > GitHub). Colab provides an instant runtime, preinstalled common packages, and is handy for sharing results.

- Local environment: use one of the following IDEs/environments:
  - Jupyter Notebook or JupyterLab (recommended for interactive exploration).
  - VS Code with the Python extension (supports opening and running notebooks inline).
  - PyCharm Professional (built-in notebook support) or use the standard Jupyter interface.

To run the notebook locally, install the notebook requirements (see below) and then:

```bash
# inside the activated virtual environment
pip install -r requirements_notebook.txt
jupyter lab   # or: jupyter notebook
```

Requirements (included here, also in `requirements_py.txt` and `requirements_notebook.txt`)

Runtime (to run the Python scripts `main_tutorial.py`, `scatter_search_tutorial.py`):

- `Python 3.8+`
- `numpy>=1.21`

Recommended for the Notebook / visualization:

- `jupyterlab>=3.0 or notebook>=6.0`
- `ipykernel`
- `numpy>=1.21`
- `matplotlib>=3.3`

