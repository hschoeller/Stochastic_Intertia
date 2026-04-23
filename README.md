# Stochastic Inertia

Code repository for the paper:
**"Noise-induced enhancement of regime lifetimes -- A data-driven approach using deterministic trajectories"**
by Henry Schoeller, Robin Chemnitz, P´eter Koltai, Maximilian Engel,
and Stephan Pfahl
(*Communications in Applied Mathematics and Computational Sciences, 2026*)

---

## Overview

This repository contains the code and resources needed to reproduce the main results from the paper.

---

## Repository Structure

```
├── code/code_deposit/            # Python source code
            ├── cdv_model/        # Code to run the CdV model
├── env.yml         # necessary python packages
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/hschoeller/Stochastic_Intertia.git
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Data

* If data is included: describe it briefly here
* If not included: explain how to obtain it

Example:

```
bash scripts/download_data.sh
```

---

## Reproducing Results

To reproduce the main results from the paper:

```
bash scripts/run_experiments.sh
```

To generate figures:

```
python scripts/plot_results.py
```

---

## Results

Expected outputs (e.g., metrics, plots) will be saved in:

```
results/
```

---

## Citation

If you use this code, please cite:

```
@article{your2026paper,
  title={Full Paper Title},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2026}
}
```

---

## License

Specify your license here (e.g., MIT, Apache 2.0).

---

## Acknowledgments

(Optional) Funding sources, collaborators, etc.
