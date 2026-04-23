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
conda env create -f env.yml
```

---

## Reproducing Results

To reproduce the main results from the paper:

The CdV trajectories can be calculated with an array of noise levels using the compandrun.sh script (on a slurm managed cluster or simply on your local machine with bash).
All trajectories and referenced data can be calculated conveniently in jupyter notebooks:

For chapter 3.1 Stochastic Inertia in a one-dimensional toy system use Toy.ipynb
For chapter 3.2 A three-dimensional example without Stochastic Inertia use Toy3d.ipynb
For chapter 3.3 Stochastic Inertia in the CdV system use PointwiseCdV.ipynb

---

## Citation

If you use this code, please cite the upcomming publication.

---

## License

MIT License

Copyright (c) 2026 [Henry Schoeller]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY.


---

## Acknowledgments

This research has been funded by Deutsche Forschungsgemeinschaft (DFG) through grant CRC
1114 "Scaling Cascades in Complex Systems", Project Number 235221301, Project A08 "Characterization and Prediction of Quasi-Stationary Atmospheric States"
