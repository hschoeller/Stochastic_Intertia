#!/usr/bin/env python3
"""
compute_reinsertion.py

Parallelization over SIGMAS (one task per sigma). Final output is a pickle
file named `reinsert_TO_3d.pkl`.

Expect a module `cdv_utils` or equivalent on PYTHONPATH that defines:
  - simulate_trajectories_per_sigma_3d
  - normal_reset_sampler_vectorized_3d
  - make_L3d
  - solve_escape_from_Q

Behaviour:
 - When run with --sigma-index IDX the script computes the row for that sigma
   (a 1D array of length len(reinsertion_radii)) and writes a partial .npz:
     <out>.part{IDX}.npz
 - Run with --merge to combine partials and write the final pickle:
     reinsert_TO_3d.pkl
 - If run without --sigma-index (and without --merge) it computes all sigmas
   sequentially and writes the final pickle directly.
"""
import os
import sys
import pickle
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from cdv_utils import (
    simulate_trajectories_per_sigma_3d,
    normal_reset_sampler_vectorized_3d,
    make_L3d,
    solve_escape_from_Q,
)

parser = argparse.ArgumentParser()
parser.add_argument("--save-every", type=int, default=1)
parser.add_argument("--dt", type=float, default=1e-1)
parser.add_argument("--n-steps", type=int, default=int(2e2) - 1)
parser.add_argument("--rng-seed", type=int, default=0)
parser.add_argument("--sigma-index", type=int,
                    help="index for this task (0-based)")
parser.add_argument("--merge", action="store_true",
                    help="merge partials into final out")
parser.add_argument("--out", default="reinsert_TO.npz",
                    help="base name for partials (final is reinsert_TO_3d.pkl)")
parser.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)

# PARAMETERS (tweak as required)
# note: use float64 for eps computation
sigmas = np.linspace(0.0, 0.3, 101, dtype=np.float64)
reinsertion_radii = np.linspace(0.01, 0.1, 4, dtype=np.float64)
dt_higher = float(args.dt)
save_every = int(args.save_every)

# deterministic points on sphere


def fibonacci_sphere(n_points=100, radius=1e-4):
    points = np.zeros((n_points, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (2.0 * i) / (n_points - 1)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        points[i] = radius * np.array([x, y, z])
    return points


points = fibonacci_sphere(100, 1e-4)

# --- 1) Simulate trajectories ONCE (shared by all sigmas & radii) ---
rng = np.random.default_rng(args.rng_seed)
traj_list = []         # list of arrays (T_k, ensemble, 3)
norms_list = []        # list of arrays (T_k,) norms of first ensemble member
lengths = []

for x_i in points:
    times, states, sigma_arr = simulate_trajectories_per_sigma_3d(
        sigmas=np.array([0.0], dtype=np.float32),
        dt=dt_higher,
        n_steps=int(args.n_steps),
        reset_sampler=normal_reset_sampler_vectorized_3d,
        rng=rng,
        save_every=save_every,
        x0=x_i,
        dtype=np.float32
    )
    first_reset = np.where(np.linalg.norm(states, axis=-1) > 2.0)[0]
    cut = first_reset[0] if first_reset.size > 0 else states.shape[0]
    traj_list.append(states[:cut, :, :])
    norms_list.append(np.linalg.norm(states[:cut, 0, :], axis=-1))
    lengths.append(cut)

# shape (T_total, ensemble, 3)
full_traj = np.concatenate(traj_list, axis=0)
# offsets to map per-point indices to full_traj indices
offsets = np.cumsum([0] + lengths[:-1]).astype(np.int64)
times = np.linspace(
    0, dt_higher * (full_traj.shape[0] - 1), full_traj.shape[0])
blocking_mask = (np.linalg.norm(
    full_traj[:-1, 0, :], axis=-1) < 1.0).ravel().astype(bool)

# Precompute reinsertion full indices and blocking indices for each reinsertion radius
n_points = len(points)
n_radii = reinsertion_radii.size
# rp_fulls[r, p] = full index in full_traj for point p and radius r
rp_fulls = np.empty((n_radii, n_points), dtype=np.int64)
# list of integer index arrays into blocking_mask
rp_in_blocking_idx = [None] * n_radii

for ri, R in enumerate(reinsertion_radii):
    for p, norms in enumerate(norms_list):
        rp_local = np.argmin(np.abs(norms - R))
        rp_fulls[ri, p] = int(rp_local + offsets[p])
    rp_in_blocking_idx[ri] = np.nonzero(
        np.isin(np.nonzero(blocking_mask)[0], rp_fulls[ri, :]))[0]

# array of eps per sigma (float64)
eps_values = (2.0 * dt_higher * (sigmas ** 2))

# --- core worker for one sigma: compute a single row (length n_radii) ---


def compute_for_sigma(s_idx):
    eps = float(eps_values[s_idx])
    print(
        f"Computing sigma index {s_idx}, sigma={sigmas[s_idx]:.6g}, eps={eps:.6g}")
    L = make_L3d(full_traj[:, 0, :], eps)
    Q = csr_matrix(L)[blocking_mask][:, blocking_mask].tocsr()
    t_escape = solve_escape_from_Q(Q)   # length blocking_mask.size
    row = np.empty(n_radii, dtype=np.float64)
    for ri in range(n_radii):
        idxs = rp_in_blocking_idx[ri]
        if idxs.size == 0:
            row[ri] = np.nan
        else:
            row[ri] = t_escape[idxs].mean()
    return row

# --- partial writer (one .npz per sigma index) ---


def write_partial(s_idx, row, out_base):
    part = out_base + f".part{s_idx}.npz"
    np.savez_compressed(part, row=row, idx=s_idx, sigma=float(sigmas[s_idx]),
                        sigmas=sigmas, reinsertion_radii=reinsertion_radii, dt=dt_higher)

# --- merge partials into final pickle ---


def merge_partials(out_base):
    parts = sorted([p for p in os.listdir(".") if p.startswith(
        os.path.basename(out_base) + ".part") and p.endswith(".npz")])
    rows = []
    idxs = []
    for p in parts:
        data = np.load(p)
        rows.append(data["row"])
        idxs.append(int(data["idx"].tolist()))
    # assemble rows into array with correct ordering by idx
    order = np.argsort(idxs)
    rows_ordered = [rows[i] for i in order]
    reinsert_TO = np.vstack(rows_ordered)   # shape (n_sigmas, n_radii)
    # write final pickle
    with open("reinsert_TO_3d.pkl", "wb") as f:
        pickle.dump(reinsert_TO, f)
    print("Merged into reinsert_TO_3d.pkl (shape: {})".format(reinsert_TO.shape))


# --- main control flow ---
if args.merge:
    merge_partials(args.out)
    sys.exit(0)

if args.sigma_index is None:
    # sequential compute all sigmas and save directly as pickle
    reinsert_TO = np.empty((sigmas.size, n_radii), dtype=np.float64)
    for s in range(sigmas.size):
        reinsert_TO[s, :] = compute_for_sigma(s)
    with open("reinsert_TO_3d.pkl", "wb") as f:
        pickle.dump(reinsert_TO, f)
    print("Saved reinsert_TO_3d.pkl (shape: {})".format(reinsert_TO.shape))
else:
    s_idx = int(args.sigma_index)
    row = compute_for_sigma(s_idx)
    write_partial(s_idx, row, args.out)
    print("Wrote partial for sigma index", s_idx)
