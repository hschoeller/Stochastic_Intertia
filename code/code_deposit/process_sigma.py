#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from cdv_utils import load_d, fit_hmm, calculate_escape_times, diffusion_maps_matrix, compute_clusters_from_TO

fixed = np.array([
    [0.7285286,   0.15657308, -0.36810966, -0.32854071, -0.08004622, 0.34314342],
    [0.94484848,  0.10714876, -0.00858286, -0.71094887, -0.17206714, 0.03996197]
])
inits = np.r_[fixed, [(fixed[0] + fixed[1]) / 2]]

parser = argparse.ArgumentParser()
parser.add_argument("input", help="trajectory .bin file")
parser.add_argument("--dims", type=int, default=6)
parser.add_argument("--sample_num", type=int, default=200000)
parser.add_argument("--n_states", type=int, default=3)
parser.add_argument("--dt", type=float, default=1.0)
args = parser.parse_args()

# load trajectory
vec = load_d(args.input, args.dims, args.sample_num)[::10, :]

# fit HMM
# state_centers, _, states = fit_hmm(
#     vec, args.n_states, initial_centers=inits)

# fit Diffusion Maps clusters
epsilon = 0.001
subsample_step = 10
n_eig = 12
n_clusters = 2
n_init = 10
eig_indices = [1, 2, 3, 4]   # zero-based eigenvector indices to use
random_state = 0

DMM, A = diffusion_maps_matrix(vec, epsilon)

# --- compute clusters from TO ---
cluster_res = compute_clusters_from_TO(DMM,
                                       n_eig=n_eig,
                                       n_clusters=n_clusters,
                                       n_init=n_init,
                                       eig_indices=eig_indices,
                                       random_state=random_state)
# calculate escape times
esc = calculate_escape_times(cluster_res["labels"], dt=args.dt)

# # mean x1 per regime
# mean_x1_by_regime = {}
# for r in range(args.n_states):
#     mask = (states == r)
#     if np.any(mask):
#         mean_x1_by_regime[r + 1] = float(np.mean(vec[mask, 0]))
#     else:
#         mean_x1_by_regime[r + 1] = np.nan

# save everything in a single file
out = args.input.rsplit(".", 1)[0] + "_DMMresults_tenth.pkl"  # _tenth.pkl"
# with open(out, "wb") as f:
#     pickle.dump({
#         "escape_times": esc,
#         "state_centers": state_centers,
#         "mean_x1_by_regime": mean_x1_by_regime
#     }, f)

with open(out, "wb") as f:
    pickle.dump({
        "escape_times": esc,
        "labels": cluster_res["labels"]}, f)

print(out)
