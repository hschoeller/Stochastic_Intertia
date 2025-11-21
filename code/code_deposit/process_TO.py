#!/usr/bin/env python3
from scipy import sparse
import os
import sys
import pickle
import numpy as np
from scipy.sparse.linalg import eigs

from cdv_utils import load_d, diffusion_maps_matrix, compute_clusters_from_TO, expected_escape_times_from_TO

# --- user-configurable constants (match your notebook) ---
subsample_step = 10
n_eig = 12
n_clusters = 2
n_init = 10
eig_indices = [1, 2, 3, 4]
random_state = 0

# path to det_traj file and where to write results
DATA_PATH = "./cdv_model/data/dataOro20_sigma0p00000000.bin"
OUT_DIR = "./cdv_model/data/eps_results"  # make sure exists
os.makedirs(OUT_DIR, exist_ok=True)

# parse SLURM_ARRAY_TASK_ID or command-line index
if len(sys.argv) >= 2:
    idx = int(sys.argv[1])
else:
    # fallback: environment variable
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

sigmas = np.linspace(0, 0.05, 501)
eps_values = 2*sigmas**2
eps = float(eps_values[idx])

# Load det_traj (subsampling here as in your code)
det_traj = load_d(DATA_PATH, 6, 200000)[::subsample_step, :]

DMM, A = diffusion_maps_matrix(det_traj, 0.001)

# --- compute clusters from TO ---
det_cluster_res = compute_clusters_from_TO(DMM,
                                           n_eig=n_eig,
                                           n_clusters=n_clusters,
                                           n_init=n_init,
                                           eig_indices=eig_indices,
                                           random_state=random_state)

# Now compute for this eps
DMM, A = diffusion_maps_matrix(det_traj, eps)

L = sparse.vstack(
    [DMM[1:], sparse.csr_matrix((1, DMM.shape[1]))], format='csr')

blocking_idx = 0
zonal_idx = 1

Q = L[det_cluster_res["labels"] == 1, :][:, det_cluster_res["labels"] == 1]
eigval, _ = eigs(Q, k=1, which='LM')

# Atomic write: write to temp then rename
out_file_tmp = os.path.join(OUT_DIR, f"Rate_{idx:03d}.pkl.tmp")
out_file = os.path.join(OUT_DIR, f"Rate_{idx:03d}.pkl")
with open(out_file_tmp, "wb") as f:
    pickle.dump({"eps": eps, "value": eigval.real[0]}, f)
os.replace(out_file_tmp, out_file)

# t_escape = expected_escape_times_from_TO(
#     L, det_cluster_res["labels"] == blocking_idx)

# # Atomic write: write to temp then rename
# out_file_tmp = os.path.join(OUT_DIR, f"TOesc_{idx:03d}.pkl.tmp")
# out_file = os.path.join(OUT_DIR, f"TOesc_{idx:03d}.pkl")
# with open(out_file_tmp, "wb") as f:
#     pickle.dump({"eps": eps, "value": t_escape}, f)
# os.replace(out_file_tmp, out_file)

# p0 = np.zeros(len(det_cluster_res["labels"]))
# p0[det_cluster_res["labels"] == zonal_idx] = 1 / \
#     (det_cluster_res["labels"] == zonal_idx).sum()
# p1 = p0 @ L
# p_regime = p1[det_cluster_res["labels"] == blocking_idx]
# p_enter = p_regime / p_regime.sum()
# regime_lifetime = float(np.dot(p_enter, t_escape))

# # Atomic write: write to temp then rename
# out_file_tmp = os.path.join(OUT_DIR, f"eps_{idx:03d}.pkl.tmp")
# out_file = os.path.join(OUT_DIR, f"eps_{idx:03d}.pkl")
# with open(out_file_tmp, "wb") as f:
#     pickle.dump({"eps": eps, "value": regime_lifetime}, f)
# os.replace(out_file_tmp, out_file)


print(f"Finished idx={idx}, eps={eps:.6f}")  # , value={regime_lifetime:.6e}")
