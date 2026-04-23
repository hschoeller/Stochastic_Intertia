from cdv_utils import (
    affinity_entry_sum, find_nearby_indices, load_d, make_L3d,
    compute_clusters_from_TO, plot_loglog_slope_analysis, find_blocking_set,
    solve_escape_from_Q, classify_new_trajectory, calculate_escape_times
)
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from itertools import cycle
import matplotlib.pyplot as plt
import importlib
import cdv_utils
import matplotlib
import glob
import re
import pickle

matplotlib.use("Agg")
fixed = np.array([
    [0.7285286,   0.15657308, -0.36810966, -0.32854071, -0.08004622, 0.34314342],
    [0.94484848, 0.10714876, -0.00858286, -0.71094887, -0.17206714, 0.03996197]
])
dt_traj = 1
sigmas = np.arange(0.0075, 0.03, 0.0025)
sig_max = 0.03
sig_step = 1e-4
n_eig = 12
n_clusters = 2
n_init = 10
eig_indices = [1, 2, 3, 4]   # zero-based eigenvector indices to use
random_state = 0

# sums = {}
for sigma in sigmas:
    dt_eff = dt_traj
    epsilon = 2 * dt_eff * sigma**2
    t_fin = int(2e4)
    print(epsilon)

    det_traj = load_d(
        "./cdv_model/data/dataOro20_sigma0p00000000.bin", 6, int(2e6))[:t_fin, :]
    L = make_L3d(det_traj, epsilon)
    # --- compute clusters from TO ---
    det_cluster_res = compute_clusters_from_TO(L.T,
                                               n_eig=n_eig,
                                               n_clusters=n_clusters,
                                               n_init=n_init,
                                               eig_indices=eig_indices,
                                               random_state=random_state)

    labels = det_cluster_res["labels"].copy()
    blocking_idx = find_blocking_set(det_traj[:-1], labels)
    plt.figure(figsize=(4, 4))
    sc = plt.scatter(det_traj[:-1, 0], det_traj[:-1, 5],
                     c=(labels == blocking_idx), s=1, linewidth=0, label='Clusters',
                     rasterized=True)
    plt.scatter(fixed[0, 0], fixed[0, 5], c='black', s=64, alpha=1,
                edgecolors='black', linewidths=1, marker="x")
    plt.scatter(fixed[1, 0], fixed[1, 5], c='red', s=64, alpha=1,
                edgecolors='black', linewidths=1, marker="x")

    cbar = plt.colorbar(sc, ticks=range(
        len(np.unique(det_cluster_res["labels"]))))
    cbar.set_ticklabels(range(len(np.unique(det_cluster_res["labels"]))))
    plt.savefig(
        f"../../Sets_subsample1_sigma_{sigma:.4f}.pdf", bbox_inches="tight", dpi=600)

    blocking_mask = np.nonzero(labels == blocking_idx)[0]
    entries = np.where(
        (labels[:-1] != blocking_idx) & (labels[1:] == blocking_idx))[0]+1
    indices_in_reduced = np.nonzero(labels == blocking_idx)[0].searchsorted(
        entries[np.isin(entries, np.nonzero(labels == blocking_idx)[0])])

    t_escape_dict = {}
    for sig in np.arange(0, sig_max + sig_step, sig_step):
        eps = (2 * dt_eff * sig**2)
        print(f"Processing sigma = {sig:.6f}")

        L = make_L3d(det_traj, eps)
        Q = L[labels == blocking_idx][:,
                                      labels == blocking_idx].tocsr()
        t_escape = solve_escape_from_Q(Q)

        # Store
        t_escape_dict[eps] = t_escape

    escape_results = {}  # plain dict
    for fname in sorted(glob.glob("./cdv_model/data/dataOro20_sigma0p*.bin")):
        # extract sigma string from filename (e.g. "0p050")
        match = re.search(r"sigma([0-9p]+)\.bin", fname)
        sigma_str = match.group(1).replace(
            "p", ".")   # e.g. "0p050" -> "0.050"
        sigma_val = float(sigma_str)
        if sigma_val > sig_max:
            break
        print(f"Processing sigma={sigma_val}")

        sig_data = load_d(fname, 6, int(2e6))[::1, :]

        # classify
        labels_new = classify_new_trajectory(
            det_traj[:-1], labels, sig_data, k=20, threshold=0.5
        )

        esc_dict = calculate_escape_times(labels_new, dt=dt_eff)

        # insert into dict safely
        for regime, times in esc_dict.items():
            if regime not in escape_results:
                escape_results[regime] = {}
            if sigma_val not in escape_results[regime]:
                escape_results[regime][sigma_val] = []
            escape_results[regime][sigma_val].extend(times)

    with open(f"subsample_1_lifetimes_sigma_{sigma:.4f}.pkl", "wb") as f:
        pickle.dump((escape_results, t_escape_dict), f)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.sqrt(np.array(sorted(t_escape_dict.keys())) / (dt_eff*2)),
               np.array([t_escape_dict[k][indices_in_reduced].mean()
                         for k in sorted(t_escape_dict.keys())]) * dt_eff,
               label=f'Markov Chain', alpha=.3, s=10,  marker="x", color='red')
    means = np.array([np.mean(v)
                      for k, v in sorted(escape_results[blocking_idx].items())])
    ax.scatter(sorted(escape_results[blocking_idx].keys()), means,
               label=f'Monte Carlo', marker="o", color='blue', s=10, alpha=.5)
    ax.set_xlabel("Noise Strength $\sigma$")
    ax.set_ylabel("Mean Regime Lifetime")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, sig_max)

    ax.legend(facecolor='white', edgecolor='black',
              loc='best', framealpha=1)
    plt.savefig(f"../../Lifetimes_subsample1_sigma_{sigma:.4f}.pdf",
                bbox_inches="tight", dpi=600)
