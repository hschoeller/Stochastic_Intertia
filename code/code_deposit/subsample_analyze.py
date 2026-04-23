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
sigma = 0.015
sig_max = 0.03
sig_step = 1e-4
n_eig = 12
n_clusters = 2
n_init = 10
eig_indices = [1, 2, 3, 4]   # zero-based eigenvector indices to use
random_state = 0
sigmas_loglog = np.logspace(-4, 0, 50)

subsample_steps = [1, 2, 5, 10]  # , 20, 50]  # <- this will be varied

# To find max slopes
# with open("./subsample_sums.pkl", "rb") as f:
#     sums = pickle.load(f)
# logx = np.log10(sigmas_loglog)

# sums = {}
for subsample_step in subsample_steps:
    # find max slope for this subsample step
    # sum_i = np.asarray(sums[subsample_step])
    # logy = np.log10(sum_i)
    # # local slopes between consecutive log points
    # slopes = np.diff(logy) / np.diff(logx)
    # imax = int(np.argmax(slopes))
    # sig_at_max = np.sqrt(sigmas_loglog[imax] * sigmas_loglog[imax + 1])
    # sigma = sig_at_max
    dt_eff = dt_traj * subsample_step
    epsilon = 2 * dt_eff * sigma**2
    t_fin = int(2e4 * subsample_step)
    print(epsilon)

    det_traj = load_d(
        "./cdv_model/data/dataOro20_sigma0p00000000.bin", 6, int(2e6))[:t_fin:subsample_step, :]

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
        f"../../Sets_subsample{subsample_step}_sigma_{sigma:.3f}.pdf", bbox_inches="tight", dpi=600)

    blocking_mask = np.nonzero(labels == blocking_idx)[0]
    entries = np.where(
        (labels[:-1] != blocking_idx) & (labels[1:] == blocking_idx))[0]+1
    indices_in_reduced = np.nonzero(labels == blocking_idx)[0].searchsorted(
        entries[np.isin(entries, np.nonzero(labels == blocking_idx)[0])])
    eps_vals = 2.0 * dt_eff * sigmas_loglog**2

    # for all points
    # sums_all = np.array([affinity_entry_sum(det_traj, eps, chunk=1000)
    #                     for eps in eps_vals]) / det_traj.shape[0]**2
    # sums[subsample_step] = sums_all
    # # for blocking set
    # eps_max = 2.0 * dt_eff * sig_max**2
    # points = find_nearby_indices(
    #     det_traj, labels, eps=eps_max, regime=blocking_idx)
    # sums_block = np.array([affinity_entry_sum(det_traj[points], eps, chunk=1000)
    #                        for eps in eps_vals]) / det_traj[points].shape[0]**2
    # fig, ax = plot_loglog_slope_analysis(sigmas_loglog, sums_all, sums_block)
    # fig.savefig(f"../../LogLog_subsample{subsample_step}.pdf",
    #             bbox_inches="tight", dpi=600)
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

        sig_data = load_d(fname, 6, int(2e6))[::subsample_step, :]

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

    with open(f"subsample_{subsample_step}_lifetimes_sigma_{sigma:.3f}.pkl", "wb") as f:
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
    plt.savefig(f"../../Lifetimes_subsample{subsample_step}_sigma_{sigma:.3f}.pdf",
                bbox_inches="tight", dpi=600)

exit()
# --- plot all loglog in one fig ---

# with open("./subsample_sums.pkl", "wb") as f:
#     pickle.dump(sums, f)

# Ensure deterministic order: sort integer keys ascending
keys = sorted(sums.keys())
n = len(keys)
cmap = plt.get_cmap('tab10')
colors = [cmap(i / (n - 1)) for i in range(n)]

# markers to cycle through if more datasets than original two
marker_list = ['x', '+', 'o', 's', 'D', '^', 'v', '<', '>', '*']
marker_cycle = cycle(marker_list)

logx = np.log10(sigmas_loglog)
fig, ax = plt.subplots(figsize=(7, 5))
x_line_log = np.linspace(logx[0], logx[-1], 300)
table_rows = []

# plot datasets and slope-lines
for idx, key in enumerate(keys):
    sum_i = np.asarray(sums[key])
    col = colors[idx]
    m = next(marker_cycle)
    logy = np.log10(sum_i)

    # local slopes between consecutive log points
    slopes = np.diff(logy) / np.diff(logx)
    imax = int(np.argmax(slopes))
    slope_max = slopes[imax]

    # geometric mean location for sigma* (between imax and imax+1)
    sig_at_max = np.sqrt(sigmas_loglog[imax] * sigmas_loglog[imax + 1])

    # anchor: midpoint in log space
    logx_mid = 0.5 * (logx[imax] + logx[imax + 1])
    logy_mid = 0.5 * (logy[imax] + logy[imax + 1])

    # line (in log space) anchored at midpoint
    y_line_log = slope_max * (x_line_log - logx_mid) + logy_mid

    # main curve: log-log
    ax.loglog(sigmas_loglog, sum_i, '-', color=col, marker=m,
              markersize=4, linewidth=1,
              label=str(key), markevery=1)
    ax.loglog(10 ** x_line_log, 10 ** y_line_log, '--', color=col,
              linewidth=1, alpha=1)

    # point of maximum slope: same colour
    y_at_max = 10 ** logy_mid  # anchored midpoint value
    ax.scatter([sig_at_max], [y_at_max], color=col, edgecolor='k',
               s=(2 * 5), zorder=10)

    # accumulate row for the summary table (first column is the integer key)
    table_rows.append((str(key), f"{sig_at_max:.3g}", f"{slope_max:.4f}"))

ax.set_xlim(sigmas_loglog[0], sigmas_loglog[-1])
ax.set_xlabel(r'Noise Strength $\sigma$')
ax.set_ylabel(r'$\sum_{i,j} K_{i,j} N^{-2}$')
ax.grid(True, which='both', ls=':', alpha=0.5)

# Legend: ensure marker+colour shown exactly as plotted
legend_handles = []
# Rebuild markers in the same order used above
# (recreate a fresh cycle so marker assignment matches plotting order)
marker_cycle = cycle(marker_list)
for idx, key in enumerate(keys):
    col = colors[idx]
    m = next(marker_cycle)
    legend_handles.append(Line2D([0], [0], color=col, marker=m, lw=1,
                                 markersize=4, label=str(key)))
ax.legend(handles=legend_handles, framealpha=1)

# Small table with results: place inside axes (bottom-right)
col_labels = (r"$\mathrm{d}t$", r"$\sigma^\ast$", r"$d$")
cell_text = [row for row in table_rows]

# make the table height scale with number of rows (keep it reasonable)
table_height = min(0.45, 0.06 + 0.05 * (n + 1))  # +1 accounts for header
bbox = [0.6, 0.02, 0.38, table_height]

tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc='center',
    colLoc='center',
    bbox=bbox,  # bottom-right corner area
    edges='closed',
    colWidths=[0.1, 0.25, 0.25]
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.set_zorder(10)

# Header: white and bold
for c in range(3):
    cell = tbl[(0, c)]
    cell.set_facecolor((1, 1, 1, 1))
    cell.set_text_props(weight='bold')

# Data rows: tinted by dataset colour
for r, col in enumerate(colors, start=1):
    rgba = list(to_rgba(col))
    rgba[3] = 0.5  # transparency for table row background
    for c in range(3):
        tbl[(r, c)].set_facecolor(rgba)

# Tidy up
plt.tight_layout()

plt.savefig(f"../../LogLog_Compare.pdf",
            bbox_inches="tight", dpi=600)
