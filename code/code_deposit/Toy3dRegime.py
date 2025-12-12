#!/usr/bin/env python3

import pickle
from cdv_utils import (simulate_trajectories_per_sigma_3d, normal_reset_sampler_vectorized_3d,
                       compute_lifetimes_per_sigma_3d,
                       solve_escape_from_Q, make_L3d,
                       affinity_entry_sum, plot_loglog_slope_analysis)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


rng = np.random.default_rng(42)
save_every = 1
sigmas = np.linspace(0.0, 0.3, 101, dtype=np.float32)
dt_compare = 1e-1
for eps_pdf in [0.005]:
    e_string = str(eps_pdf).split(".")[1]
    # times, states, sigma_arr = simulate_trajectories_per_sigma_3d(
    #     sigmas=sigmas,
    #     dt=dt_compare,
    #     n_steps=int(2e4) - 1,
    #     reset_sampler=lambda rng, size: normal_reset_sampler_vectorized_3d(
    #         rng, size, scale=eps_pdf
    #     ),
    #     rng=rng,
    #     save_every=save_every,
    #     x0=np.array([1e-5]*3, dtype=np.float32),
    #     dtype=np.float32
    # )

    # escape_times_per_sigma = compute_lifetimes_per_sigma_3d(
    #     states, sigma_arr, reset_threshold=2.0, hit_threshold=1.0)
    # escape_times_per_sigma = {sig: [et*dt_compare for et in ets]
    #                           for sig, ets in escape_times_per_sigma.items()}

    times, states, sigma_arr = simulate_trajectories_per_sigma_3d(
        sigmas=np.array([0.0], dtype=np.float32),
        dt=dt_compare,
        n_steps=int(2e4) - 1,
        reset_sampler=lambda rng, size: normal_reset_sampler_vectorized_3d(
            rng, size, scale=eps_pdf
        ),
        rng=rng,
        save_every=save_every,
        x0=np.array([1e-5]*3, dtype=np.float32),
        dtype=np.float32
    )
    blocking_mask = (np.linalg.norm(
        states[:-1, 0, :], axis=-1) < 1.0).ravel().astype(bool)
    resets = np.where(np.linalg.norm(
        states[:-1, 0, :], axis=-1) > 2.0)[0] + 1
    resets_in_block_idx = np.nonzero(
        np.isin(np.nonzero(blocking_mask)[0], resets))[0]
    eps_values = (2 * (dt_compare) * sigmas**2)

    # sigmas_loglog = np.logspace(np.log10(5e-4), np.log10(.5), 20)
    # eps_vals = 2.0 * (dt_compare) * sigmas_loglog**2
    # # compute sums
    # sums = np.array([affinity_entry_sum(states[:, 0, :], eps, chunk=1000)
    #                 for eps in eps_vals]) / states[:, 0, :].shape[0]**2
    # plot_loglog_slope_analysis(sigmas_loglog, sums)
    # plt.savefig(f"../../Toy3d_SigmaSlopeAnalysiseps{e_string}.pdf",
    #             bbox_inches="tight", dpi=600)
    # fig, ax = scatter_sphere_sampling(
    #     states[resets, 0, :], size=16, alpha=1, figsize=(5, 2))
    # fig, ax2 = add_radius_histogram_to_right(
    #     fig, states[resets, 0, :], sd=1e-2, bins=40)
    # plt.savefig("../../Toy3dSamplingDist.pdf", bbox_inches="tight", dpi=600)

    regime_dict = {}
    t_escape_dict = {}
    for j, eps in enumerate(eps_values):
        print(f"{j}th eps: {eps}")
        L = make_L3d(states[:, 0, :], eps)
        Q = L[blocking_mask][:, blocking_mask].tocsr()
        t_escape = solve_escape_from_Q(Q)
        regime_dict[eps] = t_escape[resets_in_block_idx].mean() * dt_compare
        t_escape_dict[eps] = t_escape[resets_in_block_idx] * dt_compare

    with open(f"./Toy3d_RegimeTimeTOeps{e_string}.pkl", "wb") as f:
        pickle.dump(regime_dict, f)
    with open(f"./Toy3d_tEscapeDicteps{e_string}.pkl", "wb") as f:
        pickle.dump(t_escape_dict, f)

    # means = np.array([np.mean(v)
    #                   for k, v in sorted(escape_times_per_sigma.items())])
    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.scatter(np.sqrt(np.array(sorted(regime_dict.keys())) / (dt_compare*2)),
    #            np.array([regime_dict[k]
    #                      for k in sorted(regime_dict.keys())]),
    #            color='red', marker='x', s=10, alpha=.5, label='Markov Chain')
    # ax.scatter(np.asarray(sorted(escape_times_per_sigma.keys()), dtype=float), means,
    #            color='blue', s=10, alpha=0.5, label='Monte Carlo')
    # ax.set_xlabel("Noise Strength $\sigma$")
    # ax.set_ylabel("Mean Regime Lifetime")
    # ax.grid(True, alpha=0.3)
    # ax.set_xlim(0, sigmas[-1])
    # # ax.set_title(f"Reset Radius PDF Scale: {eps_pdf}")
    # ax.legend(facecolor='white', edgecolor='black', loc='best', framealpha=1)
    # plt.tight_layout()
    # plt.savefig(f"../../Toy3d_MeanRegimeLifetimeComparisoneps{e_string}.pdf",
    #             dpi=600, bbox_inches="tight")
