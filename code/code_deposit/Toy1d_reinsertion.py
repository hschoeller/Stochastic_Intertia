#!/usr/bin/env python3


import pickle
import numpy as np
from cdv_utils import (simulate_trajectories_per_sigma,
                       constant_reset_sampler_factory,
                       make_L, solve_escape_from_Q)
rng = np.random.default_rng(6942)
dt_lower = 5e-4
reinsertion_points = np.linspace(0.01, 0.1, 4)
save_every = 1
sigmas = np.linspace(0.0, 0.3, 101, dtype=np.float32)
times, states, sigma_arr = simulate_trajectories_per_sigma(
    sigmas=np.array([0.0], dtype=np.float32),
    dt=dt_lower,
    n_steps=int(1e5) - 1,
    reset_sampler=constant_reset_sampler_factory(0.05, dtype=np.float32),
    rng=rng,
    save_every=save_every,
    x0=1e-4,
    dtype=np.float32
)

first_reset = np.where(np.abs(states[:, 0]) > 2)[0][0]
mini_traj = states[:first_reset, 0]
rp_idx = [np.argmin(np.abs(mini_traj - rp))
          for rp in reinsertion_points]
eps_values = (2 * (dt_lower) * sigmas**2)
blocking_mask = np.where(np.abs(mini_traj) < 1.0)[0]
reinsert_TO = np.empty(
    (sigmas.size, reinsertion_points.size), dtype=np.float64)

for j, eps in enumerate(eps_values):
    print(f"{j}th eps: {eps}")
    L = make_L(mini_traj, eps)
    Q = L[blocking_mask][:, blocking_mask].tocsr()
    t_escape = solve_escape_from_Q(Q)
    reinsert_TO[j, :] = t_escape[rp_idx]

with open("reinsert_TO_toy.pkl", "wb") as f:
    pickle.dump(reinsert_TO, f)
