from scipy.spatial.distance import cdist
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap, LogNorm, Normalize
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Callable
from scipy.sparse.linalg import splu, spilu, gmres, LinearOperator, bicgstab, cg, spsolve, eigs
from scipy import linalg, integrate, special, sparse
from scipy.sparse import eye, csr_matrix
from typing import Optional, Callable, List, Dict, Sequence
from numba import njit, prange
import math
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
import numpy as np
import os
from matplotlib import cm
from sklearn.cluster import KMeans
# Define the state_vector type for the state_vector
dtype = np.float64


def reconstruct_logs(history, dt):
    """
    history: (n_steps, p) array where
      history[k,j] = (1/((k+1)*dt)) * sum_{m=0}^k ell_{m+1}^{(j)}
    returns logs: (n_steps, p) array of ell_k^{(j)}
    """
    n_steps, p = history.shape
    # build k and k-1 as column vectors
    k = np.arange(1, n_steps+1)[:, None]      # shape (n_steps,1)
    km1 = (k - 1)
    # history shifted by one (with a zero‐row at the top)
    hist_prev = np.vstack([np.zeros((1, p)), history[:-1]])
    # vectorized inversion of h_k = (1/(k*dt)) sum ell  =>  ell_k = (k*h_k - (k-1)*h_{k-1})*dt
    logs = (k*history - km1*hist_prev) * dt
    return logs


def compute_ftle_vectorized(logs, dt, window):
    """
    logs:   (n_steps, p) array of instantaneous log‐stretchings ell_k^{(j)}
    window: integer N  (number of steps in the FTLE window)
    returns:
      ftles: (n_steps, p) array, where
        ftles[k,j] = (1/(window*dt)) * sum_{m=k-window+1}^k logs[m,j],
      with NaN for k<window-1
    """
    n_steps, p = logs.shape
    # prefix‐sum of logs along time:
    csum = np.vstack([np.zeros((1, p)), np.cumsum(logs, axis=0)])
    # pairwise differences:  S[k] = csum[k] - csum[k-window]
    # we want S at k=window...n_steps
    S = csum[window:] - csum[:-window]   # shape = (n_steps-window+1, p)
    # allocate output
    ftles = np.empty_like(logs)
    ftles[:window-1] = np.nan
    ftles[window-1:] = S / (window * dt)
    return ftles


def density_normalized_laplacian(positions, f, k=20, epsilon=None):
    """
    Compute the density‐normalized graph Laplacian acting on f:
        L f = (D_alpha - W_alpha) f / epsilon

    where W_alpha is the Coifman‐Lafon (alpha=1) reweighted Gaussian kernel.

    Parameters
    ----------
    positions : array, shape (n_samples, n_dims)
        Sample points x_i in R^6.
    f : array, shape (n_samples,)
        Escape‐time field at each sample.
    k : int, optional (default=20)
        Number of nearest neighbors to use.
    epsilon : float or None, optional
        Kernel width. If None, set to mean squared distance to k-th neighbor.

    Returns
    -------
    lap : ndarray, shape (n_samples,)
        Approximation to Δ_M f at each sample (unnormalized by epsilon).
    """
    n = positions.shape[0]

    # 1) k-NN search
    tree = cKDTree(positions, leafsize=1000)

    # Query k+1 nearest neighbors for all points
    dists, idxs = tree.query(positions, k=k+1, n_jobs=-1)
    dists = dists[:, 1:]      # drop self-distance
    idxs = idxs[:, 1:]       # drop self-index

    # 2) choose epsilon if not provided
    if epsilon is None:
        epsilon = np.mean(dists[:, -1]**2)

    # 3) raw kernel weights: w_ij = exp(-d_ij^2/(4 ε))
    weights = np.exp(- (dists**2) / (4 * epsilon))

    # 4) build sparse raw W (both i->j and j->i for symmetry)
    row = np.repeat(np.arange(n), k)
    col = idxs.ravel()
    data = weights.ravel()
    W_raw = sparse.coo_matrix((data, (row, col)), shape=(n, n))
    W_raw = (W_raw + W_raw.T)
    W_raw.data *= 0.5   # ensure symmetry

    # 5) compute degree for density correction
    d_raw = np.array(W_raw.sum(axis=1)).ravel()  # shape (n,)

    # 6) apply alpha=1 reweighting: W_ij <- W_raw_ij / (d_raw_i * d_raw_j)
    #    we can do this efficiently by scaling rows & columns
    inv_sqrt_d = 1.0 / np.sqrt(d_raw)
    D_inv_sqrt = sparse.diags(inv_sqrt_d)
    W_alpha = D_inv_sqrt @ (W_raw @ D_inv_sqrt)  # Efficient sparse mult

    # 7) degree of the reweighted graph
    d_alpha = np.array(W_alpha.sum(axis=1)).ravel()

    # 8) compute (D_alpha - W_alpha) f
    lap_unnorm = d_alpha * f - W_alpha.dot(f)

    # 9) divide by epsilon to approximate Δ_M f
    lap = lap_unnorm / epsilon

    return lap


def smooth_assignments_graph_laplacian(points, assignments, smoothing_param,
                                       k_neighbors=10, method='knn'):
    """
    Smooth class assignments using graph Laplacian regularization.

    Parameters:
    -----------
    points : ndarray, shape (n_obs, n_features)
        Points in state space
    assignments : ndarray, shape (n_obs,)
        Integer class assignments
    smoothing_param : float
        Smoothing parameter (0 = no smoothing, larger = more smoothing)
    k_neighbors : int, default=10
        Number of neighbors for graph construction
    method : str, default='knn'
        Graph construction method ('knn' or 'epsilon')

    Returns:
    --------
    smoothed_assignments : ndarray, shape (n_obs,)
        Smoothed integer class assignments
    """
    n_obs = points.shape[0]
    unique_classes = np.unique(assignments)
    n_classes = len(unique_classes)

    # Map assignments to consecutive integers starting from 0
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    mapped_assignments = np.array([class_mapping[cls] for cls in assignments])

    # Build k-NN graph
    if method == 'knn':
        tree = cKDTree(points)
        # Query k_neighbors + 1 neighbors (first is the point itself)
        distances, indices = tree.query(points, k=k_neighbors + 1)

        # Remove self-connections
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Build adjacency matrix with Gaussian weights
        row_idx = np.repeat(np.arange(n_obs), k_neighbors)
        col_idx = indices.flatten()

        # Use adaptive bandwidth (median distance)
        sigma = np.median(distances)
        weights = np.exp(-distances.flatten()**2 / (2 * sigma**2))

        adjacency = csr_matrix((weights, (row_idx, col_idx)),
                               shape=(n_obs, n_obs))

        # Symmetrize
        adjacency = (adjacency + adjacency.T) / 2

    # Compute graph Laplacian
    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree_matrix = csr_matrix((degree, (np.arange(n_obs), np.arange(n_obs))),
                               shape=(n_obs, n_obs))
    laplacian = degree_matrix - adjacency

    # Convert assignments to one-hot encoding
    assignment_matrix = np.zeros((n_obs, n_classes))
    assignment_matrix[np.arange(n_obs), mapped_assignments] = 1

    # Solve regularized system: (I + λL)X = Y
    identity = csr_matrix(eye(n_obs, format='csr'))
    # system_matrix = identity + smoothing_param * laplacian

    smoothed_probs = np.zeros((n_obs, n_classes))

    def matvec(x):
        # computes (I + λL) @ x without ever forming it explicitly
        return x + smoothing_param * (degree * x - adjacency.dot(x))

    A = LinearOperator((n_obs, n_obs), matvec=matvec)

    for c in range(n_classes):
        # then for each class:
        smoothed_probs[:, c], info = cg(A, assignment_matrix[:, c], rtol=1e-6)
        # smoothed_probs[:, c] = spsolve(system_matrix,
        #                                assignment_matrix[:, c])

    # Convert back to class assignments
    smoothed_mapped_assignments = np.argmax(smoothed_probs, axis=1)

    # Map back to original class labels
    reverse_mapping = {idx: cls for cls, idx in class_mapping.items()}
    smoothed_assignments = np.array([reverse_mapping[idx]
                                     for idx in smoothed_mapped_assignments])

    return smoothed_assignments


def make_scatter_backgrounds(df, variable_pairs, color_var, categorical=False,
                             cmap='viridis_r', norm=None, bins=(500, 500)):
    """
    For each (var1, var2) in variable_pairs, bin the points in 2D,
    average their mapped colormap RGBA, and return a dict
    { (var1,var2): {'img_array': HxWx4 array, 'extent': (xmin,xmax,ymin,ymax)} }.
    """
    backgrounds = {}
    # If no norm given, set one over the full color_var range
    all_vals = df[color_var].values
    if norm is None:
        norm = Normalize(all_vals.min(), all_vals.max())
    cmap = cm.get_cmap(cmap)

    if categorical:
        # determine unique integer categories
        cats = np.sort(df[color_var].unique())
        n_cats = len(cats)

        # turn your colormap into a discrete one with n_cats bins
        cmap = plt.get_cmap(cmap, n_cats)

        # build a norm that maps each integer to its own colour bin
        # boundaries run from half-integer below the min to half-integer above the max
        boundaries = np.concatenate(([cats[0] - 0.5], cats + 0.5))
        norm = BoundaryNorm(boundaries, ncolors=n_cats)

    for var1, var2 in variable_pairs:
        x = df[var1].values
        y = df[var2].values
        v = df[color_var].values

        # map values → RGBA
        rgba_pts = cmap(norm(v))  # shape (N,4)

        # define bin edges
        xedges = np.linspace(x.min(), x.max(), bins[0] + 1)
        yedges = np.linspace(y.min(), y.max(), bins[1] + 1)

        # digitize positions to bin indices
        xi = np.searchsorted(xedges, x, side='right') - 1
        yi = np.searchsorted(yedges, y, side='right') - 1

        H, W = bins[1], bins[0]
        # accumulators
        sum_rgba = np.zeros((H, W, 4), dtype=np.float64)
        count = np.zeros((H, W),       dtype=np.int64)

        # accumulate
        for xx, yy, col in zip(xi, yi, rgba_pts):
            if 0 <= xx < W and 0 <= yy < H:
                sum_rgba[yy, xx] += col
                count[yy, xx] += 1

        # compute per‑pixel mean RGBA; leave empty pixels alpha=0
        nonzero = count > 0
        img = np.zeros((H, W, 4), dtype=np.float32)
        img[nonzero] = (sum_rgba[nonzero] /
                        count[nonzero][..., None])

        # extent = (xmin, xmax, ymin, ymax)
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

        backgrounds[(var1, var2)] = {
            'img_array': img,
            'extent': extent
        }

    return backgrounds, cmap


def save_single_frame(args):
    """Create a frame with adaptive square subplots and a bottom plot with 1:4 aspect ratio."""
    (t, df_columns, data, state_vector_t, output_folder, cmap, norm,
     variable_pairs, pc_state_t, X, Y) = args

    num_pairs = len(variable_pairs)
    cols = min(5, num_pairs)
    rows = math.ceil(num_pairs / cols)

    # Define subplot size
    square_size = 3.0  # inches
    fig_width = cols * square_size
    fig_height_top = rows * square_size

    # Bottom plot aspect ratio 1:4 (height = width / 4)
    bottom_height = fig_width / 4.0
    total_height = fig_height_top + bottom_height

    # Create figure and GridSpec
    fig = plt.figure(figsize=(fig_width, total_height))
    height_ratios = [1] * rows + [bottom_height / square_size]
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=height_ratios)

    # Create and fill top subplots
    axes = [
        fig.add_subplot(gs[i // cols, i % cols])
        for i in range(num_pairs)
    ]

    for idx, (var1, var2) in enumerate(variable_pairs):
        ax = axes[idx]
        hist_data = data[(var1, var2)]

        ax.imshow(hist_data["img_array"],
                  extent=hist_data["extent"],
                  cmap=cmap,
                  aspect='equal',
                  origin='lower',
                  norm=norm)

        var1_idx = df_columns.get_loc(var1)
        var2_idx = df_columns.get_loc(var2)
        scatter_x = pc_state_t[var1_idx] if pc_state_t is not None else state_vector_t[var1_idx]
        scatter_y = pc_state_t[var2_idx] if pc_state_t is not None else state_vector_t[var2_idx]

        ax.scatter(scatter_x, scatter_y,
                   color='black' if pc_state_t is not None else 'red',
                   s=20, alpha=0.7)

        ax.set_xlabel(var1)
        ax.set_ylabel(var2)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom combined plot
    ax_beneath = fig.add_subplot(gs[rows, :])
    mode = lin_comb(state_vector_t, X, Y)
    plot_fourier_mode(mode, X, Y, ax_beneath)

    # Remove ticks from bottom plot as well
    ax_beneath.set_xticks([])
    ax_beneath.set_yticks([])

    # Save figure
    frame_filename = os.path.join(output_folder, f"frame_{t:03d}.png")
    plt.savefig(frame_filename, dpi=150)
    plt.close(fig)

    return f"Saved frame {t+1}: {frame_filename}"


def save_time_step_plots_parallel(df, data, state_vector,
                                  output_folder="frames",
                                  n_steps=1000, cmap='viridis_r', norm=None,
                                  columns=None, n_processes=None,
                                  pc_state=None):
    """
    Parallel version using multiprocessing.Pool
    """
    # Setup
    os.makedirs(output_folder, exist_ok=True)

    if columns is None:
        x_columns = [col for col in df.columns
                     if str(col).startswith(('x', 'PC'))]
    else:
        x_columns = columns

    variable_pairs = list(combinations(x_columns, 2))

    if n_processes is None:
        n_processes = min(cpu_count(), n_steps)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.linspace(0, np.pi / 2, 500)
    X, Y = np.meshgrid(x, y)
    # Prepare arguments for each time step
    args_list = []
    for t in range(n_steps):
        if pc_state is None:
            args = (t, df.columns, data, state_vector[t, :],
                    output_folder, cmap, norm, variable_pairs, pc_state,
                    X, Y)
        else:
            args = (t, df.columns, data, state_vector[t, :],
                    output_folder, cmap, norm, variable_pairs, pc_state[t, :],
                    X, Y)
        args_list.append(args)

    # Process in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(save_single_frame, args_list)

    for result in results:
        print(result)


def load_d(filename, dims, sample_num):
    # Read the state_vector from the file
    with open(filename, 'rb') as file:
        state_vector = np.fromfile(
            file, dtype=dtype).reshape(dims, sample_num).T

    return state_vector


def build_df(state_vector, dims):
    # Step 1: Create a DataFrame from the numpy array
    df = pd.DataFrame(state_vector, columns=[f'x_{
        i+1}' for i in range(dims)])
    return df


def add_density_column(df, state_vector, distance_threshold):
    # Build a KDTree for efficient neighbor search in 6D space
    tree = cKDTree(state_vector)
    # For each point, count the number of neighbors within the specified distance
    neighbor_counts = [len(tree.query_ball_point(
        point, distance_threshold)) - 1 for point in state_vector]
    print(f"Calculated neighbor counts for {state_vector.shape[0]} points")
    df['Density'] = neighbor_counts  # Add density as a new column

    return df


def compute_eof(df, n_modes):
    """
    Compute the EOF (PCA) decomposition of the input time series data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of shape (time, variables).
    n_modes : int
        Number of EOF modes (principal components) to compute.

    Returns
    -------
    eof_results : dict
        Dictionary containing:
        - 'pca': fitted PCA object
        - 'scores_df': DataFrame of shape (time, n_modes) with principal component time series
        - 'eofs': array of EOF patterns (components)
    """
    pca = PCA(n_components=n_modes)
    scores = pca.fit_transform(df.values)
    pc_cols = [f'PC{i+1}' for i in range(n_modes)]
    scores_df = pd.DataFrame(scores, columns=pc_cols, index=df.index)

    eof_results = {
        'pca': pca,
        'scores_df': scores_df,
        'eofs': pca.components_
    }
    return eof_results


def plot_scatter(df, var_name, cmap='viridis_r', categorical=False,
                 norm=None, columns=None, size=1):
    if columns is None:
        x_columns = [col for col in df.columns if str(
            col).startswith(('x', 'PC'))]
    else:
        x_columns = columns
    # Create variable pairs from x columns
    variable_pairs = list(combinations(x_columns, 2))
    n_plots = len(variable_pairs)
    ncols = min(n_plots, 5)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(size*3*ncols, size*3*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    fig.subplots_adjust(bottom=0.15)  # Adjust for color bar space

    if categorical:
        # determine unique integer categories
        cats = np.sort(df[var_name].unique())
        n_cats = len(cats)

        # turn your colormap into a discrete one with n_cats bins
        cmap = plt.get_cmap(cmap, n_cats)

        # build a norm that maps each integer to its own colour bin
        # boundaries run from half-integer below the min to half-integer above the max
        boundaries = np.concatenate(([cats[0] - 0.5], cats + 0.5))
        norm = BoundaryNorm(boundaries, ncolors=n_cats)

        # and when you draw the colourbar, label it with your integer categories:
        cbar_kwargs = dict(ticks=cats, format='%d')
    else:
        cbar_kwargs = {}

    for idx, (var1, var2) in enumerate(variable_pairs):
        ax = axes[idx]

        # Scatter plot for each variable pair
        sc = ax.scatter(df[var1], df[var2], c=df[var_name], cmap=cmap,
                        norm=norm, s=.01)

        # Set axis labels
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)

    # Add a shared color bar for Density across all subplots
    # [left, bottom, width, height]
    # Adjust the position and dimensions
    cbar_ax = fig.add_axes([0.15, 0, 0.7, 0.05])
    cbar = fig.colorbar(sc,
                        cax=cbar_ax,
                        orientation='horizontal',
                        label=var_name,
                        **cbar_kwargs)

    # plt.suptitle("Scatter Plots of Variable Pairs with Density-Based Coloring (6D)")
    # plt.tight_layout(rect=[0, .15, 1, .85])  # Adjust layout for color bar
    plt.show()
    return fig, ax


def plot_density_heatmap(df, bins=200, cmap='viridis_r', columns=None, variable_pairs=None,
                         axis_limits=None, points=None):
    """
    Plot a grid of 2D histograms for each unique pair of variables in df.
    Uses fine bins and a logarithmic color scale for density representation.
    """

    # Generate all unique variable pairs (combinations)
    if columns is None:
        x_columns = [col for col in df.columns if str(
            col).startswith(('x', 'PC'))]
    else:
        x_columns = columns
    # Create variable pairs from x columns
    if variable_pairs is None:
        variable_pairs = list(combinations(x_columns, 2))
    n_plots = len(variable_pairs)
    ncols = min(n_plots, 5)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    # Adjust for color bar space
    fig.subplots_adjust(wspace=0.31, hspace=0.0, bottom=0.25)
    histograms = {}
    for idx, (var1, var2) in enumerate(variable_pairs):
        ax = axes[idx]

        # Plot 2D histogram for each variable pair
        h = ax.hist2d(df[var1], df[var2], bins=bins,
                      cmap=cmap, norm=LogNorm(), rasterized=True)

        if points is not None:
            if isinstance(points, dict):
                x_vals = points.get(var1)
                y_vals = points.get(var2)
                if x_vals is not None and y_vals is not None:
                    colors = ['black', 'red', 'orange']
                    for i, (xv, yv) in enumerate(zip(x_vals, y_vals)):
                        color = colors[i] if i < len(colors) else 'orange'
                        ax.scatter(xv, yv, c=color, s=64, alpha=1,
                                   edgecolors=color, linewidths=1, marker="x")

        if axis_limits:
            if var1 in axis_limits:
                ax.set_xlim(axis_limits[var1])
            if var2 in axis_limits:
                ax.set_ylim(axis_limits[var2])
        else:
            # Auto-calculate limits to include both histogram and points
            x_min, x_max = df[var1].min(), df[var1].max()
            y_min, y_max = df[var2].min(), df[var2].max()

            # Extend limits if points exist
            if points is not None:
                if isinstance(points, dict):
                    x_vals = points.get(var1)
                    y_vals = points.get(var2)
                    if x_vals is not None and y_vals is not None:
                        x_min = min(x_min, np.min(x_vals))
                        x_max = max(x_max, np.max(x_vals))
                        y_min = min(y_min, np.min(y_vals))
                        y_max = max(y_max, np.max(y_vals))

            # Add small padding (5% of range)
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = 0.05 * x_range
            y_padding = 0.05 * y_range

            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        histograms[(var1, var2)] = {
            "ax": ax,
            # Store color mesh for color bar reference
            "img_array": h[3].get_array(),
            "extent": [df[var1].min(), df[var1].max(), df[var2].min(), df[var2].max()]
        }
        # Store the axis and color scale (image object)
        # Set axis labels
        ax.set_xlabel(f"${var1}$", labelpad=2)
        ax.set_ylabel(f"${var2}$", labelpad=2)
        x_min, x_max = df[var1].min(), df[var1].max()
        if x_max - x_min < .5:
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        else:
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.3))

        # ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune=None))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune=None))
        # ax.xaxis.set_major_formatter(
        #     plt.FuncFormatter(lambda val, _: f"{val:.1f}"))
        # ax.yaxis.set_major_formatter(
        #     plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    # Add a shared color bar across all subplots
    cbarAx = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Adjust position and size
    fig.colorbar(h[3], cax=cbarAx, orientation='horizontal',
                 label='Point Count')

    return fig, histograms


def generate_fourier_mode(n, m, X, Y):
    """
    Generates the 2D Fourier mode as a 2D array.

    Parameters:
    n (int): The mode number in the y direction.
    m (int): The mode number in the x direction.
    resolution (int): The number of grid points for x and y directions.

    Returns:
    tuple: A tuple containing:
        - x (ndarray): The x-coordinate grid.
        - y (ndarray): The y-coordinate grid.
        - mode (ndarray): The computed Fourier mode values.
    """

    # Compute the Fourier mode
    if m == 0:
        mode = np.sqrt(2) * np.cos(n * Y / .5)
    else:
        mode = np.sqrt(2) * np.exp(1j * m * X) * np.sin(n * Y / .5)
        mode = mode.real  # Take the real part for visualization

    return mode


def plot_fourier_mode(array, x, y, ax):
    """
    Plots the Fourier mode as a heatmap with a red-white-blue colormap.
    The white color represents the midpoint (0).

    Parameters:
    array (ndarray): The 2D array representing the Fourier mode values.
    x (ndarray): The x-coordinate grid.
    y (ndarray): The y-coordinate grid.
    """
    # Create a symmetric colormap with white as the midpoint
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    # plt.contourf(x, y, array, cmap=cmap, levels=100, norm=norm)
    # , levels=100, norm=norm)
    im = ax.contourf(x, y, array, cmap="RdBu", norm=norm)

    # plt.colorbar(label='Amplitude')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("2D Fourier Mode")
    # plt.show()
    return im


def lin_comb(x, X, Y):
    return (generate_fourier_mode(1, 0, X, Y) * x[0] +
            generate_fourier_mode(1, 1, X, Y) * x[1] +
            generate_fourier_mode(1, 2, X, Y) * x[2] +
            generate_fourier_mode(2, 0, X, Y) * x[3] +
            generate_fourier_mode(2, 1, X, Y) * x[4] +
            generate_fourier_mode(2, 2, X, Y) * x[5])


# mode = lin_comb([0.1, 0.4, -0.2, -1.1, 0.05, 0.3], X, Y)
# plotFourierMode(mode, X, Y)


def save_time_step_plots(df, histograms, stateVector, outputFolder="frames", bins=100,
                         n_steps=1000, cmap='viridis_r', columns=None):
    """
    For each time step, overlay the system's state on the histogram backgrounds and save the figure.
    """
    # Ensure output directory exists
    os.makedirs(outputFolder, exist_ok=True)
    if columns is None:
        x_columns = [col for col in df.columns if str(
            col).startswith(('x', 'PC'))]
    else:
        x_columns = columns
    # Create variable pairs from x columns
    variable_pairs = list(combinations(x_columns, 2))
    n_plots = len(variable_pairs)
    ncols = min(n_plots, 5)
    nrows = (n_plots + ncols - 1) // ncols
    # Loop through each time step
    for t in range(n_steps):
        # Use gridspec to add an extra row for the additional plot beneath
        # Adjust the figure size to accommodate the new row
        fig = plt.figure(figsize=(15, 12))
        # Extra row at the bottom for the new plot
        gs = fig.add_gridspec(4, 5, height_ratios=[3, 3, 3, 4])

        # Plot the histogram matrix in the top 3 rows (3x5 grid)
        # variablePairs = list(combinations(df.columns, 2))
        axes = [fig.add_subplot(gs[row, col])
                for row in range(3) for col in range(5)]

        for idx, (var1, var2) in enumerate(variable_pairs):
            ax = axes[idx]
            histData = histograms[(var1, var2)]

            # Plot the histogram background
            ax.imshow(histData["img"].get_array(), extent=histData["extent"],
                      cmap=cmap,
                      norm=LogNorm(), aspect='auto', origin="lower")

            # Plot current state as a red dot
            ax.scatter(stateVector[t, df.columns.get_loc(var1)],
                       stateVector[t, df.columns.get_loc(var2)],
                       color='red', s=20, alpha=0.7)

            ax.set_xlabel(var1)
            ax.set_ylabel(var2)

        # Create a new axis beneath the histograms for additional plotting
        ax_beneath = fig.add_subplot(gs[3, :])  # The entire last row

        # # Example usage
        # Define the domain
        x = np.linspace(0, 2 * np.pi, 500)
        y = np.linspace(0, np.pi / 2, 500)
        X, Y = np.meshgrid(x, y)
        mode = lin_comb(stateVector[t, :], X, Y)
        plot_fourier_mode(mode, X, Y, ax_beneath)

        # Save each frame as an individual PNG file
        frameFilename = os.path.join(outputFolder, f"frame_{t:03d}.png")
        plt.savefig(frameFilename, dpi=150)
        plt.close(fig)  # Close the figure to free memory

        print(f"Saved frame {t+1}/{n_steps}: {frameFilename}")

        from multiprocessing import Pool, cpu_count


def combine_binary_files(base_name, output_filename, dtype, dims,
                         sample_num, file_count=99):
    """
    Combine multiple binary files into a single file preserving structure.

    Args:
        base_name: Base filename pattern (e.g., 'dataOro20_')
        output_filename: Name of the combined output file
        dtype: Data type for reading files
        dims: Number of dimensions in each file
        sample_num: Number of samples per file
        file_count: Number of files to combine (default: 99)
    """
    missing_files = []
    all_data = []

    for file_idx in range(1, file_count + 1):
        input_filename = f"{base_name}{file_idx}.bin"

        if not os.path.exists(input_filename):
            missing_files.append(input_filename)
            continue

        with open(input_filename, 'rb') as file:
            file_data = np.fromfile(file, dtype=dtype).reshape(dims,
                                                               sample_num).T
            all_data.append(file_data)

    if all_data:
        combined_array = np.vstack(all_data)
        # Save in same format: transpose and flatten
        combined_binary = combined_array.T.flatten()
        combined_binary.astype(dtype).tofile(output_filename)

    if missing_files:
        print(f"Warning: {len(missing_files)} files were missing:")
        for missing_file in missing_files:
            print(f"  - {missing_file}")

    files_processed = file_count - len(missing_files)
    total_samples = files_processed * sample_num
    print(f"Combined {files_processed} files into {output_filename}")
    print(f"New sample_num for combined file: {total_samples}")


def diffusion_maps_matrix(X, epsilon):
    """
    Build diffusion-maps Markov matrix and affinity matrix using only SciPy.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix.
    epsilon : float
        Kernel scale parameter (>0). Kernel used: exp(-d^2 / epsilon).

    Returns
    -------
    DMM : scipy.sparse.csr_matrix
        Row-stochastic diffusion maps matrix (Markov matrix).
    A : scipy.sparse.csr_matrix
        Symmetric affinity matrix with Gaussian kernel (diagonal entries = 1).
    """

    n_samples = X.shape[0]
    r = 3.0 * np.sqrt(epsilon)

    tree = cKDTree(X)
    indices_list = tree.query_ball_point(X, r, return_sorted=False)

    rows = []
    cols = []
    vals = []

    for i, inds in enumerate(indices_list):

        # compute distances only for returned neighbors, sort by distance
        inds_arr = np.asarray(inds, dtype=int)
        pts = X[inds_arr]
        diffs = pts - X[i]  # shape (len(inds), n_features)
        dists = np.linalg.norm(diffs, axis=1)

        order = np.argsort(dists)
        inds_sorted = inds_arr[order]
        dists_sorted = dists[order]

        for j_idx, dist_ij in zip(inds_sorted, dists_sorted):
            # add (i, j) entry
            rows.append(i)
            cols.append(int(j_idx))
            vals.append(dist_ij)

    if len(rows) == 0:
        # no neighbors found within radius -> return trivial matrices
        A = sparse.identity(n_samples, format='csr')
        DMM = sparse.identity(n_samples, format='csr')
        return DMM, A

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    vals = np.array(vals, dtype=np.float64)

    # kernel: Gaussian using epsilon in denominator
    weights = np.exp(-(vals ** 2) / epsilon)

    # Build the symmetric affinity matrix (COO -> CSR)
    A = sparse.coo_matrix((weights, (rows, cols)),
                          shape=(n_samples, n_samples))

    A = A.tocsr()

    A = (A + A.T)

    # Ensure diagonal = 1.0 (self-affinity exp(0)=1)
    A.setdiag(1.0)

    # --- Coifman-Lafon density normalization (alpha = 1.0) ---
    alpha = 1.0
    # compute degree / density q_i = sum_j A_ij
    q = np.asarray(A.sum(axis=1)).ravel()

    D_alpha_inv = sparse.diags(q ** (-alpha), offsets=0, format='csr')
    K = D_alpha_inv.dot(A).dot(D_alpha_inv)  # K = D_alpha^{-1} A D_alpha^{-1}

    # Row-normalize K to obtain Markov matrix P (DMM)
    row_sums = np.asarray(K.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = np.finfo(float).eps
    D_row_inv = sparse.diags(1.0 / row_sums, offsets=0, format='csr')
    DMM = D_row_inv.dot(K)

    return DMM, A


def compute_clusters_from_TO(L, n_eig=12, n_clusters=2, n_init=10,
                             eig_indices=None, random_state=0):
    """
    L: scipy.sparse matrix, shape (n_samples, n_samples)  -- TO transposed
    n_eig: int -- number of eigenpairs to compute (ARPACK)
    n_clusters: int -- k for k-means
    n_init: int -- k-means n_init
    eig_indices: None or list/array of zero-based eigenvector indices to use as features.
                 If None, defaults to [1,2,3,4] (i.e. first nontrivial eigenvectors).
    random_state: int
    returns: dict with keys 'labels', 'kmeans', 'eigenvals', 'eigenvecs', 'features', 'L'
    """

    k_eigs = max(1, min(n_eig, L.shape[0] - 1))

    # try sparse eigen solver first, fallback to dense
    try:
        eigvals, eigvecs = eigs(L, k=k_eigs, which='LM')
    except Exception:
        L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        eigvals_all, eigvecs_all = np.linalg.eig(L_dense)
        idx_sort = np.argsort(-np.abs(eigvals_all))
        take = min(k_eigs, len(idx_sort))
        eigvals = eigvals_all[idx_sort[:take]]
        eigvecs = eigvecs_all[:, idx_sort[:take]]

    # default: skip trivial dominant eigenvector (index 0) and use next few
    if eig_indices is None:
        default_indices = list(range(1, min(5, eigvecs.shape[1])))
        eig_indices = default_indices if len(default_indices) > 0 else [0]

    eig_indices = [int(i) for i in eig_indices if 0 <= i < eigvecs.shape[1]]
    if len(eig_indices) == 0:
        eig_indices = [0]

    selected = eigvecs[:, eig_indices]
    if np.iscomplexobj(selected):
        features = np.hstack([selected.real, selected.imag])
    else:
        features = selected.real

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                    random_state=random_state)
    labels = kmeans.fit_predict(features)

    return {
        'labels': labels,        # clustering labels for m-1 samples
        'kmeans': kmeans,        # fitted KMeans object
        'eigenvals': eigvals,    # eigenvalues of L
        'eigenvecs': eigvecs,    # eigenvectors of L
        'features': features,    # real/imag stacked features
    }


def classify_new_trajectory(X, labels, Y, k=10, threshold=0.5, tree=None):
    """
    X: ndarray (n_samples, d)   -- original state space points
    labels: ndarray (n_samples,) -- cluster labels for X (e.g. from k-means)
    Y: ndarray (m_samples, d)   -- new trajectory to classify
    k: int -- number of nearest neighbors
    threshold: float in [0,1] -- fraction of neighbors required to assign regime

    returns: ndarray (m_samples,) of regime labels for Y
             (-1 if no cluster passes threshold)
    """
    if tree is None:
        tree = cKDTree(X)

    # Query k nearest neighbors of each point in Y
    dists, idxs = tree.query(Y, k=k)

    Y_labels = []
    for neighbors in idxs:
        neigh_labels = labels[neighbors]
        uniq, counts = np.unique(neigh_labels, return_counts=True)
        dominant = uniq[np.argmax(counts)]
        frac = counts.max() / k
        if frac > threshold:
            Y_labels.append(dominant)
        else:
            Y_labels.append(-1)
    return np.array(Y_labels)


def fit_hmm(data_array, n_states=3, initial_centers=None):
    """
    Fit Hidden Markov Model with specified number of states.

    Args:
        data_array: numpy array of shape (n_samples, n_features)
        n_states: int, number of hidden states to fit
        initial_centers: optional numpy array of shape (n_states, n_features)
                        providing initial guesses for state centers

    Returns:
        tuple: (state_means, state_assignments)
            - state_means: array of shape (n_states, n_features)
            - state_assignments: array of shape (n_samples,) with values 1 to n_states
    """
    # Validate inputs
    if not isinstance(data_array, np.ndarray) or data_array.ndim != 2:
        raise ValueError("data_array must be 2D numpy array")

    if not isinstance(n_states, int) or n_states < 2:
        raise ValueError("n_states must be integer >= 2")

    n_samples, n_features = data_array.shape

    if n_samples < n_states:
        raise ValueError(
            f"Number of samples ({n_samples}) must be >= n_states ({n_states})")

    # Validate initial_centers if provided
    if initial_centers is not None:
        if not isinstance(initial_centers, np.ndarray):
            raise ValueError("initial_centers must be numpy array")
        if initial_centers.shape != (n_states, n_features):
            expected_shape = (n_states, n_features)
            raise ValueError(
                f"initial_centers shape {initial_centers.shape} != expected {expected_shape}")

    # Standardize the data for better convergence
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_array)

    # Initialize Gaussian HMM
    if initial_centers is not None:
        # Manual initialization with provided centers
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            init_params="c",  # Only initialize covariances, not means
            params="stmc"
        )

        # Transform initial centers to scaled space
        initial_centers_scaled = scaler.transform(initial_centers)
        model.means_ = initial_centers_scaled

    else:
        # Default initialization
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            init_params="mc",
            params="stmc"
        )

    # Fit the model
    model.fit(data_scaled)

    # Get state assignments (0-indexed from hmmlearn)
    state_sequence = model.predict(data_scaled)

    # Convert to 1-indexed
    state_assignments = state_sequence + 1

    # Get state means in original scale
    state_means_scaled = model.means_
    state_means = scaler.inverse_transform(state_means_scaled)
    # Extract standard deviations from covariance matrices
    state_covs_scaled = model.covars_
    state_stds_scaled = np.sqrt(np.diagonal(
        state_covs_scaled, axis1=1, axis2=2))

    # Transform standard deviations back to original scale
    # Standard deviation scaling: std_original = std_scaled * scale_factor
    scale_factors = scaler.scale_
    state_stds = state_stds_scaled * scale_factors

    transition_matrix = model.transmat_
    print("Transition Matrix:")
    print(f"Shape: {transition_matrix.shape}")
    for i in range(n_states):
        row_str = " ".join([f"{transition_matrix[i, j]:.4f}"
                            for j in range(n_states)])
        print(f"State {i+1}: [{row_str}]")
    print()

    return state_means, state_stds, state_assignments


def plot_state_modes(state_centers, states_sd, inits):
    fig, ax = plt.subplots(3, state_centers.shape[0], figsize=[15, 6])

    # Define the domain
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.linspace(0, np.pi / 2, 500)
    X, Y = np.meshgrid(x, y)

    for i in range(state_centers.shape[0]):
        mode = lin_comb(state_centers[i, :], X, Y)
        im1 = ax[0, i].contourf(X, Y, mode, cmap="RdBu",
                                norm=mcolors.TwoSlopeNorm(
                                    vmin=-2.5, vcenter=0, vmax=2.5))

        mode = lin_comb(states_sd[i, :], X, Y)
        im2 = ax[1, i].contourf(X, Y, mode, cmap="RdBu",
                                norm=mcolors.TwoSlopeNorm(
                                    vmin=-1, vcenter=0, vmax=1))

        mode = lin_comb(inits[i, :], X, Y)
        im3 = ax[2, i].contourf(X, Y, mode, cmap="RdBu",
                                norm=mcolors.TwoSlopeNorm(
                                    vmin=-2.5, vcenter=0, vmax=2.5))

    plt.subplots_adjust(right=0.85)

    cbar1 = fig.colorbar(im1, ax=ax[0, :], orientation='vertical',
                         shrink=0.8, aspect=30, pad=0.02)
    cbar1.set_label('State Centers', rotation=270, labelpad=15)

    cbar2 = fig.colorbar(im2, ax=ax[1, :], orientation='vertical',
                         shrink=0.8, aspect=30, pad=0.02)
    cbar2.set_label('State Standard Deviations', rotation=270, labelpad=15)

    cbar3 = fig.colorbar(im3, ax=ax[2, :], orientation='vertical',
                         shrink=0.8, aspect=30, pad=0.02)
    cbar3.set_label('Initials', rotation=270, labelpad=15)

    plt.show()


def calculate_escape_times(state_sequence, dt=1.0):
    """
    Calculate escape times for each regime in a state sequence.

    Args:
        state_sequence: 1D array of state assignments (1-indexed)
        dt: float, time step between observations (default: 1.0)

    Returns:
        dict: Dictionary with keys as state numbers, values as lists of
              escape times for that state
    """
    if not isinstance(state_sequence, np.ndarray):
        state_sequence = np.array(state_sequence)

    escape_times = defaultdict(list)
    n_points = len(state_sequence)

    if n_points == 0:
        return dict(escape_times)

    # Find runs of consecutive states
    current_state = state_sequence[0]
    run_start = 0

    for i in range(1, n_points):
        if state_sequence[i] != current_state:
            # End of current run
            run_length = i - run_start
            escape_time = run_length * dt
            escape_times[current_state].append(escape_time)

            # Start new run
            current_state = state_sequence[i]
            run_start = i

    # Handle the last run
    run_length = n_points - run_start
    escape_time = run_length * dt
    escape_times[current_state].append(escape_time)

    return dict(escape_times)


def analyze_escape_statistics(escape_times_dict):
    """
    Calculate statistics for escape times of each regime.

    Args:
        escape_times_dict: Dictionary from calculate_escape_times()

    Returns:
        pd.DataFrame: Statistics for each regime
    """
    stats_data = []

    for state, times in escape_times_dict.items():
        if len(times) > 0:
            stats = {
                'state': state,
                'n_episodes': len(times),
                'mean_escape_time': np.mean(times),
                'median_escape_time': np.median(times),
                'std_escape_time': np.std(times),
                'min_escape_time': np.min(times),
                'max_escape_time': np.max(times),
                'total_time_in_state': np.sum(times)
            }
            stats_data.append(stats)

    return pd.DataFrame(stats_data)


def get_escape_times_for_points(state_sequence, target_state, dt=1.0):
    """
    For each point in target state, assign the current episode length.

    Args:
        state_sequence: 1D array of state assignments
        target_state: int, state number to analyze
        dt: float, time step between observations

    Returns:
        tuple: (indices, escape_times) where indices are positions of
               target_state points and escape_times are corresponding times
    """
    if not isinstance(state_sequence, np.ndarray):
        state_sequence = np.array(state_sequence)

    indices = []
    escape_times = []
    n_points = len(state_sequence)

    i = 0
    while i < n_points:
        if state_sequence[i] == target_state:
            # Found start of target state run
            run_start = i
            # Find end of run
            while i < n_points and state_sequence[i] == target_state:
                i += 1
            # Calculate individual escape time for each point in this run
            run_length = i - run_start
            for j in range(run_start, i):
                indices.append(j)
                # Escape time is remaining steps until end of run
                remaining_steps = i - j
                escape_time = remaining_steps * dt
                escape_times.append(escape_time)
        else:
            i += 1

    return np.array(indices), np.array(escape_times)


def plot_escape_time_distributions(escape_times_dict, episodes=True):
    """
    Plot histograms of escape time distributions for each state.

    Args:
        escape_times_dict: Dictionary from calculate_escape_times()
        bins: int, number of histogram bins
    """
    n_states = len(escape_times_dict)
    if n_states == 0:
        print("No escape times to plot")
        return

    fig, axes = plt.subplots(1, n_states, figsize=(5*n_states, 4))
    if n_states == 1:
        axes = [axes]

    for idx, (state, times) in enumerate(escape_times_dict.items()):
        if len(times) > 0:

            if episodes:
                bins = np.arange(min(times), max(times) + 2)
                axes[idx].hist(times, bins=bins, align="left", rwidth=.8,
                               alpha=0.7, edgecolor='black')
                axes[idx].set_title(
                    f'State {state} Escape Times (per episode)')
                # Add statistics text
                mean_time = np.mean(times)
                axes[idx].axvline(mean_time, color='red', linestyle='--',
                                  label=f'Mean: {mean_time:.2f}')
                axes[idx].legend()
            else:
                bins = np.arange(
                    min(times['point_escape_times']),
                    max(times['point_escape_times']) + 2)
                axes[idx].hist(times['point_escape_times'], bins=bins,
                               align="left", rwidth=.8,
                               alpha=0.7, edgecolor='black')
                axes[idx].set_title(
                    f'State {state} Escape Times (per time step)')
            axes[idx].set_xlabel('Escape Time')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def create_escape_time_sequence(state_sequence, target_state, dt=1.0):
    """
    Create a time series showing escape times for target state points.

    Args:
        state_sequence: 1D array of state assignments
        target_state: int, state to analyze
        dt: float, time step

    Returns:
        np.array: Array same length as state_sequence, with escape times
                  for target_state points, NaN elsewhere
    """
    escape_sequence = np.full(len(state_sequence), np.nan)
    indices, escape_times = get_escape_times_for_points(
        state_sequence, target_state, dt)

    escape_sequence[indices] = escape_times
    return escape_sequence


def analyze_regime_escape_times(states, dt=1.0):
    """
    Complete analysis of escape times for your HMM results.
    Args:
        states: State assignments from your HMM (1-indexed)
        dt: Time step between observations
        target_regime: Specific regime to focus analysis on
    Returns:
        dict: Comprehensive results
    """
    # Calculate escape times for all regimes
    escape_times_all = calculate_escape_times(states, dt)

    # Get statistics
    stats_df = analyze_escape_statistics(escape_times_all)
    print("Escape Time Statistics:")
    print(stats_df.to_string(index=False))
    print()

    # Analyze all regimes
    all_regime_data = {}

    for regime in escape_times_all:
        regime_escape_times = escape_times_all[regime]

        print(f"Regime {regime} Analysis:")
        print(f"Number of episodes: {len(regime_escape_times)}")
        print(f"Mean escape time: {np.mean(regime_escape_times):.3f}")
        print(f"Median escape time: {np.median(regime_escape_times):.3f}")
        print(f"Std escape time: {np.std(regime_escape_times):.3f}")
        print()

        # Get point-wise escape times
        indices, point_escape_times = get_escape_times_for_points(
            states, regime, dt)
        print(f"Points in regime {regime}: {len(indices)}")
        print(f"Escape times range: {np.min(point_escape_times):.3f} "
              f"to {np.max(point_escape_times):.3f}")
        print()

        # Store data for this regime
        all_regime_data[regime] = {
            'escape_times': regime_escape_times,
            'indices': indices,
            'point_escape_times': point_escape_times
        }

    # Create visualization
    plot_escape_time_distributions(escape_times_all, episodes=True)
    plot_escape_time_distributions(all_regime_data, episodes=False)

    # Prepare return dictionary with all regime data
    result = {
        'all_escape_times': escape_times_all,
        'statistics': stats_df,
        'all_regime_data': all_regime_data
    }

    return result


def escape_times(regimes):
    """
    Vectorized computation of escape times.

    Parameters
    ----------
    regimes : array-like of int, shape (n,)
        Regime label at each time-step.

    Returns
    -------
    escapes : ndarray of int, shape (n,)
        escapes[i] = number of steps until regimes[i] != regimes[i + k],
        or 0 if the regime never changes again.
    """
    regimes = np.asarray(regimes)
    n = regimes.size

    # 1) Find all the change‐points (the first index of each new regime segment)
    change_idx = np.flatnonzero(regimes[:-1] != regimes[1:]) + 1
    # e.g. regimes = [1,1,2,2,3] -> change_idx = [2, 4]

    # 2) For each i, find the insertion position in change_idx of i
    #    side='right' means we get the first change_idx > i
    positions = np.searchsorted(change_idx, np.arange(n), side='right')

    # 3) Build the output, defaulting to 0
    escapes = np.zeros(n, dtype=int)

    #  only those with positions < len(change_idx) actually have a next change
    mask = positions < change_idx.size
    valid_i = np.nonzero(mask)[0]
    # next change index for each valid i:
    next_changes = change_idx[positions[mask]]
    escapes[mask] = next_changes - valid_i

    return escapes


def plot_lifetime_distributions_lines(escape_results, regime, logx=True,
                                      xlabel="Sigma"):
    sigmas = sorted(escape_results[regime].keys())

    medians = []
    means = []
    q25s = []
    q75s = []
    q0s = []
    q95s = []

    for sigma in sigmas:
        data = np.array(escape_results[regime][sigma])
        if len(data) == 0:
            medians.append(np.nan)
            means.append(np.nan)
            q25s.append(np.nan)
            q75s.append(np.nan)
            q0s.append(np.nan)
            q95s.append(np.nan)
            continue

        q0, q25, q50, q75, q95 = np.percentile(data, [0, 25, 50, 75, 95])
        mean_val = np.mean(data)

        q0s.append(q0)
        q25s.append(q25)
        medians.append(q50)
        q75s.append(q75)
        q95s.append(q95)
        means.append(mean_val)

    fig, ax = plt.subplots(figsize=(5, 3))

    # median
    ax.plot(sigmas, medians, marker=None,
            color='black', linewidth=2, label='Median')

    # mean
    ax.plot(sigmas, means, marker=None, color='red',
            linewidth=1.5, label='Mean')

    # IQR shaded
    ax.fill_between(sigmas, q25s, q75s, color='lightgray',
                    alpha=0.7, label='IQR')

    # 0th–95th percentile (dotted lines)
    ax.plot(sigmas, q0s, linestyle=':', color='black', linewidth=1,
            label='0–95th percentile')
    ax.plot(sigmas, q95s, linestyle=':', color='black', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Regime Lifetime")
    # ax.set_title(f"Regime {regime} lifetime distributions")
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=1.0)
    plt.tight_layout()
    return fig, ax


def add_scaled_top_axis(fig, ax, sigmas, top_label=r"$\sqrt{\frac{\epsilon}{2}}$"):
    """
    Add a robust secondary x-axis (top) scaled as sqrt(2*x).

    Mapping:
        top = sqrt(2 * x)
        bottom = (top**2) / 2

    Handles zeros, negatives, NaNs, and Infs safely.
    """

    def scale_func(x):
        x = np.asarray(x, dtype=float)
        out = np.full_like(x, np.nan)
        valid = np.isfinite(x) & (x > 0)
        out[valid] = np.sqrt(x[valid] / 2)
        # replace remaining invalids with 0 to prevent NaN/Inf axis limits
        out[~valid] = 0
        return out

    def inverse_func(x_top):
        x_top = np.asarray(x_top, dtype=float)
        out = np.full_like(x_top, np.nan)
        valid = np.isfinite(x_top) & (x_top >= 0)
        out[valid] = (x_top[valid] ** 2) * 2
        out[~valid] = 0
        return out

    # Create top axis with safe mapping
    ax_top = ax.secondary_xaxis('top', functions=(scale_func, inverse_func))
    ax_top.set_xlabel(top_label)

    # Clip domain to avoid invalid transformations when autoscaling
    xmin, xmax = ax.get_xlim()
    xmin = max(xmin, 1e-12)  # avoid 0 or negative
    ax.set_xlim(xmin, xmax)

    fig.tight_layout()
    return fig, ax_top


def expected_escape_times_from_TO(L, mask):
    """
    L: scipy.sparse column-stochastic matrix
    mask: boolean array with True for blocking set
    """

    # Restrict to blocking set
    indices = np.where(mask)[0]
    Q = L[indices, :][:, indices].tocsr()

    # Solve (I - Q) t = 1 without forming dense inverse
    I = eye(Q.shape[0], format='csr')
    t_escape = spsolve(I - Q, np.ones(Q.shape[0]))

    return t_escape


def spy_col(Q):
    coo = Q.tocoo()
    sc = plt.scatter(coo.col, coo.row, c=np.abs(coo.data),
                     s=50, marker='s', cmap='viridis')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xlim(-0.5, Q.shape[1] - 0.5)
    ax.set_ylim(Q.shape[0] - 0.5, -0.5)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax)
    return ax


# ------ TOY functions

# Numba kernel (parallel over trajectories)


@njit(parallel=True, cache=True)
def _simulate_numba_parallel_sigma_arr(dt, sigma_arr, x0,
                                       normals, reset_samples, save_every):
    """
    Numba-parallel kernel: independent trajectories in columns.
    normals, reset_samples: shape (n_steps, n_paths)
    sigma_arr: shape (n_paths,)
    Returns saved states shape (n_saves, n_paths) dtype=float32
    """
    n_steps_local, n_paths = normals.shape
    max_saves = (n_steps_local // save_every) + 1
    saved = np.empty((max_saves, n_paths), dtype=np.float32)
    # noise = np.empty((max_saves, n_paths), dtype=np.float32)

    phi = math.exp(dt)
    var = (math.exp(2.0 * dt) - 1.0) / 2.0
    sqrt_var = math.sqrt(var)

    for p in prange(n_paths):
        x = np.float32(x0)
        save_idx = 0
        saved[save_idx, p] = x
        save_idx += 1

        s = float(sigma_arr[p])
        for i in range(n_steps_local):
            # restart at start-of-step if outside [-2,2]
            if x >= 2.0 or x <= -2.0:
                sign = 1.0 if x >= 0.0 else -1.0
                x = sign * reset_samples[i, p]
            else:
                # exact linear-step
                x = x * phi + s * normals[i, p] * sqrt_var

            if ((i + 1) % save_every) == 0:
                saved[save_idx, p] = x
                # noise[save_idx, p] = s * normals[i, p] * sqrt_var
                save_idx += 1

    return saved  # , noise


def simulate_trajectories_per_sigma(
        sigmas,
        dt: float,
        n_steps: int,
        reset_sampler: Callable[[np.random.Generator, tuple], np.ndarray],
        rng: Optional[np.random.Generator] = None,
        save_every: int = 1,
        x0: float = 0.0,
        dtype=np.float32):
    """
    Simulate one continuous trajectory per sigma value in `sigmas`.
    Args:
      sigmas: scalar or 1-D array-like of sigma values (if scalar it will be converted to a length-1 array)
      dt: time step
      n_steps: number of integration steps (trajectory length)
      reset_sampler: vectorized callable reset_sampler(rng, size) -> array shaped `size` of positive reset magnitudes
      rng: np.random.Generator (if None a new generator is created)
      save_every: save states every `save_every` steps (1 => save every step)
      x0: initial condition
      dtype: np.float32 recommended; set to np.float64 if you need double precision
      verbose: print progress/timings if True

    Returns:
      times: 1D float array of saved times (length n_saves)
      saved_states: 2D array shape (n_saves, n_paths), column j corresponds to sigmas[j]
      sigma_arr: 1D array of sigma values (dtype)
    """
    if rng is None:
        rng = np.random.default_rng()
    n_paths = sigmas.shape[0]
    n_saves = (n_steps // save_every) + 1

    # Pre-sample normals and reset samples (shape: n_steps x n_paths)
    normals = rng.normal(loc=0.0, scale=1.0, size=(
        n_steps, n_paths)).astype(dtype, copy=False)
    reset_samples = reset_sampler(rng, size=(
        n_steps, n_paths)).astype(dtype, copy=False)

    saved_states = _simulate_numba_parallel_sigma_arr(float(dt), sigmas,
                                                      float(x0),
                                                      normals, reset_samples,
                                                      int(save_every))
    times = np.arange(n_saves, dtype=float) * (save_every * dt)
    return times, saved_states,  sigmas


# Example vectorized reset samplers (you must pass one of these or your own)
def uniform_reset_sampler_vectorized(rng: np.random.Generator, size):
    """
    Returns reset magnitudes drawn uniformly on [low, high).
    size: tuple (n_steps, n_paths)
    """
    low = 0.05
    high = 0.1
    return rng.uniform(low, high, size=size)


def uniform_radius_sampler_vectorized(rng: np.random.Generator, size, low=0.05, high=0.1):
    """
    Returns radii of 3D vectors whose components are drawn uniformly from [low, high).

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator.
    size : tuple of ints
        Output shape (e.g. (n_steps, n_paths)).
    low, high : float
        Bounds for uniform distribution of each vector component.

    Returns
    -------
    radii : np.ndarray
        Array of shape `size`, containing the Euclidean norms of 3D uniform vectors.
    """
    # sample independent 3D components
    comps = rng.uniform(low, high, size=(*size, 3))
    # compute Euclidean norm along the last axis
    radii = np.linalg.norm(comps, axis=-1)/np.sqrt(3)
    return radii


def normal_radius_sampler_vectorized(rng: np.random.Generator, size, scale=1e-2):
    """
    Returns radii of 3D vectors whose components are drawn from a normal distribution.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator.
    size : tuple of ints
        Output shape (e.g. (n_steps, n_paths)).
    low, high : float
        Bounds for uniform distribution of each vector component.

    Returns
    -------
    radii : np.ndarray
        Array of shape `size`, containing the Euclidean norms of 3D uniform vectors.
    """
    # sample independent 3D components
    comps = rng.normal(loc=0, scale=scale, size=(*size, 3))
    # compute Euclidean norm along the last axis
    radii = np.linalg.norm(comps, axis=-1)/np.sqrt(3)
    return radii


def beta_symmetric_reset_sampler_vectorized(rng: np.random.Generator, size, a=2.0, b=2.0):
    """
    Return positive magnitudes u ~ Beta(a,b) in (0,1); kernel applies sign*reset_samples.
    """
    return rng.beta(a, b, size=size)


def plot_trajectory_with_reinsertions(ax: Optional[plt.Axes],
                                      times: np.ndarray,
                                      states: np.ndarray,
                                      jump_times: np.ndarray,
                                      title: str = r"Trajectory with reinsertion on $|x|\ge 1$",
                                      show_ylabel: bool = True,
                                      show_xlabel: bool = True) -> plt.Axes:
    """
    Plot a single trajectory (times, states) with reinsertion rug markers
    and display the total number of reinsertions.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(times, states, lw=0.8, label='trajectory')
    ax.axhline(1.0, ls="--", alpha=0.5)
    ax.axhline(-1.0, ls="--", alpha=0.5)
    ax.axhline(2.0, ls="--", alpha=0.5, color="grey")
    ax.axhline(-2.0, ls="--", alpha=0.5, color="grey")
    # Draw rug markers for reinsertion times
    n_reinserts = int(jump_times.size)
    if n_reinserts > 0:
        rug_y = np.full_like(jump_times, -1.08)
        ax.plot(jump_times, rug_y, linestyle='None', marker='|', markersize=8,
                label='reinsertions', zorder=5)

    # Annotate with total reinsertion count
    ax.text(
        0.02, 0.5,
        rf"\text{{reinsertions: {n_reinserts}}}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',

        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    # Axis labels and title
    if show_xlabel:
        ax.set_xlabel(r"$t$")
    if show_ylabel:
        ax.set_ylabel(r"$x(t)$")
    ax.set_title(title)

    ax.legend(loc='right', fontsize='small')
    ax.set_ylim(-2.15, 2.15)

    return ax


def plot_multiple_sigmas(times: np.ndarray,
                         states: np.ndarray,
                         sigmas: Sequence[float],
                         reset_times_per_path: Sequence[np.ndarray],
                         n_cols: int = 1,
                         figsize: tuple = (8, 3),
                         sharex: bool = True,
                         in3d: bool = False,
                         show_reinserts: bool = True):
    """
    Create subplots showing trajectories for multiple sigma values.

    Parameters
    ----------
    times : np.ndarray
        Common time vector returned by simulate_over_sigmas.
    results_dict : dict
        Dictionary mapping sigma -> {'states', 'jump_times', 'jump_states'}.
    sigmas : Sequence[float]
        Array/list of sigma values to plot (must be keys of results_dict).
    n_cols : int, optional
        Number of subplot columns. Rows are computed automatically.
    figsize : tuple, optional
        Base figure size per row.
    sharex : bool, optional
        Whether to share x-axis among subplots.
    """
    n_sigmas = len(sigmas)
    n_rows = int(np.ceil(n_sigmas / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize[0] * n_cols,
                                      figsize[1] * n_rows),
                             sharex=sharex)
    axes = np.atleast_1d(axes).flatten()

    for i, sigma in enumerate(sigmas):
        ax = axes[i]
        data = states[:, i]
        if in3d:
            plot_trajectory_with_reinsertions_3d(
                ax=ax,
                times=times,
                states=data,
                jump_times=reset_times_per_path[i],
                title=f"$\sigma = {sigma:.3f}$",
                show_xlabel=(i >= (n_rows - 1) * n_cols),
                show_ylabel=(i % n_cols == 0),
                show_reinserts=show_reinserts
            )
        else:
            plot_trajectory_with_reinsertions(
                ax=ax,
                times=times,
                states=data,
                jump_times=reset_times_per_path[i],
                title=rf"$\sigma = {sigma:.3f}$",
                show_xlabel=(i >= (n_rows - 1) * n_cols),
                show_ylabel=(i % n_cols == 0)
            )

    # Hide any unused axes (if n_sigmas < n_rows * n_cols)
    for j in range(n_sigmas, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_multiple_trajectories(ax: Optional[plt.Axes],
                               times: np.ndarray,
                               states: np.ndarray,
                               labels: List[str],
                               title: str = None,
                               text_loc: str = "center",
                               show_ylabel: bool = True,
                               show_xlabel: bool = True) -> plt.Axes:
    """
    Plot multiple trajectories given by `states` (shape: n_lines x n_times) 
    with their corresponding `labels`. Horizontal lines are annotated.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    # Plot each trajectory
    linestyles = ['-', '-.', ':']
    for i in range(states.shape[1]):
        ax.plot(times, states[:, i], lw=1.0, label=labels[i],
                ls=linestyles[i % len(linestyles)])

    ax.axhline(1.0, ls="--", alpha=0.5, color="black")
    ax.axhline(-1.0, ls="--", alpha=0.5, color="black")
    ax.axhline(2.0, ls="--", alpha=0.5, color="grey")
    ax.axhline(-2.0, ls="--", alpha=0.5, color="grey")

    if text_loc == "center":
        ax.text(times[int(.5 * len(times))], -1.0,
                f"Regime Boundary", va="bottom", ha="center")
        ax.text(times[int(.5 * len(times))], -2.0, f"Reinsertion",
                va="bottom", ha="center", color="grey")
    elif text_loc == "right":
        ax.text(times[-1], -1.0,
                f"Regime Boundary", va="bottom", ha="right")
        ax.text(times[-1], -2.0, f"Reinsertion",
                va="bottom", ha="right", color="grey")
    elif text_loc == "left":
        ax.text(times[0], -1.0,
                f"Regime Boundary", va="bottom", ha="left")
        ax.text(times[0], -2.0, f"Reinsertion",
                va="bottom", ha="left", color="grey")
    else:
        raise ValueError("text_loc must be 'center', 'right', or 'left'")

    if show_xlabel:
        ax.set_xlabel("$t$")
    if show_ylabel:
        ax.set_ylabel("$x(t)$")
    ax.set_title(title)

    ax.legend(title="$\sigma$", framealpha=1, loc="lower left")
    ax.set_ylim(-2.2, 2.2)

    return fig, ax


def estimate_pdf_from_sampler(sampler_func,
                              rng: np.random.Generator,
                              n_samples: int = 100_000,
                              n_bins: int = 200,
                              sampler_kwargs: dict = None):
    """
    Estimate the PDF of radii returned by an arbitrary sampler function.
    Works with samplers that expect `size` as a tuple.

    Parameters
    ----------
    sampler_func : callable
        Function of the form `sampler_func(rng, size, **kwargs)` returning an array of radii.
    rng : np.random.Generator
        NumPy random generator.
    n_samples : int
        Total number of samples to draw.
    n_bins : int
        Number of bins for histogram / PDF estimation.
    sampler_kwargs : dict
        Additional keyword arguments passed to sampler_func.

    Returns
    -------
    bin_centers : np.ndarray
        Centers of the histogram bins.
    pdf_norm : np.ndarray
        Normalized PDF values (max = 1).
    """
    if sampler_kwargs is None:
        sampler_kwargs = {}

    # Ensure size is a tuple for the sampler function
    size_tuple = (n_samples,)

    # Draw samples
    radii = sampler_func(rng, size=size_tuple, **sampler_kwargs).ravel()

    # Histogram as PDF
    pdf_counts, bin_edges = np.histogram(radii, bins=n_bins, density=True)

    # Normalize so max = 1
    pdf_norm = pdf_counts / pdf_counts.max() if pdf_counts.max() > 0 else pdf_counts

    # Bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, pdf_norm


def plot_multiple_trajectories_with_pdf_strip(ax: Optional[plt.Axes],
                                              times: np.ndarray,
                                              states: np.ndarray,
                                              labels: List[str],
                                              pdf_bins: np.ndarray,
                                              pdf_values: np.ndarray,
                                              title: str = None):
    """
    Plot multiple trajectories with a background strip colored according to a given PDF.
    The y-axis corresponds to the support of the PDF (pdf_bins).

    Parameters
    ----------
    ax : Optional[plt.Axes]
        Matplotlib axes to plot on.
    times : np.ndarray
        1D array of time points.
    states : np.ndarray
        2D array of shape (n_times, n_lines) containing trajectories.
    labels : List[str]
        List of labels for each line.
    pdf_bins : np.ndarray
        Bin centers corresponding to the PDF support (y-axis).
    pdf_values : np.ndarray
        Normalized PDF values (max=1) to color the strip.
    title : str
        Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))

    n_times, n_lines = states.shape

    # 1) Create background strip from PDF
    img_width = 100  # horizontal repetition
    img = np.tile(pdf_values[:, np.newaxis], (1, img_width))

    # colormap: white -> grey50
    cmap = LinearSegmentedColormap.from_list(
        "white_to_g50", [(1, 1, 1), (0.5, 0.5, 0.5)])

    # vertical extent matches pdf_bins
    y_min, y_max = pdf_bins[0], pdf_bins[-1]

    ax.imshow(
        img,
        extent=(times[0], times[-1], y_min, y_max),
        origin='lower',
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        zorder=0
    )

    # 2) Plot trajectories with different linestyles
    linestyles = ['-', '--', '-.', ':']
    for i in range(n_lines):
        style = linestyles[i % len(linestyles)]
        ax.plot(times, states[:, i], lw=1.0, linestyle=style,
                label=f"{labels[i]} ({style})", zorder=3)

    # 3) Horizontal reference lines (optional)
    ax.axhline(1.0, ls="--", alpha=0.5, color="black", zorder=1)
    ax.axhline(-1.0, ls="--", alpha=0.5, color="black", zorder=1)
    ax.axhline(2.0, ls="--", alpha=0.5, color="grey", zorder=1)
    ax.axhline(-2.0, ls="--", alpha=0.5, color="grey", zorder=1)
    ax.text(times[-1], -1.0, "Regime Boundary",
            va="bottom", ha="right", zorder=4)
    ax.text(times[-1], -2.0, "Reinsertion", va="bottom",
            ha="right", color="grey", zorder=4)

    # 4) Labels and legend
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    if title is not None:
        ax.set_title(title)
    ax.legend(title="$\\sigma$", framealpha=1,
              fontsize="small", loc="upper right")

    plt.tight_layout()
    return ax


def compute_lifetimes_per_sigma_old(
    states: np.ndarray,
    sigma_arr: np.ndarray,
    reset_threshold: float = 2.0,
    hit_threshold: float = 1.0,
) -> Dict[float, List[int]]:
    """
    Vectorized computation of lifetimes (number of saved timesteps after reinsertion until first hit |x| >= hit_threshold),
    grouped by sigma. Censored events (no hit before end) are skipped.

    Parameters
    ----------
    states : np.ndarray, shape (n_steps, n_paths)
    sigma_arr : np.ndarray, shape (n_paths,)
    reset_threshold : float
        Threshold to detect reset (|x| > reset_threshold).
    hit_threshold : float
        Threshold that constitutes a hit (|x| >= hit_threshold).
    Returns
    -------
    dict
        {sigma_value: [lifetimes_as_ints, ...]}
    """
    n_paths = states.shape[1]
    n_steps = states.shape[0]

    abs_states = np.abs(states)
    lifetimes_per_sigma = defaultdict(list)

    for j in range(n_paths):
        col = abs_states[:, j]

        # indices where reset occurs (|x| > reset_threshold)
        reset_idxs = np.nonzero(col > reset_threshold)[0]
        if reset_idxs.size == 0:
            continue

        # indices where hit occurs (|x| >= hit_threshold)
        hit_idxs = np.nonzero(col >= hit_threshold)[0]
        if hit_idxs.size == 0:
            # no hits at all -> all resets are censored, skip
            continue

        # reinsertion is immediately after reset index
        reinsertion_idxs = reset_idxs + 1

        # any reinsertion beyond saved data are censored
        valid_mask = reinsertion_idxs < n_steps
        if not np.any(valid_mask):
            continue
        reinsertion_idxs = reinsertion_idxs[valid_mask]
        reset_idxs = reset_idxs[valid_mask]

        # For each reinsertion index find the first hit index >= reinsertion using searchsorted
        # searchsorted returns insertion positions in hit_idxs; if pos == len(hit_idxs) => no hit after reinsertion
        pos = np.searchsorted(hit_idxs, reinsertion_idxs, side="left")

        # filter out censored (pos == len(hit_idxs))
        valid_pos_mask = pos < hit_idxs.size
        if not np.any(valid_pos_mask):
            continue

        # corresponding hit indices for valid reinsertion events
        hit_for_reset = hit_idxs[pos[valid_pos_mask]]

        # lifetimes = hit_index - reinsertion_index (number of saved timesteps after reinsertion until hit)
        lifetimes = (hit_for_reset -
                     reinsertion_idxs[valid_pos_mask]).astype(int)

        sigma = float(sigma_arr[j])
        lifetimes_per_sigma[sigma].extend(lifetimes.tolist())

    return dict(lifetimes_per_sigma)


def compute_lifetimes_per_sigma(
    states: np.ndarray,
    sigma_arr: np.ndarray,
    reset_threshold: float = 1.0,
    hit_threshold: float = 1.0,
) -> Dict[float, List[int]]:
    """
    Vectorized computation of lifetimes (number of saved timesteps after reinsertion until first hit |x| >= hit_threshold),
    grouped by sigma. Censored events (no hit before end) are skipped.

    NOTE: reinsertion (reset) detection now uses the drop in absolute |x| between consecutive saved timesteps:
        reinsertion occurs where abs(states[t]) - abs(states[t+1]) > reset_threshold
    This is useful when you've subsampled and the reset shows up as a large downward jump in |x|.

    Parameters
    ----------
    states : np.ndarray, shape (n_steps, n_paths)
    sigma_arr : np.ndarray, shape (n_paths,)
    reset_threshold : float
        Threshold on drop in |x| between consecutive saved timesteps to detect reinsertion (abs(t) - abs(t+1) > reset_threshold).
    hit_threshold : float
        Threshold that constitutes a hit (|x| >= hit_threshold).
    Returns
    -------
    dict
        {sigma_value: [lifetimes_as_ints, ...]}
    """
    n_paths = states.shape[1]
    n_steps = states.shape[0]

    abs_states = np.abs(states)
    lifetimes_per_sigma = defaultdict(list)

    for j in range(n_paths):
        col = abs_states[:, j]

        # new criterion: look for large drops between consecutive saved timesteps:
        # drop at time t is abs_states[t] - abs_states[t+1]; we detect indices t where drop > reset_threshold
        drops = col[:-1] - col[1:]
        # these are t indices; reinsertion is at t+1
        reset_idxs = np.nonzero(drops > reset_threshold)[0]
        if reset_idxs.size == 0:
            continue

        # indices where hit occurs (|x| >= hit_threshold)
        hit_idxs = np.nonzero(col >= hit_threshold)[0]
        if hit_idxs.size == 0:
            # no hits at all -> all resets are censored, skip
            continue

        # reinsertion is immediately after detected drop (t -> reinsertion at t+1)
        reinsertion_idxs = reset_idxs + 1

        # any reinsertion beyond saved data are censored (shouldn't happen because reinsertion_idxs <= n_steps-1)
        valid_mask = reinsertion_idxs < n_steps
        if not np.any(valid_mask):
            continue
        reinsertion_idxs = reinsertion_idxs[valid_mask]
        reset_idxs = reset_idxs[valid_mask]

        # For each reinsertion index find the first hit index >= reinsertion using searchsorted
        pos = np.searchsorted(hit_idxs, reinsertion_idxs, side="left")

        # filter out censored (pos == len(hit_idxs))
        valid_pos_mask = pos < hit_idxs.size
        if not np.any(valid_pos_mask):
            continue

        # corresponding hit indices for valid reinsertion events
        hit_for_reset = hit_idxs[pos[valid_pos_mask]]

        # lifetimes = hit_index - reinsertion_index (number of saved timesteps after reinsertion until hit)
        lifetimes = (hit_for_reset -
                     reinsertion_idxs[valid_pos_mask]).astype(int)

        sigma = float(sigma_arr[j])
        lifetimes_per_sigma[sigma].extend(lifetimes.tolist())

    return dict(lifetimes_per_sigma)


def constant_reset_sampler_factory(value, dtype=np.float32):
    def reset_sampler(rng, size):
        return np.full(size, value, dtype=dtype)
    return reset_sampler


def mean_escape_times_via_reinsertion_points(
        sigmas: np.ndarray,
        reinsertion_points: np.ndarray,
        dt: float,
        n_steps: int,
        rng: np.random.Generator = None,
        save_every: int = 1,
        dtype=np.float32):
    """
    Uses compute_lifetimes_per_sigma to get lifetimes (in saved timesteps) per sigma,
    converts to seconds via dt * save_every, and returns mean and counts arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigmas = np.asarray(sigmas, dtype=float)
    reinsertion_points = np.asarray(reinsertion_points, dtype=float)

    n_sig = sigmas.size
    n_r = reinsertion_points.size

    mean_matrix = np.full((n_sig, n_r), np.nan, dtype=np.float64)
    counts = np.zeros((n_sig, n_r), dtype=np.int64)

    time_per_saved = float(dt * save_every)

    for j, r in enumerate(reinsertion_points):
        reset_sampler = constant_reset_sampler_factory(r, dtype=dtype)

        times, states, sigma_arr = simulate_trajectories_per_sigma(
            sigmas=sigmas.astype(np.float32),
            dt=dt,
            n_steps=n_steps,
            reset_sampler=reset_sampler,
            rng=rng,
            save_every=save_every,
            x0=float(r),
            dtype=dtype
        )

        # Compute lifetimes per sigma using the helper you already have.
        # lifetimes_dict: { sigma_value (float) : [lifetimes_in_saved_timesteps, ...] }
        lifetimes_dict = compute_lifetimes_per_sigma_old(
            states=states,
            sigma_arr=sigma_arr,
            reset_threshold=2.0,
            hit_threshold=1.0
        )
        # fill mean_matrix and counts in the same order as `sigmas`
        for i, sigma in enumerate(sigmas):
            key = float(sigma)
            lifesteps = lifetimes_dict.get(key, [])
            cnt = len(lifesteps)
            counts[i, j] = cnt
            if cnt == 0:
                mean_matrix[i, j] = np.nan
            else:
                durations_sec = np.asarray(
                    lifesteps, dtype=float) * time_per_saved
                mean_matrix[i, j] = float(np.mean(durations_sec))

    return mean_matrix, counts


def plot_mean_escape_by_reinsertion(sigmas, reinsertion_points, mean_matrix,
                                    logx=True, xlabel="Sigma"):
    """
    Plot mean escape time vs sigma for several reinsertion points.
    mean_matrix shape: (n_sigmas, n_reinsertion_points)
    """
    sigmas = np.asarray(sigmas)
    # ensure sigmas sorted and reorder mean_matrix accordingly
    order = np.argsort(sigmas)
    sigmas_sorted = sigmas[order]
    means_sorted = np.asarray(mean_matrix)[order, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, r in enumerate(reinsertion_points):
        y = means_sorted[:, j]
        mask = np.isfinite(y)
        ax.plot(sigmas_sorted[mask], y[mask], marker='o', linewidth=1.8,
                label=f"reinsert = {r}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean escape time")
    ax.set_title("Mean escape time vs sigma (by reinsertion point)")
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Reinsertion", loc="best")
    plt.tight_layout()
    return fig, ax


# asymptotic expansion of erfi(u) for large u:
# erfi(u) ~ exp(u^2)/(sqrt(pi)*u) * (1 + 1/(2u^2) + 3/(4u^4) + ...)
# so exp(-u^2)*erfi(u) ~ 1/(sqrt(pi)*u) * (1 + 1/(2u^2) + ...)


def tail_integrand_asym(u, terms=3):
    # returns approximation to exp(-u^2)*erfi(u)
    # using expansion up to O(u^{-2*terms})
    # safe for |u| > ~5
    s = 1.0 / (np.sqrt(np.pi) * u)
    uu2 = u*u
    # include a few correction terms
    if terms >= 1:
        s *= (1.0 + 1.0/(2.0*uu2))
    if terms >= 2:
        # not strict series-multiplicative, but good numeric
        s *= (1.0 + 3.0/(4.0*uu2))
    return s


def f_scientific(x, sigma, U=6.0):
    """Stable evaluation using domain split at +/-U (in u-space)."""
    if sigma == 0:
        return -np.log(np.abs(x)) if x != 0 else np.nan

    u1 = -1.0 / np.sqrt(2.0 * sigma)
    u2 = x / np.sqrt(2.0 * sigma)

    # integrand: -sqrt(pi) * exp(-u^2) * erfi(u)
    def integrand(u):
        return -np.sqrt(np.pi) * np.exp(-u*u) * special.erfi(u)

    # choose split points within [u1,u2] away from huge values
    a = min(u1, u2)
    b = max(u1, u2)

    # if interval is small and within safe zone, integrate directly
    SAFE_BOUND = U
    if b <= SAFE_BOUND and a >= -SAFE_BOUND:
        res, err = integrate.quad(
            integrand, a, b, epsabs=1e-10, epsrel=1e-10, limit=200)
        return res

    # otherwise split into: [a, -U], [-U, U], [U, b], skipping empty pieces
    total = 0.0

    # left tail [a, -U]
    left_end = -SAFE_BOUND
    if a < left_end:
        # approximate integral of integrand on [a, left_end] using asymptotic
        # integrand ~ -sqrt(pi) * (1/(sqrt(pi)*u)) = -1/u  (first term)
        # so integral approx = -log(|u|) between bounds plus small corrections.
        # We'll integrate the asymptotic approximation numerically for safety.
        def asymp(u):
            return -np.sqrt(np.pi) * tail_integrand_asym(u)
        res, err = integrate.quad(
            asymp, a, left_end, epsabs=1e-10, epsrel=1e-10, limit=200)
        total += res

    # center [-U, U]
    mid_a = max(a, -SAFE_BOUND)
    mid_b = min(b, SAFE_BOUND)
    if mid_b > mid_a:
        res, err = integrate.quad(
            integrand, mid_a, mid_b, epsabs=1e-10, epsrel=1e-10, limit=400)
        total += res

    # right tail [U, b]
    right_start = SAFE_BOUND
    if b > right_start:
        def asymp(u):
            return -np.sqrt(np.pi) * tail_integrand_asym(u)
        res, err = integrate.quad(
            asymp, right_start, b, epsabs=1e-10, epsrel=1e-10, limit=200)
        total += res

    return total


def diffusion_maps_matrix_1d(X, epsilon):
    """
    1D-specialized version of diffusion_maps_matrix.
    X: ndarray of shape (n_samples,) or (n_samples, 1)
    epsilon: float > 0
    Returns: (DMM, A) as scipy.sparse.csr_matrices (row-stochastic DMM, affinity A)
    """
    # flatten / validate 1D input
    X_arr = np.asarray(X)
    if X_arr.ndim == 2 and X_arr.shape[1] == 1:
        X_flat = X_arr.ravel()
    elif X_arr.ndim == 1:
        X_flat = X_arr
    else:
        raise ValueError(
            "diffusion_maps_matrix_1d expects 1D input (shape (n,) or (n,1)).")

    n_samples = X_flat.shape[0]
    if n_samples == 0:
        # empty inputs
        A = sparse.csr_matrix((n_samples, n_samples))
        return A, A

    # radius from epsilon (same choice as your original)
    r = np.sqrt(epsilon)*3

    # sort points and use searchsorted to get neighbor windows efficiently
    order = np.argsort(X_flat)
    Xs = X_flat[order]   # sorted coordinates

    rows = []
    cols = []
    vals = []

    # For each sorted index, find neighbors in [x - r, x + r]
    for i_sorted, x in enumerate(Xs):
        left = np.searchsorted(Xs, x - r, side='left')
        right = np.searchsorted(Xs, x + r, side='right')
        if right <= left:
            # no neighbors found (shouldn't happen since self is within window), but guard
            continue
        idxs_sorted = np.arange(left, right)
        orig_rows = np.full(idxs_sorted.shape, order[i_sorted], dtype=np.int32)
        orig_cols = order[idxs_sorted].astype(np.int32)
        dists = np.abs(x - Xs[idxs_sorted]).astype(np.float64)

        rows.append(orig_rows)
        cols.append(orig_cols)
        vals.append(dists)

    # Concatenate lists
    rows = np.concatenate(rows).astype(np.int32)
    cols = np.concatenate(cols).astype(np.int32)
    vals = np.concatenate(vals).astype(np.float64)

    # Gaussian kernel: same convention as original: exp(-d^2 / epsilon)
    A = sparse.coo_matrix((np.exp(-(vals**2) / epsilon),
                          (rows, cols)), shape=(n_samples, n_samples)).tocsr()

    # Ensure self-loop weight (set diagonal to 1.0)
    A.setdiag(1.0)

    # density normalization (Coifman–Lafon style with alpha = 1.0)
    row_means = np.asarray(A.mean(axis=1)).ravel()
    row_means[row_means == 0] = np.finfo(float).eps
    q = 1.0 / row_means

    alpha = 1.0
    kalpha = q ** alpha
    D_k = sparse.diags(kalpha, offsets=0, format='csr')
    Adensnorm = D_k.dot(A).dot(D_k)

    # row-normalize to get Markov matrix DMM
    row_sums = np.asarray(Adensnorm.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = np.finfo(float).eps
    inv_row_sums = 1.0 / row_sums
    D_norm = sparse.diags(inv_row_sums, offsets=0, format='csr')
    DMM = D_norm.dot(Adensnorm)

    return DMM, A


def make_L(X, epsilon):
    """
    1D-specialized version of diffusion_maps_matrix.
    X: ndarray of shape (n_samples,) or (n_samples, 1)
    epsilon: float > 0
    Returns: (DMM, A) as scipy.sparse.csr_matrices (row-stochastic DMM, affinity A)
    """
    # flatten / validate 1D input
    X_arr = np.asarray(X)
    if X_arr.ndim == 2 and X_arr.shape[1] == 1:
        X_flat_full = X_arr.ravel()
    elif X_arr.ndim == 1:
        X_flat_full = X_arr
    else:
        raise ValueError(
            "diffusion_maps_matrix_1d expects 1D input (shape (n,) or (n,1)).")

    last_p = X_flat_full[-1]
    X_flat = X_flat_full[:-1]

    n_samples = X_flat.shape[0]

    if epsilon > 0:
        # radius from epsilon (same choice as your original)
        r = np.sqrt(epsilon)*3

        # sort points and use searchsorted to get neighbor windows efficiently
        order = np.argsort(X_flat)
        Xs = X_flat[order]   # sorted coordinates

        rows = []
        cols = []
        vals = []

        # For each sorted index, find neighbors in [x - r, x + r]
        for i_sorted, x in enumerate(Xs):
            left = np.searchsorted(Xs, x - r, side='left')
            right = np.searchsorted(Xs, x + r, side='right')

            idxs_sorted = np.arange(left, right)
            orig_rows = np.full(idxs_sorted.shape,
                                order[i_sorted], dtype=np.int32)
            orig_cols = order[idxs_sorted].astype(np.int32)
            dists = np.abs(x - Xs[idxs_sorted]).astype(np.float64)

            rows.append(orig_rows)
            cols.append(orig_cols)
            vals.append(dists)

        # Concatenate lists
        rows = np.concatenate(rows).astype(np.int32)
        cols = np.concatenate(cols).astype(np.int32)
        vals = np.concatenate(vals).astype(np.float64)

        # Gaussian kernel: same convention as original: exp(-d^2 / epsilon)
        A = sparse.coo_matrix((np.exp(-(vals**2) / epsilon),
                               (rows, cols)), shape=(n_samples, n_samples)).tocsr()
        A.setdiag(1.0)
        # density normalization (Coifman–Lafon style with alpha = 1.0)
        row_means = np.asarray(A.mean(axis=1)).ravel()
        row_means[row_means == 0] = np.finfo(float).eps
        q = 1.0 / row_means

        alpha = 1.0
        kalpha = q ** alpha
        D_k = sparse.diags(kalpha, offsets=0, format='csr')
        Adensnorm = D_k.dot(A).dot(D_k)

        # row-normalize to get Markov matrix DMM
        row_sums = np.asarray(Adensnorm.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = np.finfo(float).eps
        inv_row_sums = 1.0 / row_sums
        D_norm = sparse.diags(inv_row_sums, offsets=0, format='csr')
        DMM = D_norm.dot(Adensnorm)
        # construct L without last point
        L = sparse.vstack(
            [DMM[1:], sparse.csr_matrix((1, DMM.shape[1]))], format='csr')
        # now the last point
        order = np.argsort(X_flat_full)
        Xs = X_flat_full[order]   # sorted coordinates
        left = np.searchsorted(Xs, last_p - r, side='left')
        right = np.searchsorted(Xs, last_p + r, side='right')
        idxs_sorted = np.arange(left, right)
        orig_rows = np.full(
            idxs_sorted.shape[0] - 1, L.shape[0]-1, dtype=np.int32)
        orig_cols = order[idxs_sorted].astype(np.int32)

        dists = np.abs(last_p - Xs[idxs_sorted]).astype(np.float64)
        dists = dists[orig_cols != L.shape[0]]
        orig_cols = orig_cols[orig_cols != L.shape[0]]
        values = np.exp(-(dists**2)/epsilon)

        updates = csr_matrix(
            (values/values.sum(), (orig_rows, orig_cols)),
            shape=L.shape
        )

        L = L + updates
    else:
        # epsilon == 0 case: use identity matrix
        DMM = sparse.eye(n_samples, format='csr')

        # construct L without last point
        L = sparse.vstack(
            [DMM[1:], sparse.csr_matrix((1, DMM.shape[1]))], format='csr')

    return L


def solve_escape_from_Q(Q):
    """
    Solve (I - Q) x = 1 for x (expected escape times).
    Input:
      Q : scipy.sparse matrix or numpy.ndarray (square, shape (n,n))
    Output:
      x : 1D numpy array, solution to (I - Q) x = 1
    Behavior:
      - If Q is dense ndarray -> uses dense LAPACK solve.
      - If Q is sparse -> uses an internal heuristic then:
          1) try sparse direct LU (splu)
          2) if that fails, try iterative GMRES with ILU preconditioner
          3) if that fails, fall back to dense solve (convert to ndarray)
      The function prints short diagnostic messages on which path was taken.
      No user tuning required.
    """

    # Sanity checks
    if not (hasattr(Q, "shape") and Q.shape[0] == Q.shape[1]):
        raise ValueError("Q must be square")

    n = Q.shape[0]
    ones = np.ones(n, dtype=float)

    # If Q is a dense ndarray -> dense solve immediately
    if isinstance(Q, np.ndarray):
        A = np.eye(n, dtype=float) - Q
        # solve with LAPACK
        x = linalg.solve(A, ones, assume_a='gen')
        print(f"[dense] solved dense system (n={n}) via LAPACK.")
        return x

    # Otherwise treat as sparse
    # ensure CSR for nnz/density queries, and CSC for LU
    Q = sparse.csr_matrix(Q)
    nnz = Q.nnz
    density = float(nnz) / (n*n)

    # estimate memory for dense array (bytes)
    estimated_dense_bytes = n * n * 8
    # Heuristic decision: if matrix is quite dense or dense memory is modest, prefer dense path.
    # These are internal heuristics only — you don't need to change them.
    prefer_dense = (density > 0.05) or (
        estimated_dense_bytes <= 200 * 1024 * 1024)

    A_sparse = sparse.eye(n, format='csr') - Q

    # 1) Try sparse direct LU if we don't strongly prefer dense
    if not prefer_dense:
        try:
            A_csc = A_sparse.tocsc()
            # may raise MemoryError or take very long if matrix effectively dense
            lu = splu(A_csc)
            x = lu.solve(ones)
            print(
                f"[sparse-direct] used splu (n={n}, nnz={nnz}, density={density:.4g}).")
            return x
        except Exception as e:
            # fallback to iterative or dense
            print(
                f"[sparse-direct] splu failed or not feasible: {type(e).__name__}: {e}")

    # 2) Try iterative solver with ILU preconditioner (good for large sparse)
    #    Use defaults; if it fails we move to dense fallback.
    try:
        A_csc = A_sparse.tocsc()
        # build ILU preconditioner; default parameters used (no user tuning)
        # may raise MemoryError on too-dense matrices
        ilu = spilu(A_csc)
        M = LinearOperator(A_csc.shape, ilu.solve)
        x, info = gmres(A_csc, ones, M=M, rtol=1e-8)
        if info == 0:
            print(
                f"[iterative] GMRES converged with ILU preconditioner (n={n}, nnz={nnz}, density={density:.4g}).")
            return x
        else:
            print(
                f"[iterative] GMRES did not converge (info={info}); falling back to dense. (n={n})")
    except Exception as e:
        print(f"[iterative] iterative path failed: {type(e).__name__}: {e}")

    # 3) Dense fallback: convert to array and use LAPACK
    try:
        A_dense = (sparse.eye(n, format='csr') - Q).toarray()
        x = linalg.solve(A_dense, ones, assume_a='gen')
        print(
            f"[dense-fallback] converted to dense and solved (n={n}, nnz={nnz}, density={density:.4g}).")
        return x
    except Exception as e:
        # last resort: raise informative error
        raise RuntimeError("All solve attempts failed. Last error: " + repr(e))


def expected_escape_times_from_TO_fast(L, mask,
                                       direct_cutoff=4000,
                                       ilu_drop_tol=1e-3,
                                       krylov_tol=1e-8,
                                       krylov_maxiter_factor=10,
                                       richardson_tol=1e-8,
                                       richardson_maxiter=5000):
    """
    Solve (I - Q) t = 1 where Q = L[indices, :][:, indices] and 'mask' selects the blocking set.

    Strategy:
      - If m <= direct_cutoff: use direct sparse solve (spsolve).
      - Else: try ILU preconditioner + bicgstab.
      - If ILU or bicgstab fails, fallback to simple Richardson iteration:
            t_{k+1} = 1 + Q t_k
        (equivalent to Neumann series) until convergence.

    Parameters:
      L : scipy.sparse matrix (transfer operator)
      mask : boolean array selecting the blocking indices (source nodes)
      direct_cutoff : int, size threshold below which a direct solve is used
      ilu_drop_tol : float, drop tolerance for spilu preconditioner
      krylov_tol : float, tolerance for bicgstab
      krylov_maxiter_factor : int, max iterations = factor * m
      richardson_tol : float, tolerance for Richardson residual
      richardson_maxiter : int, max Richardson iterations

    Returns:
      t_escape : 1D numpy array of length m (floats)
    """
    indices = np.nonzero(mask)[0]
    m = indices.size
    if m == 0:
        return np.array([], dtype=float)

    # Build Q in csr (note: L may be row- or column-stochastic depending on pipeline;
    # this matches your original usage of L[1:,:-1] indexing.)
    Q = L[indices, :][:, indices].tocsr()

    # build I - Q
    IminusQ = eye(m, format='csr') - Q

    # RHS vector
    b = np.ones(m, dtype=float)

    # 1) small problems: direct sparse solver
    if m <= direct_cutoff:
        t = spsolve(IminusQ, b)
        return t

    # 2) try ILU preconditioner + bicgstab
    # spilu requires csc matrix
    try:
        # convert to csc for spilu and compute ILU factorization
        ilu = spilu(IminusQ.tocsc(), drop_tol=ilu_drop_tol)
        # create preconditioner as LinearOperator
        M = LinearOperator((m, m), matvec=ilu.solve)

        maxiter = int(krylov_maxiter_factor * m)
        t, info = bicgstab(IminusQ, b, rtol=krylov_tol, maxiter=maxiter, M=M)
        if info == 0:
            return t
        else:
            # info > 0 means no convergence in maxiter; info < 0 means breakdown
            # proceed to fallback
            print(
                f"bicgstab did not converge (info={info}), falling back to Richardson.")
    except Exception as e:
        # ILU failed (memory/zero pivot etc) -> fallback
        print("ILU preconditioning failed or unavailable, falling back to Richardson. Exception:", e)

    # 3) fallback: Richardson / Neumann fixed-point iteration
    # iterate t_{k+1} = 1 + Q t_k, starting from t0 = 0
    # stop when ||residual||/||b|| < tol, where residual = (I-Q)t - 1
    t = np.zeros(m, dtype=float)
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        bnorm = 1.0
    for it in range(richardson_maxiter):
        t_new = b + Q.dot(t)
        # compute residual r = (I-Q) t_new - b = -Q(t_new - t)  -> easier compute r = t_new - Q.dot(t_new) - b
        r = t_new - Q.dot(t_new) - b
        res_norm = np.linalg.norm(r)
        if res_norm / bnorm <= richardson_tol:
            return t_new
        t = t_new
    # if still not converged, return last iterate with a warning
    print(
        f"Richardson did not converge in {richardson_maxiter} iterations; residual={res_norm:.3e}")
    return t


# --- Numba kernel (parallel over trajectories) for 3D states ---


@njit(parallel=True, cache=True)
def _simulate_numba_parallel_sigma_arr_3d(dt, sigma_arr, x0,
                                          normals, reset_samples, save_every):
    """
    Numba-parallel kernel for 3D trajectories.
    normals, reset_samples: shape (n_steps, n_paths, 3)
    sigma_arr: shape (n_paths,)
    Returns saved states shape (n_saves, n_paths, 3) dtype=float32
    """
    n_steps_local, n_paths, _ = normals.shape
    max_saves = (n_steps_local // save_every) + 1
    saved = np.empty((max_saves, n_paths, 3), dtype=np.float32)

    phi = math.exp(dt)
    var = (math.exp(2.0 * dt) - 1.0) / 2.0
    sqrt_var = math.sqrt(var)

    for p in prange(n_paths):
        # initialize state x as a local 3-vector
        x0_local0 = x0[0]
        x0_local1 = x0[1]
        x0_local2 = x0[2]

        save_idx = 0
        saved[save_idx, p, 0] = x0_local0
        saved[save_idx, p, 1] = x0_local1
        saved[save_idx, p, 2] = x0_local2
        save_idx += 1

        s = float(sigma_arr[p])
        for i in range(n_steps_local):
            # compute squared norm
            norm2 = x0_local0 * x0_local0 + x0_local1 * x0_local1 + x0_local2 * x0_local2

            # restart if outside ball of radius 2 (i.e., norm^2 >= 4)
            if norm2 >= 4.0:
                # reset_samples shape: (n_steps, n_paths, 3)
                x0_local0 = reset_samples[i, p, 0]
                x0_local1 = reset_samples[i, p, 1]
                x0_local2 = reset_samples[i, p, 2]
            else:
                # exact linear step for each component + additive noise
                # normals[i, p, k] ~ N(0,1)
                x0_local0 = x0_local0 * phi + s * sqrt_var * normals[i, p, 0]
                x0_local1 = x0_local1 * phi + s * sqrt_var * normals[i, p, 1]
                x0_local2 = x0_local2 * phi + s * sqrt_var * normals[i, p, 2]
            if ((i + 1) % save_every) == 0:
                saved[save_idx, p, 0] = x0_local0
                saved[save_idx, p, 1] = x0_local1
                saved[save_idx, p, 2] = x0_local2
                save_idx += 1
    return saved


def simulate_trajectories_per_sigma_3d(
        sigmas,
        dt: float,
        n_steps: int,
        reset_sampler: Callable[[np.random.Generator, tuple], np.ndarray],
        rng: Optional[np.random.Generator] = None,
        save_every: int = 1,
        x0: Optional[np.ndarray] = None,
        dtype=np.float32):
    """
    Simulate one 3D continuous trajectory per sigma value in `sigmas`.

    Args
      sigmas: scalar or 1-D array-like of sigma values (if scalar it will be converted to length-1 array)
      dt: time step
      n_steps: number of integration steps (trajectory length)
      reset_sampler: vectorized callable reset_sampler(rng, size) -> array shaped `size`
                     For 3D this should return shape (n_steps, n_paths, 3) containing **vectors** to reset to.
      rng: np.random.Generator (if None a new generator is created)
      save_every: save states every `save_every` steps (1 => save every step)
      x0: initial condition (array-like of length 3). If None defaults to [0.1, 0.0, 0.0]
      dtype: np.float32 recommended; set to np.float64 if you need double precision

    Returns:
      times: 1D float array of saved times (length n_saves)
      saved_states: 3D array shape (n_saves, n_paths, 3), column j corresponds to sigmas[j]
      sigma_arr: 1D array of sigma values (dtype)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_paths = sigmas.shape[0]
    n_saves = (n_steps // save_every) + 1

    # Pre-sample normals and reset samples (shape: n_steps x n_paths x 3)
    normals = rng.normal(loc=0.0, scale=1.0, size=(
        n_steps, n_paths, 3)).astype(dtype, copy=False)
    reset_samples = reset_sampler(rng, size=(
        n_steps, n_paths, 3)).astype(dtype, copy=False)
    saved_states = _simulate_numba_parallel_sigma_arr_3d(float(dt), sigmas.astype(dtype),
                                                         x0.astype(
        dtype),
        normals, reset_samples,
        int(save_every))
    times = np.arange(n_saves, dtype=float) * (save_every * dt)
    return times, saved_states, sigmas


def sample_on_sphere(rng, size, radius=1.0):
    """Sample points uniformly on the surface of a sphere with given radius."""
    shape = size[:-1]
    phi = rng.uniform(0, 2*np.pi, size=shape)
    cos_theta = rng.uniform(-1, 1, size=shape)
    sin_theta = np.sqrt(1 - cos_theta**2)
    x = radius * sin_theta * np.cos(phi)
    y = radius * sin_theta * np.sin(phi)
    z = radius * cos_theta
    return np.stack([x, y, z], axis=-1)


def normal_reset_sampler_vectorized_3d(rng: np.random.Generator, size, scale=5e-2):
    """
    Returns reset vectors with independent components drawn from a normal distribution.
    size: tuple (n_steps, n_paths, 3)
    """
    return rng.normal(loc=0.0, scale=scale, size=size)


def uniform_reset_sampler_vectorized_3d(rng: np.random.Generator, size, low=-0.1, high=0.1):
    """
    Returns reset vectors with independent components drawn uniformly on [low, high).
    size: tuple (n_steps, n_paths, 3)
    """
    return rng.uniform(low, high, size=size)


def uniform_in_ball_reset_sampler_vectorized(rng: np.random.Generator, size, radius=0.1):
    """
    Returns reset vectors uniformly distributed in the 3D ball of given radius.
    Uses rejection sampling; size is (n_steps, n_paths, 3) and the function returns that shape.
    For moderate radius and typical sizes this is fine; for extreme performance consider direct radial sampling.
    """
    n_steps, n_paths, _ = size
    out = np.empty(size, dtype=np.float64)
    for i in range(n_steps):
        for j in range(n_paths):
            # rejection sampling per vector
            while True:
                v = rng.uniform(-radius, radius, size=3)
                if v[0]*v[0] + v[1]*v[1] + v[2]*v[2] <= radius*radius:
                    out[i, j, :] = v
                    break
    return out


def plot_trajectory_with_reinsertions_3d(ax: Optional[plt.Axes],
                                         times: np.ndarray,
                                         states: np.ndarray,
                                         jump_times: np.ndarray,
                                         path_index: int = 0,
                                         title: str = None,
                                         show_ylabel: bool = True,
                                         show_xlabel: bool = True,
                                         show_reinserts: bool = True,
                                         text_loc: str = "left") -> plt.Axes:
    """
    Plot a single 3D trajectory's three components on one axes and draw reinsertion rug markers.

    Parameters
    ----------
    ax : Optional[plt.Axes]
        Matplotlib axes to draw on. If None, a new figure/axes is created.
    times : 1D array_like
        Saved times (length n_saves).
    states : array_like
        Trajectory states. Accepted shapes:
          - (n_saves, 3)          : single 3D trajectory
          - (n_saves, n_paths, 3) : many trajectories; `path_index` selects which path to plot
    jump_times : 1D array_like
        Times of reinsertion events (rug markers).
    path_index : int
        If `states` has shape (n_saves, n_paths, 3), choose this path (default 0).
    title : str
        Plot title.
    show_ylabel, show_xlabel : bool
        Whether to show axis labels.

    Returns
    -------
    ax : plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    times = np.asarray(times)
    jump_times = np.asarray(jump_times)

    states = np.asarray(states)
    if states.ndim == 3:
        # choose the requested path index
        if path_index < 0 or path_index >= states.shape[1]:
            raise IndexError(
                "path_index out of range for states with shape (n_saves, n_paths, 3)")
        traj = states[:, path_index, :]   # shape (n_saves, 3)
    elif states.ndim == 2 and states.shape[1] == 3:
        traj = states  # shape (n_saves, 3)
    else:
        raise ValueError(
            "states must have shape (n_saves, 3) or (n_saves, n_paths, 3)")

    # Plot the three components with distinct colours and labels
    comp_labels = ["$x_1$", "$x_2$", "$x_3$"]
    colors = ["C0", "C1", "C2"]
    linestyles = ['-', '-.', ':']

    for k in range(3):
        ax.plot(times, traj[:, k], lw=0.9,
                label=comp_labels[k], color=colors[k], linestyle=linestyles[k])

    # Optional reference horizontal lines (analogous to 1D case)
    ax.axhline(1.0, ls="--", alpha=0.5, color="black")
    ax.axhline(-1.0, ls="--", alpha=0.5, color="black")
    ax.axhline(2.0, ls="--", alpha=0.5, color="grey")
    ax.axhline(-2.0, ls="--", alpha=0.5, color="grey")

    # Annotate with total reinsertion count
    if show_reinserts:
        # rug markers for reinsertion times: place them slightly below the plotted data
        n_reinserts = int(jump_times.size)
        if n_reinserts > 0:
            ymin = np.min(traj)
            ymax = np.max(traj)
            yrange = ymax - ymin if (ymax > ymin) else 1.0
            rug_y = np.full_like(jump_times, ymin - 0.08 * yrange)
            ax.plot(jump_times, rug_y, linestyle='None', marker='|', markersize=8,
                    label='reinsertions', color='k', zorder=5)
        ax.text(
            0.02, 0.55,
            f"reinsertions: {n_reinserts}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    if text_loc == "center":
        ax.text(times[int(.5 * len(times))], -1.0,
                f"Regime Boundary", va="bottom", ha="center")
        ax.text(times[int(.5 * len(times))], -2.0, f"Reinsertion",
                va="bottom", ha="center", color="grey")
    elif text_loc == "right":
        ax.text(times[-1], -1.0,
                f"Regime Boundary", va="bottom", ha="right")
        ax.text(times[-1], -2.0, f"Reinsertion",
                va="bottom", ha="right", color="grey")
    elif text_loc == "left":
        ax.text(times[int(.01 * len(times))], -1.0,
                f"Regime Boundary", va="bottom", ha="left")
        ax.text(times[int(.01 * len(times))], -2.0, f"Reinsertion",
                va="bottom", ha="left", color="grey")
    # Labels / title
    if show_xlabel:
        ax.set_xlabel("$t$")
    if show_ylabel:
        ax.set_ylabel("$x_i(t)$")
    ax.set_title(title)

    ax.legend(loc='upper left', framealpha=1)
    # set reasonable ylim with small margin
    ax.set_ylim(-2.2, 2.2)

    return ax


def compute_lifetimes_per_sigma_3d(
    states: np.ndarray,
    sigma_arr: np.ndarray,
    reset_threshold: float = 2.0,
    hit_threshold: float = 1.0,
) -> Dict[float, List[int]]:
    """
    Compute lifetimes for 3D trajectories.

    Parameters
    ----------
    states : np.ndarray, shape (n_steps, n_paths, 3)
        Saved states for each path (last axis = 3 components).
    sigma_arr : np.ndarray, shape (n_paths,)
        Sigma value for each path (same column ordering as states).
    reset_threshold : float
        Radius threshold for detecting a reset (norm > reset_threshold).
    hit_threshold : float
        Radius threshold that constitutes a hit (norm >= hit_threshold).

    Returns
    -------
    dict
        { sigma_value: [lifetimes_in_saved_timesteps, ...] }
        Lifetimes are integers (number of saved timesteps after reinsertion until first hit).
        Censored reinjections (no hit before end) are skipped.
    """
    if states.ndim != 3 or states.shape[2] != 3:
        raise ValueError(
            "states must be a 3D array with shape (n_steps, n_paths, 3)")
    n_steps, n_paths, _ = states.shape
    if sigma_arr.shape[0] != n_paths:
        raise ValueError(
            "sigma_arr length must match number of paths (states.shape[1])")

    # compute radius for every saved time and path: shape (n_steps, n_paths)
    radii = np.linalg.norm(states, axis=2)

    lifetimes_per_sigma = defaultdict(list)

    for j in range(n_paths):
        col_r = radii[:, j]

        # indices where reset occurs (radius > reset_threshold)
        reset_idxs = np.nonzero(col_r > reset_threshold)[0]
        if reset_idxs.size == 0:
            continue

        # indices where hit occurs (radius >= hit_threshold)
        hit_idxs = np.nonzero(col_r >= hit_threshold)[0]
        if hit_idxs.size == 0:
            # no hits at all -> all resets are censored, skip
            continue

        # reinsertion is immediately after reset index
        reinsertion_idxs = reset_idxs + 1

        # any reinsertion beyond saved data are censored
        valid_mask = reinsertion_idxs < n_steps
        if not np.any(valid_mask):
            continue
        reinsertion_idxs = reinsertion_idxs[valid_mask]

        # For each reinsertion find the first hit index >= reinsertion using searchsorted
        pos = np.searchsorted(hit_idxs, reinsertion_idxs, side="left")

        # filter out censored (pos == len(hit_idxs))
        valid_pos_mask = pos < hit_idxs.size
        if not np.any(valid_pos_mask):
            continue

        hit_for_reset = hit_idxs[pos[valid_pos_mask]]
        reinj_valid = reinsertion_idxs[valid_pos_mask]

        # lifetimes = hit_index - reinsertion_index (number of saved timesteps after reinsertion until hit)
        lifetimes = (hit_for_reset - reinj_valid).astype(int)

        sigma = float(sigma_arr[j])
        lifetimes_per_sigma[sigma].extend(lifetimes.tolist())

    return dict(lifetimes_per_sigma)


def constant_radius_sampler_factory(radius, dtype=np.float32):
    def sample_on_sphere(rng, size):
        """Sample points uniformly on the surface of a sphere with given radius."""
        shape = size[:-1]
        phi = rng.uniform(0, 2*np.pi, size=shape)
        cos_theta = rng.uniform(-1, 1, size=shape)
        sin_theta = np.sqrt(1 - cos_theta**2)
        x = radius * sin_theta * np.cos(phi)
        y = radius * sin_theta * np.sin(phi)
        z = radius * cos_theta
        return np.stack([x, y, z], axis=-1)
    return sample_on_sphere


def mean_escape_times_via_reinsertion_radius3d(
        sigmas: np.ndarray,
        reinsertion_radius: np.ndarray,
        dt: float,
        n_steps: int,
        rng: np.random.Generator = None,
        save_every: int = 1,
        dtype=np.float32):
    """
    Uses compute_lifetimes_per_sigma to get lifetimes (in saved timesteps) per sigma,
    converts to seconds via dt * save_every, and returns mean and counts arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigmas = np.asarray(sigmas, dtype=float)
    reinsertion_radius = np.asarray(reinsertion_radius, dtype=float)

    n_sig = sigmas.size
    n_r = reinsertion_radius.size

    mean_matrix = np.full((n_sig, n_r), np.nan, dtype=np.float64)
    counts = np.zeros((n_sig, n_r), dtype=np.int64)

    time_per_saved = float(dt * save_every)

    for j, r in enumerate(reinsertion_radius):
        reset_sampler = constant_radius_sampler_factory(r, dtype=dtype)

        times, states, sigma_arr = simulate_trajectories_per_sigma_3d(
            sigmas=sigmas,
            dt=dt,
            n_steps=n_steps,
            reset_sampler=reset_sampler,
            rng=rng,
            save_every=save_every,
            x0=np.array([r/np.sqrt(3), r/np.sqrt(3), r /
                        np.sqrt(3)], dtype=np.float32),
            dtype=dtype
        )
        # Compute lifetimes per sigma using the helper you already have.
        # lifetimes_dict: { sigma_value (float) : [lifetimes_in_saved_timesteps, ...] }
        lifetimes_dict = compute_lifetimes_per_sigma_3d(
            states=states,
            sigma_arr=sigma_arr
        )
        # fill mean_matrix and counts in the same order as `sigmas`
        for i, sigma in enumerate(sigmas):
            key = float(sigma)
            lifesteps = lifetimes_dict.get(key, [])
            cnt = len(lifesteps)
            counts[i, j] = cnt
            if cnt == 0:
                mean_matrix[i, j] = np.nan
            else:
                durations_sec = np.asarray(
                    lifesteps, dtype=float) * time_per_saved
                mean_matrix[i, j] = float(np.mean(durations_sec))

    return mean_matrix, counts


def plot_means_with_inset(
    means_ana, means, reinsert_TO, sigmas,
    s=20, figsize=(6, 4),
    inset_coords=(0.19, 0.23, 2.25, 2.65),
    inset_width="55%", inset_height="55%",
    legend_loc="lower left", legend_ncol=4,
    axinset_xticks=None, axinset_yticks=None,
    reinsertion_points=None,
    dt_lower=1.0
):
    """
    Plot analytic and approximate curves with an inset zoom.

    Args:
        means_ana: analytic mean values (n_sigmas x n_curves)
        means: approximate means (n_sigmas x n_curves)
        reinsert_TO: reinsertion values (n_sigmas x n_curves)
        sigmas: noise strength array
        s: marker size
        figsize: figure size
        inset_coords: (x1, x2, y1, y2) limits of inset
        inset_width, inset_height: inset size
        legend_loc: legend location
        legend_ncol: legend columns
        axinset_xticks, axinset_yticks: tick locators for inset axes (Optional)
        reinsertion_points: values to label the legend
        dt_lower: scaling for reinsert_TO
    """
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(6)]
    markers = ['o', 's', '^', 'D', 'v', 'P']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot analytic curves
    if means_ana is not None:
        for j in range(means_ana.shape[1]):
            ax.plot(sigmas, means_ana[:, j], linestyle='-',
                    linewidth=1, color=colors[j])

    # Plot approximations
    for j in range(means.shape[1]):
        ax.scatter(sigmas, means[:, j], marker=markers[j], s=s, linewidths=1,
                   edgecolors=colors[j], alpha=1, zorder=3)

    # Plot reinsertion points
    for j in range(reinsert_TO.shape[1]):
        ax.scatter(sigmas, reinsert_TO[:, j]*dt_lower, marker=markers[j], s=s,
                   linewidths=1, edgecolors=colors[j], facecolors='none', alpha=1)

    # Legend
    if reinsertion_points is None:
        reinsertion_points = range(reinsert_TO.shape[1])
    labels = [f"{val:g}" for val in reinsertion_points]
    handles = [Line2D([0], [0], color=colors[i], marker=markers[i],
                      linestyle='-', markersize=7, markeredgewidth=0.8)
               for i in range(len(labels))]

    ax.legend(handles, labels, title="Reinsertion Points", loc=legend_loc,
              frameon=False, ncols=legend_ncol)

    # Inset axes
    x1, x2, y1, y2 = inset_coords
    axins = inset_axes(ax, width=inset_width, height=inset_height, loc="upper right",
                       bbox_to_anchor=(0.0, 0.0, 1, 1), bbox_transform=ax.transAxes)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    for j in range(means.shape[1]):
        if means_ana is not None:
            axins.plot(sigmas, means_ana[:, j],
                       linestyle='-', linewidth=1, color=colors[j])
        axins.scatter(sigmas, means[:, j], marker=markers[j], s=s, linewidths=1,
                      edgecolors=colors[j], alpha=1, zorder=3)
        axins.scatter(sigmas, reinsert_TO[:, j]*dt_lower, marker=markers[j], s=s,
                      linewidths=1, edgecolors=colors[j], facecolors='none', alpha=1)

    if axinset_xticks is not None:
        axins.xaxis.set_major_locator(axinset_xticks)
    if axinset_yticks is not None:
        axins.yaxis.set_major_locator(axinset_yticks)

    axins.grid(alpha=0.2)

    # Rectangle + connector
    rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.0, edgecolor='gray',
                     facecolor='none', linestyle='--', zorder=5)
    ax.add_patch(rect)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", linewidth=0.8)

    ax.set_xlabel("Noise Strength $\sigma$")
    ax.set_ylabel("Mean Escape Time")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, sigmas[-1])
    plt.tight_layout()
    return fig, ax, axins


def make_L3d(X, epsilon):
    """
    3D-specialized version of the original make_L.
    X: ndarray of shape (n_samples, 3)
       The last row of X is treated specially (mirrors original function logic).
    epsilon: float > 0
    Returns: L as scipy.sparse.csr_matrix (row-stochastic Markov matrix)

    Notes:
      - Uses cKDTree for efficient neighbor queries within radius r = sqrt(epsilon)*3.
      - Builds an affinity matrix A using Gaussian kernel exp(-d^2/epsilon).
      - Applies Coifman–Lafon density normalization with alpha = 1.0 (same as original).
      - Builds DMM (row-stochastic) for the first (n-1) points, then constructs L by
        stacking DMM[1:] and a bottom row computed from the last point's affinities
        (matching logic from your provided 1D function).
    """
    X = np.asarray(X, dtype=np.float64)

    # Separate last special point
    last_p = X[-1]
    X_flat = X[:-1]               # shape (n_samples, 3)
    n_samples = X_flat.shape[0]

    if epsilon > 0:
        # radius consistent with your original choice
        r = np.sqrt(epsilon) * 3.0

        # Build KD-tree for the (n) points (excluding the last special one)
        tree = cKDTree(X_flat)

        # query_ball_point on all points returns list of neighbor indices for each point
        # shape (m,2), each row (i,j) with i<j
        pairs = tree.query_pairs(r=r, output_type='ndarray')

        rows = []
        cols = []
        vals = []

        if pairs.size > 0:
            # compute distances for each unordered pair exactly once
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]

            diffs = X_flat[i_idx] - X_flat[j_idx]
            dists = np.sqrt((diffs * diffs).sum(axis=1)).astype(np.float64)

            # add both directions (i->j and j->i)
            rows.append(i_idx.astype(np.int32))
            cols.append(j_idx.astype(np.int32))
            vals.append(dists)

            rows.append(j_idx.astype(np.int32))
            cols.append(i_idx.astype(np.int32))
            vals.append(dists)

        # add self-loops once (distance = 0)
        diag_idx = np.arange(n_samples, dtype=np.int32)
        rows.append(diag_idx)
        cols.append(diag_idx)
        vals.append(np.zeros(n_samples, dtype=np.float64))

        # concatenate arrays
        rows = np.concatenate(rows).astype(np.int32)
        cols = np.concatenate(cols).astype(np.int32)
        vals = np.concatenate(vals).astype(np.float64)
        # affinity weights using Gaussian kernel
        A = sparse.coo_matrix((np.exp(-(vals**2) / epsilon), (rows, cols)),
                              shape=(n_samples, n_samples)).tocsr()

        # ensure self-loop weight (set diagonal to 1.0)
        A.setdiag(1.0)

        # density normalization (Coifman–Lafon style with alpha = 1.0)
        row_means = np.asarray(A.mean(axis=1)).ravel()
        q = 1.0 / row_means

        alpha = 1.0
        kalpha = q ** alpha
        D_k = sparse.diags(kalpha, offsets=0, format='csr')
        Adensnorm = D_k.dot(A).dot(D_k)

        # row-normalize to get Markov matrix DMM
        row_sums = np.asarray(Adensnorm.sum(axis=1)).ravel()
        inv_row_sums = 1.0 / row_sums
        D_norm = sparse.diags(inv_row_sums, offsets=0, format='csr')
        # shape (n_samples, n_samples), row-stochastic
        DMM = D_norm.dot(Adensnorm)

        L = sparse.vstack(
            [DMM[1:], sparse.csr_matrix((1, DMM.shape[1]))], format='csr')

        tree_full = cKDTree(X)

        neighbors_full = np.array(
            tree_full.query_ball_point(last_p, r=r), dtype=int)
        cols_last = neighbors_full[neighbors_full < X_flat.shape[0]]

        # if only one neighbor, add the next nearest
        if cols_last.size == 1:
            print(
                "Only one neighbor found for last point; adding second nearest neighbor.")
            dists_knn, inds_knn = tree_full.query(last_p, k=3)
            # keep only the neighbor(s) in X_flat, exclude last point itself
            mask = inds_knn < X_flat.shape[0]
            cols_last = inds_knn[mask]
            dists_last = dists_knn[mask]
        else:
            diffs_last = X_flat[cols_last] - last_p
            dists_last = np.linalg.norm(diffs_last, axis=1)

        # compute affinities and normalize
        vals_last = np.exp(-(dists_last**2) / epsilon)
        normalized_vals = vals_last / vals_last.sum()

        orig_rows = np.full(normalized_vals.shape, L.shape[0] - 1, dtype=int)
        orig_cols = cols_last.astype(int)
        updates = csr_matrix(
            (normalized_vals, (orig_rows, orig_cols)), shape=L.shape)
        L = L + updates

    else:
        DMM = sparse.eye(n_samples, format='csr')
        L = sparse.vstack(
            [DMM[1:], sparse.csr_matrix((1, DMM.shape[1]))], format='csr')

    return L


# Maxwell PDF function (same as before)


def maxwell_pdf(r, sd):
    r = np.asarray(r)
    coeff = np.sqrt(2/np.pi) / sd**3
    pdf = coeff * r**2 * np.exp(-r**2 / (2*sd**2))
    return np.where(r >= 0, pdf, 0.0)


def add_radius_histogram_to_right(fig, points, pdf="Maxwell",
                                  param=0.1, bins=30, hist_alpha=1,
                                  r_max=2):
    """
    Add a second subplot to the right of the existing one in `fig`.
    Plots a density histogram of radii and overlays the Maxwell pdf with given sd.
    """
    r = np.linalg.norm(points, axis=1)
    N = r.size
    # ---- Create a wider layout using GridSpec ----
    # more room for subplot 2
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])

    # Reassign original ax into gs[0]
    ax_old = fig.axes[0]
    ax_old.set_position(gs[0].get_position(fig))
    ax_old.set_subplotspec(gs[0])

    # Create second subplot on the right
    ax2 = fig.add_subplot(gs[1])
    # pdf grid
    r_grid = np.linspace(0, r_max, 400)

    # histogram as density
    n, bins, patches = ax2.hist(r, bins=bins, density=False, alpha=hist_alpha,
                                label='Empirical (hist)', edgecolor='none')
    bin_widths = np.diff(bins)
    avg_bw = bin_widths.mean()
    if pdf == "Maxwell":
        sd = param
        # Maxwell pdf
        ax2.plot(r_grid, maxwell_pdf(r_grid, sd) * N * avg_bw,
                 lw=2, label=f'Maxwell pdf (sd={sd})')
    elif pdf == "uniform":
        scale = param
        un_pdf = 3*(r_grid**2)/(scale**3)

        ax2.plot(r_grid, un_pdf * N * avg_bw, lw=2,
                 label=f'Uniform pdf (scale={scale})')
    # right-side y-axis
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax2.set_xlabel('$r$')
    ax2.set_ylabel('Count')
    ax2.set_ylim(0, max(n))
    ax2.set_xlim(0, r_max)
    plt.tight_layout()
    return fig, ax2


def cartesian_to_spherical(points):
    """Convert Nx3 cartesian coordinates to spherical (theta, phi)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def visualize_sphere_sampling(points, n_theta_bins=30, n_phi_bins=60, figsize=(6, 4)):
    """
    Create equal-area 2D histogram of 3D point distribution on sphere.

    Parameters
    ----------
    points : ndarray of shape (n_samples, 3)
        3D cartesian coordinates
    n_theta_bins : int
        Number of bins for polar angle (latitude-like)
    n_phi_bins : int
        Number of bins for azimuthal angle (longitude-like)
    """
    theta, phi = cartesian_to_spherical(points)

    cos_theta = np.cos(theta)
    theta_edges = np.arccos(np.linspace(1, -1, n_theta_bins + 1))
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

    histogram, _, _ = np.histogram2d(
        theta, phi, bins=[theta_edges, phi_edges]
    )

    fig, ax = plt.subplots(figsize=figsize)

    hist_int = histogram.astype(int)
    max_count = hist_int.max()

    n_categories = 8
    boundaries = np.linspace(-0.5, max_count + 0.5, n_categories + 1)

    colors = plt.cm.viridis(np.linspace(0, 1, n_categories))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    category_maxima = boundaries[1:]
    category_labels = [f'$\\leq{int(np.ceil(m - 0.5))}$'
                       for m in category_maxima]

    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

    mesh = ax.pcolormesh(
        np.degrees(phi_centers),
        -np.cos(theta_centers),
        histogram,
        cmap=cmap,
        norm=norm,
        shading='nearest'
    )

    cbar = plt.colorbar(mesh, ax=ax, spacing='proportional')
    cbar.set_label('Sample Count', rotation=270, labelpad=20)
    tick_positions = (boundaries[:-1] + boundaries[1:]) / 2
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(category_labels)

    ax.set_xlabel('$\\varphi$')
    ax.set_ylabel('$\\cos(\\theta)$')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    return fig, ax


def scatter_sphere_sampling(points, size=8, alpha=.5, figsize=(6, 4)):
    """
    Create scatter plot of 3D points in angular coordinates.

    Parameters
    ----------
    points : ndarray of shape (n_samples, 3)
        3D cartesian coordinates
    """
    theta, phi = cartesian_to_spherical(points)

    fig, ax = plt.subplots(figsize=figsize)

    n_points = len(points)

    ax.scatter(
        np.degrees(phi),
        # np.degrees(theta),
        -np.cos(theta),
        s=size,
        alpha=alpha,
        c='steelblue',
        edgecolors='none',
        rasterized=True
    )

    ax.set_xlabel('$\\varphi$')
    ax.set_ylabel('$\\cos(\\theta)$')
    ax.set_xlim(-180, 180)
    # ax.set_ylim(0, 180)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    return fig, ax


def direction_homogeneity_metric(points, n_theta_bins=30, n_phi_bins=60):
    """
    Calculate homogeneity metric for directional sampling.

    Returns
    -------
    float
        Coefficient of variation (std/mean) of bin counts.
        0 = perfectly homogeneous
        Higher values = more inhomogeneous
    """
    theta, phi = cartesian_to_spherical(points)

    theta_edges = np.arccos(np.linspace(1, -1, n_theta_bins + 1))
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

    histogram, _, _ = np.histogram2d(
        theta, phi, bins=[theta_edges, phi_edges]
    )

    bin_counts = histogram.flatten()

    return np.std(bin_counts) / np.mean(bin_counts)


def direction_homogeneity_ratio(points, n_theta_bins=30, n_phi_bins=60,
                                n_reference_samples=10):
    """
    Calculate homogeneity ratio compared to uniform random sampling.

    Parameters
    ----------
    points : ndarray of shape (n_samples, 3)
        3D cartesian coordinates
    n_theta_bins : int
        Number of bins for polar angle
    n_phi_bins : int
        Number of bins for azimuthal angle
    n_reference_samples : int
        Number of random samples to average over for reference

    Returns
    -------
    float
        Ratio of CV to reference CV.
        1.0 = same as uniform random
        < 1.0 = more homogeneous than random
        > 1.0 = less homogeneous than random
    """
    n_samples = len(points)

    cv_observed = direction_homogeneity_metric(
        points, n_theta_bins, n_phi_bins
    )

    cv_reference_values = []
    for _ in range(n_reference_samples):
        random_dirs = np.random.randn(n_samples, 3)
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
        cv_ref = direction_homogeneity_metric(
            random_dirs, n_theta_bins, n_phi_bins
        )
        cv_reference_values.append(cv_ref)

    cv_reference = np.mean(cv_reference_values)

    return cv_observed / cv_reference


def affinity_entry_sum(X, epsilon, chunk=1000):
    """sum_{i,j} exp(-||x_i-x_j||^2 / epsilon)"""
    X = np.asarray(X)
    n = X.shape[0]
    total = 0.0
    for i in range(0, n, chunk):
        d2 = cdist(X[i:i+chunk], X, metric='sqeuclidean')
        total += np.exp(-d2 / epsilon).sum()
    return float(total)


def plot_loglog_slope_analysis(sig_vals, sums):
    # compute log-log slopes and find maximum
    logx = np.log10(sig_vals)
    logy = np.log10(sums)
    slopes = np.diff(logy) / np.diff(logx)
    imax = int(np.argmax(slopes))
    slope_max = slopes[imax]
    sig_at_max = np.sqrt(sig_vals[imax] * sig_vals[imax+1])
    # line (in log space) anchored at midpoint
    logx_mid = 0.5*(logx[imax]+logx[imax+1])
    logy_mid = 0.5*(logy[imax]+logy[imax+1])
    x_line_log = np.linspace(logx[0], logx[-1], 300)
    y_line_log = slope_max*(x_line_log - logx_mid) + logy_mid

    # plot
    plt.figure(figsize=(4, 2.5))
    plt.loglog(sig_vals, sums, '-o', markersize=4, label='entry sum')
    plt.loglog(10**x_line_log, 10**y_line_log, '--',
               linewidth=2, label='maximum slope')
    plt.scatter([sig_at_max], [
                10**(slope_max*(logx_mid-logx_mid)+logy_mid)], color='red', zorder=10)
    plt.xlim(sig_vals[0], sig_vals[-1])
    plt.xlabel('$\sigma$')
    plt.ylabel('$\sum_{i,j} K_{i,j} N^{-2}$')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.text(
        .95, 0.05,
        rf'$\sigma^\ast \approx {sig_at_max:.3g}$' '\n'
        rf'$d \approx {slope_max:.4f}$',
        ha='right',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=1, edgecolor='none')
    )
    plt.legend(framealpha=1)
    plt.tight_layout()


def cdv(u, t):
    params = [0.95, -0.76095, 0.1, 1.25, 0.2, 0.5]

    x1, x2, x3, x4, x5, x6 = u
    x1star, x4star, C, beta, gamma, b = params

    def alpha_m(m, b):
        return (8.*math.sqrt(2.)/math.pi)*(m*m/(4*m*m-1))*((b*b + m*m - 1)/(b*b+m*m))

    def beta_m(m, b):
        return beta*b*b/(b*b+m*m)

    def gamma_m(m, b):
        return gamma*(4*m*m*m/(4*m*m-1))*(math.sqrt(2)*b/(math.pi*(b*b + m*m)))

    def gamma_m_tld(m, b):
        return gamma*(4*m/(4*m*m-1))*(math.sqrt(2)*b/math.pi)

    def delta(m, b):
        return (64.*math.sqrt(2.)/(15.*math.pi))*((b*b - m*m+1)/(b*b+m*m))

    # epsilon = 16.*math.sqrt(2)/(5*math.pi)
    epsilon = 1.44050610585137

    x1dot = gamma_m_tld(1, b)*x3 - C*(x1 - x1star)
    x4dot = gamma_m_tld(2, b)*x6 - C*(x4 - x4star) + epsilon*(x2*x6 - x3*x5)

    x2dot = -(alpha_m(1, b)*x1 - beta_m(1, b))*x3 - C*x2 - delta(1, b)*x4*x6
    x5dot = -(alpha_m(2, b)*x1 - beta_m(2, b))*x6 - C*x5 - delta(2, b)*x4*x3

    x3dot = (alpha_m(1, b)*x1 - beta_m(1, b))*x2 - \
        gamma_m(1, b)*x1 - C*x3 + delta(1, b)*x4*x5
    x6dot = (alpha_m(2, b)*x1 - beta_m(2, b))*x5 - \
        gamma_m(2, b)*x4 - C*x6 + delta(2, b)*x4*x2

    return np.array([x1dot, x2dot, x3dot, x4dot, x5dot, x6dot])


def cdvjac(u, t):
    params = [0.95, -0.76095, 0.1, 1.25, 0.2, 0.5]

    x1, x2, x3, x4, x5, x6 = u
    x1star, x4star, C, beta, gamma, b = params

    def alpha_m(m, b):
        return (8.*math.sqrt(2.)/math.pi)*(m*m/(4*m*m-1))*((b*b + m*m - 1)/(b*b+m*m))

    def beta_m(m, b):
        return beta*b*b/(b*b+m*m)

    def gamma_m(m, b):
        return gamma*(4*m*m*m/(4*m*m-1))*(math.sqrt(2)*b/(math.pi*(b*b + m*m)))

    def gamma_m_tld(m, b):
        return gamma*(4*m/(4*m*m-1))*(math.sqrt(2)*b/math.pi)

    def delta(m, b):
        return (64.*math.sqrt(2.)/(15.*math.pi))*((b*b - m*m+1)/(b*b+m*m))

    # epsilon = 16.*math.sqrt(2)/(5*math.pi)
    epsilon = 1.44050610585137

    dx1dot = np.array([-C, 0., gamma_m_tld(1, b), 0., 0., 0.])
    dx2dot = np.array([-alpha_m(1, b)*x3, -C, -alpha_m(1, b)
                      * x1 + beta_m(1, b), -delta(1, b)*x6, 0., -delta(1, b)*x4])
    dx3dot = np.array([alpha_m(1, b)*x2-gamma_m(1, b), alpha_m(1, b)
                      * x1 - beta_m(1, b), -C, delta(1, b)*x5, delta(1, b)*x4, 0.])
    dx4dot = np.array([0., epsilon*x6, -epsilon*x5, -C, -
                      epsilon*x3, gamma_m_tld(2, b)+epsilon*x2])
    dx5dot = np.array([-alpha_m(2, b)*x6, 0., -delta(2, b) *
                      x4, -delta(2, b)*x3, -C, -alpha_m(2, b)*x1 + beta_m(2, b)])
    dx6dot = np.array([alpha_m(2, b)*x5, delta(2, b)*x4, 0, -gamma_m(2,
                      b) + delta(2, b)*x2, alpha_m(2, b)*x1 - beta_m(2, b), -C])

    return np.array([dx1dot, dx2dot, dx3dot, dx4dot, dx5dot, dx6dot])


def classify_new_trajectory_radius(X, labels, Y, l, r, not_l=-1, tree=None):
    """
    X: ndarray (n_samples, d)       -- original state space points
    labels: ndarray (n_samples,)   -- labels for X
    Y: ndarray (m_samples, d)       -- new trajectory to classify
    l: int                          -- target label to check for
    r: float                        -- radius for neighborhood search
    not_l: int                      -- label assigned if l is not found

    returns: ndarray (m_samples,)   -- labels for Y (pointwise)
    """
    if tree is None:
        tree = cKDTree(X)
    Y_labels = np.empty(len(Y), dtype=labels.dtype)

    for i, y in enumerate(Y):
        idxs = tree.query_ball_point(y, r)
        if idxs and np.any(labels[idxs] == l):
            Y_labels[i] = l
        else:
            Y_labels[i] = not_l

    return Y_labels


def find_nearby_indices(points: np.ndarray, labels_TO: np.ndarray, eps: float, regime: int) -> np.ndarray:
    """
    Return sorted unique indices of trajectory points that are:
      - in regime 1 (labels_TO == 1), or
      - one step after a regime-1 point,
    and are within radius r of any of those seed points.

    Assumes well-formed inputs: points.shape == (N, D), labels_TO.shape == (N,), labels in {0,1}.
    """
    points = np.asarray(points)
    labels = np.asarray(labels_TO)
    r = np.sqrt(eps)*3
    N = points.shape[0]

    idx_regime1 = np.nonzero(labels == regime)[0]
    idx_after = (idx_regime1 + 1)[(idx_regime1 + 1) < N]
    seed_idx = np.unique(np.concatenate(
        [idx_regime1, idx_after])) if idx_regime1.size else np.array([], dtype=int)

    if seed_idx.size == 0:
        return np.array([], dtype=int)

    tree = cKDTree(points)
    neighbors_lists = tree.query_ball_point(points[seed_idx], r)

    # Minimal, common-case flattening:
    if seed_idx.size == 1:
        # query_ball_point returns a single list of ints for a single query point
        neighbors = np.asarray(neighbors_lists, dtype=int)
    else:
        # returns list-of-lists for multiple query points
        neighbors = np.concatenate(neighbors_lists).astype(int)

    return np.unique(neighbors)


def plot_loglog_slope_analysis(sigmas_loglog, sums_all, sums_block):
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    logx = np.log10(sigmas_loglog)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    x_line_log = np.linspace(logx[0], logx[-1], 300)
    table_rows = []
    # plot datasets and slope-lines
    for idx, (sums, lab, col, m) in enumerate(zip([sums_all, sums_block],
                                                  ["All", "Blocking"],
                                                  ['tab:blue', 'tab:orange'],
                                                  ['x', '+'])):
        sums = np.asarray(sums)
        logy = np.log10(sums)

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
        ax.loglog(sigmas_loglog, sums, '-', color=col, marker=m,
                  markersize=4, linewidth=1,
                  label=lab, markevery=1)
        ax.loglog(10 ** x_line_log, 10 ** y_line_log, '--', color=col,
                  linewidth=1, alpha=1)

        # point of maximum slope: same colour
        y_at_max = 10 ** (slope_max * (logx_mid - logx_mid) +
                          logy_mid)  # == 10**logy_mid
        ax.scatter([sig_at_max], [y_at_max], color=col, edgecolor='k',
                   s=(2*5), zorder=10)

        # accumulate row for the summary table
        table_rows.append((lab, f"{sig_at_max:.3g}", f"{slope_max:.4f}"))
    ax.set_xlim(sigmas_loglog[0], sigmas_loglog[-1])
    ax.set_xlabel(r'Noise Strength $\sigma$')
    ax.set_ylabel(r'$\sum_{i,j} K_{i,j} N^{-2}$')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # Legend: ensure marker+colour shown exactly as plotted
    legend_handles = []
    for col, m, lab in zip(['tab:blue', 'tab:orange'],
                           ['x', '+'],
                           ["All", "Blocking"]
                           ):
        legend_handles.append(Line2D([0], [0], color=col, marker=m, lw=1,
                                     markersize=4, label=lab))
    ax.legend(handles=legend_handles, framealpha=1)

    # Small table with results: place inside axes (upper left)
    col_labels = ("", r"$\sigma^\ast$", r"$d$")
    cell_text = [row for row in table_rows]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        colLoc='center',
        bbox=[0.45, 0.02, 0.53, 0.35],  # bottom-right corner
        edges='closed',
        colWidths=[0.4, 0.3, 0.3]
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
    for r, col in enumerate(['tab:blue', 'tab:orange'], start=1):
        rgba = list(to_rgba(col))
        rgba[3] = .5  # transparency

        for c in range(3):
            tbl[(r, c)].set_facecolor(rgba)
    # Tidy up
    plt.tight_layout()
    return fig, ax


def find_blocking_set(traj, labels):
    mean_x1_label_0 = traj[labels == 0, 0].mean()
    mean_x1_label_1 = traj[labels == 1, 0].mean()

    return 1 if mean_x1_label_1 < mean_x1_label_0 else 0
