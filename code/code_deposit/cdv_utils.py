from scipy.sparse.linalg import spsolve
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse import csr_matrix, eye
import math
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from matplotlib.colors import LogNorm, BoundaryNorm
from itertools import combinations
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import Normalize
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
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', leaf_size=1000,
                            n_jobs=-1).fit(positions)
    dists, idxs = nbrs.kneighbors(positions)
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
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1,
                                algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)

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
    (t, df_columns, data, state_vector_t, output_folder, cmap,
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
                  origin='lower')

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
                                  n_steps=1000, cmap='viridis_r',
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
                    output_folder, cmap, variable_pairs, pc_state,
                    X, Y)
        else:
            args = (t, df.columns, data, state_vector[t, :],
                    output_folder, cmap, variable_pairs, pc_state[t, :],
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
    tree = KDTree(state_vector)
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


def plot_density_heatmap(df, bins=200, cmap='viridis_r', columns=None,
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
    variable_pairs = list(combinations(x_columns, 2))
    n_plots = len(variable_pairs)
    ncols = min(n_plots, 5)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    fig.subplots_adjust(bottom=0.15)  # Adjust for color bar space
    histograms = {}
    for idx, (var1, var2) in enumerate(variable_pairs):
        ax = axes[idx]

        # Plot 2D histogram for each variable pair
        h = ax.hist2d(df[var1], df[var2], bins=bins,
                      cmap=cmap, norm=LogNorm())

        if points is not None:
            if isinstance(points, dict):
                # If dict, assume it has the same structure as df columns
                x_vals = points.get(var1)
                y_vals = points.get(var2)
                if x_vals is not None and y_vals is not None:
                    n_points = len(x_vals)
                    colors = plt.cm.Set1(np.linspace(0, 1, n_points))
                    ax.scatter(x_vals, y_vals, c=colors, s=50, alpha=0.8,
                               edgecolors='black', linewidths=0.5)

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
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)

    # Add a shared color bar across all subplots
    cbarAx = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Adjust position and size
    fig.colorbar(h[3], cax=cbarAx, orientation='horizontal',
                 label='Point Count (log scale)')

    plt.show()
    plt.close(fig)
    return histograms


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
    ax.contourf(x, y, array, cmap="RdBu", norm=norm)

    # plt.colorbar(label='Amplitude')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("2D Fourier Mode")
    # plt.show()


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
    X: ndarray, shape (n_samples, n_features)
    epsilon: float
    returns: (DMM, A) as scipy.sparse matrices
    """
    n_samples = X.shape[0]
    r = np.sqrt(5.0 * epsilon)

    nbrs = NearestNeighbors(radius=r).fit(X)
    distances_list, indices_list = nbrs.radius_neighbors(
        X, return_distance=True)

    # accumulate COO entries
    total = sum(len(idx) for idx in indices_list)
    rows = np.empty(total, dtype=np.int32)
    cols = np.empty(total, dtype=np.int32)
    vals = np.empty(total, dtype=np.float64)

    p = 0
    for i, (idxs, dists) in enumerate(zip(indices_list, distances_list)):
        li = len(idxs)
        if li == 0:
            continue
        rows[p:p+li] = i
        cols[p:p+li] = idxs
        vals[p:p+li] = dists
        p += li

    if p < total:
        rows = rows[:p]
        cols = cols[:p]
        vals = vals[:p]

    # Gaussian kernel (note: using epsilon in denominator like exp(-d^2/epsilon))
    A = sparse.coo_matrix((np.exp(-(vals**2) / epsilon),
                          (rows, cols)), shape=(n_samples, n_samples)).tocsr()

    # ensure self-loop weight (set diagonal to 1.0)
    A.setdiag(1.0)

    # density normalization (Coifman–Lafon style with alpha = 1.0)
    row_means = np.asarray(A.mean(axis=1)).ravel()
    # avoid division by zero
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


def compute_clusters_from_TO(DMM, n_eig=12, n_clusters=2, n_init=10,
                             eig_indices=None, random_state=0):
    """
    DMM: scipy.sparse matrix, shape (n_samples, n_samples)  -- diffusion-maps Markov matrix (row-stochastic)
    n_eig: int -- number of eigenpairs to compute (ARPACK)
    n_clusters: int -- k for k-means
    n_init: int -- k-means n_init
    eig_indices: None or list/array of zero-based eigenvector indices to use as features.
                 If None, defaults to [1,2,3,4] (i.e. first nontrivial eigenvectors).
    random_state: int
    returns: dict with keys 'labels', 'kmeans', 'eigenvals', 'eigenvecs', 'features', 'L'
    """
    import numpy as np
    from scipy.sparse.linalg import eigs
    from sklearn.cluster import KMeans

    n = DMM.shape[0]

    # build time-shifted transfer operator (size (n-1, n-1))
    L = DMM[1:, :-1].transpose()

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
        'L': L                   # transfer operator used
    }


def classify_new_trajectory(X, labels, Y, k=10, threshold=0.5):
    """
    X: ndarray (n_samples, d)   -- original state space points
    labels: ndarray (n_samples,) -- cluster labels for X (e.g. from k-means)
    Y: ndarray (m_samples, d)   -- new trajectory to classify
    k: int -- number of nearest neighbors
    threshold: float in [0,1] -- fraction of neighbors required to assign regime

    returns: ndarray (m_samples,) of regime labels for Y
             (-1 if no cluster passes threshold)
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, idxs = nn.kneighbors(Y)

    Y_labels = []
    for neighbors in idxs:
        neigh_labels = labels[neighbors]
        uniq, counts = np.unique(neigh_labels, return_counts=True)
        dominant = uniq[np.argmax(counts)]
        frac = counts.max() / k
        if frac >= threshold:
            Y_labels.append(dominant)
        else:
            Y_labels.append(-1)   # mark as "uncertain / outside"
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

    fig, ax = plt.subplots(figsize=(10, 6))

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
            label='0–95th percentile' if regime == 0 else "")
    ax.plot(sigmas, q95s, linestyle=':', color='black', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lifetime (time steps)")
    ax.set_title(f"Regime {regime} lifetime distributions")
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
