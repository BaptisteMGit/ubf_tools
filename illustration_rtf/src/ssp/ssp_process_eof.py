#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ssp_process_eof.py
@Time    :   2026/09/08 16:11:11
@Author  :   Menetrier Baptiste
@Version :   1.1 (reviewed)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Generate realistic, synthetic sound-speed profiles (SSP)
             from a set of real (CMEMS-derived, see load_ssp_data.py)
             profiles, via an EOF/PCA decomposition: fit a PCA on the
             real profiles (get_ssp_eof), then sample new profiles by
             drawing random PC scores from the SAME per-component
             variance the real data showed (generate_new_ssp_profiles)
             and projecting back to depth space. process_ssp_profiles()
             ties this together end-to-end for one input '.nc' file
             (read -> fit -> sample -> save -> plot), and the
             '__main__' block below runs it for every seasonal/
             all-data file load_ssp_data.py produces.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from publication.publication_figure import PubFigure

PubFigure(ticks_fontsize=22, label_fontsize=24)


def get_ssp_eof(
    ssp, cumulative_variance_threshold=None, n_components=None, verbose=False
):
    """Fit a PCA (EOF decomposition) on a set of sound-speed profiles.

    Depths that are NaN in 'ssp' (typically below the local seafloor,
    since CMEMS-style products use a fixed depth grid covering the
    deepest point anywhere in the dataset) are dropped first, since
    PCA cannot handle missing values at all.

    Args:
        ssp (xr.DataArray): sound-speed profiles, dims (time, depth)
            (matching load_ssp_data.py's own output) -- one profile
            per time step.
        cumulative_variance_threshold (float|None): if given (and
            'n_components' is not), an exploratory PCA is fit first to
            find the SMALLEST number of components whose cumulative
            explained-variance ratio reaches this threshold (e.g. 0.99
            for 99%), and that count is then used for the real fit.
            Mutually exclusive with 'n_components' -- exactly one of
            the two must be given.
        n_components (int|None): if given (and
            'cumulative_variance_threshold' is not), fit the PCA with
            exactly this many components directly.
        verbose (bool): print the number of components chosen when
            'cumulative_variance_threshold' is used.

    Returns:
        tuple(xr.DataArray, np.ndarray, np.ndarray, sklearn.decomposition.PCA, sklearn.preprocessing.StandardScaler):
        ssp (the input, reduced to only the NaN-free depths actually
        used for the fit -- use THIS one downstream, not the original
        'ssp', so depths stay aligned with 'eof''s own columns), eof
        (pca.components_, shape (n_components, n_valid_depths)), the
        PC scores (pca.fit_transform's own output, shape
        (n_time, n_components)), the fitted PCA object, and the fitted
        StandardScaler (needed together to project new PC scores back
        to depth space later -- see generate_new_ssp_profiles()).

    Raises:
        ValueError: if both or neither of
            'cumulative_variance_threshold'/'n_components' are given.
    """
    X_original = ssp

    # A depth is only kept if it is NaN-free at EVERY time step.
    not_nan_depth_mask = ~X_original.isnull().any(dim="time").values
    not_nan_depth_idx = np.nonzero(not_nan_depth_mask)[0]
    X_original = X_original.isel(depth=not_nan_depth_idx)

    if cumulative_variance_threshold is not None and n_components is not None:
        raise ValueError(
            "Provide either cumulative_variance_threshold or n_components, not both."
        )
    if cumulative_variance_threshold is None and n_components is None:
        raise ValueError(
            "Provide either cumulative_variance_threshold or n_components."
        )

    if cumulative_variance_threshold is not None:
        # 1. Fit PCA original data (X_scaled)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)
        # Use all components to compute cumulative variance
        pca = PCA()
        pca.fit(X_scaled)

        # Compute cumulative variance ratio
        cumul_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        # Determine the number of components needed to reach the threshold (lazy implementation -> recompute PCA)
        n_components = (
            np.argmax(cumul_explained_variance >= cumulative_variance_threshold) + 1
        )

        if verbose:
            print(
                f"Number of components to reach {cumulative_variance_threshold*100}% cumulative variance: {n_components}"
            )

    # else: n_components was already given directly -- use as-is.

    # Fit PCA original data (X_scaled)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    pca = PCA(n_components)  # Choose number of components

    # Project data onto the principal components
    X_pca = pca.fit_transform(X_scaled)
    eof = pca.components_  # EOFs

    return X_original, eof, X_pca, pca, scaler


def plot_input_profiles_and_eofs(ssp, eof):
    """Plot every input profile (with the mean overlaid) next to each
    EOF/mode shape, side by side.

    Args:
        ssp (xr.DataArray): the (NaN-free) profiles get_ssp_eof()
            returned, dims (time, depth).
        eof (np.ndarray): get_ssp_eof()'s own 'eof' return value,
            shape (n_components, n_depth) -- must match 'ssp''s own
            depth grid (i.e. come from the SAME get_ssp_eof() call).

    Returns:
        tuple(matplotlib.figure.Figure, np.ndarray): the figure and
        its array of axes (1 profile panel + 1 panel per EOF).
    """
    n_components = eof.shape[0]
    fig, axs = plt.subplots(
        1,
        n_components + 1,
        figsize=(16, 8),
        sharey=True,
        gridspec_kw={"width_ratios": [1] + [0.5] * n_components},
    )

    # Plot original profiles
    for it in range(ssp.sizes["time"]):
        ssp.isel(time=it).plot(y="depth", yincrease=False, alpha=0.25, ax=axs[0])
    ssp.mean(dim="time").plot(
        y="depth", yincrease=False, color="k", lw=2, label="Mean profile", ax=axs[0]
    )
    axs[0].set_xlabel(r"[m~s$^{-1}$]")
    axs[0].set_ylabel("")
    axs[0].set_title("")

    # Plot EOFs
    for ic in range(n_components):
        idx_comp = ic + 1
        axs[idx_comp].plot(eof[ic, :], ssp.depth.values, color="k", lw=2)
        axs[idx_comp].set_title(rf"$\psi_{{{idx_comp}}}$")

    fig.supylabel("Depth [m]")

    return fig, axs


def plot_synthetic_profiles(ssp_original, ssp_synthetic, n_profiles_max=50):
    """Plot a sample of the synthetic profiles alongside the original
    data's mean profile.

    Args:
        ssp_original (xr.DataArray): the real profiles get_ssp_eof()
            returned, dims (time, depth) -- only its mean (over time)
            is actually plotted, as a reference.
        ssp_synthetic (xr.DataArray): generate_new_ssp_profiles()'s
            output (via convert_synthetic_to_xarray()), dims
            (time, depth).
        n_profiles_max (int): if 'ssp_synthetic' has more profiles than
            this, a random subset of this size is plotted instead (for
            legibility) -- the full set is still used elsewhere (e.g.
            when it's saved to disk), only this plot is subsampled.

    Returns:
        tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), sharey=True)

    # Limit the number of synthetic profiles plotted for clarity
    if ssp_synthetic.sizes["time"] > n_profiles_max:
        ssp_synthetic = ssp_synthetic.isel(
            time=np.random.choice(
                ssp_synthetic.sizes["time"], n_profiles_max, replace=False
            )
        )

    ssp_synthetic.plot(
        ax=ax,
        y="depth",
        yincrease=False,
        hue="time",
    )
    ssp_original.mean(dim="time").plot(
        ax=ax, y="depth", yincrease=False, color="k", lw=2, label="Mean profile"
    )
    # Remove legend for clarity
    ax.legend().remove()
    # No title
    ax.set_title("")

    return fig, ax


def generate_new_ssp_profiles(pca, scaler, n_new_samples=100):
    """Generate new synthetic profiles by sampling random PC scores
    from the SAME per-component variance the real data showed (see
    get_ssp_eof()), then projecting them back to depth space.

    Args:
        pca (sklearn.decomposition.PCA): fitted on the real profiles
            (see get_ssp_eof()'s own return value).
        scaler (sklearn.preprocessing.StandardScaler): fitted on the
            SAME real profiles, alongside 'pca' (see get_ssp_eof()).
        n_new_samples (int): number of synthetic profiles to generate.

    Returns:
        np.ndarray: shape (n_new_samples, n_depth) -- sound speed
        (m/s) at each of the depths get_ssp_eof() kept (i.e. matching
        'ssp''s own depth grid, NOT necessarily the original,
        un-filtered one).
    """
    # Calculate standard deviation of each component
    pca_stds = np.sqrt(pca.explained_variance_)

    # Sample new random coords from a normal distribution
    new_pca_coords = np.random.normal(
        loc=0.0, scale=pca_stds, size=(n_new_samples, pca.n_components_)
    )

    # 3. Transform back to original feature space
    X_new_scaled = pca.inverse_transform(new_pca_coords)

    # 4. Invert the scaling to return to the original data scale
    X_new_generated = scaler.inverse_transform(X_new_scaled)

    return X_new_generated


def convert_synthetic_to_xarray(X_ssp_synthetic, ssp_original, n_components):
    """Wrap generate_new_ssp_profiles()'s raw array into an xr.DataArray,
    matching 'ssp_original''s own depth grid, with metadata describing
    it as synthetic (rather than the real 'ssp_original''s own attrs,
    which no longer apply as-is).

    Args:
        X_ssp_synthetic (np.ndarray): shape (n_new_samples, n_depth),
            as returned by generate_new_ssp_profiles().
        ssp_original (xr.DataArray): the real profiles the synthetic
            ones were derived from (get_ssp_eof()'s own return value)
            -- only its depth grid and attrs (as a starting point) are
            used here.
        n_components (int): how many PCA components were used to
            generate 'X_ssp_synthetic' -- recorded in the returned
            DataArray's attrs, for traceability.

    Returns:
        xr.DataArray: dims (time, depth) -- 'time' here is just a
        0-based sample index (these are SYNTHETIC profiles, not tied
        to any real date), 'depth' matches 'ssp_original''s own.
    """

    attrs = ssp_original.attrs.copy()
    for key in ("source", "cell_methods"):
        attrs.pop(key, None)
    attrs["description"] = "Synthetic profiles generated using PCA"
    attrs["history"] = "Generated using PCA on original profiles"
    attrs["pca_components"] = n_components

    ssp_synthetic = xr.DataArray(
        X_ssp_synthetic,
        dims=["time", "depth"],
        coords={
            "depth": ssp_original.depth.values,
            "time": np.arange(X_ssp_synthetic.shape[0]),
        },
        attrs=attrs,
    )

    return ssp_synthetic


def process_ssp_profiles(
    root_ssp_nc,
    filename_ssp,
    root_img,
    cumulative_variance_threshold=0.9999,
    n_new_samples=1000,
):
    """End-to-end pipeline for one input profile set: read
    '<root_ssp_nc>/<filename_ssp>.nc' (as produced by load_ssp_data.py),
    fit an EOF/PCA decomposition (get_ssp_eof()), generate
    'n_new_samples' synthetic profiles from it
    (generate_new_ssp_profiles()), save them to
    '<root_ssp_nc>/synthetic_<filename_ssp>_<n_new_samples>.nc', and
    save 2 diagnostic figures (the input profiles + EOFs, and a sample
    of the synthetic profiles) under 'root_img'.

    Args:
        root_ssp_nc (str): directory holding '<filename_ssp>.nc'
            (input) and where the synthetic '.nc' file is written.
        filename_ssp (str): input file's name, WITHOUT the '.nc'
            extension (e.g. "ssp_profiles_sw_winter").
        root_img (str): directory the 2 diagnostic figures are saved
            into.
        cumulative_variance_threshold (float): forwarded to
            get_ssp_eof() -- the number of PCA components is chosen
            automatically to reach this fraction of explained
            variance.
        n_new_samples (int): forwarded to generate_new_ssp_profiles().

    Returns:
        tuple(xr.DataArray, np.ndarray, np.ndarray, sklearn.decomposition.PCA, sklearn.preprocessing.StandardScaler, xr.DataArray):
        see get_ssp_eof()'s own return value (ssp, eof, X_pca, pca,
        scaler), plus ssp_synthetic (the generated profiles, as
        returned by convert_synthetic_to_xarray()).
    """
    os.makedirs(root_ssp_nc, exist_ok=True)
    os.makedirs(root_img, exist_ok=True)

    fpath = os.path.join(root_ssp_nc, filename_ssp + ".nc")
    ssp = xr.open_dataset(fpath).ssp

    # Get EOFs
    ssp, eof, X_pca, pca, scaler = get_ssp_eof(
        ssp, cumulative_variance_threshold=cumulative_variance_threshold, verbose=True
    )

    # Generate synthetic profiles
    X_ssp_synthetic = generate_new_ssp_profiles(
        pca, scaler, n_new_samples=n_new_samples
    )
    ssp_synthetic = convert_synthetic_to_xarray(
        X_ssp_synthetic=X_ssp_synthetic,
        ssp_original=ssp,
        n_components=pca.n_components_,
    )
    # Save synthetic profiles to NetCDF
    synthetic_filename = f"synthetic_{filename_ssp}_{n_new_samples}.nc"
    ssp_synthetic.to_netcdf(os.path.join(root_ssp_nc, synthetic_filename))

    # Plot EOFs and original profiles
    fig, axs = plot_input_profiles_and_eofs(ssp, eof)
    fig.savefig(os.path.join(root_img, f"synthetic_{filename_ssp}_eofs.png"))
    plt.close(fig)

    # Plot synthetic profiles
    fig, ax = plot_synthetic_profiles(
        ssp_original=ssp, ssp_synthetic=ssp_synthetic, n_profiles_max=50
    )
    fig.savefig(os.path.join(root_img, f"synthetic_{filename_ssp}_profiles.png"))
    plt.close(fig)

    return ssp, eof, X_pca, pca, scaler, ssp_synthetic


if __name__ == "__main__":
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\img\ssp"
    root_ssp_nc = (
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\ssp"
    )

    n_new_samples = 1000
    cumulative_variance_threshold = 0.9999

    filenames = [
        "ssp_profiles_sw",
        "ssp_profiles_dw",
        "ssp_profiles_sw_winter",
        "ssp_profiles_sw_spring",
        "ssp_profiles_sw_summer",
        "ssp_profiles_sw_automn",
        "ssp_profiles_dw_winter",
        "ssp_profiles_dw_spring",
        "ssp_profiles_dw_summer",
        "ssp_profiles_dw_automn",
    ]

    for filename in filenames:
        print(f"Processing {filename}...")
        process_ssp_profiles(
            root_ssp_nc=root_ssp_nc,
            filename_ssp=filename,
            root_img=root_img,
            cumulative_variance_threshold=cumulative_variance_threshold,
            n_new_samples=n_new_samples,
        )
