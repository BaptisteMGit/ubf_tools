# Poubelle avec les versions de code qui n'ont pas été retenues.


def rtf_cs(f, n_rcv, Rx, Rv, first_column=False):
    """
    Derive RTF vector using covariance subtraction method described in Markovich-Golan, S., & Gannot, S. (2015).
    Reference receiver is assumed to be the first one.

    Parameters:
    f : ndarray
        Frequencies vector.
    n_rcv : int
        Number of receivers
    Rx : ndarray
        3D CSD matrix for signal at receiver positions (frequency bins x num_receivers x num_receivers).
    Rv : ndarray
        3D CSD matrix for noise at receiver positions (frequency bins x num_receivers x num_receivers).

    Returns:
    f : ndarray
        Frequencies vector.
    rtf : ndarray
        Relative Transfer Function (RTF) matrix (len(f) x num_receivers).
    """

    R_delta = Rx - Rv  # Equation (9)

    # for k in range(R_delta.shape[0]):
    #     print(k, np.alltrue(np.diag(R_delta[k]) >= 0))
    # pos_diags = np.array(
    #     [np.alltrue(np.diag(R_delta[k]) >= 0) for k in range(R_delta.shape[0])]
    # )
    # print(np.sum(pos_diags))

    # Faster implementation
    # Reference receiver is assumed to be the first one
    if first_column:
        e1 = np.eye(n_rcv)[:, 0]

        # Vectorized computation of rtf across all frequencies
        R_delta_e1 = R_delta @ e1  # First columns of CSDMs (for all freqs)
        e1_TR_delta_e1 = (
            e1.T @ R_delta @ e1
        )  # First entry of first column of CSDMs (for all freqs)

        eps = np.finfo(float).eps
        rtf = R_delta_e1 / (e1_TR_delta_e1[:, np.newaxis] + eps)

    else:
        for k in range(R_delta.shape[0]):
            eigva, eigve = scipy.linalg.eigh(R_delta[k, ...], check_finite=False)

            _, rtf_f = sort_eigenvectors_get_major(eigva, eigve)
            rtf_f = normalize_to_1(rtf_f)

            if k == 0:
                rtf = rtf_f[np.newaxis, :]
            else:
                rtf = np.vstack((rtf, rtf_f[np.newaxis, :]))

        # _, rtf = sort_eigenvectors_get_major(eigva, eigve)
        # rtf = normalize_to_1(rtf)

    # rtf[~pos_diags, :] = np.ones(R_delta.shape[1]) * np.nan
    # print(f"Ellapsed time (fast) = {time()-t0}")

    return f, rtf


def sort_eigenvectors_get_major(eigva, eigve, num_to_keep=1, squeeze=True):
    """
    Return eigenvector corresponding to eigenvalue with maximum norm. if eigenvalues are not ALL finite, return NaN
    """

    if num_to_keep == -1:
        num_to_keep = len(eigva)  # keep all eigenvectors

    if not np.all(np.isfinite(eigva)):
        return (
            np.ones_like(eigva)[:num_to_keep] * np.nan,
            np.ones_like(eigve)[:, :num_to_keep] * np.nan,
        )

    # Sort eigenvalues and eigenvectors in ascending order
    idx_largest_eigvas_sorted = np.argsort(np.real(eigva))
    eigva, eigve = (
        eigva[idx_largest_eigvas_sorted],
        eigve[:, idx_largest_eigvas_sorted],
    )

    if squeeze:
        return np.squeeze(eigva[-num_to_keep:]), np.squeeze(eigve[:, -num_to_keep:])
    else:
        return eigva[-num_to_keep:], eigve[:, -num_to_keep:]


def normalize_to_1(eigve_single_column):
    idx_ref_mic = 0
    eps = np.finfo(float).eps

    # normalize vector to get 1 at reference microphone
    if np.abs(eigve_single_column[idx_ref_mic]) < eps:
        eigve_normalized = np.zeros_like(eigve_single_column)
    else:
        eigve_normalized = eigve_single_column / eigve_single_column[idx_ref_mic]

    return eigve_normalized


def rtf_cw(f, n_rcv, stft_x, Rv):

    # Loop over frequencies
    rtf = np.zeros((len(f), n_rcv), dtype=complex)
    # First receiver is considered as the reference
    e1 = np.eye(n_rcv)[:, 0]

    for i, f_i in enumerate(f):
        Rv_f = Rv[i]
        # Rs_f = Rs[i]
        # Rx_f = Rx[i]
        stft_x_f = stft_x[:, i, :]

        # Cholesky decomposition of the noise csdm and its inverse : Equation (25a) and (25b)
        Rv_half = scipy.linalg.cholesky(Rv_f, lower=False)
        Rv_half_inv = np.linalg.inv(
            Rv_half
        ).T  # Theoreticaly equivalent but leads to greater numerical errors
        # Rv_inv_f = np.linalg.inv(Rv_f)
        # Rv_half_inv = scipy.linalg.cholesky(Rv_inv_f, lower=False)

        # Compute the whitened signal csdm : Equation (26)
        stft_y_f = Rv_half_inv @ stft_x_f

        # Compute the whitened signal csdm : Equation (31)
        # Reshape to the required shape for the computation
        stft_y_f = [
            stft_y_f[i, np.newaxis, :] for i in range(n_rcv)
        ]  # List of stft at frequency f : n_rcv element of shape (n_freq=1, n_seg)
        Ry_f = compute_csd_matrix_fast(
            stft_y_f, n_seg_cov=0
        )  # Covariance matrix at frequency f
        Ry_f = (
            Ry_f.squeeze()
        )  # Remove useless frequency dimension to get shape (n_rcv, n_rcv)

        # Eigenvalue decomposition of Ry_f to get q (major eingenvector) : Equation (32)
        eig_val, eig_vect = np.linalg.eig(Ry_f)
        # We can check that the Ry_f can be diagonalized np.round(np.abs(np.linalg.inv(eig_vect) @ Ry_f @ eig_vect), 5)

        i_max_eig = np.argmax(np.abs(eig_val))
        q = eig_vect[:, i_max_eig]

        rtf_f = (Rv_half @ q) / (e1.T @ Rv_half @ q)  # Equation (32)
        rtf[i, :] = rtf_f

    return f, rtf


def estimate_rtf_parallel(
    ds_sig_noise,
    i_ref,
    library_props,
    nperseg=2**11,
    noverlap=2**10,
    verbose=False,
):
    """
    Estimate the RTF vector using Covariance Substraction method (CS).

    10/02/2025 : Dummy implementation looping over x and y axis.

    """

    # By default rtf estimation method assumed the first receiver as the reference -> need to roll along the receiver axis
    idx_pos_ref = np.argmin(np.abs(ds_sig_noise.idx_rcv.values - i_ref))
    npos_to_roll = ds_sig_noise.sizes["idx_rcv"] - idx_pos_ref
    ds_sig_noise_rolled = ds_sig_noise.roll(
        idx_rcv=npos_to_roll,
        roll_coords=True,
    )
    ## Derive event RTF ##
    # Extract useful noisy signals
    x_e = ds_sig_noise_rolled.x_e  # Noisy event signals
    t = ds_sig_noise_rolled.t.values  # Time vector
    # Extract noise signals (different noise realisation than the one use to pollute the signals)
    n_e = ds_sig_noise_rolled.n_e_bis  # Event noise
    f_rtf, rtf_e, _, _, _ = rtf_covariance_subtraction(
        t, noisy_signal=x_e.T, noise_only=n_e.T, nperseg=nperseg, noverlap=noverlap
    )

    ### Derive library RTF ###
    # Split dataset in blocks to parallelize computation
    ds_sig_noise_blocks = build_ds_block(ds_sig_noise_rolled)
    iterable_args = [(ds_block, nperseg, noverlap) for ds_block in ds_sig_noise_blocks]

    # Create multiprocessing Pool of workers
    with Pool(N_WORKERS) as pool:
        res = pool.starmap(
            func=estimate_rtf_parallel_process_block, iterable=iterable_args
        )

    # Gather results
    rtf = gather_res_blocks(res, ds_sig_noise_rolled, f_rtf)

    # Restict to the frequency band of interest
    idx_band = (f_rtf >= library_props["f0"]) & (f_rtf <= library_props["f1"])
    f_rtf = f_rtf[idx_band]
    rtf_l = rtf[idx_band]
    rtf_e = rtf_e[idx_band]

    return f_rtf, rtf_l, rtf_e


def estimate_rtf_parallel_process_block(ds_block, nperseg, noverlap):
    t = ds_block.t.values
    results_cs = []
    # results_cw = []
    for x_i in ds_block.x:
        for y_i in ds_block.y:
            # Transpose to fit rtf estimation required input shape (ns, nrcv)
            noisy_sig = ds_block.x_l.sel(x=x_i, y=y_i).T
            noise_only = ds_block.n_l_bis.sel(x=x_i, y=y_i).T

            # Derive rtf
            _, rtf_cs_l, _, _, _ = rtf_covariance_subtraction(
                t, noisy_sig, noise_only, nperseg, noverlap
            )
            # _, rtf_cs_l, _, _, _ = rtf_covariance_whitening(
            #     t, noisy_sig, noise_only, nperseg, noverlap
            # )

            # Store
            results_cs.append(rtf_cs_l)

    rtf_cs_l = np.array(results_cs)

    ### Reshape to the required shape ###
    # Step 1 : reshape to (nx, ny, nf, n_rcv)
    shape = (len(ds_block.x), len(ds_block.y)) + rtf_cs_l.shape[1:]
    rtf_cs_l = rtf_cs_l.reshape(shape)

    # Step 2 : permute to (nf, nx, ny, n_rcv)
    axis_permutation = (2, 0, 1, 3)
    rtf_cs_l = np.transpose(rtf_cs_l, axis_permutation)
    ### End reshape ###

    return rtf_cs_l


def estimate_rtf(
    ds_sig_noise,
    i_ref,
    library_props,
    nperseg=2**11,
    noverlap=2**10,
    verbose=False,
    roll=True,
):
    """
    Estimate the RTF vector using Covariance Substraction method (CS).

    10/02/2025 : Dummy implementation looping over x and y axis.

    """

    # By default rtf estimation method assumed the first receiver as the reference -> need to roll along the receiver axis
    if roll:
        idx_pos_ref = np.argmin(np.abs(ds_sig_noise.idx_rcv.values - i_ref))
        npos_to_roll = ds_sig_noise.sizes["idx_rcv"] - idx_pos_ref
        ds_sig_noise_rolled = ds_sig_noise.roll(
            idx_rcv=npos_to_roll,
            roll_coords=True,
        )

    # Extract useful noisy signals
    x_l = ds_sig_noise_rolled.x_l  # Noisy library signals
    x_e = ds_sig_noise_rolled.x_e  # Noisy event signals
    t = ds_sig_noise_rolled.t.values  # Time vector

    # Extract noise signals (different noise realisation than the one use to pollute the signals)
    n_l = ds_sig_noise_rolled.n_l_bis  # Library noise
    n_e = ds_sig_noise_rolled.n_e_bis  # Event noise

    # NOTE : inputs to rtf estimation function need to be transposed to fit required input shape (ns, nrcv)
    ## Derive event RTF ##
    # f_rtf, rtf_cs_e, _, _, _ = rtf_covariance_subtraction(
    f_rtf, rtf_cs_e, _, _, _ = rtf_covariance_subtraction(
        t, noisy_signal=x_e.T, noise_only=n_e.T, nperseg=nperseg, noverlap=noverlap
    )

    # f_rtf, rtf_cs_e, _, _, _ = rtf_covariance_whitening(
    #     t, noisy_signal=x_e.T, noise_only=n_e.T, nperseg=nperseg, noverlap=noverlap
    # )

    # f_rtf, rtf_cs_e, _, _, _ = rtf_covariance_whitening(
    #     t, noisy_signal=x_e.T, noise_only=n_e.T, nperseg=nperseg, noverlap=noverlap
    # )
    # Dask used at higher level to parallelize the computation

    results_cs = []
    # results_cw = []
    for x_i in ds_sig_noise.x:
        for y_i in ds_sig_noise.y:
            # Transpose to fit rtf estimation required input shape (ns, nrcv)
            noisy_sig = x_l.sel(x=x_i, y=y_i).T
            noise_only = n_l.sel(x=x_i, y=y_i).T

            # Derive rtf
            _, rtf_cs_l, _, _, _ = rtf_covariance_subtraction(
                t, noisy_sig, noise_only, nperseg, noverlap
            )
            # _, rtf_cs_l, _, _, _ = rtf_covariance_whitening(
            #     t, noisy_sig, noise_only, nperseg, noverlap
            # )

            # Store
            results_cs.append(rtf_cs_l)

    rtf_cs_l = np.array(results_cs)

    ### Reshape to the required shape ###
    # Step 1 : reshape to (nx, ny, nf, n_rcv)
    shape = (len(ds_sig_noise.x), len(ds_sig_noise.y)) + rtf_cs_l.shape[1:]
    rtf_cs_l = rtf_cs_l.reshape(shape)

    # Step 2 : permute to (nf, nx, ny, n_rcv)
    axis_permutation = (2, 0, 1, 3)
    rtf_cs_l = np.transpose(rtf_cs_l, axis_permutation)
    ### End reshape ###

    # Restict to the frequency band of interest
    idx_band = (f_rtf >= library_props["f0"]) & (f_rtf <= library_props["f1"])
    f_rtf = f_rtf[idx_band]
    rtf_cs_l = rtf_cs_l[idx_band]
    rtf_cs_e = rtf_cs_e[idx_band]

    return f_rtf, rtf_cs_l, rtf_cs_e
