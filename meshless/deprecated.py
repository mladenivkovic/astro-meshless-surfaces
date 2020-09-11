#!/usr/bin/env python3

###########################################################################################
#  package:   astro-meshless-surfaces
#  file:      deprecated.py
#  copyright: GPLv3
#             Copyright (C) 2019 EPFL (Ecole Polytechnique Federale de Lausanne)
#             LASTRO - Laboratory of Astrophysics of EPFL
#  author:    Mladen Ivkovic <mladen.ivkovic@epfl.ch>
#
# This file is part of astro-meshless-surfaces.
###########################################################################################

# =========================================================
# This file contains deprecated functions that I
# still have a look into every now and then and am
# too lazy to look in the git history every time.
# =========================================================


# =====================================================================================================================
def Aij_Ivanova_approximate_gradients(
    pind, x, y, h, m, rho, kernel="cubic_spline", fact=1, L=1, periodic=True
):
    # =====================================================================================================================
    """
    Compute A_ij as defined by Ivanova 2013
    pind:           particle index for which to work for (The i in A_ij)
    x, y, m, rho:   full data arrays as read in from hdf5 file
    h:              kernel support radius array
    kernel:         which kernel to use
    fact:           factor for h for limit of neighbour search; neighbours are closer than fact*h
    L:              boxsize
    periodic:       whether to assume periodic boundaries

    returns:
        A_ij: array of A_ij, containing x and y component for every neighbour j of particle i
    """

    print(
        "Warning: This method hasn't been checked thoroughly in a while. Results might not be right."
    )

    npart = x.shape[0]

    neighbours = [[] for i in x]

    for l in range(npart):
        # find and store all neighbours;
        neighbours[l] = find_neighbours(l, x, y, h, fact=fact, L=L, periodic=periodic)

    # compute all psi_k(x_l) for all l, k
    # first index: index k of psi: psi_k(x)
    # second index: index of x_l: psi(x_l)
    psi_k_at_l = np.zeros((npart, npart), dtype=my_float)

    for k in range(npart):
        for l in neighbours[k]:
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_k_at_l[k, l] = psi(
                x[l],
                y[l],
                x[k],
                y[k],
                h[l],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )

        # self contribution part: k = l +> h[k] = h[l], so use h[k] here
        psi_k_at_l[k, k] = psi(
            0, 0, 0, 0, h[k], kernel=kernel, fact=fact, L=L, periodic=periodic
        )

    omega = np.zeros(npart, dtype=my_float)

    for l in range(npart):
        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[neighbours[l], l]) + psi_k_at_l[l, l]
        # omega_k = sum_l W(x_k - x_l) = sum_l psi_l(x_k) as it is currently stored in memory

    # normalize psi's and convert to float for linalg module
    for k in range(npart):
        psi_k_at_l[:, k] /= omega[k]
    if my_float != np.float:
        psi_k_at_l = psi_k_at_l.astype(np.float)

    # compute all matrices B_k
    B_k = np.zeros((npart), dtype=np.matrix)
    for k in range(npart):
        nbors = neighbours[k]
        # nbors now contains all neighbours l
        B_k[k] = get_matrix(
            x[k], y[k], x[nbors], y[nbors], psi_k_at_l[nbors, k], L=L, periodic=periodic
        )

    # compute all psi_tilde_k at every l
    psi_tilde_k_at_l = np.zeros((npart, npart, 2))
    for k in range(npart):
        # can't just go over neighbours here!
        for l in range(npart):

            dx = np.array([x[k] - x[l], y[k] - y[l]])
            psi_tilde_k_at_l[k, l] = np.dot(B_k[l], dx) * psi_k_at_l[k, l]

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]

    A_ij = np.zeros((len(nbors), 2), dtype=np.float)

    for i, j in enumerate(nbors):

        A = np.array([0.0, 0.0])
        for k in range(npart):
            psi_i_xk = psi_k_at_l[pind, k]
            psi_j_xk = psi_k_at_l[j, k]
            Vk = V(k, m, rho)
            temp = np.array([0.0, 0.0])
            for l in range(npart):
                psi_i_xl = psi_k_at_l[pind, l]
                psi_j_xl = psi_k_at_l[j, l]
                psi_tilde_l = psi_tilde_k_at_l[l, k]

                temp += (psi_j_xk * psi_i_xl - psi_i_xk * psi_j_xl) * psi_tilde_l

            temp *= Vk
            A += temp

        A_ij[i] = A

    # return -A_ij: You will actually use A_ji . F in the formula
    # for the hydrodynamics, not A_ij . F
    return -A_ij


# ==================================================================================================================
def Aij_Ivanova_analytical_gradients(
    pind, x, y, h, m, rho, kernel="cubic_spline", fact=1, L=1, periodic=True
):
    # ==================================================================================================================
    """
    Compute A_ij as defined by Ivanova 2013. Use analytical expressions for the
    gradient of the kernels instead of the matrix representation.
    Not the recommended way to do it, needs extra computation.

    pind:           particle index for which to work for (The i in A_ij)
    x, y, m, rho:   full data arrays as read in from hdf5 file
    h:              kernel support radius array
    kernel:         which kernel to use
    fact:           factor for h for limit of neighbour search; neighbours are closer than fact*h
    L:              boxsize
    periodic:       whether to assume periodic boundaries

    returns:
        A_ij: array of A_ij, containing x and y component for every neighbour j of particle i
    """

    print(
        "Warning: This method hasn't been checked thoroughly in a while. Results might not be right."
    )

    npart = x.shape[0]

    neighbours = [[] for i in x]

    for l in range(npart):
        # find and store all neighbours;
        neighbours[l] = find_neighbours(l, x, y, h, fact=fact, L=L, periodic=periodic)

    # compute all psi_k(x_l) for all l, k
    # first index: index k of psi: psi_k(x)
    # second index: index of x_l: psi(x_l)
    psi_k_at_l = np.zeros((npart, npart), dtype=my_float)

    for k in range(npart):
        for l in neighbours[k]:
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_k_at_l[k, l] = psi(
                x[l],
                y[l],
                x[k],
                y[k],
                h[l],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )

        # self contribution part: k = l +> h[k] = h[l], so use h[k] here
        psi_k_at_l[k, k] = psi(
            0, 0, 0, 0, h[k], kernel=kernel, fact=fact, L=L, periodic=periodic
        )

    omega = np.zeros(npart, dtype=my_float)

    for l in range(npart):
        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[neighbours[l], l]) + psi_k_at_l[l, l]
        # omega_k = sum_l W(x_k - x_l) = sum_l psi_l(x_k) as it is currently stored in memory

    grad_psi_k_at_l = get_grad_psi_j_at_i_analytical(
        x, y, h, omega, psi_k_at_l, neighbours, kernel=kernel, fact=fact
    )

    # normalize psi's and convert to float for linalg module
    for k in range(npart):
        psi_k_at_l[:, k] /= omega[k]
    if my_float != np.float:
        psi_k_at_l = psi_k_at_l.astype(np.float)

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]

    A_ij = np.zeros((len(nbors), 2), dtype=np.float)

    for i, j in enumerate(nbors):

        A = np.array([0.0, 0.0], dtype=np.float)
        for k in range(npart):
            #  for
            psi_i_xk = psi_k_at_l[pind, k]
            psi_j_xk = psi_k_at_l[j, k]
            grad_psi_i_xk = grad_psi_k_at_l[pind, k]
            grad_psi_j_xk = grad_psi_k_at_l[j, k]
            V_k = 1 / omega[k]

            A += (psi_j_xk * grad_psi_i_xk - psi_i_xk * grad_psi_j_xk) * V_k

        A_ij[i] = A

    # return -A_ij: You will actually use A_ji . F in the formula
    # for the hydrodynamics, not A_ij . F
    return -A_ij


# =========================================================================================================================
def Integrand_Aij_Ivanova(
    iind,
    jind,
    xx,
    yy,
    hh,
    x,
    y,
    h,
    m,
    rho,
    kernel="cubic_spline",
    fact=1,
    L=1,
    periodic=True,
):
    # =========================================================================================================================
    """
    Compute the effective area integrand for the particles iind jind at
    the positions xx, yy

    (Note that this should be integrated to get the proper effective surface)

    integrand A_ij  = psi_j(x) \nabla psi_i(x) - psi_i (x) \nabla psi_j(x)
                    = sum_k [ psi_j(x_k) psi_i(x) - psi_i(x_k) psi_j(x) ] * psi_tilde_k(x)
                    = psi_i(x) * sum_k psi_j(x_k) * psi_tilde_k(x) - psi_j(x) * sum_k psi_i(x_k) * psi_tilde_k(x)

    The last line is what is actually computed here, with the expression for the gradient
    inserted.


    iind, jind:     particle index for which to work for (The i and j in A_ij)
    xx, yy:         position at which to evaluate
    hh:             kernel support radius at xx, yy
    x, y, m, rho:   full data arrays as read in from hdf5 file
    h:              kernel support radius array
    kernel:         which kernel to use
    fact:           factor for h for limit of neighbour search; neighbours are closer than fact*h
    L:              boxsize
    periodic:       whether to assume periodic boundaries

    returns:
        A_ij: array of integrands A_ij, containing x and y component for every neighbour j of particle i



    """

    print(
        "Warning: This method hasn't been checked thoroughly in a while. Results might not be right."
    )

    nbors = find_neighbours_arbitrary_x(
        xx, yy, x, y, h, fact=fact, L=L, periodic=periodic
    )

    xk = x[nbors]
    yk = y[nbors]
    hk = h[nbors]

    # ----------------------
    # compute psi_i/j(x)
    # ----------------------

    # compute all psi(x)
    psi_x = compute_psi(
        xx, yy, xk, yk, hh, kernel=kernel, fact=fact, L=L, periodic=periodic
    )

    # normalize psis
    omega = np.sum(psi_x)
    psi_x /= omega
    if my_float != np.float:
        psi_x = psi_x.astype(np.float)

    # find where psi_i and psi_j are in that array
    try:
        inb = nbors.index(iind)
        psi_i_of_x = psi_x[inb]  # psi_i(xx, yy)
    except ValueError:
        psi_i_of_x = 0  # can happen for too small smoothing lengths
        print("Exception in psi_i_of_x: iind not found in neighbour list")

    try:
        jnb = nbors.index(jind)
        psi_j_of_x = psi_x[jnb]  # psi_j(xx, yy)
    except ValueError:
        psi_j_of_x = 0  # can happen for too small smoothing lengths
        print("Exception in psi_j_of_x: jind not found in neighbour list")

    # ------------------------------------------------
    # Compute psi_i/j(x_k) at neighbouring positions
    # ------------------------------------------------

    psi_i_xk = [None for n in nbors]
    psi_j_xk = [None for n in nbors]

    omegas = [0, 0]

    for i, n in enumerate([iind, jind]):
        # first compute all psi(xl) from neighbour's neighbours to get weights omega
        nneigh = find_neighbours(n, x, y, h, fact=fact, L=L, periodic=periodic)

        xl = x[nneigh]
        yl = y[nneigh]

        for j, nn in enumerate(nneigh):
            psi_l = compute_psi(
                x[n],
                y[n],
                xl,
                yl,
                h[n],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )

        omegas[i] = np.sum(psi_l) + psi(
            0, 0, 0, 0, h[iind], kernel=kernel, fact=fact, L=L, periodic=periodic
        )

    # now compute psi_i/j(x_k)
    for i, n in enumerate(nbors):
        psi_i_xk[i] = (
            psi(
                xk[i],
                yk[i],
                x[iind],
                y[iind],
                h[iind],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )
            / omegas[0]
        )
        psi_j_xk[i] = (
            psi(
                xk[i],
                yk[i],
                x[jind],
                y[jind],
                h[jind],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )
            / omegas[1]
        )

    # ---------------------------------------------
    # Compute psi_tilde_k(x)
    # ---------------------------------------------

    # compute matrix B
    B = get_matrix(xx, yy, xk, yk, psi_x, L=L, periodic=periodic)

    # compute psi_tilde_k(xx)
    psi_tilde_k = np.empty((2, xk.shape[0]))
    for i in range(xk.shape[0]):
        dx = np.array([xk[i] - xx, yk[i] - yy])
        psi_tilde_k[:, i] = np.multiply(np.dot(B, dx), psi_x[i])

    # ----------------------------------
    # Compute A_ij
    # ----------------------------------

    sum_i = np.sum(np.multiply(psi_tilde_k, psi_i_xk), axis=1)
    sum_j = np.sum(np.multiply(psi_tilde_k, psi_j_xk), axis=1)

    A_ij = psi_i_of_x * sum_j - psi_j_of_x * sum_i

    return A_ij
