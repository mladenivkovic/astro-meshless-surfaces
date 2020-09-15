#!/usr/bin/env python3

###########################################################################################
#  package:   astro-meshless-surfaces
#  file:      meshless.py
#  brief:     main file, contains the interesting routines.
#  copyright: GPLv3
#             Copyright (C) 2019 EPFL (Ecole Polytechnique Federale de Lausanne)
#             LASTRO - Laboratory of Astrophysics of EPFL
#  author:    Mladen Ivkovic <mladen.ivkovic@hotmail.com>
#
# This file is part of astro-meshless-surfaces.
###########################################################################################

"""
A module containing common routines for the meshless effective area 
visualisiation with 2d datasets.
"""

from .kernels import *
from .particles import *
from .optional_packages import jit, prange

from typing import Union
import numpy as np


def Aij_Hopkins(
    pind: int,
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    m: np.ndarray,
    rho: np.ndarray,
    tree: Union[None, cKDTree] = None,
    kernel="cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic=True,
):
    """
    Compute A_ij as defined by Hopkins 2015

    Parameters
    ----------

    pind:   int
        particle index that you want solution for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    m: numpy.ndarray
        particle masses

    rho: numpy.ndarray
        particle densities

    tree: scipy.spatial.cKDTree or None
        tree used for neighbour searches. If none, one will be generated.

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    A_ij: np.ndarray
        array of A_ij, containing x and y component for every neighbour j of 
        particle i. Returns only neighbours within compact support radius of
        particle i.
    """

    # find neighbours
    tree, nbors = find_neighbours(pind, x, y, H, tree=tree, L=L, periodic=periodic)
    nneigh = nbors.shape[0]
    xj = x[nbors]
    yj = y[nbors]

    # -------------------------------------------------------
    # Part 1: For particle at x_i (Our chosen particle)
    # -------------------------------------------------------

    # compute psi_j(x_i)
    psi_j = compute_psi_j(
        x[pind], y[pind], xj, yj, H[pind], kernel=kernel, L=L, periodic=periodic
    )

    # normalize psi_j
    omega_xi = np.sum(psi_j) + psi(
        0.0, 0.0, 0.0, 0.0, H[pind], kernel=kernel, periodic=periodic
    )
    psi_j /= omega_xi
    psi_j = np.atleast_1d(psi_j)

    # compute B_i
    B_i = get_matrix(x[pind], y[pind], xj, yj, psi_j, L=L, periodic=periodic)

    # compute psi_tilde_j(x_i)
    psi_tilde_j = np.empty(nneigh * 2).reshape((nneigh, 2))
    for i in prange(nneigh):
        dx = np.array([xj[i] - x[pind], yj[i] - y[pind]])
        psi_tilde_j[i] = np.dot(B_i, dx) * psi_j[i]

    # --------------------------------------------------------------------------
    # Part 2: values of psi/psi_tilde of particle i at neighbour positions x_j
    # --------------------------------------------------------------------------

    psi_i = np.zeros(nneigh)  # psi_i(xj)
    psi_tilde_i = np.empty(nneigh * 2).reshape((nneigh, 2))  # psi_tilde_i(x_j)

    for j in prange(nneigh):
        nj = nbors[j]
        # first compute all psi(xj) from neighbour's neighbours to get weight omega
        tree, newneigh = find_neighbours(nj, x, y, H, tree=tree, L=L, periodic=periodic)
        xk = x[newneigh]
        yk = y[newneigh]
        psi_k = np.zeros(newneigh.shape[0])
        for k, nn in enumerate(newneigh):
            psi_k = compute_psi_j(
                x[nj], y[nj], xk, yk, H[nj], kernel, L=L, periodic=periodic
            )
            if nn == pind:
                # store psi_i, which is the psi for the particle whe chose at
                # position xj; psi_i(xj)
                psi_i[j] = psi_k[k]

        omega_xj = np.sum(psi_k) + psi(
            0.0, 0.0, 0.0, 0.0, H[nj], kernel=kernel, L=L, periodic=periodic
        )

        psi_i[j] /= omega_xj
        psi_k /= omega_xj

        # now compute B_j^{\alpha \beta}
        B_j = get_matrix(x[nj], y[nj], xk, yk, psi_k, L=L, periodic=periodic)

        # get psi_i_tilde(x = x_j)
        dx, dy = get_dx(x[pind], x[nj], y[pind], y[nj], L=L, periodic=periodic)
        dvecx = np.array([dx, dy])
        psi_tilde_i[j] = np.dot(B_j, dvecx) * psi_i[j]

    # -------------------------------
    # Part 3: Compute A_ij
    # -------------------------------

    A_ij = np.empty(nneigh * 2).reshape((nneigh, 2))

    for i in prange(nneigh):
        n = nbors[i]
        A_ij[i] = V(pind, m, rho) * psi_tilde_j[i] - V(n, m, rho) * psi_tilde_i[i]

    return A_ij


def Aij_Hopkins_v2(
    pind: int,
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    m: np.ndarray,
    rho: np.ndarray,
    tree: Union[None, cKDTree],
    kernel: str = "cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic=True,
):
    """
    Compute A_ij as defined by Hopkins 2015, second version.
    Very slow, as it computes psi_k(x_l) for ALL k, l first.
    Mainly intended for debugging and consistency checks.
    Not intended for use.

    Parameters
    ----------

    pind:   int
        particle index that you want solution for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    m: numpy.ndarray
        particle masses

    rho: numpy.ndarray
        particle densities

    tree: scipy.spatial.cKDTree or None
        tree used for neighbour searches. If none, one will be generated.

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    A_ij: np.ndarray
        array of A_ij, containing x and y component for every neighbour j of particle i
    """

    npart = x.shape[0]

    # compute all psi_k(x_l) for all l, k
    # first index: index k of psi: psi_k(x)
    # second index: index of x_l: psi(x_l)
    psi_k_at_l = np.zeros(npart * npart).reshape((npart, npart))

    for k in prange(npart):
        for l in range(k, npart):
            # kernels are symmetric in x_i, x_j, but h can vary.
            # So we can't do a symmetric thingy.
            psi_k_at_l[k, l] = psi(
                x[l], y[l], x[k], y[k], H[l], kernel=kernel, L=L, periodic=periodic,
            )
            psi_k_at_l[l, k] = psi(
                x[l], y[l], x[k], y[k], H[k], kernel=kernel, L=L, periodic=periodic,
            )

    tree, neighbours, nneigh = get_neighbours_for_all(
        x, y, H, tree=tree, L=L, periodic=periodic
    )
    omega = np.zeros(npart)

    for l in prange(npart):
        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[:, l])
        # omega_k = sum_l W(x_k - x_l, h_k) = sum_l psi_l(x_k) as it is currently stored in memory

    # normalize psi's and convert to float for linalg module
    for k in prange(npart):
        psi_k_at_l[:, k] /= omega[k]

    # compute all matrices B_k
    B_k = np.empty(npart * 2 * 2).reshape((npart, 2, 2))
    for k in prange(npart):
        nbors = neighbours[k][: nneigh[k]]
        # nbors now contains all neighbours l
        B_k[k] = get_matrix(
            x[k], y[k], x[nbors], y[nbors], psi_k_at_l[nbors, k], L=L, periodic=periodic
        )

    # compute all psi_tilde_k at every l
    psi_tilde_k_at_l = np.zeros(npart * npart * 2).reshape((npart, npart, 2))
    for k in prange(npart):
        for l in range(npart):
            dx, dy = get_dx(x[k], x[l], y[k], y[l], L=L, periodic=periodic)
            dvecx = np.array([dx, dy])
            psi_tilde_k_at_l[k, l] = np.dot(B_k[l], dvecx) * psi_k_at_l[k, l]

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind][: nneigh[pind]]

    A_ij = np.zeros(nneigh[pind] * 2).reshape((nneigh[pind], 2))

    for i in prange(nneigh[pind]):
        j = nbors[i]
        A_ij[i] = (
            V(pind, m, rho) * psi_tilde_k_at_l[j, pind]
            - V(j, m, rho) * psi_tilde_k_at_l[pind, j]
        )

    return A_ij


#  @jit(nopython=False, parallel=True, forceobj=True)
def Aij_Ivanova_all(
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    tree: Union[None, cKDTree] = None,
    kernel: str = "cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Compute A_ij as defined by Ivanova 2013, using the discretization by Taylor
    expansion as Hopkins does it. Use analytical expressions for the
    gradient of the kernels instead of the matrix representation.
    This function computes the effective surfaces of all particles for all their
    respective neighbours.


    Parameters
    ----------
    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    tree: scipy.spatial.cKDTree or None
        tree used for neighbour searches. If none, one will be generated.

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    A_ij: np.ndarray
        array of A_ij, containing x and y component for every neighbour j of every particle i
        ! important: indices i and j are switched compared to definition in Ivanova 2013

    neighbours: np.ndarray
        array of shape (npart, maxneigh) of neighbour indices for every particle i
    """

    npart = x.shape[0]

    tree, neighbours, nneigh = get_neighbours_for_all(
        x, y, H, tree=tree, L=L, periodic=periodic
    )

    maxneigh = neighbours.shape[1]

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    W_j_at_i, omega = get_W_j_at_i(
        x, y, H, neighbours, nneigh, kernel=kernel, L=L, periodic=periodic
    )

    # get gradients
    grad_psi_j_at_i = get_grad_psi_j_at_i_analytical(
        x, y, H, omega, W_j_at_i, neighbours, nneigh, kernel=kernel, periodic=periodic,
    )

    A_ij = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))

    # precompute all volumes
    Vol = np.zeros(npart)
    for i in range(npart):
        Vol[i] = 1.0 / omega[i]

    # now compute A_ij for all neighbours j of i
    for i in prange(npart):

        nbors = neighbours[i]

        V_i = Vol[i]

        for j, nj in enumerate(nbors):

            A_ij[i, j] += V_i * grad_psi_j_at_i[i, j]

            if i in neighbours[nj]:  # j may be neighbour of i, but not vice versa
                iind = neighbours[nj] == i
                iind[nneigh[nj] :] = False
                A_ij[nj, iind] -= V_i * grad_psi_j_at_i[i, j]

    return A_ij, neighbours


def Aij_Ivanova(
    pind: int,
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    tree: Union[None, cKDTree] = None,
    kernel="cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Compute A_ij as defined by Ivanova 2013, using the discretization by Taylor
    expansion as Hopkins does it. Use analytical expressions for the
    gradient of the kernels instead of the matrix representation.


    Parameters
    ----------

    pind: int
        particle index that you want solution for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    tree: scipy.spatial.cKDTree or None
        tree used for neighbour searches. If none, one will be generated.

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    A_ij: np.ndarray
        array of A_ij, containing x and y component for every neighbour j of particle i
        !! important: indices i and j are switched compared to definition in Ivanova 2013
    """

    # find neighbours
    tree, nbors = find_neighbours(pind, x, y, H, tree=tree, L=L, periodic=periodic)
    nneigh = nbors.shape[0]
    xj = x[nbors]
    yj = y[nbors]

    # -------------------------------------------------------
    # Part 1: For particle at x_i (Our chosen particle)
    # -------------------------------------------------------

    # compute W_j(x_i)
    W_j_at_i = compute_psi_j(
        x[pind], y[pind], xj, yj, H[pind], kernel=kernel, L=L, periodic=periodic
    )

    # normalization for psi_j
    omega_xi = np.sum(W_j_at_i) + psi(
        0.0, 0.0, 0.0, 0.0, H[pind], kernel=kernel, L=L, periodic=periodic
    )

    # get gradient of psi_j's at x_i
    grad_psi_j_at_i = np.zeros(nneigh * 2).reshape((nneigh, 2))
    grad_W_j_at_i = np.zeros(nneigh * 2).reshape((nneigh, 2))
    sum_grad_Wj_at_i = np.zeros(2)
    for j in prange(nneigh):
        nj = nbors[j]
        # order is important here!
        dx, dy = get_dx(x[pind], x[nj], y[pind], y[nj], L=L, periodic=periodic)
        r = np.sqrt(dx ** 2 + dy ** 2)
        dwdr = dWdr(r / H[pind], H[pind], kernel)

        grad_W_j_at_i[j, 0] = dwdr * dx / r
        grad_W_j_at_i[j, 1] = dwdr * dy / r

        sum_grad_Wj_at_i += grad_W_j_at_i[j]

    for j in prange(nneigh):
        grad_psi_j_at_i[j, :] = (
            grad_W_j_at_i[j, :] / omega_xi
            - W_j_at_i[j] * sum_grad_Wj_at_i[:] / omega_xi ** 2
        )

    # --------------------------------------------------------------------------
    # Part 2: get data for all neighbours
    # --------------------------------------------------------------------------

    W_i_at_j = compute_psi_i(
        x[pind], y[pind], xj, yj, H[nbors], kernel=kernel, L=L, periodic=periodic
    )
    omega_xj = np.zeros(nneigh)
    grad_W_i_at_j = np.zeros(nneigh * 2).reshape((nneigh, 2))
    sum_grad_Wk_at_j = np.zeros(nneigh * 2).reshape((nneigh, 2))

    for j in prange(nneigh):

        # first get omega
        nj = nbors[j]
        tree, newneigh = find_neighbours(nj, x, y, H, tree=tree, L=L, periodic=periodic)
        xk = x[newneigh]
        yk = y[newneigh]
        W_k_at_j = compute_psi_j(
            x[nj], y[nj], xk, yk, H[nj], kernel, L=L, periodic=periodic
        )

        omega_xj[j] = np.sum(W_k_at_j) + psi(
            0.0, 0.0, 0.0, 0.0, H[nj], kernel=kernel, L=L, periodic=periodic
        )

        for k, nk in enumerate(newneigh):

            # now get the gradient
            # order is important here!
            dx, dy = get_dx(x[nj], x[nk], y[nj], y[nk], L=L, periodic=periodic)
            r = np.sqrt(dx ** 2 + dy ** 2)
            dwdr = dWdr(r / H[nj], H[nj], kernel)

            sum_grad_Wk_at_j[j, 0] += dwdr * dx / r
            sum_grad_Wk_at_j[j, 1] += dwdr * dy / r

            if nk == pind:
                grad_W_i_at_j[j, 0] = dwdr * dx / r
                grad_W_i_at_j[j, 1] = dwdr * dy / r

    grad_psi_i_at_j = np.zeros(nneigh * 2).reshape((nneigh, 2))
    for j in prange(nneigh):
        grad_psi_i_at_j[j, :] = (
            grad_W_i_at_j[j, :] / omega_xj[j]
            - W_i_at_j[j] * sum_grad_Wk_at_j[j, :] / omega_xj[j] ** 2
        )

    # -------------------------------
    # Part 3: Compute A_ij
    # -------------------------------

    A_ij = np.empty(nneigh * 2).reshape((nneigh, 2))

    for j in prange(nneigh):
        A_ij[j] = (
            1.0 / omega_xi * grad_psi_j_at_i[j] - 1.0 / omega_xj[j] * grad_psi_i_at_j[j]
        )

    return A_ij


@jit(nopython=True)
def x_ij(
    pind: int,
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    nbors: Union[np.ndarray, None] = None,
    which: Union[int, None] = None,
):
    """
    compute x_ij for all neighbours of particle with index pind.
    If `which` is given, instead compute only for specific particle
    where `which` is that particle's index.

    Parameters
    ----------

    pind: int
        particle index that you want solution for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    nbors: np.ndarray or None
        array of particle with index `pind`'s neighbours.
        If `which` is `None`, `nbors` must be provided.

    which: int or None
        for which part


    Returns
    -------
    
    x_ij: np.ndarray
        array of particle x_ij's. Is always 2D.

    """

    if which is not None:
        hfact = H[pind] / (H[pind] + H[which])
        xij = np.array(
            [
                x[pind] - hfact * (x[pind] - x[which]),
                y[pind] - hfact * (y[pind] - y[which]),
            ]
        )
        return xij

    elif nbors is not None:
        xij = np.empty(len(nbors) * 2).reshape(len(nbors), 2)
        for i, n in enumerate(nbors):
            hfact = H[pind] / (H[pind] + H[n])
            xij[i] = np.array(
                [x[pind] - hfact * (x[pind] - x[n]), y[pind] - hfact * (y[pind] - y[n])]
            )

        return xij

    else:
        raise ValueError(
            "Gotta give me either an array of neighbours or a single particle info for x_ij"
        )


@jit(nopython=True, parallel=True)
def get_W_j_at_i(
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    neighbours: np.ndarray,
    nneigh: np.ndarray,
    kernel: str = "cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Evaluate kernels between all neighbour pairs at their positions,
    and find normalisation omega.


    Parameters
    ----------
    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    neighbours: np.ndarray
        array of particle neighbour indexes for all particles

    nneigh: np.ndarray
        array containing how many neighbours each particle has

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    W_j_at_i: np.ndarray
        array of shape (part, maxneigh) with evaluated kernel values
        for all neighbours.

    omega: np.ndarray
        array of normalisations of psi

    """

    npart = neighbours.shape[0]
    maxneigh = neighbours.shape[1]

    W_j_at_i = np.zeros(npart * maxneigh).reshape(npart, maxneigh)
    omega = np.zeros(npart)

    for j in prange(npart):
        for i in range(nneigh[j]):
            ind_n = neighbours[j][i]
            # kernels are symmetric in x_i, x_j, but h can vary.
            # So we can't make this symmetric.
            W_j_at_i[j, i] = psi(
                x[j],
                y[j],
                x[ind_n],
                y[ind_n],
                H[j],
                kernel=kernel,
                L=L,
                periodic=periodic,
            )
            omega[j] += W_j_at_i[j, i]

        # add self-contribution
        omega[j] += psi(0.0, 0.0, 0.0, 0.0, H[j], kernel=kernel, L=L, periodic=periodic)
    return W_j_at_i, omega


@jit(nopython=True, parallel=True)
def get_grad_psi_j_at_i_analytical(
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    omega: np.ndarray,
    W_j_at_i: np.ndarray,
    neighbours: np.ndarray,
    nneigh: np.ndarray,
    kernel="cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Compute `\\nabla \\psi_k (x_l)` for all particles k and l

    Parameters
    ----------

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    omega: np.ndarray
        array of normalisations of psi

    W_j_at_i: np.ndarray
        W_k(x_l) : npart x npart array

    neighbours: np.ndarray
        array of particle neighbour indexes for all particles

    nneigh: np.ndarray
        array containing how many neighbours each particle has

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    grad_psi_j_at_i: np.ndarray
        npart x max_neigh x 2 array; grad psi_j (x_i) for 
        all i,j for both x and y direction
    """
    npart = x.shape[0]
    maxneigh = neighbours.shape[1]

    # gradient of psi_j at neighbour i's position
    grad_psi_j_at_i = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))
    # gradient of W_j at neighbour i's position
    grad_W_j_at_i = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))
    # gradient sum for the same h_i
    sum_grad_W = np.zeros(npart * 2).reshape((npart, 2))

    for j in prange(npart):
        for i in range(nneigh[j]):
            ind_n = neighbours[j, i]
            dx, dy = get_dx(x[j], x[ind_n], y[j], y[ind_n], L=L, periodic=periodic)
            r = np.sqrt(dx ** 2 + dy ** 2)

            dwdr = dWdr(r / H[j], H[j], kernel)

            grad_W_j_at_i[j, i, 0] = dwdr * dx / r
            grad_W_j_at_i[j, i, 1] = dwdr * dy / r

            sum_grad_W[j] += grad_W_j_at_i[j, i]

    # finish computing the gradients: Need W(r, h), which is currently stored as psi
    for j in prange(npart):
        for i in range(nneigh[j]):
            grad_psi_j_at_i[j, i, :] = (
                grad_W_j_at_i[j, i, :] / omega[j]
                - W_j_at_i[j, i] * sum_grad_W[j, :] / omega[j] ** 2
            )

    return grad_psi_j_at_i


@jit(nopython=True, parallel=True)
def compute_psi_j(
    xi: float,
    yi: float,
    xj: np.ndarray,
    yj: np.ndarray,
    Hi: float,
    kernel: str = "cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Compute psi_j(x_i) for all j, i.e.
    W(|xj - xi|, Hi)


    Parameters
    ----------
    
    xi: float
        x position of particle i

    yi: float
        y position of particle i

    xj: np.ndarray
        x positions of particles j

    yj: np.ndarray
        y positions of particles j

    Hi: float
        kernel support radius/radii at position xi, yi

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    psi_j: np.ndarray
        numpy array of psi_j(x)
    """

    psi_j = np.zeros(xj.shape[0])

    for j in prange(xj.shape[0]):
        psi_j[j] = psi(xi, yi, xj[j], yj[j], Hi, kernel=kernel, L=L, periodic=periodic)

    return psi_j


@jit(nopython=True, parallel=True)
def compute_psi_i(
    xi: float,
    yi: float,
    xj: np.ndarray,
    yj: np.ndarray,
    Hj: np.ndarray,
    kernel: str = "cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Compute psi_i(x_j) for all j, i.e.
    W(|xj - xi|, Hj)


    Parameters
    ----------
    
    xi: float
        x position of particle i

    yi: float
        y position of particle i

    xj: np.ndarray
        x positions of particles j

    yj: np.ndarray
        y positions of particles j

    Hj: np.ndarray
        array of kernel support radius/radii at position xj, yj

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    psi_i: np.ndarray
        numpy array of psi_i(xj) for all j
    """

    psi_i = np.zeros(xj.shape[0])

    for j in prange(xj.shape[0]):
        psi_i[j] = psi(
            xj[j], yj[j], xi, yi, Hj[j], kernel=kernel, L=L, periodic=periodic
        )

    return psi_i


@jit(nopython=True)
def psi(
    xi: float,
    yi: float,
    xj: float,
    yj: float,
    Hi: float,
    kernel: str = "cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    UNNORMALIZED Volume fraction at position x of some particle
    with coordinates xi, yi, smoothing length h(x)

    i.e. psi_i(x) = W([x - xi, y - yi], h(x))


    Parameter
    ---------

    xi: float
        particle xi position

    yi: float
        particle yi position

    xj: float
        particle xj position

    yj: float
        particle yj position

    Hi: float
        kernel support radius at position (x, y)

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------
    psi_i: float
        UNNORMALIZED volume fraction psi_i(x)
    """

    dx, dy = get_dx(xi, xj, yi, yj, L=L, periodic=periodic)

    q = np.sqrt(dx ** 2 + dy ** 2) / Hi

    return W(q, Hi, kernel)


@jit(nopython=True, parallel=True)
def get_matrix(
    xi: float,
    yi: float,
    xj: np.ndarray,
    yj: np.ndarray,
    psi_j: np.ndarray,
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Get B_i ^{alpha beta}

    Parameters
    ----------
    
    xi: float
        x position of particle i

    yi: float
        y position of particle i

    xj: np.ndarray
        x positions of particles j

    yj: np.ndarray
        y positions of particles j

    psi_j: np.ndarray  
        volume fraction of neighbours at position x_i; psi_j(x_i)
 
    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------
    B: np.matrix
        2 x 2 matrix
    """

    nn = xj.shape[0]
    dx = np.zeros(nn)
    dy = np.zeros(nn)

    for i in range(nn):
        dx[i], dy[i] = get_dx(xj[i], xi, yj[i], yi, L=L, periodic=periodic)

    E00 = np.sum(dx * dx * psi_j)
    E01 = np.sum(dx * dy * psi_j)
    E11 = np.sum(dy * dy * psi_j)

    E = np.array([[E00, E01], [E01, E11]])
    B = np.linalg.inv(E)
    return B


@jit(nopython=True)
def h_of_x(
    xx: float,
    yy: float,
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    kernel="cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Compute h(x) at position (xx, yy), where there is
    not necessariliy a particle
    by approximating it as h(x) = sum_j h_j * psi_j(x)


    Parameters
    ----------

    xx: float
        x position to compute for

    yy: float
        y position to compute for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    h: numpy.ndarray
        kernel support radii

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool


    Returns
    -------

    hh: float
        smoothing length at position (xx, yy)
    """

    # TODO: rewrite

    nbors = find_neighbours_arbitrary_x(xx, yy, x, y, h, L=L, periodic=periodic)

    xj = x[nbors]
    yj = y[nbors]
    hj = h[nbors]

    psi_j = compute_psi_j(xx, yy, xj, yj, hj, kernel=kernel, L=L, periodic=periodic)
    psi_j /= np.sum(psi_j)

    hh = np.sum(hj * psi_j)

    return hh
