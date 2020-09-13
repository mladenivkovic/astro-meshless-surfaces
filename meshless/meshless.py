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
from .optional_packages import jit, List

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
        L: List = [1.0, 1.0],
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
        array of A_ij, containing x and y component for every neighbour j of particle i
    """

    tree, nbors = find_neighbours(pind, x, y, H, tree=tree, L=L, periodic=periodic)

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
    omega_xi = np.sum(psi_j) + psi(0.0, 0.0, 0.0, 0.0, H[pind], kernel)
    psi_j /= omega_xi
    psi_j = np.atleast_1d(psi_j)

    # compute B_i
    B_i = get_matrix(x[pind], y[pind], xj, yj, psi_j, L=L, periodic=periodic)

    # compute psi_tilde_j(x_i)
    psi_tilde_j = np.empty(len(nbors) * 2).reshape((len(nbors), 2))
    for i, n in enumerate(nbors):
        dx = np.array([xj[i] - x[pind], yj[i] - y[pind]])
        psi_tilde_j[i] = np.dot(B_i, dx) * psi_j[i]

    # ---------------------------------------------------------------------------
    # Part 2: values of psi/psi_tilde of particle i at neighbour positions x_j
    # ---------------------------------------------------------------------------

    psi_i = np.zeros(len(nbors))  # psi_i(xj)
    psi_tilde_i = np.empty(len(nbors) * 2).reshape((len(nbors), 2))  # psi_tilde_i(x_j)

    for i, n in enumerate(nbors):
        # first compute all psi(xj) from neighbour's neighbours to get weight omega
        tree, nneigh = find_neighbours(n, x, y, H, tree=tree, L=L, periodic=periodic)
        xk = x[nneigh]
        yk = y[nneigh]
        nn = None
        psi_k = np.zeros(len(nneigh))
        for j, nn in enumerate(nneigh):
            psi_k = compute_psi_j(
                x[n], y[n], xk, yk, H[n], kernel, L=L, periodic=periodic
            )
            if nn == pind:
                # store psi_i, which is the psi for the particle whe chose at
                # position xj; psi_i(xj)
                psi_i[i] = psi_k[j]

        omega_xj = np.sum(psi_k)
        if nn is not None:
            omega_xj += psi(0, 0, 0, 0, H[nn], kernel)

        psi_i[i] /= omega_xj
        psi_k /= omega_xj

        # now compute B_j^{\alpha \beta}
        B_j = get_matrix(x[n], y[n], xk, yk, psi_k, L=L, periodic=periodic)

        # get psi_i_tilde(x = x_j)
        dx = np.array([x[pind] - x[n], y[pind] - y[n]])
        psi_tilde_i[i] = np.dot(B_j, dx) * np.float(psi_i[i])

    # -------------------------------
    # Part 3: Compute A_ij
    # -------------------------------

    A_ij = np.empty(len(nbors) * 2).reshape((len(nbors), 2))

    for i, n in enumerate(nbors):
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
        L: List = [1.0, 1.0],
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

    for k in range(npart):
        for l in range(npart):
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_k_at_l[k, l] = psi(
                x[l], y[l], x[k], y[k], H[l], kernel=kernel, L=L, periodic=periodic,
            )

    tree, neighbours, maxneigh = get_neighbours_for_all(
        x, y, H, tree=tree, L=L, periodic=periodic
    )
    omega = np.zeros(npart)

    for l in range(npart):
        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[:, l])
        # omega_k = sum_l W(x_k - x_l, h_k) = sum_l psi_l(x_k) as it is currently stored in memory

    # normalize psi's and convert to float for linalg module
    for k in range(npart):
        psi_k_at_l[:, k] /= omega[k]

    # compute all matrices B_k
    B_k = List([np.array([[0.0, 0.0], [0.0, 0.0]]) for _ in range(npart)])
    for k in range(npart):
        nbors = neighbours[k]
        # nbors now contains all neighbours l
        psi_to_use = psi_k_at_l[nbors, k]
        B_k[k] = get_matrix(
            x[k], y[k], x[nbors], y[nbors], psi_to_use, L=L, periodic=periodic
        )

    # compute all psi_tilde_k at every l
    psi_tilde_k_at_l = np.zeros(npart * npart * 2).reshape((npart, npart, 2))
    for k in range(npart):
        for l in range(npart):
            dx, dy = get_dx(x[k], x[l], y[k], y[l], L=L, periodic=periodic)
            dvecx = np.array([dx, dy])
            psi_tilde_k_at_l[k, l] = np.dot(B_k[l], dvecx) * psi_k_at_l[k, l]

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]

    A_ij = np.zeros(len(nbors) * 2).reshape((len(nbors), 2))

    for i, j in enumerate(nbors):
        A_ij[i] = (
                V(pind, m, rho) * psi_tilde_k_at_l[j, pind]
                - V(j, m, rho) * psi_tilde_k_at_l[pind, j]
        )

    return A_ij


def Aij_Ivanova_all(
        x: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        tree: Union[None, cKDTree] = None,
        kernel: str = "cubic_spline",
        L: List = [1.0, 1.0],
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

    neighbours: list 
        list of lists of neighbour indices for every particle i
    """

    npart = x.shape[0]

    tree, neighbours, maxneigh = get_neighbours_for_all(
        x, y, H, tree=tree, L=L, periodic=periodic
    )

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    W_j_at_i, omega = get_W_j_at_i(
        x, y, H, neighbours, maxneigh, kernel=kernel, L=L, periodic=periodic
    )

    # get gradients
    grad_psi_j_at_i = get_grad_psi_j_at_i_analytical(
        x,
        y,
        H,
        omega,
        W_j_at_i,
        neighbours,
        maxneigh,
        kernel=kernel,
        periodic=periodic,
    )

    A_ij = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))

    # precompute all volumes
    Vol = np.zeros(npart)
    for i in range(npart):
        Vol[i] = 1.0 / omega[i]

    # now compute A_ij for all neighbours j of i
    for i in range(npart):

        nbors = neighbours[i]

        V_i = Vol[i]

        for j, nj in enumerate(nbors):

            A_ij[i, j] += V_i * grad_psi_j_at_i[i, j]
            if i in neighbours[nj]:  # j may be neighbour of i, but not vice versa
                iind = neighbours[nj].index(i)
                A_ij[nj, iind] -= V_i * grad_psi_j_at_i[i, j]

    return A_ij, neighbours


def Aij_Ivanova(
        pind: int,
        x: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        tree: Union[None, cKDTree] = None,
        kernel="cubic_spline",
        L: List = [1.0, 1.0],
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

    # first get neighbour data
    tree, neighbours, maxneigh = get_neighbours_for_all(
        x, y, H, tree=tree, L=L, periodic=periodic
    )

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    W_j_at_i, omega = get_W_j_at_i(
        x, y, H, neighbours, maxneigh, kernel=kernel, L=L, periodic=periodic
    )

    # get gradients
    grad_psi_j_at_i = get_grad_psi_j_at_i_analytical(
        x,
        y,
        H,
        omega,
        W_j_at_i,
        neighbours,
        maxneigh,
        kernel=kernel,
        periodic=periodic,
    )

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]
    A_ij = np.zeros(len(nbors) * 2).reshape((len(nbors), 2))

    V_i = 1.0 / omega[pind]

    for j, nj in enumerate(nbors):

        grad_psi_j_xi = grad_psi_j_at_i[pind, j]
        grad_psi_i_xj = 0.0
        if pind in neighbours[nj]:  # j may be neighbour of i, but not vice versa
            iind = neighbours[nj].index(pind)
            grad_psi_i_xj = grad_psi_j_at_i[nj, iind]

        V_j = 1.0 / omega[nj]

        A_ij[j] = V_i * grad_psi_j_xi - V_j * grad_psi_i_xj

    return A_ij


@jit(nopython=True)
def x_ij(
        pind: int,
        x: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        nbors: Union[List, None] = None,
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

    nbors: list or None
        list of particle with index `pind`'s neighbours.
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
            "Gotta give me either a list of neighbours or a single particle info for x_ij"
        )


@jit(nopython=True)
def get_W_j_at_i(
        x: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        neighbours: List,
        maxneigh: int,
        kernel: str = "cubic_spline",
        L: List = [1.0, 1.0],
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

    neighbours: List
        List of lists of particle neighbour indexes

    maxneigh: integer
        maximal number of neighbours any particle has

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    W_j_at_i: np.ndarray
        array of shape (npart, maxneigh) with evaluated kernel values
        for all neighbours.

    omega: np.ndarray
        array of normalisations of psi

    """

    npart = x.shape[0]

    W_j_at_i = np.zeros(npart * maxneigh).reshape(npart, maxneigh)
    omega = np.zeros(npart)

    for j in range(npart):
        nn = len(neighbours[j])
        for i in range(nn):
            ind_n = neighbours[j][i]
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
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


@jit(nopython=True)
def get_grad_psi_j_at_i_analytical(
        x: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        omega: np.ndarray,
        W_j_at_i: np.ndarray,
        neighbours: List,
        maxneigh: int,
        kernel="cubic_spline",
        L: List = [1.0, 1.0],
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

    neighbours: List
        list of lists of neighbour indices

    maxneigh: integer
        highest number of neighbours any particle has

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

    # gradient of psi_j at neighbour i's position
    grad_psi_j_at_i = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))
    # gradient of W_j at neighbour i's position
    grad_W_j_at_i = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))
    # gradient sum for the same h_i
    sum_grad_W = np.zeros(npart * 2).reshape((npart, 2))

    for j in range(npart):
        for i, ind_n in enumerate(neighbours[j]):
            dx, dy = get_dx(x[j], x[ind_n], y[j], y[ind_n], L=L, periodic=periodic)
            r = np.sqrt(dx ** 2 + dy ** 2)

            dwdr = dWdr(r / H[j], H[j], kernel)

            grad_W_j_at_i[j, i, 0] = dwdr * dx / r
            grad_W_j_at_i[j, i, 1] = dwdr * dy / r

            sum_grad_W[j] += grad_W_j_at_i[j, i]

    # finish computing the gradients: Need W(r, h), which is currently stored as psi
    for j in range(npart):
        for i, ind_n in enumerate(neighbours[j]):
            grad_psi_j_at_i[j, i, 0] = (
                    grad_W_j_at_i[j, i, 0] / omega[j]
                    - W_j_at_i[j, i] * sum_grad_W[j, 0] / omega[j] ** 2
            )
            grad_psi_j_at_i[j, i, 1] = (
                    grad_W_j_at_i[j, i, 1] / omega[j]
                    - W_j_at_i[j, i] * sum_grad_W[j, 1] / omega[j] ** 2
            )

    return grad_psi_j_at_i


@jit(nopython=True, fastmath=True)
def compute_psi_j(
        xi: float,
        yi: float,
        xj: np.ndarray,
        yj: np.ndarray,
        Hi: float,
        kernel: str = "cubic_spline",
        L: List = [1.0, 1.0],
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

    for j in range(xj.shape[0]):
        psi_j[j] = psi(xi, yi, xj[j], yj[j], Hi, kernel=kernel, L=L, periodic=periodic)

    return psi_j


@jit(nopython=True, fastmath=True)
def compute_psi_i(
        xi: float,
        yi: float,
        xj: np.ndarray,
        yj: np.ndarray,
        Hj: np.ndarray,
        kernel: str = "cubic_spline",
        L: List = [1.0, 1.0],
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

    for j in range(xj.shape[0]):
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
        L: List = [1.0, 1.0],
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


def get_matrix(
        xi: float,
        yi: float,
        xj: np.ndarray,
        yj: np.ndarray,
        psi_j: np.ndarray,
        L: List = [1.0, 1.0],
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
        L: List = [1.0, 1.0],
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
