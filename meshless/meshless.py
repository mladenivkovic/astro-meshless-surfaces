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

try:
    from .meshlessio import *
    from .kernels import *
    from .particles import *
    from .optional_packages import jit, prange, List
    from typing import Union
except ImportError:
    # in case you're not using it as a package, but directly in the pythonpath
    from meshlessio import *
    from kernels import *
    from particles import *


import numpy as np

# define global float precision here
my_float = np.float64


def Aij_Hopkins(
    pind: int,
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    m: np.ndarray,
    rho: np.ndarray,
    kernel="cubic_spline",
    L: List = (1.0, 1.0),
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

    m: numpy.ndarray
        particle masses

    rho: numpy.ndarray
        particle densities

    h: numpy.ndarray
        kernel support radii

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

    debug = False

    nbors = find_neighbours(pind, x, y, h, L=L, periodic=periodic)

    xj = x[nbors]
    yj = y[nbors]
    hj = h[nbors]

    # -------------------------------------------------------
    # Part 1: For particle at x_i (Our chosen particle)
    # -------------------------------------------------------

    # compute psi_j(x_i)
    psi_j = compute_psi(
        x[pind], y[pind], xj, yj, h[pind], kernel, L=L, periodic=periodic
    )

    # normalize psi_j
    omega_xi = np.sum(psi_j) + psi(0, 0, 0, 0, h[pind], kernel)
    psi_j /= omega_xi
    if my_float != np.float:
        psi_j = np.atleast_1d(psi_j.astype(np.float))
    else:
        psi_j = np.atleast_1d(psi_j)

    # compute B_i
    B_i = get_matrix(x[pind], y[pind], xj, yj, psi_j, L=L, periodic=periodic)

    # compute psi_tilde_j(x_i)
    psi_tilde_j = np.empty((len(nbors), 2), dtype=np.float)
    for i, n in enumerate(nbors):
        dx = np.array([xj[i] - x[pind], yj[i] - y[pind]])
        psi_tilde_j[i] = np.dot(B_i, dx) * psi_j[i]

    # ---------------------------------------------------------------------------
    # Part 2: values of psi/psi_tilde of particle i at neighbour positions x_j
    # ---------------------------------------------------------------------------

    psi_i = np.zeros(len(nbors), dtype=my_float)  # psi_i(xj)
    psi_tilde_i = np.empty((len(nbors), 2), dtype=np.float)  # psi_tilde_i(x_j)

    for i, n in enumerate(nbors):
        # first compute all psi(xj) from neighbour's neighbours to get weight omega
        nneigh = find_neighbours(n, x, y, h, L=L, periodic=periodic)
        xk = x[nneigh]
        yk = y[nneigh]
        for j, nn in enumerate(nneigh):
            psi_k = compute_psi(
                x[n], y[n], xk, yk, h[n], kernel, L=L, periodic=periodic
            )
            if (
                nn == pind
            ):  # store psi_i, which is the psi for the particle whe chose at position xj; psi_i(xj)
                psi_i[i] = psi_k[j]

        omega_xj = np.sum(psi_k) + psi(0, 0, 0, 0, h[nn], kernel)

        psi_i[i] /= omega_xj
        psi_k /= omega_xj
        if my_float != np.float:
            psi_k = psi_k.astype(np.float)

        # now compute B_j^{\alpha \beta}
        B_j = get_matrix(x[n], y[n], xk, yk, psi_k, L=L, periodic=periodic)

        # get psi_i_tilde(x = x_j)
        dx = np.array([x[pind] - x[n], y[pind] - y[n]])
        psi_tilde_i[i] = np.dot(B_j, dx) * np.float(psi_i[i])

    # -------------------------------
    # Part 3: Compute A_ij
    # -------------------------------

    A_ij = np.empty((len(nbors), 2), dtype=np.float)

    for i, n in enumerate(nbors):
        A_ij[i] = V(pind, m, rho) * psi_tilde_j[i] - V(n, m, rho) * psi_tilde_i[i]

    return A_ij


def Aij_Hopkins_v2(
    pind, x, y, h, m, rho, kernel="cubic_spline", L: List = (1.0, 1.0), periodic=True,
):
    """
    Compute A_ij as defined by Hopkins 2015, second version

    Parameters
    ----------

    pind:   int
        particle index that you want solution for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    m: numpy.ndarray
        particle masses

    rho: numpy.ndarray
        particle densities

    h: numpy.ndarray
        kernel support radii

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
    psi_k_at_l = np.zeros((npart, npart), dtype=my_float)

    for k in range(npart):
        for l in range(npart):
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_k_at_l[k, l] = psi(
                x[l], y[l], x[k], y[k], h[l], kernel=kernel, L=L, periodic=periodic,
            )

    neighbours = [[] for i in x]
    omega = np.zeros(npart, dtype=my_float)

    for l in range(npart):

        # find and store all neighbours;
        neighbours[l] = find_neighbours(l, x, y, h, L=L, periodic=periodic)

        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[:, l])
        # omega_k = sum_l W(x_k - x_l, h_k) = sum_l psi_l(x_k) as it is currently stored in memory

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
        for l in range(npart):

            dx = np.array([x[k] - x[l], y[k] - y[l]])
            psi_tilde_k_at_l[k, l] = np.dot(B_k[l], dx) * psi_k_at_l[k, l]

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]

    A_ij = np.zeros((len(nbors), 2), dtype=np.float)

    for i, j in enumerate(nbors):

        A_ij[i] = (
            V(pind, m, rho) * psi_tilde_k_at_l[j, pind]
            - V(j, m, rho) * psi_tilde_k_at_l[pind, j]
        )

    return A_ij


def Aij_Ivanova_all(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    m: np.ndarray,
    rho: np.ndarray,
    kernel: str = "cubic_spline",
    L: List = (1.0, 1.0),
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

    m: numpy.ndarray
        particle masses

    rho: numpy.ndarray
        particle densities

    h: numpy.ndarray
        kernel support radii

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

    # first get neighbour data
    neighbour_data = get_neighbour_data_for_all(x, y, h, L=L, periodic=periodic)

    maxneigh = neighbour_data.maxneigh
    neighbours = neighbour_data.neighbours
    nneigh = neighbour_data.nneigh
    iinds = neighbour_data.iinds

    npart = len(neighbours)

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    W_j_at_i = np.zeros((npart, maxneigh), dtype=np.float)
    omega = np.zeros(npart, dtype=my_float)

    for j in prange(npart):
        for i, ind_n in enumerate(neighbours[j]):
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            W_j_at_i[j, i] = psi(
                x[j],
                y[j],
                x[ind_n],
                y[ind_n],
                h[j],
                kernel=kernel,
                L=L,
                periodic=periodic,
            )
            omega[j] += W_j_at_i[j, i]

        # add self-contribution
        omega[j] += psi(0.0, 0.0, 0.0, 0.0, h[j], kernel=kernel, L=L, periodic=periodic)

    # get gradients
    grad_psi_j_at_i = get_grad_psi_j_at_i_analytical(
        x, y, h, omega, W_j_at_i, neighbour_data, kernel=kernel, periodic=periodic,
    )

    maxn = max([len(n) for n in neighbours])
    A_ij = np.zeros((npart, maxn, 2), dtype=np.float)

    # precompute all volumes
    Vol = np.zeros((npart), dtype=np.float)
    for i in range(npart):
        Vol[i] = 1 / omega[i]

    # now compute A_ij for all neighbours j of i
    for i in prange(npart):

        nbors = neighbours[i]

        V_i = Vol[i]

        for j, nj in enumerate(nbors):

            grad_psi_j_xi = grad_psi_j_at_i[i, j]
            iind = iinds[i, j]
            grad_psi_i_xj = grad_psi_j_at_i[nj, iind]
            #  try:
            #      grad_psi_i_xj = grad_psi_j_at_i[nj, iind]
            #  except IndexError:
            #      dx, dy = get_dx(x[i], x[nj], y[i], y[nj], L=L, periodic=periodic)
            #      r = np.sqrt(dx ** 2 + dy ** 2)
            #      if r / h[j] > 1.0:
            #          grad_psi_i_xj = 0.0
            #      else:
            #          print("Didn't find index i=", i, "as neighbour of j=", nj)
            #          print("nbors i:", neighbours[i])
            #          print("nbors j:", neighbours[nj])
            #          print("r/H_i", r / h[i], "r/H[j]", r / h[nj])
            #  raise IndexError

            V_j = Vol[nj]

            A_ij[i, j] = V_i * grad_psi_j_xi - V_j * grad_psi_i_xj

    return A_ij, neighbours


def Aij_Ivanova(
    pind: int,
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    m: np.ndarray,
    rho: np.ndarray,
    kernel="cubic_spline",
    L: List = (1.0, 1.0),
    periodic: bool = True,
):
    """
    Compute A_ij as defined by Ivanova 2013, using the discretization by Taylor
    expansion as Hopkins does it. Use analytical expressions for the
    gradient of the kernels instead of the matrix representation.


    Parameters
    ----------

    pind:   int
        particle index that you want solution for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    m: numpy.ndarray
        particle masses

    rho: numpy.ndarray
        particle densities

    h: numpy.ndarray
        kernel support radii

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

    npart = x.shape[0]

    # first get neighbour data
    neighbour_data = get_neighbour_data_for_all(x, y, h, L=L, periodic=periodic)

    maxneigh = neighbour_data.maxneigh
    neighbours = neighbour_data.neighbours
    nneigh = neighbour_data.nneigh
    iinds = neighbour_data.iinds

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    W_j_at_i = np.zeros((npart, maxneigh), dtype=np.float)
    omega = np.zeros(npart, dtype=my_float)

    for j in range(npart):
        for i, ind_n in enumerate(neighbours[j]):
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            W_j_at_i[j, i] = psi(
                x[ind_n],
                y[ind_n],
                x[j],
                y[j],
                h[ind_n],
                kernel=kernel,
                L=L,
                periodic=periodic,
            )
            omega[ind_n] += W_j_at_i[j, i]

        # add self-contribution
        omega[j] += psi(0.0, 0.0, 0.0, 0.0, h[j], kernel=kernel, L=L, periodic=periodic)

    # get gradients
    grad_psi_j_at_i = get_grad_psi_j_at_i_analytical(
        x, y, h, omega, W_j_at_i, neighbour_data, kernel=kernel, periodic=periodic,
    )

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]
    A_ij = np.zeros((len(nbors), 2), dtype=np.float)

    V_i = 1 / omega[pind]

    for j, nj in enumerate(nbors):

        grad_psi_j_xi = grad_psi_j_at_i[pind, j]
        iind = iinds[pind, j]
        try:
            grad_psi_i_xj = grad_psi_j_at_i[nj, iind]
        except IndexError:
            dx, dy = get_dx(x[i], x[nj], y[i], y[nj], L=L, periodic=periodic)
            r = np.sqrt(dx ** 2 + dy ** 2)
            if r / h[j] > 1.0:
                grad_psi_i_xj = 0.0
            else:
                print("Didn't find index i=", i, "as neighbour of j=", nj)
                print("nbors i:", neighbours[i])
                print("nbors j:", neighbours[nj])
                print("r/H_i", r / h[i], "r/H[j]", r / h[nj])
                raise IndexError

        V_j = 1 / omega[nj]

        A_ij[j] = V_i * grad_psi_j_xi - V_j * grad_psi_i_xj

    return A_ij


@jit(nopython=True)
def x_ij(
    pind: int,
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
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

    h: numpy.ndarray
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
        hfact = h[pind] / (h[pind] + h[which])
        x_ij = np.array(
            [
                x[pind] - hfact * (x[pind] - x[which]),
                y[pind] - hfact * (y[pind] - y[which]),
            ]
        )
        return x_ij

    elif nbors is not None:
        x_ij = np.empty((len(nbors), 2), dtype=np.float)
        for i, n in enumerate(nbors):
            hfact = h[pind] / (h[pind] + h[n])
            x_ij[i] = np.array(
                [x[pind] - hfact * (x[pind] - x[n]), y[pind] - hfact * (y[pind] - y[n])]
            )

        return x_ij

    else:
        raise ValueError(
            "Gotta give me either a list of neighbours or a single particle info for x_ij"
        )

    return


#  @jit()
def get_grad_psi_j_at_i_analytical(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    omega: np.ndarray,
    W_j_at_i: np.ndarray,
    neighbour_data,
    kernel="cubic_spline",
    L: List = (1.0, 1.0),
    periodic: bool = True,
):
    """
    Compute \nabla \psi_k (x_l) for all particles k and l

    Parameters
    ----------

    pind:   int
        particle index that you want solution for

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    h: numpy.ndarray
        kernel support radii

    kernel: str
        which kernel to use

    W_j_at_i: np.ndarray
        W_k(x_l) : npart x npart array

    neighbour_data: 
        neighbour_data object. See function get_neighbour_data_for_all

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    grad_psi_j_at_i: np.ndarray
        npart x max_neigh x 2 array; grad psi_j (x_i) for all i,j for both x and y direction
    """

    npart = x.shape[0]
    maxneigh = neighbour_data.maxneigh
    neighbours = neighbour_data.neighbours
    nneigh = neighbour_data.nneigh
    iinds = neighbour_data.iinds

    # gradient of psi_j at neighbour i's position
    grad_psi_j_at_i = np.zeros((npart, maxneigh, 2), dtype=my_float)
    # gradient of W_j at neighbour i's position
    grad_W_j_at_i = np.zeros((npart, maxneigh, 2), dtype=my_float)
    # gradient sum for the same h_i
    sum_grad_W = np.zeros((npart, 2), dtype=my_float)

    for j in prange(npart):
        for i, ind_n in enumerate(neighbours[j]):
            dx, dy = get_dx(x[j], x[ind_n], y[j], y[ind_n], L=L, periodic=periodic)
            r = np.sqrt(dx ** 2 + dy ** 2)

            dwdr = dWdr(r / h[j], h[j], kernel)

            grad_W_j_at_i[j, i, 0] = dwdr * dx / r
            grad_W_j_at_i[j, i, 1] = dwdr * dy / r

            sum_grad_W[j] += grad_W_j_at_i[j, i]

    # finish computing the gradients: Need W(r, h), which is currently stored as psi
    for j in prange(npart):
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
def compute_psi(
    xi: float,
    yi: float,
    xj: np.ndarray,
    yj: np.ndarray,
    h: Union[float, np.ndarray],
    kernel: str = "cubic_spline",
    L: List = (1.0, 1.0),
    periodic: bool = True,
):
    """
    Compute psi_j(x_i) for all j


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

    h: float or np.ndarray
        smoothing length at position xi, yi
        or array of h for xj, yj [used to compute h(x)]

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

    psi_j = np.zeros(xj.shape[0], dtype=my_float)

    if isinstance(h, np.ndarray):
        for i in range(xj.shape[0]):
            psi_j[i] = psi(
                xi, yi, xj[i], yj[i], h[i], kernel=kernel, L=L, periodic=periodic,
            )

    else:
        for i in range(xj.shape[0]):
            psi_j[i] = psi(
                xi, yi, xj[i], yj[i], h, kernel=kernel, L=L, periodic=periodic,
            )

    return psi_j


@jit()
def psi(
    x: float,
    y: float,
    xi: float,
    yi: float,
    h: float,
    kernel: str = "cubic_spline",
    L: List = (1.0, 1.0),
    periodic: bool = True,
):
    """
    UNNORMALIZED Volume fraction at position x of some particle
    with coordinates xi, yi, smoothing length h(x)

    i.e. psi_i(x) = W([x - xi, y - yi], h(x))


    Parameter
    ---------

    x: float
        particle x position

    y: float
        particle y position

    xi: float
        particle xi position

    yi: float
        particle yi position

    h: float
        kernel support radius at position (x, y)

    kernel: str
        which kernel to use

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------
    psi_i: my_float
        !!!! returns type my_float!
        Needed to prevent precision errors for normalisation
    """

    dx, dy = get_dx(x, xi, y, yi, L=L, periodic=periodic)

    q = my_float(np.sqrt(dx ** 2 + dy ** 2) / h)

    return W(q, h, kernel)


@jit(nopython=True)
def get_matrix(
    xi: float,
    yi: float,
    xj: np.ndarray,
    yj: np.ndarray,
    psi_j: np.ndarray,
    L: List = (1.0, 1.0),
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

    dx = np.zeros(xj.shape[0])
    dy = np.zeros(xj.shape[0])

    for i in range(xj.shape[0]):
        dx[i], dy[i] = get_dx(xj[i], xi, yj[i], yi, L=L, periodic=periodic)

    E00 = np.sum(dx * dx * psi_j)
    E01 = np.sum(dx * dy * psi_j)
    E11 = np.sum(dy * dy * psi_j)

    E = np.matrix([[E00, E01], [E01, E11]])

    try:
        return E.getI()
    except np.linalg.LinAlgError:
        print("Exception: Singular Matrix")
        print("E:", E)
        print("dx:", xj - xi)
        print("dy:", yj - yi)
        print("psi:", psi_j)
        quit(2)

    return


@jit(nopython=True)
def h_of_x(
    xx: float,
    yy: float,
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    m: np.ndarray,
    rho: np.ndarray,
    kernel="cubic_spline",
    L: List = (1.0, 1.0),
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

    m: numpy.ndarray
        particle masses

    rho: numpy.ndarray
        particle densities

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

    nbors = find_neighbours_arbitrary_x(xx, yy, x, y, h, L=L, periodic=periodic)

    xj = x[nbors]
    yj = y[nbors]
    hj = h[nbors]

    psi_j = compute_psi(xx, yy, xj, yj, hj, kernel=kernel, L=L, periodic=periodic)
    psi_j /= np.sum(psi_j)

    hh = np.sum(hj * psi_j)

    return hh
