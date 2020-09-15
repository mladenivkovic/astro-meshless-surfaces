#!/usr/bin/env python3

###########################################################################################
#  package:   astro-meshless-surfaces
#  file:      particles.py
#  brief:     particle related functions
#  copyright: GPLv3
#             Copyright (C) 2019 EPFL (Ecole Polytechnique Federale de Lausanne)
#             LASTRO - Laboratory of Astrophysics of EPFL
#  author:    Mladen Ivkovic <mladen.ivkovic@hotmail.com>
#
# This file is part of astro-meshless-surfaces.
###########################################################################################

import numpy as np
from .optional_packages import jit, prange
from typing import Union
from scipy.spatial import cKDTree


@jit(nopython=True)
def find_index(x: np.ndarray, y: np.ndarray, pcoord: np.ndarray):
    """
    Find the index in the read-in arrays where
    the particle with coordinates of your choice is


    Parameters
    ----------
    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    pcoord: numpy.ndarray
        coordinates of the particle that you want to look for


    Returns
    -------

    pind: integer
        Particle index in `x`, `y` arrays
    """

    diff = (x - pcoord[0]) ** 2 + (y - pcoord[1]) ** 2
    pind = np.argmin(diff)
    return pind


def find_index_by_id(ids: np.ndarray, id_to_look_for: int):
    """
    Find the index in the read-in arrays where
    the particle with id_to_look_for is

    Parameters
    ----------
   
    ids:    numpy.ndarray
        particle IDs
    
    id_to_look_for : int
        which ID to find


    Returns
    -------
    pind:  integer
        index of particle with id_to_look_for

    """

    pind = np.asscalar(np.where(ids == id_to_look_for)[0])

    return pind


def find_neighbours(
    ind: int,
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    tree: Union[None, cKDTree] = None,
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Find indices of all neighbours of a particle with index ind
    within kernel suppor radius H (where kernel != 0)


    Parameters
    ----------

    ind: integer
        index of particle whose neighbours you want

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    tree: None or scipy.spatial.cKDTree
        tree used to look for neighbours. If `None`, will be generated.

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    tree: scipy.spatial.cKDTree
        tree used to look for neighbours.

    neigh: np.ndarray
        array of neighbour indices.

    """

    if tree is None:
        tree = get_tree(x, y, L=L, periodic=periodic)

    ns = tree.query_ball_point([x[ind], y[ind]], H[ind])

    ns.remove(ind)  # remove yourself
    ns.sort()
    neigh = np.array(ns, dtype=np.int)

    return tree, neigh


def find_neighbours_arbitrary_x(
    x0: float,
    y0: float,
    x: np.ndarray,
    y: np.ndarray,
    H: float,
    tree: Union[cKDTree, None] = None,
    L: np.ndarray = np.ones(2),
    periodic=True,
):
    """
    Find indices of all neighbours around position x0, y0
    within H (where kernel != 0)

    Parameters
    ----------

    x0: float
        x position for which to look

    y0: float
        y position for which to look

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: float
       kernel support radius around (x0, y0)

    tree: scipy.spatial.cKDTree object or None
        tree to query. If none, one will be generated.

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    tree: scipy.spatial.cKDTree
        tree used to look up neighbours.

    neighbours: np.ndarray
        array containing indices of neighbours

    """

    if tree is None:
        tree = get_tree(x, y, L=L, periodic=periodic)

    coord = np.array([x0, y0])

    ns = tree.query_ball_point(coord, H)
    ns.sort()

    return tree, np.array(ns)


def V(ind: int, m: np.ndarray, rho: np.ndarray):
    """
    Volume estimate for particle with index ind.

    Paramters
    ---------

    ind: int
        particle index to work for

    m: np.ndarray
        array of all particle masses

    rho: np.ndarray
        array of all particle densities

    
    Returns
    -------

    V: float
        associated particle volume

    """
    V = m[ind] / rho[ind]
    if V > 1:
        print("Got particle volume V=", V)
        print("Did you put the arguments in the correct places?")
    return V


def find_central_particle(L: np.ndarray, ids: np.ndarray):
    """
    Find the index of the central particle at (0.5, 0.5)

    Parameters
    ----------

    L: np.ndarray
        Boxlen

    ids: np.ndarray
        particle IDs
    

    Returns
    -------

    cind: int
        index of central particle
    """

    i = L[0] // 2 - 1
    j = L[1] // 2 - 1
    cid = i * L + j + 1
    cind = np.asscalar(np.where(ids == cid)[0])

    return cind


def find_added_particle(ids: np.ndarray):
    """
    Find the index of the added particle (has highest ID).

    Parameters
    ----------

    ids: np.ndarray
        particle IDs

    Returns
    -------

    pind: int
        index of added particle
    """

    pid = ids.shape[0]
    pind = np.asscalar(np.where(ids == pid)[0])

    return pind


@jit(nopython=True)
def get_dx(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Compute difference of vectors [x1 - x2, y1 - y2] while
    checking for periodicity if necessary.

    Parameters
    ----------
    x1: float
        x position of particle 1

    x2: float
        x position of particle 2

    y1: float
        y position of particle 1

    y2: float
        y position of particle 2

    L: np.ndarray
        boxsize

    periodic: bool
        whether to assume periodic boundaries

    
    Returns
    -------

    dx: Tuple
        Tuple (dx, dy) particle difference
    """
    dx = x1 - x2
    dy = y1 - y2

    if periodic:

        Lxhalf = L[0] * 0.5
        Lyhalf = L[1] * 0.5
        if dx > Lxhalf:
            dx -= L[0]
        elif dx < -Lxhalf:
            dx += L[0]

        if dy > Lyhalf:
            dy -= L[1]
        elif dy < -Lyhalf:
            dy += L[1]

    return np.array((dx, dy))


def get_neighbours_for_all(
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    tree: Union[cKDTree, None] = None,
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Find neighbours for all particles.


    Parameters
    ----------

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    H: numpy.ndarray
        kernel support radii

    tree: scipy.spatial.cKDTree object or None
        tree to query. If none, one will be generated.

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries


    Returns
    -------

    tree: scipy.spatial.cKDTree
        tree used to look up neighbours.

    neighbours: np.ndarray
        array of shape (npart, maxneigh) containing neighbour indices of each particle

    nneigh: np.ndarray(dtype=int)
        number of neighbours each particle has.

    """

    if tree is None:
        tree = get_tree(x, y, L=L, periodic=periodic)

    nparts = x.shape[0]

    nlist = [[] for _ in range(nparts)]
    nneigh = np.zeros(nparts, dtype=np.int)

    for p in prange(nparts):
        ns = tree.query_ball_point([x[p], y[p]], H[p])
        ns.sort()
        ns.remove(p)  # remove yourself

        nneigh[p] = len(ns)

    maxneigh = nneigh.max()
    neighbours = np.zeros(nparts * maxneigh, dtype=np.int).reshape((nparts, maxneigh))

    for p in prange(nparts):
        neighbours[p][: nneigh[p]] = nlist[p][: nneigh[p]]

    return tree, neighbours, nneigh


def get_tree(
    x: np.ndarray, y: np.ndarray, L: np.ndarray = np.ones(2), periodic: bool = True
):
    """
    Generate the KDTree to be queried for neighbour searches.

    Parameters
    ----------

    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    L: Tuple
        boxsize

    periodic: bool
        whether to assume periodic boundaries

    Returns
    -------

    tree: scipy.spatial.cKDTree
        tree that can be queried for neighbour searches.
    """

    if periodic:
        boxsize = L
    else:
        boxsize = None

    fullarr = np.stack((x, y), axis=1)
    tree = cKDTree(fullarr, boxsize=boxsize)
    return tree
