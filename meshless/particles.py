#!/usr/bin/env python3

###########################################################################################
#  package:   astro-meshless-surfaces
#  file:      particles.py
#  brief:     particle related functions
#  copyright: GPLv3
#             Copyright (C) 2019 EPFL (Ecole Polytechnique Federale de Lausanne)
#             LASTRO - Laboratory of Astrophysics of EPFL
#  author:    Mladen Ivkovic <mladen.ivkovic@epfl.ch>
#
# This file is part of astro-meshless-surfaces.
###########################################################################################

import numpy as np


# ===============================================
def find_index(x, y, pcoord, tolerance=1e-3):
    # ===============================================
    """
    Find the index in the read-in arrays where
    the particle with coordinates of your choice is

    x, y:       arrays of x, y positions of all particles
    pcoors:     array/list of x,y position of particle to look for
    tolerance:  how much of a tolerance to use to identify particle
                useful when you have random perturbations of a
                uniform grid with unknown exact positions
    """

    for i in range(x.shape[0]):
        if abs(x[i] - pcoord[0]) < tolerance and abs(y[i] - pcoord[1]) < tolerance:
            pind = i
            break

    return pind


# ===============================================
def find_index_by_id(ids, id_to_look_for):
    # ===============================================
    """
    Find the index in the read-in arrays where
    the particle with id_to_look_for is

        ids:    numpy array of particle IDs
        id_to_look_for : which ID to find

    returns:
        pind:  index of particle with id_to_look_for

    """

    pind = np.asscalar(np.where(ids == id_to_look_for)[0])

    return pind


# ================================================================
def find_neighbours(ind, x, y, h, fact=1.0, L=1.0, periodic=True):
    # ================================================================
    """
    Find indices of all neighbours of a particle with index ind
    within fact*h (where kernel != 0)
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize
    periodic:   Whether you assume periodic boundary conditions

    returns list of neighbour indices in x,y,h array
    """

    # None for Gaussian
    if fact is not None:

        x0 = x[ind]
        y0 = y[ind]
        fhsq = h[ind] * h[ind] * fact * fact
        neigh = [None for i in x]

        j = 0
        for i in range(x.shape[0]):
            if i == ind:
                continue

            dx, dy = get_dx(x0, x[i], y0, y[i], L=L, periodic=periodic)

            dist = dx ** 2 + dy ** 2

            if dist < fhsq:
                neigh[j] = i
                j += 1

        return neigh[:j]

    else:
        neigh = [i for i in range(x.shape[0])]
        neigh.remove(ind)
        return neigh


# =================================================================================
def find_neighbours_arbitrary_x(x0, y0, x, y, h, fact=1.0, L=1.0, periodic=True):
    # =================================================================================
    """
    Find indices of all neighbours around position x0, y0
    within fact*h (where kernel != 0)
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize
    periodic:   Whether you assume periodic boundary conditions

    returns list of neighbour indices
    """

    # None for Gaussian
    if fact is not None:
        neigh = [None for i in x]
        j = 0

        if isinstance(h, np.ndarray):
            fsq = fact * fact

            for i in range(x.shape[0]):

                dx, dy = get_dx(x0, x[i], y0, y[i], L=L, periodic=periodic)

                dist = dx ** 2 + dy ** 2

                fhsq = h[i] * h[i] * fsq
                if dist < fhsq:
                    neigh[j] = i
                    j += 1

        else:
            fhsq = fact * fact * h * h
            for i in range(x.shape[0]):

                dx, dy = get_dx(x0, x[i], y0, y[i], L=L, periodic=periodic)

                dist = dx ** 2 + dy ** 2

                if dist < fhsq:
                    neigh[j] = i
                    j += 1

        return neigh[:j]

    else:
        neigh = [i for i in range(x.shape[0])]
        return neigh


# ===================
def V(ind, m, rho):
    # ===================
    """
    Volume estimate for particle with index ind
    """
    V = m[ind] / rho[ind]
    if V > 1:
        print(
            "Got particle volume V=",
            V,
            ". Did you put the arguments in the correct places?",
        )
    return V


# ======================================
def find_central_particle(L, ids):
    # ======================================
    """
    Find the index of the central particle at (0.5, 0.5)
    assumes Lx = Ly = L
    """

    i = L // 2 - 1
    cid = i * L + i + 1
    cind = np.asscalar(np.where(ids == cid)[0])

    return cind


# ======================================
def find_added_particle(ids):
    # ======================================
    """
    Find the index of the added particle (has highest ID)
    """

    pid = ids.shape[0]
    pind = np.asscalar(np.where(ids == pid)[0])

    return pind


# =====================================================
def get_dx(x1, x2, y1, y2, L=1.0, periodic=True):
    # =====================================================
    """
    Compute difference of vectors [x1 - x2, y1 - y2] while
    checking for periodicity if necessary
    L:          boxsize
    periodic:   whether to assume periodic boundaries
    """
    dx = x1 - x2
    dy = y1 - y2

    if periodic:

        if hasattr(L, "__len__"):
            Lxhalf = L[0] / 2.0
            Lyhalf = L[1] / 2.0
        else:
            Lxhalf = L / 2.0
            Lyhalf = L / 2.0
            L = [L, L]

        if dx > Lxhalf:
            dx -= L[0]
        elif dx < -Lxhalf:
            dx += L[0]

        if dy > Lyhalf:
            dy -= L[1]
        elif dy < -Lyhalf:
            dy += L[1]

    return dx, dy


# ========================================================================
def get_neighbour_data_for_all(x, y, h, fact=1.0, L=1.0, periodic=True):
    # ========================================================================
    """
    Gets all the neighbour data for all particles ready.
    Assumes domain is a rectangle with boxsize L[0], L[1].
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize. List/Array or scalar.
    periodic:   Whether you assume periodic boundary conditions

    returns neighbour_data object:
        self.neighbours :   List of lists of every neighbour of every particle
        self.maxneigh :     Highest number of neighbours
        self.nneigh:        integer array of number of neighbours for every particle
        self.iinds:         iinds[i, j] = which index does particle i have in the neighbour
                            list of particle j, where j is the j-th neighbour of i
                            Due to different smoothing lengths, particle j can be the
                            neighbour of i, but i not the neighbour of j.
                            In that case, the particles will be assigned indices j > nneigh[i]

    """

    import copy

    # if it isn't in a list already, create one
    # do this before function/class definition
    if not hasattr(L, "__len__"):
        L = [L, L]

    # -----------------------------------------------
    class neighbour_data:
        # -----------------------------------------------
        def __init__(self, neighbours=None, maxneigh=None, nneigh=None, iinds=None):
            self.neighbours = neighbours
            self.maxneigh = maxneigh
            self.nneigh = nneigh
            self.iinds = iinds

    # -----------------------------------------------
    class cell:
        # -----------------------------------------------
        """
        A cell object to store particles in.
        Stores particle indexes, positions, compact support radii
        """

        def __init__(self):
            self.npart = 0
            self.size = 100
            self.parts = np.zeros(self.size, dtype=np.int)
            self.x = np.zeros(self.size, dtype=np.float)
            self.y = np.zeros(self.size, dtype=np.float)
            self.h = np.zeros(self.size, dtype=np.float)
            self.xmin = 1e300
            self.xmax = -1e300
            self.ymin = 1e300
            self.ymax = -1e300
            self.hmax = -1e300
            return

        def add_particle(self, ind, xp, yp, hp):
            """
            Add a particle, store the index, positions and h
            """
            if self.npart == self.size:
                self.parts = np.append(self.parts, np.zeros(self.size, dtype=np.int))
                self.x = np.append(self.x, np.zeros(self.size, dtype=np.float))
                self.y = np.append(self.y, np.zeros(self.size, dtype=np.float))
                self.h = np.append(self.h, np.zeros(self.size, dtype=np.float))
                self.size *= 2

            self.parts[self.npart] = ind
            self.x[self.npart] = xp
            self.y[self.npart] = yp
            self.h[self.npart] = hp
            self.npart += 1

            if self.xmax < xp:
                self.xmax = xp
            if self.xmin > xp:
                self.xmin = xp
            if self.ymax < yp:
                self.ymax = yp
            if self.ymin > yp:
                self.ymin = yp
            if self.hmax < hp:
                self.hmax = hp

            return

        def is_within_h(self, xp, yp, hp):
            """
            Check whether any particle of this cell is within
            compact support of particle with x, y, h = xp, yp, hp
            """
            dx1, dy1 = get_dx(xp, self.xmax, yp, self.ymax, L=L, periodic=periodic)
            dx2, dy2 = get_dx(xp, self.xmin, yp, self.ymin, L=L, periodic=periodic)
            dxsq = min(dx1 * dx1, dx2 * dx2)
            dysq = min(dy1 * dy1, dy2 * dy2)
            if dxsq / hp ** 2 <= 1 or dysq / hp ** 2 <= 1:
                return True
            else:
                return False

    # ---------------------------------------------------------------
    def find_neighbours_in_cell(i, j, p, xx, yy, hh, is_self):
        # ---------------------------------------------------------------
        """
        Find neighbours of a particle in the cell with indices i,j
        of the grid
        p:      global particle index to work with
        xx, yy: position of particle x
        hh:     compact support radius for p
        is_self: whether this is the cell where p is in
        """
        n = 0
        neigh = [0 for i in range(1000)]
        ncell = grid[i][j]  # neighbour cell we're checking for

        if not is_self:
            if not ncell.is_within_h(xx, yy, hh):
                return []

        N = ncell.npart

        fhsq = hh * hh * fact * fact

        for c, cp in enumerate(ncell.parts[:N]):
            if cp == p:
                # skip yourself
                continue

            dx, dy = get_dx(xx, ncell.x[c], yy, ncell.y[c], L=L, periodic=periodic)

            dist = dx ** 2 + dy ** 2

            if dist <= fhsq:
                try:
                    neigh[n] = cp
                except IndexError:
                    nneigh += [0 for i in range(1000)]
                    nneigh[n] = cp
                n += 1

        return neigh[:n]

    npart = x.shape[0]

    # first find cell size
    ncells_x = int(L[0] / h.max()) + 1
    ncells_y = int(L[1] / h.max()) + 1
    cell_size_x = L[0] / ncells_x
    cell_size_y = L[1] / ncells_y

    # create grid
    grid = [[cell() for j in range(ncells_y)] for i in range(ncells_x)]

    # sort out particles
    for p in range(npart):
        i = int(x[p] / cell_size_x)
        j = int(y[p] / cell_size_y)
        grid[i][j].add_particle(p, x[p], y[p], h[p])

    neighbours = [[] for i in x]
    nneigh = np.zeros(npart, dtype=np.int)

    # main loop: find and store all neighbours;
    # go cell by cell
    for row in range(ncells_y):
        for col in range(ncells_x):

            cell = grid[col][row]
            N = cell.npart
            parts = cell.parts
            if N == 0:
                continue

            hmax = cell.h[:N].max()

            # find over how many cells to loop in every direction
            maxdistx = int(cell_size_x / hmax + 0.5) + 1
            maxdisty = int(cell_size_y / hmax + 0.5) + 1

            xstart = -maxdistx
            xstop = maxdistx + 1
            ystart = -maxdisty
            ystop = maxdisty + 1

            # exception handling: if ncells < 4, just loop over
            # all of them so that you don't add neighbours multiple
            # times
            if ncells_x < 4:
                xstart = 0
                xstop = ncells_x
            if ncells_y < 4:
                ystart = 0
                ystop = ncells_y

            checked_cells = [
                (None, None) for i in range((2 * maxdistx + 1) * (2 * maxdisty + 1))
            ]
            it = 0

            # loop over all neighbours
            # need to loop over entire square. You need to consider
            # the maximal distance from the edges/corners, not from
            # the center of the cell!
            for i in range(xstart, xstop):
                for j in range(ystart, ystop):

                    if ncells_x < 4:
                        iind = i
                    else:
                        iind = col + i

                    if ncells_y < 4:
                        jind = j
                    else:
                        jind = row + j

                    if periodic:
                        while iind < 0:
                            iind += ncells_x
                        while iind >= ncells_x:
                            iind -= ncells_x
                        while jind < 0:
                            jind += ncells_y
                        while jind >= ncells_y:
                            jind -= ncells_y
                    else:
                        if iind < 0 or iind >= ncells_x:
                            continue
                        if jind < 0 or jind >= ncells_y:
                            continue

                    it += 1
                    if (iind, jind) in checked_cells[: it - 1]:
                        continue
                    else:
                        checked_cells[it - 1] = (iind, jind)

                    # loop over all particles in THIS cell
                    for pc, pg in enumerate(cell.parts[:N]):

                        xp = cell.x[pc]
                        yp = cell.y[pc]
                        hp = cell.h[pc]

                        neighbours[pg] += find_neighbours_in_cell(
                            iind, jind, pg, xp, yp, hp, iind == col and jind == row
                        )

    # sort neighbours by index
    for p in range(npart):
        neighbours[p].sort()
        nneigh[p] = len(neighbours[p])

    # max number of neighbours; needed for array allocation
    maxneigh = nneigh.max()

    # store the index of particle i when required as the neighbour of particle j in arrays[npart, maxneigh]
    # i.e. find index 0 <= i < maxneigh for ever j
    iinds = np.zeros((npart, 2 * maxneigh), dtype=np.int)
    current_count = copy.copy(nneigh)

    for i in range(npart):
        for jc, j in enumerate(neighbours[i]):

            try:
                iinds[i, jc] = (neighbours[j]).index(i)
            except ValueError:
                # it is possible that j is a neighbour for i, but i is not a neighbour
                # for j depending on their respective smoothing lengths
                dx, dy = get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)
                r = np.sqrt(dx ** 2 + dy ** 2)
                if r / h[j] < 1:
                    print(
                        "something went wrong when computing iinds in get_neighbour_data_for_all."
                    )
                    print("i=", i, "j=", j, "r=", r, "H=", h[j], "r/H=", r / h[j])
                    print("neighbours i:", neighbours[i])
                    print("neighbours j:", neighbours[j])
                    print("couldn't find i as neighbour of j")
                    print("exiting")
                    quit()
                else:
                    # append after nneigh[j]
                    iinds[i, jc] = current_count[j]
                    current_count[j] += 1

    nd = neighbour_data(
        neighbours=neighbours, maxneigh=maxneigh, nneigh=nneigh, iinds=iinds
    )

    return nd


# ===============================================================================
def get_neighbour_data_for_all_naive(x, y, h, fact=1.0, L=1.0, periodic=True):
    # ===============================================================================
    """
    Gets all the neighbour data for all particles ready.
    Naive way: Loop over all particles for each particle
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize
    periodic:   Whether you assume periodic boundary conditions

    returns neighbour_data object:
        self.neighbours :   List of lists of every neighbour of every particle
        self.maxneigh :     Highest number of neighbours
        self.nneigh:        integer array of number of neighbours for every particle
        self.iinds:         iinds[i, j] = which index does particle i have in the neighbour
                            list of particle j, where j is the j-th neighbour of i
                            Due to different smoothing lengths, particle j can be the
                            neighbour of i, but i not the neighbour of j.
                            In that case, the particles will be assigned indices j > nneigh[i]

    """

    import copy

    npart = x.shape[0]

    # find and store all neighbours;
    neighbours = [[] for i in x]
    for i in range(npart):
        neighbours[i] = find_neighbours(i, x, y, h, fact=fact, L=L, periodic=periodic)

    # get neighbour counts array
    nneigh = np.zeros((npart), dtype=np.int)
    for i in range(npart):
        nneigh[i] = len(neighbours[i])

    # max number of neighbours; needed for array allocation
    maxneigh = nneigh.max()

    # store the index of particle i when required as the neighbour of particle j in arrays[npart, maxneigh]
    # i.e. find index 0 <= i < maxneigh for ever j
    iinds = np.zeros((npart, 2 * maxneigh), dtype=np.int)
    current_count = copy.copy(nneigh)

    for i in range(npart):
        for jc, j in enumerate(neighbours[i]):

            try:
                iinds[i, jc] = (neighbours[j]).index(i)
            except ValueError:
                # it is possible that j is a neighbour for i, but i is not a neighbour
                # for j depending on their respective smoothing lengths
                dx, dy = get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)
                r = np.sqrt(dx ** 2 + dy ** 2)
                if r / h[j] < 1:
                    print("something went wrong when computing iinds.")
                    print("i=", i, "j=", j, "r=", r, "H=", h[j], "r/H=", r / h[j])
                    print("neighbours i:", neighbours[i])
                    print("neighbours j:", neighbours[j])
                    print("couldn't find i as neighbour of j")
                    print("exiting")
                    quit()
                else:
                    # append after nneigh[j]
                    iinds[i, jc] = current_count[j]
                    current_count[j] += 1

    class neighbour_data:
        def __init__(self, neighbours=None, maxneigh=None, nneigh=None, iinds=None):
            self.neighbours = neighbours
            self.maxneigh = maxneigh
            self.nneigh = nneigh
            self.iinds = iinds

    nd = neighbour_data(
        neighbours=neighbours, maxneigh=maxneigh, nneigh=nneigh, iinds=iinds
    )

    return nd
