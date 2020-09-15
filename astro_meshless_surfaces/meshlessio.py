#!/usr/bin/env python3

###########################################################################################
#  package:   astro-meshless-surfaces
#  file:      meshlessio.py
#  brief:     contains IO routines
#  copyright: GPLv3
#             Copyright (C) 2019 EPFL (Ecole Polytechnique Federale de Lausanne)
#             LASTRO - Laboratory of Astrophysics of EPFL
#  author:    Mladen Ivkovic <mladen.ivkovic@hotmail.com>
#
# This file is part of astro-meshless-surfaces.
###########################################################################################

"""
All io related functions.
"""

import os
import numpy as np
from typing import Union
import h5py


def read_file(srcfile: str, ptype: str = "PartType0", sort: bool = False):
    """
    Read swift output hdf5 file.

    Parameters
    ----------
    
    srcfile: str
        string of file to be read in

    ptype: str
        which particle type to work with
 
    sort: bool
        whether to sort read in arrays by particle ID


    Returns
    -------
    x: numpy.ndarray
        particle x positions

    y: numpy.ndarray
        particle y positions

    h: numpy.ndarray
        kernel support radii

    rho: numpy.ndarray
        particle densities

    m: numpy.ndarray
        particle masses

    ids: numpy.ndarray
        particle IDs

    npart: int
        number of particles
    """

    f = h5py.File(srcfile, "r")

    x = f[ptype]["Coordinates"][:, 0]
    y = f[ptype]["Coordinates"][:, 1]
    m = f[ptype]["Masses"][:]
    ids = f[ptype]["ParticleIDs"][:]

    try:
        # old SWIFT header versions
        h = f[ptype]["SmoothingLength"][:]
        rho = f[ptype]["Density"][:]
    except KeyError:
        # new SWIFT header versions
        h = f[ptype]["SmoothingLengths"][:]
        rho = f[ptype]["Densities"][:]
    npart = x.shape[0]

    f.close()

    if sort:
        inds = np.argsort(ids)
        x = x[inds]
        y = y[inds]
        h = h[inds]
        rho = rho[inds]
        m = m[inds]
        ids = ids[inds]

    return x, y, h, rho, m, ids, npart


def get_sample_size(prefix: Union[str, None] = None):
    """
    Count how many files we're dealing with.
    Assumes snapshots start with "snapshot-" string and contain
    two numbers: snapshot-XXX-YYY_ZZZZ.hdf5, where both XXX and YYY
    are integers, have the same minimal, maximal value and same
    difference between two consecutive numbers.

    If prefix is given, it will prepend it to snapshots.

    This is intended for numbered output.

    Parameters
    ----------

    prefix: str or None
        prefix/directory to prepend to 'snapshot-' when looking for files.

    Returns
    -------

    nx : int
        number of files (in one direction)

    filenummax: int 
        highest XXX

    fileskip: int
        integer difference between two XXX or YYY
    """

    if prefix is not None:
        filelist = os.listdir(prefix)
    else:
        filelist = os.listdir()

    snaplist = []
    for f in filelist:
        if f.startswith("snapshot-"):
            snaplist.append(f)

    snaplist.sort()
    first = snaplist[0]
    s, dash, rest = first.partition("-")
    num, dash, junk = rest.partition("-")
    lowest = int(num)

    finalsnap = snaplist[-1]
    s, dash, rest = finalsnap.partition("-")
    num, dash, junk = rest.partition("-")

    highest = int(num)

    steps = int(np.sqrt(len(snaplist)))

    nx = steps
    filenummax = highest
    fileskip = int((highest - lowest) / (steps - 1))

    return nx, filenummax, fileskip


def snapstr(number):
    """
    return formatted string for snapshot number
    (4 digit, zero padded string). Can take both
    strings and integers as argument.

    Parameters
    ----------

    number: int or str
        which snapshot number we want

    Returns
    -------
    snapstr: str
        4 digit zero padded string of snapshot number
    """

    if hasattr(number, "__len__") and (not isinstance(number, str)):
        print(
            "You've given me a list for a snapshot number? I'm going to try element 0:",
            number[0],
        )
        number = number[0]

    errormsg = "'" + str(number) + "' is not an integer."

    if isinstance(number, float):
        raise ValueError(errormsg)

    try:
        n = int(number)
    except ValueError:
        raise ValueError(errormsg)
    return "{0:04d}".format(n)


def read_boxsize(fnamestr="_0000.hdf5"):
    """
    Looks for a swift hdf5 file that contains `fnamestr` and reads in the
    boxsize.

    Parameters
    ----------

    fnamestr: str
        string to look for in snapshot filename.

    Returns
    -------

    boxsize: np.ndarray
        [xdim, ydim, zdim] array, where xdim, ydim, zdim are floats
    """

    filelist = os.listdir()
    for f in filelist:
        if fnamestr in f:
            # read in boxsize
            f5 = h5py.File(f)
            h = f5["Header"]
            boxsize = h.attrs["BoxSize"]
            f5.close()
            return boxsize

    # if you're out and haven't found file, say it
    raise IOError("Haven't found any file that contains", fnamestr)
