Meshless Python Module
======================

A small, inefficient python module for computing effective surfaces for 
meshless methods of particle hydrodynamics the Hopkins 2015 way 
(https://arxiv.org/abs/1409.7395) or the Ivanova 2013 
(https://arxiv.org/abs/1209.4302) way.

It's only written for 2d computations.

It's meant to test things out, generate some plots etc, not to be used for 
serious computations. (Although I have recently extended it to work with numba, 
so bigger computations are feasible now.)

Also includes simple reading-in functions for SWIFT (https://ascl.net/1805.020) 
hdf5 output.


I haven't written any sort of documentation apart from docstrings. If you
want to know something, feel free to contact me (mladen.ivkovic[at]hotmail[dot]com) 
anytime, or leave a bug/issue on https://github.com/mladenivkovic/astro-meshless-surfaces. 
Otherwise, there are some usecase examples in ``/examples``.



Requirements
------------

- numpy > 1.15
- matplotlib > 3.0
- h5py > 2.0
- scipy > 1.1

Optional:

- numba
