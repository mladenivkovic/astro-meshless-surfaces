from setuptools import setup
#  from distutils.core import setup

setup(
        name='astro-meshless-surfaces',
        version='0.1.16',
        author='Mladen Ivkovic',
        author_email='mladen.ivkovic@hotmail.com',
        packages=['astro_meshless_surfaces'],
        license='GLPv3',
        scripts=[
                ],
        long_description=open('README.rst').read(),
        install_requires=[
            'numpy',
            'matplotlib',
            'h5py',
            'scipy'
        ],
        extras_requires = ['numba'],
        urt="https://github.com/mladenivkovic/astro-meshless-surfaces", 
     )
