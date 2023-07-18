from setuptools import setup

setup(
    name='npdrot',
    version='0.0.1',
    author='Jacob Lindbäck',
    author_email='jlindbac@kth.se',
    description='Packaged version of DROT built on numpy, to enable installation in conda env',
    install_requires=[
        "numpy==1.21.5",
        "numba==0.56.3",
        "scipy==1.7.3"
]
)