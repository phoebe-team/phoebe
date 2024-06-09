Introduction
============

Phoebe is a software for the ab-initio prediction of transport properties, such as electron-phonon limited conductivity and phonon thermal conductivity.

Currently, we support integration with the ab-initio software suite Quantum ESPRESSO for electron-phonon properties and anharmonic force constants through ShengBTE and Phono3py. 

Additionally, Phoebe is written in C++ and designed for use on modern HPC infrastructures through hybrid MPI/OpenMP parallelization, distributed memory computation via ScaLAPACK, and support for GPU acceleration using Kokkos.

For more information, or to cite Phoebe, please refer to:

Phoebe: a high-performance framework for solving phonon and electron Boltzmann transport equations.
A. Cepellotti, J. Coulter, A. Johansson, N. S. Fedorova, B. Kozinsky.
J. Phys. Mater. 5 035003. (2022).
`DOI:10.1088/2515-7639/ac86f6 <https://doi.org/10.1088/2515-7639/ac86f6>`_, `arXiv:2111.14999 <https://arxiv.org/abs/2111.14999>`_.

.. raw:: html

  <h3>Features:</h3>

* Electron-phonon limited electronic transport coefficients, including the electrical conductivity, Seebeck coefficient, and electronic thermal conductivity

* Lattice thermal conductivity, including contributions from 3-phonon, phonon-isotope, phonon-electron, and phonon-boundary scattering

* BTE solutions by RTA, iterative, variational, and relaxon solvers

* Electron-phonon transport via Wannier interpolation or EPA approximation

* Wigner transport equation correction for electrons and phonons 

* Calculation of electron and phonon lifetimes/linewidths (including projected onto a band path)

* Electron and phonon viscosities

* Plots of electron band structures/phonon dispersion, electron and phonon density of states

This documentation contains a brief description of the formalism used to compute these quantities. We expect that the user is already familiar with ab-initio codes and with the fundamentals of solid state physics. A good introduction to the topic is the book "Electrons and Phonons: The Theory of Transport Phenomena in Solids" by Ziman. In the theory section, we add references to contemporary literature for the latest method/algorithm developments.

This guide will help you through code's installation, calculation of properties, descriptions of the code input files, postprocessing of results, and a theory handbook. A tutorials section will explain how to obtain the transport properties of Silicon. The Developer's documentation section contains a brief overview of the code structure, and is complemented with a detailed description of purpose and methods for each class.
