![Build and Test](https://github.com/phoebe-team/phoebe/actions/workflows/buildandtest_push.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/phoebe/badge/?version=develop)](https://phoebe.readthedocs.io/en/develop/?badge=develop)

# Phoebe <img src="doc/sphinx/source/_static/icon.png" width="25"/>

### A high-performance framework for solving phonon and electron Boltzmann transport equations

Phoebe is an open-source code for the ab-initio computation of electron and phonon transport properties of crystalline materials.

It is designed to take advantage of HPC systems via MPI-OpenMP hybrid parallelism, memory-distributed computing via ScaLAPACK, and GPU accelerated calculation of scattering rates.

For more details, see:

* Phoebe: a high-performance framework for solving phonon and electron Boltzmann transport equations.  
  A. Cepellotti, J. Coulter, A. Johansson, N. S. Fedorova, B. Kozinsky. (2022).   
  [DOI:10.1088/2515-7639/ac86f6](https://doi.org/10.1088/2515-7639/ac86f6).

Tutorials, documentation of functionality and underlying theory can be found at:
  * [Homepage](https://phoebe-team.github.io/phoebe/)
  * [Tutorials/Documentation](https://phoebe.readthedocs.io/en/develop/introduction.html)

For further questions and feature requests, please post on the discussions page for the git repo.
If you feel you've found a bug or seen some unexpected behavior, please let us know by opening a git issue. 

-------------------------
### Current functionalities
#### Electronic Transport

   * Electron-phonon and phonon-electron scattering rates by Wannier interpolation
   * Electron-phonon scattering within the electron-phonon averaged (EPA) approximation
   * Electronic transport coefficients (mobility, conductivity, thermal conductivity, and Seebeck coefficient)

#### Phonon Transport

   * Phonon (lattice) thermal conductivity, including: 
     * 3-phonon scattering from thirdOrder.py/ShengBTE or Phono3py force constants
     * Boundary, isotope, and phonon-electron scattering contributions
     * Lattice thermal conductivity calculations including both ph-ph and ph-el scattering 

#### And more:

   * BTE solutions by RTA, iterative, variational, and relaxon solvers
   * Phonon drag effects via the coupled BTE
   * Calculation of electron and phonon linewidths or relaxation times on a path
   * Wigner transport equation correction for electrons and phonons 
   * Hydrodynamic transport properties (viscosity) for electrons and phonons
