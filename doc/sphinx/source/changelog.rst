Changelog
==========

Version 1.1.0
-------------

Summary 
^^^^^^^^
A long, long overdue versioning update. Going forward, we will regularly follow semantic versioning.
Many changes have been made since the original release, which are detailed below and have been added over time. 

**Major changes:**
  * Phonon-electron lifetime predictions (PR #182)
  * Thermal conductivity including a phonon-electron scattering contribution (PR #182)
  * Major improvements to the cost and quality of relaxons BTE solutions.

**New features:**
  * It is now possible to run phonopy force constants and ShengBte for the anharmonic force constants. (PR #140)
  * Added input variable `hdf5ElPhFileFormat <https://phoebe.readthedocs.io/en/develop/inputFiles.html#hdf5elphfileformat>`_. This triggers an HDF5 file format for very large el-ph matrix element files, to be used in the qeToPhoebe application. When triggered, the el-ph tensor is saved while being split along R_e indices. (PR #143)
  * Added support for Wannier-dependent shifts in the phase factors of the Fourier transform of the electron Hamiltonian for more accurate Wannier interpolation. For theory details, see the paragraph about the construction of the R_irr vectors `here <https://phoebe.readthedocs.io/en/develop/theory/wannier.html#wannier-interpolation-of-band-structure>`_. The use of these shifts is triggered by the input variable `wsVecFileName <https://phoebe.readthedocs.io/en/develop/inputFiles.html#wsvecfilename>`_ (PR #145)
  * Added the capability to include the non-analytic correction to the FCs in the phonopy case, in an input variable phonopyBORNFileName which is set to the path of a phonopy BORN file. (PR #147 and PR #164)
  * Added functionality to output the phonon eigendisplacements from the phonon bands app, dumping them to JSON when the outputEigendisplacements variable is set to true in the phononBands app input file. A plot script is included to view these eigendisplacements, ``phoebe/scripts/plotScripts/plotEigendisplacements.py``. A documented example is found `here <https://phoebe.readthedocs.io/en/develop/tutorials/bands.html#plotting-phonon-eigendisplacements>`_ (PR #169). 

**Performance updates:**
  * We use Kokkos with OMP the default behavior of Phoebe, so that users are not accidentally slowed down if they forget the ``Kokkos_ENABLE_OMP`` flag during CMake. (PR #148)
  * Phoebe now works properly with Intel compilers (PR #148) and also Clang compilers (PR #202) (thanks to @Yaraslaut).
  * Alleviated bottlenecks related to the filtering of points in active band structure construction (PR #150)
  * Batched diagonalization in the phonon and electronWannier Hamiltonians improves the speed of the interpolation of energies, velocities, and eigenvectors and allows them to be accelerated using GPUs. (PR #160)
  * Relaxons solutions are now as much as 1-2 orders of magnitude faster on the scattering matrix diagonalization due to improvements in the ScaLAPACK calls. (PR #191, #192)
  * The option to only compute transport using a certain number of relaxons states has been added with the input variable numRelaxonsEigenvalues. (PR #191)
  * Replaced a 5D kokkos for loop with a kokkos::gemv call from kokkos-kernels, which in some cases can result in a dramatic speedup of elph calculations. (PR #199)
  * Earlier, we were using a transpose of the distributed scattering matrix with an explicit/hand written call. This was super slow, and should have been implemented using the ScaLAPACK pdtran function. (PR #201)

**Bugfixes:**
  * Segfault fix in epaTransport app -- QE file parsing failed when used with DFT calculations using smearing (PR #151). 
  * Stability improvements to electronic exact BTE solvers by using the full scattering matrix with symmetrization (PR #152 and #153). 
  * Changes to the git urls used by CMake to download and set up dependencies for the JSON library (PR #159). 
  * Updated phono3py file parser to handle newer phono3py file formats (such as those from the python interface) (PR #162).
  * The cutoff for determining which vectors are in the WS cell (in the wsWeight function of the phononH0 class) was sometimes too strict, and gave zero weight to vectors which were actually in the WS cell. Fixed in (PR #165).
  * Comments behind the # symbol were not always properly ignored (PR #171).
  * Fixed a minor bug in the printing of the band path to JSON -- sometimes earlier, high symmetry points were duplicated in output (PR #171).
  * A bugfix due to the use of non-blocking MPI collective which caused segfaults related to MPI_Ireduce calls in interaction_elph. (PR #194) 
  * When using the trio of inputs deltaChemicalPotential, minChemicalPotential, maxChemicalPotential, the electron Wannier transport app was mistakenly throwing an error saying that the chemical potential had not been set. (PR #197)
  * Switch to a stable branch of the Eigen repository (PR #203).
  * Viscosity outputs were previously incorrect due to two bugs. (PR #207)
  * Bugfix to isotope and boundary scattering terms. (PR #208)
  * Minor bugfix to scripts for plotting lifetimes, required due to a change in python. (PR #212)

#### Bugfixes
* Segfault fix in epaTransport app -- QE file parsing failed when used with DFT calculations using smearing (PR #151). 
* Stability improvements to electronic exact BTE solvers by using the full scattering matrix with symmetrization (PR #152 and #153). 
* Changes to the git urls used by CMake to download and set up dependencies for the JSON library (PR #159). 
* Updated phono3py file parser to handle newer phono3py file formats (such as those from the python interface) (PR #162).
* The cutoff for determining which vectors are in the WS cell (in the wsWeight function of the phononH0 class) was sometimes too strict, and gave zero weight to vectors which were actually in the WS cell. Fixed in (PR #165).
* Comments behind the # symbol were not always properly ignored (PR #171).
* Fixed a minor bug in the printing of the band path to JSON -- sometimes earlier, high symmetry points were duplicated in output (PR #171).
* A bugfix due to the use of non-blocking MPI collective which caused segfaults related to MPI_Ireduce calls in interaction_elph. (PR #194) 
* When using the trio of inputs deltaChemicalPotential, minChemicalPotential, maxChemicalPotential, the electron Wannier transport app was mistakenly throwing an error saying that the chemical potential had not been set. (PR #197)
* Switch to a stable branch of the Eigen repository (PR #203).
* Viscosity outputs were previously incorrect due to two bugs. (PR #207)
* Bugfix to isotope and boundary scattering terms. (PR #208)
* Minor bugfix (making an energy cutoff 10x stricter) to account for possible divergence in the phononLifetimes app for generate states. 

**Interface changes:**
  * The requirement to use disp_phono3py.yaml with phono3py calculations is deprecated (supplying this file does nothing.) Now, only the ``phonopy_disp.yaml`` and ``fc*.hdf5`` files are required. (PR #140)

**Documentation updates:** 
  * Added the MLFF thermal conductivity tutorial created by @YuuuXie (PR #149)
  * Updated the phonon transport example files to include example for phono3py. (PR #162)
  * Added the phonon-electron linewidth tutorial. (PR #182)
  * Documentation for building Phoebe on Perlmutter at NERSC. (PR #196)
  * Documentation for building on SLURM based compute clusters. (PR #196)
  * Updated to RTD v2 (PR #209) 

**Miscelaneous changes**
  * c++ std is now set to 17 (PR #196)
  * Kokkos-kernels is now a submodule. (PR #199)
  * In general, there was also added OMP parallelism, loop refactoring, further commenting of functions, typo fixes additional citations, and minor additions to the documentation. 

**Contributors:** 
  * Jenny Coulter (@jcoulter12)
  * Anders Johansson (@anjohan)
  * Andrea Cepellotti (@cepelotti)
  * Michele Simoncelli (@MSimoncelli)
  * Yu Xie (@YuuuXie)
  * Changpeng Lin (@cplin)
