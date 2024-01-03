.. _installation:

Installing Phoebe
=================

Download
--------

The code is available (free of charge with an open-source MIT license) at its `github page <https://github.com/mir-group/phoebe>`__.
When checking out from the GitHub repository, make sure to use the master branch. The other branches may be functional, but are not guaranteed to work and should be used with caution.

Prerequisites
-------------

The installation requires the following packages are available:

* CMake (a recent version)

* A C++ compiler with C++17 support, whether GCC, Intel, or Clang

* MPI (although the code can compile without)

* Optional: OpenMP

* Optional: CUDA (for GPU acceleration)

* Optional: ScaLAPACK (this will be built for you if you don't have it)

* Optional: HDF5 (see below, a copy of parallel/MPI-enabled HDF5 is highly recommended)

* An internet connection, to download external libraries


Cmake build
-----------

Basic build
^^^^^^^^^^^

To install Phoebe, type::

  # clone the git repository, including the Kokkos submodule
  git clone --recurse-submodules https://github.com/mir-group/phoebe.git
  cd phoebe
  # build the code
  mkdir build
  cd build
  cmake ..
  make -j $(nproc)

where you should substitute ``nproc`` with the number of cores available for parallel compilation. This will create the executable ``phoebe`` in the ``build`` directory. This will build a copy of Phoebe with the default CMake options -- it assumes MPI, OMP, and parallel HDF5 are available (``-DMPI_AVAIL=ON -DOMP_AVAIL=ON -DHDF5_AVAIL=ON``).

CMake will inspect the paths found in the environmental variable ``LD_LIBRARY_PATH`` to verify the existence of an installed copy of the ScaLAPACK library and link it. If not found, the installation will download and compile a copy of ScaLAPACK.

.. note::
   Phoebe often uses OMP through Kokkos. For best performance, you may want to follow the advice from Kokkos regarding OMP env variables:
   "In general, for best performance with OpenMP 4.0 or better set ``OMP_PROC_BIND=spread`` and ``OMP_PLACES=threads``. For best performance with OpenMP 3.1 set OMP_PROC_BIND=true". However, you should check for yourself that this improves performance as it's system dependent.  

HDF5 build
^^^^^^^^^^

Phoebe can make use of HDF5 through the `HighFive <https://github.com/BlueBrain/HighFive>`__ library to read and write the electron-phonon matrix elements.
**This is highly recommended**, as it speeds up what can be time consuming I/O, and also significantly reduces file sizes.
When built using CMake with the flag ``-DHDF5_AVAIL=ON``, Phoebe will be built with HDF5. If MPI is also present,
Phoebe will be built to perform HDF5 operations in parallel. The default behavior is to build for parallel HDF5 operations, with ``-DHDF5_AVAIL=ON`` and ``-DMPI_AVAIL=ON`` both being default CMake flags.

If, for some reason, a user has MPI present, but has built a copy of serial HDF5, which does not link to MPI and therefore cannot
perform parallel read/write operations, they must build Phoebe using the ``-DHDF5_SERIAL=ON`` CMake option to force serial HDF5 operations.

.. note::
  When building on Ubuntu, one may need to specify the location of HDF5 to CMake. This can be done, for example, when using libhdf5-openmpi-dev::

   cmake .. -DCMAKE_CXX_STANDARD_LIBRARIES="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/" -DCMAKE_CXX_FLAGS="-I/usr/include/hdf5/openmpi/"

.. note::
   When building with Intel compilers, it seems sometimes it's necessary to set compiler flags to CC=mpiicc, CXX=mpiicpc, and FC=mpiifort explicitly. Try this if you have issues.


Build without OpenMP (Not recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use flags ``DOMP_AVAIL`` and ``DKokkos_ENABLE_OPENMP`` to toggle OpenMP functionally, with the first flag controlling typical OMP use, and the second applying to OMP functionality provided through Kokkos. The default is ``DOMP_AVAIL=ON`` and ``DKokkos_ENABLE_OPENMP=ON``. Setting ``DOMP_AVAIL=OFF`` will disable the use of OMP and OMP through Kokkos, though performance will take a hit if you must do this.


Kokkos build (For GPU use)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Phoebe utilizes Kokkos to generate either OpenMP or CUDA code to accelerate parts of the code.
Currently, this can be used to accelerate the calculation of phonon-phonon or electron-phonon scattering rates (the phononTransport and electronWannierTransport apps):

To build for use with Kokkos on GPUs, specify the flags to enable CUDA and OpenMP functionality. In the case where one wants to utilize GPUs::

  # CMake flags to build Kokkos with CUDA for GPU architectures
  cmake .. -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DOMP_AVAIL=ON
  make -j$(nproc)

Additionally, replace VOLTA70 (e.g. for V100s GPUs) as shown with the arch of your GPU.

To build with Kokkos using OpenMP instead of GPUs (recommended if you don't have GPU architecture), use the OpenMP build described above::

  # CMake flags to build Kokkos with OMP for CPU architectures
  cmake .. -DKokkos_ENABLE_OPENMP=ON -DOMP_AVAIL=ON
  make -j$(nproc)

Phoebe also accepts all the CMake arguments of Kokkos, which can improve performance.
For example, to attain better performance, you could specify ``-DKokkos_ARCH_KNL=ON`` in the above line when building for Knight's Landing nodes.

.. note::

   A Kokkos build compiled for GPUs won't necessarily work on CPU architecture,
   though apps which do not use Kokkos (all but phononTransport and
   electronWannierTransport) will of course still work on CPU regardless.
   It may be useful to build two copies of Phoebe if you want to occasionally use either kind of architecture for phonon-phonon/electron-phonon scattering calculations.

Compiling the documentation
---------------------------

In order to compile the documentation locally (the same documentation as on the Phoebe website), you need to have the following available on your machine:

* doxygen

* graphviz

* pdflatex (to render equations)

Then type::

  cd build
  make doc

Note that compiling the documentation doesn't require compiling the code.


Installation instructions for specific systems
--------------------------------------------------------------------

SLURM-based compute clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many compute clusters currently use SLURM and the related module system to manage the dependencies
you need to build Phoebe. If your cluster uses SLURM, you should try to build Phoebe by running "module spider HDF5" (or perhaps "hdf5"). Then, use "module spider" to look up specific versions of HDF5, until you identify one which requires an MPI verison to be loaded (like OpenMPI, Intel mpi/impi, or MPICH). Load all the modules related to that HDF5 version, plus "module load CMake". In total, you will want something similar to::

  module load gcc openmpi HDF5 cmake
  # or
  module load intel impi HDF5 cmake

If your cluster also has a module with a name like "intel-mkl" or "imkl", we suggest loading that as well, because CMake will use it for the ScaLAPACK dependency.

While the capitalization/names of these modules may vary, once you have a module set with parallel HDF5 (one which requires an MPI version) you will almost certainly be able to build the code using the "Basic Build" instructions above.

NERSC (Perlmutter)
^^^^^^^^^^^^^^^^^^

In `/phoebe/scripts/sampleBuildScripts/perlmutter.sh` in the Phoebe github repository, we have instructions which should work for building Phoebe on Perlmutter (for GPUs or cpus) as tested in Sept. 2023.

Ubuntu
^^^^^^

To install (without GPU support)::

  sudo apt install cmake gcc doxygen graphviz libomp-dev libopenmpi3 libhdf5-openmpi-dev
  git submodule update --init
  mkdir build
  cd build
  cmake .. -DKokkos_ENABLE_OPENMP=ON -DOMP_AVAIL=ON -DCMAKE_CXX_STANDARD_LIBRARIES="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/" -DCMAKE_CXX_FLAGS="-I/usr/include/hdf5/openmpi/"
  make -j$(nproc)
  make doc

Note that paths to the HDF5 library may need to be updated.
Tested on Ubuntu 20.04.

MacOS
^^^^^

* We have encountered difficulty linking the ScaLAPACK library, especially when linking with libgfortran. If libgfortran is not found, try adding it specifically to ``LD_LIBRARY_PATH`` or ``LIBRARY_PATH`` as follows:
  ::

    export LIBRARY_PATH=$LIBRARY_PATH:/path/to/libgfortran/

  In particular, if you are using a version of gcc installed using homebrew, you might need to link the "Cellar" copy of libgfortran. As an example working for gcc 12 is::

    export LIBRARY_PATH=$LIBRARY_PATH:/opt/homebrew/Cellar/gcc/12.2.0/lib/gcc/12/

* Additonally, there exists an issue when building with the Apple Clang compiler
  and the Eigen library, specifically when Eigen is built using OpenMP with a c++ std>11. We recommend either building without OpenMP (``cmake -DOMP_AVAIL=OFF ../``), or using a different compiler.
