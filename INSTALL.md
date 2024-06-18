# Building Phoebe
The installation requires:
* CMake
* C++ compiler with C++14 support. We regularly test with GCC, but Intel and Clang have also worked.
* Internet connection (we will download external libraries)

Optionally, Phoebe can also use MPI, OpenMP and CUDA, as described below.

Two executables will be created:
* phoebe
* runTests (for developers)

## Basic build procedure
```
git submodule update --init
mkdir build
cd build
cmake ..
make -j$(nproc)
```

This will build a basic version of Phoebe, with MPI and OpenMP if available.
Often, you can receive additional acceleration by following the OMP env variable
instructions from Kokkos:

  "In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
  For best performance with OpenMP 3.1 set OMP_PROC_BIND=true."

## CUDA build
```
git submodule update --init
mkdir build
cd build
cmake .. -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON
make -j$(nproc)
```
Replace `VOLTA70` (which is for V100s) with the arch of your GPU.

## Kokkos
Phoebe utilizes Kokkos to generate either OpenMP or CUDA code to accelerate parts of the code.
Thus Phoebe accepts all the CMake arguments of Kokkos, which can improve performance.
For example, you should specify `-DKokkos_ARCH_KNL=ON` when building for Knight's Landing nodes.


## Documentation
In order to compile the documentation, you need to have installed:
* doxygen
* graphviz
Then type
* `make doc`


## Notes for Mac Users:

#### For older, pre-M1 Macs: 
There can be trouble linking to the libgfortran files required by scalapack.
If libgfortran is not found, try adding it specifically to LD_LIBRARY_PATH:
```
export LIBRARY_PATH=/path/to/libgfortran/
```
If using homebrew installed gcc, you might need to add the cellar copy rather than the
one located with gcc:
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/homebrew/Cellar/gcc/12.2.0/lib/gcc/12/
```

#### For newer Macs with Apple Silicon chips: 

```
# if you don't have these already, use homebrew to install: 

brew install cmake
brew install gcc
brew install llvm
# stop here and follow the instructions llvm provides regarding setting path variables --
# run whatever line it tells you that looks like the one below, where username should be your laptop username. 
#echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> /Users/username/.bash_profile

brew install open-mpi
brew install hdf5-mpi
brew install scalapack
brew install libomp

# before building, set these flags
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-no_compact_unwind,-rpath,/opt/homebrew/opt/llvm/lib/c++"
export CXXFLAGS="-I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/libomp/include -fopenmp"
export CFLAGS="-I/usr/local/opt/libomp/include -I/opt/homebrew/opt/llvm/include"
export LIBRARY_PATH=$LIBRARY_PATH:/opt/homebrew/Cellar/gcc/13.2.0/lib/gcc/13/
export SDKROOT=$(xcrun --show-sdk-path)

mkdir build
cd build
cmake ../
make -j 4 phoebe  # change number to appropriate number of cpus
```
