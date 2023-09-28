# Build instructions for perlmutter
# run this in your home directory

# load relevant modules
# important to load cudatoolkit/11.5
module load PrgEnv-gnu gcc cudatoolkit/11.5 cray-libsci craype cray-mpich cray-hdf5-parallel cray-netcdf-hdf5parallel cray-parallel-netcdf cmake

# build a copy of openblas in your .local dir
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make PREFIX=$HOME/.local install
cd ../

# clone phoebe
git clone --recursive https://github.com/mir-group/phoebe.git
cd phoebe
mkdir build
cd build

# run cmake
CXX=CC CC=cc cmake .. -DCMAKE_PREFIX_PATH="$PWD/../deps/install;$HOME/.local;/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5" -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON -DOMP_AVAIL=ON -DMPI_AVAIL=ON

# make the code -- can run with more tasks if you're in an interactive session
make -j 2 phoebe

# save the module list if everything works, for easy future use
module save phoebe

# Perhaps, you could also build the test suite with make -j 2 runTests, and then test them in the build dir with ./runTests, to ensure the build was successful.

# Running Phoebe with GPUs ---------------------------------

 #    export MAXMEM=40  # max mem per gpu
 #    # perlmutter seemed to suggest this should be set
 #    export OMP_NUM_THREADS=1
 #    # for optimal kokkos performance
 #    export OMP_PROC_BIND=spread
 #    export OMP_PLACES=threads
 #
 #    # you can run Phoebe using the srun line, where n=numGPUs, and G=numGPUs
 #    # Right now, sometimes I see bottlenecks in the OMP/mpi heavy parts of the code. I need to experiment with -c numThreads + OMP_NUM_THREADS, or running with more than one mpi task / gpu to see if I can remedy this.
 #
 #    srun -n 2 --cpu_bind=cores -G 2 --gpu-bind=single:1 ../phoebe/build/phoebe -in lifetimes.in
 #
