# run this in an empty directory, or in one directory outside your 
# phoebe directory

# change the items below this line ===============

export OMP_ON="ON"
export MPI_ON="ON"

# choose if you want to set up the code, download the 
# test data, or run the tests
BUILD=false
DOWNLOAD=false
RUN=false

if [ "$MPI_ON" == "ON" ]
then
  mpiCommand="mpirun -np 4"
else
  mpiCommand=""
fi

export OMP_NUM_THREADS=4
export MPI_PROCS=4

# use this if on a slurm system, replace with
# your modules
module restore phoebe

BUILD_DIR="build/"
PHOEBE_DIR="phoebe/"

# SET UP AND RUN THE TESTS ============

# pull down the code
if ${BUILD}
then
  git clone git@github.com:mir-group/phoebe.git
  cd phoebe
  git submodule update --init
  mkdir build
  cd build
  cmake ../ -DMPI_AVAIL=${MPI_ON} -DOMP_AVAIL=${OMP_ON}
  make -j 4 phoebe
  #make -j 4 runTests
  cd ../../
fi

# get the test files from online
if ${DOWNLOAD}
then
  cd phoebe
  wget github.com/mir-group/phoebe-data/archive/master.zip
  unzip -j master.zip "phoebe-data-master/example/Silicon-ph/qe-phonons/*" -d "example/Silicon-ph/qe-phonons"
  unzip -j master.zip "phoebe-data-master/example/Silicon-ph/qe-ph-anharmonic/*" -d "example/Silicon-ph/thirdorder.py-anharmonic"
  unzip -j master.zip "phoebe-data-master/example/Silicon-el/qe-elph/*" -d "example/Silicon-el/qe-elph"
  unzip 'example/Silicon-el/qe-elph/silicon.phoebe.*.dat.zip' -d example/Silicon-el/qe-elph/
  cp example/Silicon-el/qe-elph/* example/Silicon-epa/qe-elph
  mkdir example/Silicon-epa/qe-elph/out
  unzip -j master.zip "phoebe-data-master/example/Silicon-epa/qe-elph/out/*" -d "example/Silicon-epa/qe-elph/out"
  cd ../
fi

if ${RUN}
then

  cd phoebe

  echo "Run epa example"
  cd example/Silicon-epa
  rm elpaTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in qeToPhoebeEPA.in >> elpaTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in epaTransport.in >> elpaTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in electronFourierBands.in >> elpaTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in electronFourierDos.in >> elpaTest.out
  python3 reference/run_check.py
  cd ../../

  echo "Run ph example"
  cd example/Silicon-ph
  rm phTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in phononTransport.in >> phTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in phononBands.in >> phTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in phononDos.in >> phTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in phononLifetimes.in >> phTest.out
  python3 reference/run_check.py
  cd ../../

  echo "Run el example"
  cd example/Silicon-el
  rm elTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in qeToPhoebeWannier.in >> elTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in electronWannierTransport.in >> elTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in electronWannierBands.in >> elTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in electronWannierDos.in >> elTest.out
  ${mpiCommand} ../../${BUILD_DIR}/phoebe -in electronLifetimes.in >> elTest.out
  python3 reference/run_check.py

  # if we have mpi also check these with pools
  if [ "$MPI_ON" == "ON" ]
  then
    ${mpiCommand} ../../${BUILD_DIR}/phoebe -ps 2 -in electronWannierTransport.in >> elTest.out
    ${mpiCommand} ../../${BUILD_DIR}/phoebe -ps 2 -in electronLifetimes.in >> elTest.out
    python3 reference/run_check.py
  fi
  cd ../
fi
