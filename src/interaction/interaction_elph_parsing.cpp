#include "interaction_elph.h"
#include <fstream>

#ifdef HDF5_AVAIL
#include <highfive/H5Easy.hpp>
#endif

// specific parse function for the case where there is no
// HDF5 available
InteractionElPhWan parseNoHDF5(Context &context, Crystal &crystal,
                               PhononH0& phononH0_) {

  std::string fileName = context.getElphFileName();

  int numElectrons, numSpin;
  int numElBands, numElBravaisVectors, numPhBands, numPhBravaisVectors;
  numElBravaisVectors = 0; // suppress initialization warning
  Eigen::MatrixXd phBravaisVectors_, elBravaisVectors_;
  Eigen::VectorXd phBravaisVectorsDegeneracies_, elBravaisVectorsDegeneracies_;
  Eigen::Tensor<std::complex<double>, 5> couplingWannier_;

  // Open ElPh file
  if (mpi->mpiHeadPool()) {
    std::ifstream infile(fileName);
    if (not infile.is_open()) {
      Error("ElPh file not found");
    }

    // Read the bravais lattice vectors info for q mesh.
    infile >> numElectrons >> numSpin;

    int kx, ky, kz;
    int qx, qy, qz;
    infile >> kx >> ky >> kz;
    infile >> qx >> qy >> qz;

    int iCart;

    infile >> iCart >> numPhBravaisVectors;
    phBravaisVectors_.resize(3, numPhBravaisVectors);
    phBravaisVectorsDegeneracies_.resize(numPhBravaisVectors);
    phBravaisVectors_.setZero();
    phBravaisVectorsDegeneracies_.setZero();
    for (int i : {0, 1, 2}) {
      for (int j = 0; j < numPhBravaisVectors; j++) {
        infile >> phBravaisVectors_(i, j);
      }
    }
    for (int i = 0; i < numPhBravaisVectors; i++) {
      infile >> phBravaisVectorsDegeneracies_(i);
    }

    int totalNumElBravaisVectors;
    infile >> iCart >> totalNumElBravaisVectors;

    auto localElVectors = mpi->divideWorkIter(totalNumElBravaisVectors, mpi->intraPoolComm);
    numElBravaisVectors = int(localElVectors.size());

    elBravaisVectors_.resize(3, numElBravaisVectors);
    elBravaisVectors_.setZero();
    for (int i : {0, 1, 2}) {
      for (int j = 0; j < totalNumElBravaisVectors; j++) {
        double x;
        infile >> x;
        if (std::find(localElVectors.begin(),localElVectors.end(), j) != localElVectors.end() ) {
          auto localIrE = int(j - localElVectors[0]);
          elBravaisVectors_(i, localIrE) = x;
        }
      }
    }
    elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
    elBravaisVectorsDegeneracies_.setZero();
    for (int i = 0; i < totalNumElBravaisVectors; i++) {
      double x;
      infile >> x;
      if (std::find(localElVectors.begin(),localElVectors.end(), i) != localElVectors.end() ) {
        auto localIrE = int(i - localElVectors[0]);
        elBravaisVectorsDegeneracies_(localIrE) = x;
      }
    }
    std::string line;
    std::getline(infile, line);

    // Read real space matrix elements for el-ph coupling
    int tmpI;
    infile >> numElBands >> tmpI >> numPhBands >> tmpI >> tmpI;

    // user info about memory
    {
      std::complex<double> cx;
      (void) cx;
      double x = numElBands * numElBands * numPhBands * numPhBravaisVectors *
                 numElBravaisVectors / pow(1024., 3) * sizeof(cx);
      std::cout << "Allocating " << x
                << " (GB) (per MPI process) for the el-ph coupling matrix."
                << std::endl;
    }

    couplingWannier_.resize(numElBands, numElBands, numPhBands,
                            numPhBravaisVectors, numElBravaisVectors);
    couplingWannier_.setZero();
    double re, im;
    for (int i5 = 0; i5 < totalNumElBravaisVectors; i5++) {
      int localIrE = -1;
      if (std::find(localElVectors.begin(),localElVectors.end(), i5) != localElVectors.end() ) {
        localIrE = int(i5 - localElVectors[0]);
      }
      for (int i4 = 0; i4 < numPhBravaisVectors; i4++) {
        for (int i3 = 0; i3 < numPhBands; i3++) {
          for (int i2 = 0; i2 < numElBands; i2++) {
            for (int i1 = 0; i1 < numElBands; i1++) {
              infile >> re >> im;
              // note: in qe2Phoebe, the first index is on k+q bands,
              // and the second is on the bands of k. Here I invert them
              // similarly, in qe2Phoebe I inverted the order of R_el and R_ph
              if (localIrE >= 0) {
                couplingWannier_(i1, i2, i3, i4, localIrE) = {re, im};
              }
            }
          }
        }
      }
    }
  } // mpiHead done reading file

  mpi->bcast(&numElectrons);
  mpi->bcast(&numSpin);

  mpi->bcast(&numElBands);
  mpi->bcast(&numPhBands);
  mpi->bcast(&numElBravaisVectors, mpi->interPoolComm);
  mpi->bcast(&numPhBravaisVectors);

  if (numSpin == 2) {
    Error("Spin-polarized and spin-non-colinear calculations are not currently supported.");
  }
  context.setNumOccupiedStates(numElectrons);

  if (!mpi->mpiHeadPool()) { // head already allocated these
    phBravaisVectors_.resize(3, numPhBravaisVectors);
    phBravaisVectorsDegeneracies_.resize(numPhBravaisVectors);
    elBravaisVectors_.resize(3, numElBravaisVectors);
    elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
    couplingWannier_.resize(numElBands, numElBands, numPhBands,
                            numPhBravaisVectors, numElBravaisVectors);
  }
  mpi->bcast(&elBravaisVectors_, mpi->interPoolComm);
  mpi->bcast(&elBravaisVectorsDegeneracies_, mpi->interPoolComm);
  mpi->bcast(&phBravaisVectors_);
  mpi->bcast(&phBravaisVectorsDegeneracies_);
  mpi->bcast(&couplingWannier_, mpi->interPoolComm);

  // there is no JDFTx parsing option for this function, so we will always have
  // phase convention 0, in which we use Re and Rp
  int phaseConvention_ = 0;

  InteractionElPhWan output(crystal, couplingWannier_, elBravaisVectors_,
                            elBravaisVectorsDegeneracies_, phBravaisVectors_,
                            phBravaisVectorsDegeneracies_, phaseConvention_, phononH0_);
  return output;
}

#ifdef HDF5_AVAIL

std::tuple<int, int, int, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<size_t>,
    Eigen::VectorXd, Eigen::VectorXd, int> parseHeaderHDF5(Context &context) {

  std::string fileName = context.getElphFileName();

  // Here, phase convention corresponds to using
  // g(Re,Rp) (phase convention = 1) or g(Re',Re) (phase convention = 2)
  int numElectrons, numSpin, phaseConvention;
  int numElBands, numElBravaisVectors, totalNumElBravaisVectors, numPhBands, numPhBravaisVectors;
  // suppress initialization warning
  numElBravaisVectors = 0; totalNumElBravaisVectors = 0; numPhBravaisVectors = 0;
  Eigen::MatrixXd phBravaisVectors_, elBravaisVectors_;
  Eigen::VectorXd phBravaisVectorsDegeneracies_, elBravaisVectorsDegeneracies_;
  Eigen::Tensor<std::complex<double>, 5> couplingWannier_;
  std::vector<size_t> localElVectors;

  try {
    // Use MPI head only to read in the small data structures
    // then distribute them below this
    if (mpi->mpiHeadPool()) {
      // need to open the files differently if MPI is available or not
      // NOTE: do not remove the braces inside this if -- the file must
      // go out of scope, so that it can be reopened for parallel
      // read in the next block.
      {
        // Open the HDF5 ElPh file
        HighFive::File file(fileName, HighFive::File::ReadOnly);

        // read in the number of electrons and the spin
        HighFive::DataSet dnelec = file.getDataSet("/numElectrons");
        HighFive::DataSet dnspin = file.getDataSet("/numSpin");
        dnelec.read(numElectrons);
        dnspin.read(numSpin);

        // read in the number of phonon and electron bands
        HighFive::DataSet dnElBands = file.getDataSet("/numElBands");
        HighFive::DataSet dnModes = file.getDataSet("/numPhModes");
        dnElBands.read(numElBands);
        dnModes.read(numPhBands);

        // if phaseConvention not in the file, it's a typical QE file and phaseConv = 0
        // if it's set, this is a jdftx file and it should be set to zero
        try {
          HighFive::DataSet dPhaseConvention = file.getDataSet("/phaseConvention");
          dPhaseConvention.read(phaseConvention);
        } catch (std::exception &error) {
          phaseConvention = 0;
        }

        std::string datasetPhBravaisVectors = "/phBravaisVectors";
        std::string datasetPhDegeneracies = "/phDegeneracies";
        std::string datasetElBravaisVectors = "/elBravaisVectors";
        std::string datasetElDegeneracies = "/elDegeneracies";

        // if this is a jdftx elph file, we need to read in elphBravaisVectors
        // here, instead of el and ph bravais vectors, we just have one set of R
        // vectors saved under elphBravaisVectors
        if(phaseConvention == 1) {
          datasetPhBravaisVectors = "/elphBravaisVectors";
          datasetPhDegeneracies = "/elphDegeneracies";
          datasetElBravaisVectors = "/elphBravaisVectors";
          datasetElDegeneracies = "/elphDegeneracies";
        }

        // read phonon bravais lattice vectors and degeneracies
        HighFive::DataSet dphbravais = file.getDataSet(datasetPhBravaisVectors);
        HighFive::DataSet dphDegeneracies = file.getDataSet(datasetPhDegeneracies);
        dphbravais.read(phBravaisVectors_);
        dphDegeneracies.read(phBravaisVectorsDegeneracies_);
        numPhBravaisVectors = int(phBravaisVectors_.cols());

        // read electron Bravais lattice vectors and degeneracies
        HighFive::DataSet delDegeneracies = file.getDataSet(datasetElDegeneracies);
        delDegeneracies.read(elBravaisVectorsDegeneracies_);
        totalNumElBravaisVectors = int(elBravaisVectorsDegeneracies_.size());
        numElBravaisVectors = int(elBravaisVectorsDegeneracies_.size());
        HighFive::DataSet delbravais = file.getDataSet(datasetElBravaisVectors);
        delbravais.read(elBravaisVectors_);

        // redistribute in case of pools are present
        if (mpi->getSize(mpi->intraPoolComm) > 1) {
          localElVectors = mpi->divideWorkIter(totalNumElBravaisVectors, mpi->intraPoolComm);
          numElBravaisVectors = int(localElVectors.size());
          // copy a subset of elBravaisVectors
          Eigen::VectorXd tmp1 = elBravaisVectorsDegeneracies_;
          Eigen::MatrixXd tmp2 = elBravaisVectors_;
          elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
          elBravaisVectors_.resize(3, numElBravaisVectors);
          int i = 0;
          for (auto irE : localElVectors) {
            elBravaisVectorsDegeneracies_(i) = tmp1(irE);
            elBravaisVectors_.col(i) = tmp2.col(irE);
            i++;
          }
        }
      }
    }
    // broadcast to all MPI processes
    mpi->bcast(&numElectrons);
    mpi->bcast(&numSpin);
    mpi->bcast(&numPhBands);
    mpi->bcast(&numPhBravaisVectors);
    mpi->bcast(&numElBands);
    mpi->bcast(&numElBravaisVectors, mpi->interPoolComm);
    mpi->bcast(&totalNumElBravaisVectors, mpi->interPoolComm);
    mpi->bcast(&numElBravaisVectors, mpi->interPoolComm);
    mpi->bcast(&phaseConvention);

    // phase convention 1 is used by JDFTx, which supports spin
    if (numSpin == 2 && phaseConvention == 0) {
      Error("Spin is not currently supported when using QE.");
    } else if (numSpin == 2 && phaseConvention == 1) { // this is spin polarized JDFTx
      Warning("Spin-polarized JDFTx calculations should be closely monitored, as they are not well tested!");
      bool spinOrbit = true;
      context.setHasSpinOrbit(spinOrbit);
    }
    context.setNumOccupiedStates(numElectrons);

    if (!mpi->mpiHeadPool()) {// head already allocated these
      localElVectors = mpi->divideWorkIter(totalNumElBravaisVectors, mpi->intraPoolComm);
      phBravaisVectors_.resize(3, numPhBravaisVectors);
      phBravaisVectorsDegeneracies_.resize(numPhBravaisVectors);
      elBravaisVectors_.resize(3, numElBravaisVectors);
      elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
      couplingWannier_.resize(numElBands, numElBands, numPhBands,
                              numPhBravaisVectors, numElBravaisVectors);
    }
    mpi->bcast(&elBravaisVectors_, mpi->interPoolComm);
    mpi->bcast(&elBravaisVectorsDegeneracies_, mpi->interPoolComm);
    mpi->bcast(&phBravaisVectors_, mpi->interPoolComm);
    mpi->bcast(&phBravaisVectorsDegeneracies_, mpi->interPoolComm);
  } catch (std::exception &error) {
    Error("Issue reading elph Wannier representation header data from hdf5.");
  }

  return std::make_tuple(numElBands, numPhBands, totalNumElBravaisVectors, elBravaisVectors_,
          phBravaisVectors_, localElVectors, elBravaisVectorsDegeneracies_,
          phBravaisVectorsDegeneracies_, phaseConvention);
}

// specific parse function for the case where parallel HDF5 is available
InteractionElPhWan parseHDF5V1(Context &context, Crystal &crystal,
                               PhononH0& phononH0_) {

  Kokkos::Profiling::pushRegion("parseHDF5V1");

  std::string fileName = context.getElphFileName();

  auto t = parseHeaderHDF5(context);
  int numElBands = std::get<0>(t);
  int numPhBands = std::get<1>(t);
  int totalNumElBravaisVectors = std::get<2>(t);
  Eigen::MatrixXd elBravaisVectors_ = std::get<3>(t);
  Eigen::MatrixXd phBravaisVectors_ = std::get<4>(t);
  std::vector<size_t> localElVectors = std::get<5>(t);
  Eigen::VectorXd elBravaisVectorsDegeneracies_ = std::get<6>(t);
  Eigen::VectorXd phBravaisVectorsDegeneracies_ = std::get<7>(t);
  int numElBravaisVectors = elBravaisVectorsDegeneracies_.size();
  int numPhBravaisVectors = phBravaisVectorsDegeneracies_.size();
  int phaseConvention = std::get<8>(t);

  if(numElBravaisVectors < mpi->getSize() && phaseConvention == 1) {
    Error("JDFTx input files cannot be used when nMPI processes > numElBravaisVectors = "
     + std::to_string(numElBravaisVectors));
  }

  Eigen::Tensor<std::complex<double>, 5> couplingWannier_;

  try {
    // Define the eph matrix element containers

    // This is broken into parts, otherwise it can overflow if done all at once
    size_t totElems = numElBands * numElBands * numPhBands;
    totElems *= numPhBravaisVectors;
    totElems *= numElBravaisVectors;

    // user info about memory
    {
      std::complex<double> cx;
      auto x = double(totElems / pow(1024., 3) * sizeof(cx));
      if (mpi->mpiHead()) {
        std::cout << "Allocating " << x
                  << " (GB) (per MPI process) for the el-ph coupling matrix."
                  << std::endl;
      }
    }

    couplingWannier_.resize(numElBands, numElBands, numPhBands,
                            numPhBravaisVectors, numElBravaisVectors);
    couplingWannier_.setZero();

// Regular parallel read
#if defined(MPI_AVAIL) && !defined(HDF5_SERIAL)

    // Set up buffer to receive full matrix data
    Eigen::VectorXcd gWanFlat(couplingWannier_.size());

    // we will either divide the work over ranks, or we will divide the work
    // over the processes in the head pool
    int comm;
    size_t start, stop, offset, numElements;

    // if there's only one pool (aka, no pools) each process reads
    // in a piece of the matrix from file.
    // Then, at the end, we gather the pieces into one big gWan matrix.
    if(mpi->getSize(mpi->intraPoolComm) == 1) {
      comm = mpi->worldComm;
      // start and stop points use divideWorkIter in the case without pools
      start = mpi->divideWorkIter(numElBravaisVectors, comm)[0] * numElBands *
              numElBands * numPhBands * numPhBravaisVectors;
      stop = (mpi->divideWorkIter(numElBravaisVectors, comm).back() + 1) *
              numElBands* numElBands * numPhBands * numPhBravaisVectors - 1;
      offset = start;
      numElements = stop - start + 1;
    // else we have the pools case, in which each process on the head
    // pool reads in a piece of the matrix (associated with whatever chunk
    // of the bravais vectors it has), then we broadcast this information to
    // all pools.
    } else {
      comm = mpi->intraPoolComm;
      // each process has its own chunk of bravais vectors,
      // and we need to read in all the elements associated with
      // those vectors
      start = 0;
      stop = numElBravaisVectors * numElBands *
              numElBands * numPhBands * numPhBravaisVectors;
      // offset indexes the chunk we want to read in within the elph hdf5
      // file, and indicates where this block starts in the full matrix
      offset = localElVectors[0] * numElBands *
              numElBands * numPhBands * numPhBravaisVectors;;
      numElements = stop - start + 1;
    }

    // Reopen the HDF5 ElPh file for parallel read of eph matrix elements
    HighFive::FileAccessProps fapl = HighFive::FileAccessProps{};
    fapl.add(HighFive::MPIOFileAccess<MPI_Comm, MPI_Info>(mpi->getComm(comm), MPI_INFO_NULL));
    HighFive::File file(fileName, HighFive::File::ReadOnly, fapl);

    // Set up dataset for gWannier
    HighFive::DataSet dgWannier = file.getDataSet("/gWannier");

    // if this chunk of elements to be written by this process
    // is greater than 2 GB, we must split it further due to a
    // limitation of HDF5 which prevents read/write of
    // more than 2 GB at a time.

    // below, note the +1/-1 indexing on the start/stop numbers.
    // This has to do with the way divideWorkIter sets the range
    // of work to be done -- it uses indexing from 0 and doesn't
    // include the last element as a result.
    //
    // start/stop points and the number of the total number of elements
    // to be written by this process

    // maxSize represents ~1 GB worth of std::complex<doubles>
    // this is the maximum amount we feel is safe to read at once.
    auto maxSize = int(pow(1000, 3)) / sizeof(std::complex<double>);
    // the size of all elements associated with one electronic BV
    size_t sizePerBV =
        numElBands * numElBands * numPhBands * numPhBravaisVectors;
    std::vector<int> irEBunchSizes;

    // determine the # of eBVs to be read by this process.
    // the bunchSizes vector tells us how many BVs each process will read
    int numEBVs = int(mpi->divideWorkIter(totalNumElBravaisVectors, comm).back() + 1 -
           mpi->divideWorkIter(totalNumElBravaisVectors, comm)[0]);

    // loop over eBVs and add them to the current bunch until
    // we reach the maximum read size
    int irEBunchSize = 0;
    for (int irE = 0; irE < numEBVs; irE++) {
      irEBunchSize++;
      // this bunch is as big as possible, stop adding to it
      if ((irEBunchSize + 1) * sizePerBV > maxSize) {
         irEBunchSizes.push_back(irEBunchSize);
         irEBunchSize = 0;
      }
    }
    // push the last one, no matter the size, to the list of bunch sizes
    irEBunchSizes.push_back(irEBunchSize);

    // Set up buffer to be filled from hdf5, enough for total # of elements
    // to be read in by this process
    Eigen::VectorXcd gWanSlice(numElements);

    // determine the number of bunches -- not necessarily evenly sized
    auto numBunches = int(irEBunchSizes.size());

    // counter for offset from first element on this rank to current element
    size_t bunchOffset = 0;
    // we now loop over these bunch of eBVs, and read each bunch of
    // bravais vectors in parallel
    for (int iBunch = 0; iBunch < numBunches; iBunch++) {

      // we need to determine the start, stop and offset of this
      // sub-slice of the dataset available to this process
      size_t bunchElements = irEBunchSizes[iBunch] * sizePerBV;
      size_t totalOffset = offset + bunchOffset;

      Eigen::VectorXcd gWanBunch(bunchElements);

      // Read in the elements for this process
      // into this bunch's location in the slice which will
      // hold all the elements to be read by this process
      dgWannier.select({0, totalOffset}, {1, bunchElements}).read(gWanBunch);

      // Perhaps this could be more effective.
      // however, HiFive doesn't seem to allow me to pass
      // a slice of gWanSlice, so we have instead read to gWanBunch
      // then copy it over
      //
      // copy bunch data into gWanSlice
      for (size_t i = 0; i<bunchElements; i++) {
        // if we're using pool, each pool proc has its own gwanFlat
        if(comm == mpi->intraPoolComm) {
          gWanFlat[i+bunchOffset] = gWanBunch[i];
        }
        // if no pools, each proc writes to a slice of the matrix
        // which is later gathered to build the full one
        else { gWanSlice[i+bunchOffset] = gWanBunch[i]; }
      }
      // calculate the offset for the next bunch
      bunchOffset += bunchElements;
    }

    // collect and broadcast the matrix elements now that they have been read in

    // We have the standard case of 1 pool (aka no pools),
    // and we need to gather the components of the matrix into one big matrix
    if(comm != mpi->intraPoolComm)  {
      // collect the information about how many elements each mpi rank has
      std::vector<size_t> workDivisionHeads(mpi->getSize());
      mpi->allGather(&offset, &workDivisionHeads);
      std::vector<size_t> workDivs(mpi->getSize());
      size_t numIn = gWanSlice.size();
      mpi->allGather(&numIn, &workDivs);

      // Gather the elements read in by each process
      mpi->bigAllGatherV(gWanSlice.data(), gWanFlat.data(),
        workDivs, workDivisionHeads, comm);
    }
    // In the case of pools, where we read in only on the head pool,
    // we now send it to all the other pools
    //if(mpi->getSize(mpi->intraPoolComm) != 1) {
    else {
      mpi->bcast(&gWanFlat, mpi->interPoolComm);
    }

    // Map the flattened matrix back to tensor structure
    Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>> gWanTemp(
        gWanFlat.data(), numElBands, numElBands, numPhBands,
        numPhBravaisVectors, numElBravaisVectors);
    couplingWannier_ = gWanTemp;

#else
    // Reopen serial version, either because MPI does not exist
    // or because we forced HDF5 to run in serial.

    // Set up buffer to receive full matrix data
    std::vector<std::complex<double>> gWanFlat(totElems);

    if (mpi->getSize(mpi->intraPoolComm) == 1) {
      if (mpi->mpiHead()) {
        HighFive::File file(fileName, HighFive::File::ReadOnly);

        // Set up dataset for gWannier
        HighFive::DataSet dgWannier = file.getDataSet("/gWannier");
        // Read in the elements for this process
        dgWannier.read(gWanFlat);
      }
      mpi->bcast(&gWanFlat);

    } else {
      if (mpi->mpiHeadPool()) {
        HighFive::File file(fileName, HighFive::File::ReadOnly);

        // Set up dataset for gWannier
        HighFive::DataSet dgWannier = file.getDataSet("/gWannier");
        // Read in the elements for this process
        size_t offset = localElVectors[0] * pow(numElBands, 2) * numPhBravaisVectors * numPhBands;
        size_t extent = numElBravaisVectors * pow(numElBands, 2) * numPhBravaisVectors * numPhBands;
        dgWannier.select({0, offset}, {1, extent}).read(gWanFlat);
      }
      mpi->bcast(&gWanFlat, mpi->interPoolComm);
    }

    // Map the flattened matrix back to tensor structure
    Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>> gWanTemp(
        gWanFlat.data(), numElBands, numElBands, numPhBands,
        numPhBravaisVectors, numElBravaisVectors);
    couplingWannier_ = gWanTemp;

#endif

  } catch (std::exception &error) {
    Error("Issue reading elph Wannier representation from hdf5.");
  }

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("Interaction constructor");
  InteractionElPhWan output(crystal, couplingWannier_, elBravaisVectors_,
                            elBravaisVectorsDegeneracies_, phBravaisVectors_,
                            phBravaisVectorsDegeneracies_, phaseConvention, phononH0_);
  Kokkos::Profiling::popRegion();
  return output;

}

// specific parse function for the case where parallel HDF5 is available
InteractionElPhWan parseHDF5V2(Context &context, Crystal &crystal, PhononH0 &phononH0_) {

  std::string fileName = context.getElphFileName();

  auto t = parseHeaderHDF5(context);
  int numElBands = std::get<0>(t);
  int numPhBands = std::get<1>(t);
  Eigen::MatrixXd elBravaisVectors_ = std::get<3>(t);
  Eigen::MatrixXd phBravaisVectors_ = std::get<4>(t);
  std::vector<size_t> localElVectors = std::get<5>(t);
  Eigen::VectorXd elBravaisVectorsDegeneracies_ = std::get<6>(t);
  Eigen::VectorXd phBravaisVectorsDegeneracies_ = std::get<7>(t);
  int numElBravaisVectors = elBravaisVectorsDegeneracies_.size();
  int numPhBravaisVectors = phBravaisVectorsDegeneracies_.size();
  int phaseConvention = std::get<8>(t);
  if(phaseConvention == 1) {
    DeveloperError("There is no phaseConvention=1 possibility for parseHDF5 version 2!");
  }

  Eigen::Tensor<std::complex<double>, 5> couplingWannier_;

  try {
    // Define the eph matrix element containers

    // This is broken into parts, otherwise it can overflow if done all at once
    size_t totElems = numElBands * numElBands * numPhBands;
    totElems *= numPhBravaisVectors;
    totElems *= numElBravaisVectors;

    // user info about memory
    {
      std::complex<double> cx;
      auto x = double(totElems / pow(1024., 3) * sizeof(cx));
      if (mpi->mpiHead()) {
        std::cout << "Allocating " << x
                  << " (GB) (per MPI process) for the el-ph coupling matrix."
                  << std::endl;
      }
    }

    couplingWannier_.resize(numElBands, numElBands, numPhBands,
                            numPhBravaisVectors, numElBravaisVectors);
    couplingWannier_.setZero();

    // Set up buffer to receive full matrix data
    size_t sliceElements = pow(numElBands,2) * numPhBands * numPhBravaisVectors;
    Eigen::VectorXcd slice(sliceElements);

    if (mpi->getSize(mpi->intraPoolComm) > 1) { // case with pools
      if (mpi->mpiHeadPool()) {
        HighFive::File file(fileName, HighFive::File::ReadOnly);
        for (int irE : localElVectors) {
          int irELocal = irE - localElVectors[0];
          std::string datasetName = "/gWannier_" + std::to_string(irE);
          // Set up dataset for gWannier
          HighFive::DataSet dslice = file.getDataSet(datasetName);
          dslice.read(slice);

          // Map the flattened matrix back to tensor structure
          Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 4>> sliceTemp(
              slice.data(), numElBands, numElBands, numPhBands, numPhBravaisVectors);

          for (int irP = 0; irP < numPhBravaisVectors; irP++) {
            for (int nu = 0; nu < numPhBands; nu++) {
              for (int i = 0; i < numElBands; i++) {
                for (int j = 0; j < numElBands; j++) {
                  couplingWannier_(i, j, nu, irP, irELocal) = sliceTemp(i, j, nu, irP);
                }
              }
            }
          }
        }
      }
      mpi->bcast(&couplingWannier_, mpi->interPoolComm);

    } else { // case without pools
      HighFive::File file(fileName, HighFive::File::ReadOnly);
      for (int irE : mpi->divideWorkIter(numElBravaisVectors)) {
        std::string datasetName = "/gWannier_" + std::to_string(irE);
        // Set up dataset for gWannier
        HighFive::DataSet dslice = file.getDataSet(datasetName);
        dslice.read(slice);

        // Map the flattened matrix back to tensor structure
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 4>> sliceTemp(
            slice.data(), numElBands, numElBands, numPhBands, numPhBravaisVectors);

        for (int irP = 0; irP < numPhBravaisVectors; irP++) {
          for (int nu = 0; nu < numPhBands; nu++) {
            for (int i = 0; i < numElBands; i++) {
              for (int j = 0; j < numElBands; j++) {
                couplingWannier_(i, j, nu, irP, irE) = sliceTemp(i, j, nu, irP);
              }
            }
          }
        }
      }
      mpi->bcast(&couplingWannier_);
    }
  } catch (std::exception &error) {
    Error("Issue reading elph Wannier representation from hdf5.");
  }

  InteractionElPhWan output(crystal, couplingWannier_, elBravaisVectors_,
                            elBravaisVectorsDegeneracies_, phBravaisVectors_,
                            phBravaisVectorsDegeneracies_, phaseConvention, phononH0_);
  return output;
}


// specific parse function for the case where parallel HDF5 is available
InteractionElPhWan parseHDF5(Context &context, Crystal &crystal,
                             PhononH0 &phononH0_) {
  // check for existence of file
  std::string fileName = context.getElphFileName();
  {
    std::ifstream infile(fileName);
    if (not infile.is_open()) {
      Error("Required electron-phonon file ***.phoebe.elph.hdf5 "
            "not found at " + fileName);
    }
  }

  int fileFormat = 1;
  try {
    // Use MPI head only to read in the small data structures
    if (mpi->mpiHead()) {
      // need to open the files differently if MPI is available or not
      // NOTE: do not remove the braces inside this if -- the file must
      // go out of scope, so that it can be reopened for parallel
      // read in the next block.
      {
        // Open the HDF5 ElPh file
        HighFive::File file(fileName, HighFive::File::ReadOnly);
        if ( file.exist("/fileFormat") ) {
          // note: the earlier versions of the HDF5 file didn't have a format id
          HighFive::DataSet dsFileFormat = file.getDataSet("/fileFormat");
          dsFileFormat.read(fileFormat);
        }
      }
    }
    mpi->bcast(&fileFormat);

  } catch (std::exception &error) {
    Error("Something wrong determining the HDF5 format.");
  }

  if (fileFormat==1) {
    return parseHDF5V1(context, crystal, phononH0_);
  } else {
    return parseHDF5V2(context, crystal, phononH0_);
  }
}

#endif

// General parse function
InteractionElPhWan InteractionElPhWan::parse(Context &context, Crystal &crystal,
                                             PhononH0& phononH0_) {
  if (mpi->mpiHead()) {
    std::cout << "\n";
    std::cout << "Started parsing of el-ph interaction." << std::endl;
  }
#ifdef HDF5_AVAIL
  auto output = parseHDF5(context, crystal, phononH0_);
#else
  auto output = parseNoHDF5(context, crystal, phononH0_);
#endif

  if (mpi->mpiHead()) {
    std::cout << "Finished parsing of el-ph interaction." << std::endl;
  }

  return output;
}
