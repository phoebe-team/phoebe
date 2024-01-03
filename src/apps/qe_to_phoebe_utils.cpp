#include "bandstructure.h"
#include "eigen.h"
#include "elph_qe_to_phoebe_app.h"
#include "interaction_elph.h"
#include "io.h"
#include "qe_input_parser.h"
#include "utilities.h"
#include <Kokkos_Core.hpp>
#include <exception>
#include <sstream>
#include <string>

#ifdef HDF5_AVAIL
#include <highfive/H5Easy.hpp>
#endif

void ElPhQeToPhoebeApp::testElectronicTransform(
    Points &kPoints, const std::string &wannierPrefix,
    const Eigen::MatrixXd &elBravaisVectors,
    const Eigen::Tensor<std::complex<double>, 3> &uMatrices,
    const Eigen::VectorXd &elDegeneracies, ElectronH0Wannier &electronH0) {
  /** This is a simple test:
   * 1) Fourier Transform the electronic Hamiltonian to Wannier representation
   *    Here I use the U matrices from file
   * 2) FT back to Bloch representation, using the U matrices from ElectronH0
   *    on the original grid of k-points
   * If everything works, I expect to find the same electronic energies
   * Phases of rotation matrices in the back-FT will be random.
   */

  int numBands = int(uMatrices.dimension(0));
  int numWannier = int(uMatrices.dimension(1));
  assert(numBands >= numWannier);

  Eigen::MatrixXd blochEnergies(numBands, kPoints.getNumPoints());
  blochEnergies.setZero();

  auto t = kPoints.getMesh();
  auto kMesh = std::get<0>(t);

  // I try the FFT of the energies
  for (int ik = 0; ik < kPoints.getNumPoints(); ik++) {
    auto kCrystal = kPoints.getPointCoordinates(ik);
    kCrystal(0) *= kMesh(0);
    kCrystal(1) *= kMesh(1);
    kCrystal(2) *= kMesh(2);

    int ikOld = int(kCrystal[0] * kMesh(2) * kMesh(1) + kCrystal[1] * kMesh(2) + kCrystal[2]);
    {
      std::string eigFileName = wannierPrefix + ".eig";
      std::ifstream eigenFile(eigFileName);
      if (not eigenFile.is_open()) {
        Error("Wannier90 *.eig file not found");
      }
      int ib, ikk;
      double x;
      while (eigenFile >> ib >> ikk >> x) {
        if (ikk - 1 == ikOld) {
          // Note: this causes some warnings from Eigen
          blochEnergies(ib - 1, ik) = x;
        }
      }
    }
  }

  //----------------------------------------------------------------------------
  // Now FT to Wannier representation
  Eigen::Tensor<std::complex<double>, 3> h0R(elBravaisVectors.cols(),
                                             numWannier, numWannier);
  h0R.setZero();
  for (int ik1 = 0; ik1 < kPoints.getNumPoints(); ik1++) {
    auto k1C = kPoints.getPointCoordinates(ik1, Points::cartesianCoordinates);

    // u has size (numBands, numWannier, numKPoints)
    Eigen::MatrixXcd uK(numBands, numWannier);
    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < numWannier; j++) {
        uK(i, j) = uMatrices(i, j, ik1);
      }
    }

    // Eq. 26 of Giustino PRB 2007. Note that the U are inverted
    Eigen::MatrixXcd h0K1(numBands, numBands);
    for (int ib = 0; ib < numBands; ib++) {
      h0K1(ib, ib) = {blochEnergies(ib, ik1), 0};
    }
    Eigen::MatrixXcd h0K(numWannier, numWannier);
    h0K = uK.transpose() * h0K1 * uK.adjoint().transpose();

    for (int iR = 0; iR < elBravaisVectors.cols(); iR++) {
      Eigen::Vector3d R = elBravaisVectors.col(iR);
      double arg = k1C.dot(R);
      std::complex<double> phase =
          exp(-complexI * arg) / double(kPoints.getNumPoints());
      for (int m = 0; m < numWannier; m++) {
        for (int n = 0; n < numWannier; n++) {
          h0R(iR, m, n) += phase * h0K(m, n);
        }
      }
    }
  }

  //  --------------------------------------------------------------------------
  // FFT back

  for (int ik = 0; ik < kPoints.getNumPoints(); ik++) {
    // get U
    auto k1C = kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);
    auto t3 = electronH0.diagonalizeFromCoordinates(k1C);
    auto en = std::get<0>(t3);
    auto u = std::get<1>(t3);

    Eigen::MatrixXcd h0K(numWannier, numWannier);
    h0K.setZero();
    for (int iR = 0; iR < elBravaisVectors.cols(); iR++) {
      Eigen::Vector3d R = elBravaisVectors.col(iR);
      double arg = k1C.dot(R);
      std::complex<double> phase = exp(complexI * arg) / elDegeneracies(iR);
      for (int m = 0; m < numWannier; m++) {
        for (int n = 0; n < numWannier; n++) {
          h0K(m, n) += phase * h0R(iR, m, n);
        }
      }
    }

    h0K = u.adjoint() * h0K * u;

    for (int ib = 0; ib < numWannier; ib++) {
      assert(abs(h0K(ib, ib).real() - blochEnergies(ib, ik)) < 1.0e-4);
    }
  }
}

void ElPhQeToPhoebeApp::testPhononTransform(
    Crystal &crystal, PhononH0 &phononH0, Points &qPoints,
    const Eigen::Tensor<std::complex<double>, 3> &phEigenvectors,
    const Eigen::MatrixXd &phBravaisVectors,
    const Eigen::VectorXd &phDegeneracies, const Eigen::MatrixXd &phEnergies) {
  /** Like the test above, we
   * 1) FT to Wannier representation.
   *    Since these are force constants, they should be real.
   * 2) FT back to Bloch space and check that we find the same results.
   *
   * We also verify that the eigenvectors are normalized by masses
   * Note that the test works fine for non-polar systems.
   */

  int numPhBands = int(phononH0.getNumBands());

  // Bloch To Wannier transform

  auto atomicPositions = crystal.getAtomicPositions();
  int numAtoms = int(atomicPositions.rows());
  auto atomicMasses = crystal.getAtomicMasses();

  // test mass normalization as expected
  for (int iq = 0; iq < qPoints.getNumPoints(); iq++) {
    Eigen::MatrixXcd norm(numPhBands, numPhBands);
    norm.setZero();
    for (int ib1 = 0; ib1 < numPhBands; ib1++) {
      for (int ib2 = 0; ib2 < numPhBands; ib2++) {
        for (int k1 = 0; k1 < numAtoms; k1++) {
          for (int iCart : {0, 1, 2}) {
            int i = compress2Indices(k1, iCart, numAtoms, 3);
            norm(ib1, ib2) +=
                phEigenvectors(i, ib1, iq) * sqrt(atomicMasses(k1)) * phEigenvectors(i, ib2, iq) * sqrt(atomicMasses(k1));
          }
        }
      }
      norm(ib1) -= 1.;// It should be an identity matrix
    }
    for (int ib1 = 0; ib1 < numPhBands; ib1++) {
      assert(abs(norm(ib1)) < 1.0e-6);
    }
  }

  // FT to Wannier representation

  Eigen::Tensor<std::complex<double>, 5> h0R(
      numAtoms * numAtoms * phBravaisVectors.size(), numAtoms, numAtoms, 3, 3);
  h0R.setZero();

  for (int iq = 0; iq < qPoints.getNumPoints(); iq++) {
    auto qC = qPoints.getPointCoordinates(iq, Points::cartesianCoordinates);
    qC = qPoints.bzToWs(qC, Points::cartesianCoordinates);

    // u has size (numBands, numWannier, numKPoints)
    Eigen::MatrixXcd uK(numPhBands, numPhBands);
    for (int k1 = 0; k1 < numAtoms; k1++) {
      for (int iCart : {0, 1, 2}) {
        int i = compress2Indices(k1, iCart, numAtoms, 3);
        for (int j = 0; j < numPhBands; j++) {
          uK(i, j) = phEigenvectors(i, j, iq) * sqrt(atomicMasses(k1));
        }
      }
    }
    assert(uK.inverse() == uK.adjoint());// check is unitary matrix

    // build dynamical matrix
    Eigen::MatrixXcd h0K(numPhBands, numPhBands);
    h0K.setZero();
    for (int ib = 0; ib < numPhBands; ib++) {
      h0K(ib, ib) = {phEnergies(ib, iq) * phEnergies(ib, iq), 0.};
    }
    h0K = uK * h0K * uK.adjoint();
    // if here multiply by mass, we get the QE results

    for (int iR = 0; iR < phBravaisVectors.cols(); iR++) {
      Eigen::Vector3d R0 = phBravaisVectors.col(iR);
      for (int k1 = 0; k1 < numAtoms; k1++) {
        for (int k2 = 0; k2 < numAtoms; k2++) {
          // Eigen::Vector3d R = R0; // - atomicPositions.col(k1)
          //+ atomicPositions.col(k2);
          double arg = qC.dot(R0);
          std::complex<double> phase =
              exp(-complexI * arg) / double(qPoints.getNumPoints());
          for (int iCart : {0, 1, 2}) {
            for (int jCart : {0, 1, 2}) {
              int m = compress2Indices(k1, iCart, numAtoms, 3);
              int n = compress2Indices(k2, jCart, numAtoms, 3);
              h0R(iR, k1, k2, iCart, jCart) += phase * h0K(m, n);
            }
          }
        }
      }
    }
  }

  // check that h0R, the force constants, are real
  {
    double realSum = 0.;
    double imaginarySum = 0.;
    for (int iR0 = 0; iR0 < phBravaisVectors.cols(); iR0++) {
      for (int k1 = 0; k1 < numAtoms; k1++) {
        for (int k2 = 0; k2 < numAtoms; k2++) {
          for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
              double x = std::real(h0R(iR0, k1, k2, i, j));
              realSum += pow(x, 2);
              imaginarySum += pow(std::imag(h0R(iR0, k1, k2, i, j)), 2);
              // set to zero the imaginary part to clean noise
              // this is also what QE does
              h0R(iR0, k1, k2, i, j) = {x, 0.};
            }
          }
        }
      }
    }
    // I want the imaginary part to be much smaller than the real
    assert(imaginarySum * pow(10, 6) < realSum);
  }

  //--------------------------------------------------------------------------
  // FFT back

  for (int iq = 0; iq < qPoints.getNumPoints(); iq++) {
    // get U
    auto qC = qPoints.getPointCoordinates(iq, Points::cartesianCoordinates);
    auto t = phononH0.diagonalizeFromCoordinates(qC, false);
    auto en = std::get<0>(t);
    auto u = std::get<0>(t);

    Eigen::MatrixXcd hWK(numPhBands, numPhBands);
    hWK.setZero();
    for (int iR = 0; iR < phBravaisVectors.cols(); iR++) {
      Eigen::Vector3d R0 = phBravaisVectors.col(iR);
      for (int k1 = 0; k1 < numAtoms; k1++) {
        for (int k2 = 0; k2 < numAtoms; k2++) {
          // Eigen::Vector3d R = R0; // - atomicPositions.col(k1)
          //+ atomicPositions.col(k2);
          double arg = qC.dot(R0);
          std::complex<double> phase = exp(complexI * arg) / phDegeneracies(iR);
          for (int iCart : {0, 1, 2}) {
            for (int jCart : {0, 1, 2}) {
              int m = compress2Indices(k1, iCart, numAtoms, 3);
              int n = compress2Indices(k2, jCart, numAtoms, 3);
              hWK(m, n) += phase * h0R(iR, k1, k2, iCart, jCart);
            }
          }
        }
      }
    }

    // diagonalize it, using the matrices from phononH0
    auto dq = u.adjoint() * hWK * u;
    (void) dq;
    // check I found again the same eigenvalues
    for (int ib = 0; ib < numPhBands; ib++) {
      assert(abs(std::sqrt(dq(ib, ib).real()) - phEnergies(ib, iq)) < 1.0e-6);
    }
  }
}

void ElPhQeToPhoebeApp::testBackTransform(
    Context &context, PhononH0 &phononH0, Points &kPoints, Points &qPoints,
    ElectronH0Wannier &electronH0, Crystal &crystal,
    Eigen::Tensor<std::complex<double>, 5> &gFull) {
  /** This is the important test of el-ph Wannier interpolation
   * We compute the band structure
   * Read the el-ph interaction from file
   * Check that the el-ph coupling, interpolated on the same initial grid,
   * is the same of the el-ph coupling read from QE.
   */
  bool withVelocities = true;
  bool withEigenvectors = true;
  FullBandStructure bandStructure =
      electronH0.populate(kPoints, withVelocities, withEigenvectors);
  int numKPoints = int(kPoints.getNumPoints());
  int numModes = int(phononH0.getNumBands());

// needed by ::parse()
#ifdef HDF5_AVAIL
  context.setElphFileName(context.getQuantumEspressoPrefix() + ".phoebe.elph.hdf5");
#else
  context.setElphFileName(context.getQuantumEspressoPrefix() + ".phoebe.elph.dat");
#endif

  auto couplingElPh = InteractionElPhWan::parse(context, crystal, &phononH0);

  for (int ik1 = 0; ik1 < numKPoints; ik1++) {
    Eigen::Vector3d k1C =
        kPoints.getPointCoordinates(ik1, Points::cartesianCoordinates);
    auto ik1Index = WavevectorIndex(ik1);
    Eigen::MatrixXcd eigenVector1 = bandStructure.getEigenvectors(ik1Index);

    couplingElPh.cacheElPh(eigenVector1, k1C);

    for (int ik2 = 0; ik2 < numKPoints; ik2++) {
      Eigen::Vector3d k2C =
          kPoints.getPointCoordinates(ik2, Points::cartesianCoordinates);

      std::vector<Eigen::Vector3d> k2Cs;
      k2Cs.push_back(k2C);

      Eigen::Vector3d q3C = k2C - k1C;
      Eigen::Vector3d q3Crystal = qPoints.cartesianToCrystal(q3C);
      int iq3 = int(qPoints.getIndex(q3Crystal));
      std::vector<Eigen::Vector3d> q3Cs;
      q3Cs.push_back(q3C);

      auto ik2Index = WavevectorIndex(ik2);
      Eigen::MatrixXcd eigenVector2 = bandStructure.getEigenvectors(ik2Index);
      std::vector<Eigen::MatrixXcd> eigenVectors2;
      eigenVectors2.push_back(eigenVector2);

      auto t = phononH0.diagonalizeFromCoordinates(q3C);
      auto eigenVector3 = std::get<1>(t);
      std::vector<Eigen::MatrixXcd> eigenVectors3;
      eigenVectors3.push_back(eigenVector3);

      std::vector<Eigen::VectorXcd> polarData;
      Eigen::VectorXcd polar = couplingElPh.polarCorrectionPart1(q3C, eigenVector3);
      polarData.push_back(polar);

      couplingElPh.calcCouplingSquared(eigenVector1, eigenVectors2,
                                       eigenVectors3, q3Cs, polarData);
      auto coupling2 = couplingElPh.getCouplingSquared(0);

      double sum1 = 0.;
      double sum2 = 0.;
      for (int ib1 = 0; ib1 < 4; ib1++) {
        for (int ib2 = 0; ib2 < 4; ib2++) {
          for (int ib3 = 0; ib3 < numModes; ib3++) {
            sum1 += std::norm(gFull(ib2, ib1, ib3, ik1, iq3));
            sum2 += coupling2(ib1, ib2, ib3);
          }
        }
      }
      // note that I change the meaning of the indices
      assert(abs((sum1 - sum2) / sum1) < 0.0001);
    }
  }
}

#ifdef HDF5_AVAIL

void writeHeaderHDF5(
    const std::string &outFileName, const int &numFilledWannier,
    const int &numSpin, const int &numModes, const int &numWannier,
    const Eigen::VectorXd &phDegeneracies,
    const Eigen::VectorXd &elDegeneracies,
    const Eigen::MatrixXd &phBravaisVectors,
    const Eigen::MatrixXd &elBravaisVectors, const Eigen::Vector3i &qMesh,
    const Eigen::Vector3i &kMesh, const int &fileFormat) {
  // note: call this function after the file has been erased of previous content

  try {
    // need to open the files differently if MPI is available or not
    // NOTE: do not remove the braces inside this if -- the file must
    // go out of scope, so that it can be reopened/written by head for the
    // small quantities as in the next block.

    // we write the small quantities only with MPI head
    if (mpi->mpiHead()) {

      HighFive::File file(outFileName, HighFive::File::ReadWrite);

      HighFive::DataSet dnFileFormat = file.createDataSet<int>(
          "/fileFormat", HighFive::DataSpace::From(fileFormat));
      dnFileFormat.write(fileFormat);

      // write out the number of electrons and the spin
      HighFive::DataSet dnElectrons = file.createDataSet<int>(
          "/numElectrons", HighFive::DataSpace::From(numFilledWannier));
      HighFive::DataSet dnSpin = file.createDataSet<int>(
          "/numSpin", HighFive::DataSpace::From(numSpin));
      dnElectrons.write(numFilledWannier);// # of occupied wannier functions
      dnSpin.write(numSpin);

      HighFive::DataSet dnElBands = file.createDataSet<int>(
          "/numElBands", HighFive::DataSpace::From(numWannier));
      HighFive::DataSet dnModes = file.createDataSet<int>(
          "/numPhModes", HighFive::DataSpace::From(numModes));
      dnElBands.write(numWannier);
      dnModes.write(numModes);

      // write out the kMesh and qMesh
      HighFive::DataSet dkMesh =
          file.createDataSet<int>("/kMesh", HighFive::DataSpace::From(kMesh));
      HighFive::DataSet dqMesh =
          file.createDataSet<int>("/qMesh", HighFive::DataSpace::From(qMesh));
      dkMesh.write(kMesh);
      dqMesh.write(qMesh);

      // write bravais lattice vectors
      HighFive::DataSet dPhBravais = file.createDataSet<double>(
          "/phBravaisVectors", HighFive::DataSpace::From(phBravaisVectors));
      HighFive::DataSet dElBravais = file.createDataSet<double>(
          "/elBravaisVectors", HighFive::DataSpace::From(elBravaisVectors));
      dPhBravais.write(phBravaisVectors);
      dElBravais.write(elBravaisVectors);

      // write electron and phonon degeneracies
      HighFive::DataSet dPhDegeneracies = file.createDataSet<double>(
          "/phDegeneracies", HighFive::DataSpace::From(phDegeneracies));
      HighFive::DataSet dElDegeneracies = file.createDataSet<double>(
          "/elDegeneracies", HighFive::DataSpace::From(elDegeneracies));
      dPhDegeneracies.write(phDegeneracies);
      dElDegeneracies.write(elDegeneracies);
    }
  } catch (std::exception &error) {
    Error("Issue writing elph Wannier header to hdf5.");
  }
}

void writeElPhCouplingHDF5v1(
    Context &context, Eigen::Tensor<std::complex<double>, 5> &gWannier,
    const int &numFilledWannier, const int &numSpin, const int &numModes,
    const int &numWannier, const Eigen::VectorXd &phDegeneracies,
    const Eigen::VectorXd &elDegeneracies,
    const Eigen::MatrixXd &phBravaisVectors,
    const Eigen::MatrixXd &elBravaisVectors, const Eigen::Vector3i &qMesh,
    const Eigen::Vector3i &kMesh) {

  const int fileFormat = 1;

  std::string phoebePrefixQE = context.getQuantumEspressoPrefix();

  int numPhBravaisVectors = int(phDegeneracies.size());
  int numElBravaisVectors = int(elDegeneracies.size());

  bool matrixDistributed;
  if (gWannier.dimension(4) != numElBravaisVectors) {
    matrixDistributed = true;
  } else {
    matrixDistributed = false;
  }

  std::string outFileName = phoebePrefixQE + ".phoebe.elph.hdf5";
  // if the hdf5 file is there already, we want to delete it. Occasionally
  // these files seem to get stuck open when a process dies while writing to
  // them, (even if a python script dies) and then they can't be overwritten
  // properly.
  std::remove(&outFileName[0]);

  try {
    // need to open the files differently if MPI is available or not
    // NOTE: do not remove the braces inside this if -- the file must
    // go out of scope, so that it can be reopened/written by head for the
    // small quantities as in the next block.
#if defined(MPI_AVAIL) && !defined(HDF5_SERIAL)

    {
      // open the hdf5 file
      HighFive::FileAccessProps fapl;// = HighFive::FileAccessProps{};
      fapl.add(HighFive::MPIOFileAccess<MPI_Comm, MPI_Info>(MPI_COMM_WORLD, MPI_INFO_NULL));
      HighFive::File file(outFileName, HighFive::File::Overwrite, fapl);

      // flatten the tensor (tensor is not supported) and create the data set
      Eigen::VectorXcd gwan = Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(
          gWannier.data(), gWannier.size());

      // note: gwan is distributed
      size_t globalSize = size_t(numWannier) * numWannier * numModes * numPhBravaisVectors * numElBravaisVectors;

      // Create the data-space to write gWannier to
      std::vector<size_t> dims(2);
      dims[0] = 1;
      dims[1] = globalSize;
      HighFive::DataSet dgwannier = file.createDataSet<std::complex<double>>(
          "/gWannier", HighFive::DataSpace(dims));

      // start point and the number of the total number of elements
      // to be written by this process
      size_t start = mpi->divideWorkIter(numElBravaisVectors)[0] * size_t(numWannier) * numWannier * numModes * numPhBravaisVectors;
      size_t offset = start;

      // Note: HDF5 < v1.10.2 cannot write datasets larger than 2 Gbs
      // ( due to max(int 32 bit))/1024^3 = 2Gb overflowing in MPI)
      // In order to be compatible with older versions, we split the tensor
      // into smaller chunks and write them to separate datasets
      // slower, but it will work more often.

      // maxSize represents 2GB worth of std::complex<doubles>, since that's
      // what we write
      auto maxSize = int(pow(1000, 3)) / sizeof(std::complex<double>);
      size_t smallestSize =
          size_t(numWannier) * numWannier * numModes * numPhBravaisVectors;
      std::vector<int> irEBunchSizes;

      // determine the size of each bunch of electronic bravais vectors
      // the BunchSizes vector tells us how many are in each set
      int numBVs = mpi->divideWorkIter(numElBravaisVectors).back() + 1 - mpi->divideWorkIter(numElBravaisVectors)[0];

      int irEBunchSize = 0;
      for (int irE = 0; irE < numBVs; irE++) {
        irEBunchSize++;
        // this bunch is as big as possible, stop adding to it
        if ((irEBunchSize + 1) * smallestSize > maxSize) {
          irEBunchSizes.push_back(irEBunchSize);
          irEBunchSize = 0;
        }
      }
      // push the last one, no matter the size, to the list of bunch sizes
      irEBunchSizes.push_back(irEBunchSize);

      // determine the number of bunches. The last bunch may be smaller
      // than the rest
      int numDatasets = irEBunchSizes.size();

      // we now loop over these data sets, and write each chunk of
      // bravais vectors in parallel
      size_t netOffset = 0;// offset from first bunch in this set to current bunch
      for (int iBunch = 0; iBunch < numDatasets; iBunch++) {

        // we need to determine the start, stop and offset of this
        // sub-slice of the dataset available to this process
        size_t bunchElements = irEBunchSizes[iBunch] * smallestSize;
        size_t bunchStart = start + netOffset;
        size_t bunchOffset = offset + netOffset;
	// we will use this to update the offset for the bunch during the
	// next loop
        netOffset += bunchElements;

        int gwanSliceStart;
        if (matrixDistributed) {
          // here, no need to slice the gwan tensor (it's already distributed)
          // but we have to use the right offsets to identify tensor elements.
          gwanSliceStart = 0;
        } else {
          // here we slice the gWannier tensor (it's not distributed)
          gwanSliceStart = bunchStart;
        }

        // Each process writes to hdf5
        // The format is ((startRow,startCol),(numRows,numCols)).write(data)
        // Because it's a vector (1 row) all processes write to row=0,
        // col=startPoint
        // with nRows = 1, nCols = number of items this process will write.
        dgwannier.select({0, bunchOffset}, {1, bunchElements}).write_raw(&gwan(gwanSliceStart));
      }
    }

#else
    {// do not remove these braces, see above note.
      // case where mpi exists, but HDF5 was built serially

      if (matrixDistributed) {
        // so, in this case, we must reduce the tensor before writing it
        Eigen::Tensor<std::complex<double>, 5> gWannierRed(
            numWannier, numWannier, numModes, numPhBravaisVectors,
            numElBravaisVectors);
        gWannierRed.setZero();

        auto irEIterator = mpi->divideWorkIter(elDegeneracies.size());
        for (int irE : irEIterator) {
          int irELocal = irE - irEIterator[0];
          for (int irP = 0; irP < numPhBravaisVectors; irP++) {
            for (int nu = 0; nu < numModes; nu++) {
              for (int i = 0; i < numWannier; i++) {
                for (int j = 0; j < numWannier; j++) {
                  gWannierRed(i, j, nu, irP, irE) +=
                      gWannier(i, j, nu, irP, irELocal);
                }
              }
            }
          }
        }
        mpi->allReduceSum(&gWannierRed);
        gWannier = gWannierRed;
      }

      if (mpi->mpiHead()) {
        // open the hdf5 file
        HighFive::File file(outFileName, HighFive::File::Overwrite);

        // flatten the tensor in a vector
        Eigen::VectorXcd gwan = Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(
            gWannier.data(), gWannier.size());

        // create dataset
        HighFive::DataSet dgwannier = file.createDataSet<std::complex<double>>(
            "/gWannier", HighFive::DataSpace::From(gwan));

        // write to hdf5
        dgwannier.write(gwan);
      }
    }
#endif
  } catch (std::exception &error) {
    Error("Issue writing elph Wannier representation to hdf5.");
  }

  writeHeaderHDF5(outFileName, numFilledWannier, numSpin, numModes, numWannier,
                  phDegeneracies, elDegeneracies, phBravaisVectors,
                  elBravaisVectors, qMesh, kMesh, fileFormat);
}

void writeElPhCouplingHDF5v2(
    Context &context, Eigen::Tensor<std::complex<double>, 5> &gWannier,
    const int &numFilledWannier, const int &numSpin, const int &numModes,
    const int &numWannier, const Eigen::VectorXd &phDegeneracies,
    const Eigen::VectorXd &elDegeneracies,
    const Eigen::MatrixXd &phBravaisVectors,
    const Eigen::MatrixXd &elBravaisVectors, const Eigen::Vector3i &qMesh,
    const Eigen::Vector3i &kMesh) {

  const int fileFormat = 2;

  std::string phoebePrefixQE = context.getQuantumEspressoPrefix();

  int numPhBravaisVectors = int(phDegeneracies.size());
  int numElBravaisVectors = int(elDegeneracies.size());

  bool matrixDistributed;
  if (gWannier.dimension(4) != numElBravaisVectors) {
    matrixDistributed = true;
  } else {
    matrixDistributed = false;
  }

  std::string outFileName = phoebePrefixQE + ".phoebe.elph.hdf5";
  // if the hdf5 file is there already, we want to delete it. Occasionally
  // these files seem to get stuck open when a process dies while writing to
  // them, (even if a python script dies) and then they can't be overwritten
  // properly.
  std::remove(&outFileName[0]);

  try {
    // need to open the files differently if MPI is available or not
    // NOTE: do not remove the braces inside this if -- the file must
    // go out of scope, so that it can be reopened/written by head for the
    // small quantities as in the next block.

    {
      // open the hdf5 file and remove existing files
      if (mpi->mpiHead()) {
        HighFive::File file(outFileName, HighFive::File::Overwrite);
      }
      mpi->barrier(); // wait for file to be overwritten

      // now open the file in serial mode
      // because we do the parallelization by hand
      HighFive::File file(outFileName, HighFive::File::Overwrite);

      // create buffer to save a slice of the tensor, at fixed irE
      Eigen::Tensor<std::complex<double>, 4> slice;
      slice.resize(numWannier, numWannier, numModes, numPhBravaisVectors);

      auto irEIterator = mpi->divideWorkIter(elDegeneracies.size());
      for (int irE : irEIterator) {
        int irELocal;
        if (matrixDistributed) {
          irELocal = irE - irEIterator[0];
        } else {
          irELocal = irE;
        }

        // select a slice
        for (int irP = 0; irP < numPhBravaisVectors; irP++) {
          for (int nu = 0; nu < numModes; nu++) {
            for (int i = 0; i < numWannier; i++) {
              for (int j = 0; j < numWannier; j++) {
                slice(i, j, nu, irP) = gWannier(i, j, nu, irP, irELocal);
              }
            }
          }
        }

        // flatten the tensor in a vector
        Eigen::VectorXcd flatSlice = Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(
            slice.data(), slice.size());

        std::string datasetName = "/gWannier_" + std::to_string(irE);

        // create dataset
        HighFive::DataSet dslice = file.createDataSet<std::complex<double>>(
            datasetName, HighFive::DataSpace::From(flatSlice));

        // write to hdf5
        dslice.write(flatSlice);
      }
    }
  } catch (std::exception &error) {
    Error("Issue writing elph Wannier representation to hdf5.");
  }

  writeHeaderHDF5(outFileName, numFilledWannier, numSpin, numModes, numWannier,
                  phDegeneracies, elDegeneracies, phBravaisVectors,
                  elBravaisVectors, qMesh, kMesh, fileFormat);
}

#else

void writeElPhCouplingNoHDF5(
    Context &context, Eigen::Tensor<std::complex<double>, 5> &gWannier,
    const int &numFilledWannier, const int &numSpin, const int &numModes,
    const int &numWannier, const Eigen::VectorXd &phDegeneracies,
    const Eigen::VectorXd &elDegeneracies,
    const Eigen::MatrixXd &phBravaisVectors,
    const Eigen::MatrixXd &elBravaisVectors, const Eigen::Vector3i &qMesh,
    const Eigen::Vector3i &kMesh) {

  std::string phoebePrefixQE = context.getQuantumEspressoPrefix();

  int numPhBravaisVectors = int(phDegeneracies.size());
  int numElBravaisVectors = int(elDegeneracies.size());

  bool matrixDistributed;
  if (gWannier.dimension(4) != numElBravaisVectors) {
    matrixDistributed = true;
  } else {
    matrixDistributed = false;
  }

  std::string outFileName = "./" + phoebePrefixQE + ".phoebe.elph.dat";

  if (mpi->mpiHead()) {
    std::ofstream outfile(outFileName);
    if (not outfile.is_open()) {
      Error("Output file couldn't be opened");
    }
    outfile << numFilledWannier << " " << numSpin << "\n";
    outfile << kMesh << "\n";
    outfile << qMesh << "\n";
    outfile << phBravaisVectors.rows() << " " << phBravaisVectors.cols()
            << "\n";
    outfile << phBravaisVectors << "\n";
    outfile << phDegeneracies << "\n";
    outfile << elBravaisVectors.rows() << " " << elBravaisVectors.cols()
            << "\n";
    outfile << elBravaisVectors << "\n";
    outfile << elDegeneracies << "\n";
    outfile << "\n";
    for (auto x : gWannier.dimensions()) {
      outfile << x << " ";
    }
    outfile << "\n";
  }

  int numPhBands = gWannier.dimension(2);

  if (matrixDistributed) {

    for (int iRank = 0; iRank < mpi->getSize(); iRank++) {
      if (iRank == mpi->getRank()) {

        std::fstream outfile(outFileName,
                             std::fstream::out | std::fstream::app);
        outfile << std::setprecision(16);

        auto i5Iterator = mpi->divideWorkIter(elDegeneracies.size());
        for (int i5 : i5Iterator) {

          int i5Local;
          if (matrixDistributed) {
            i5Local = i5 - i5Iterator[0];
          } else {
            i5Local = i5;
          }

          for (int i4 = 0; i4 < phDegeneracies.size(); i4++) {
            for (int i3 = 0; i3 < numPhBands; i3++) {
              for (int i2 = 0; i2 < numWannier; i2++) {
                for (int i1 = 0; i1 < numWannier; i1++) {
                  outfile << std::setw(22)
                          << gWannier(i1, i2, i3, i4, i5Local).real() << " "
                          << std::setw(22)
                          << gWannier(i1, i2, i3, i4, i5Local).imag() << "\n";
                }
              }
            }
          }
        }
      }
      mpi->barrier();
    }
  } else {// matrix not distributed
    if (mpi->mpiHead()) {
      std::fstream outfile(outFileName, std::fstream::out | std::fstream::app);
      outfile << std::setprecision(16);
      for (int i5 = 0; i5 < elDegeneracies.size(); i5++) {
        for (int i4 = 0; i4 < phDegeneracies.size(); i4++) {
          for (int i3 = 0; i3 < numPhBands; i3++) {
            for (int i2 = 0; i2 < numWannier; i2++) {
              for (int i1 = 0; i1 < numWannier; i1++) {
                outfile << std::setw(22) << gWannier(i1, i2, i3, i4, i5).real()
                        << " " << std::setw(22)
                        << gWannier(i1, i2, i3, i4, i5).imag() << "\n";
              }
            }
          }
        }
      }
    }
    mpi->barrier();
  }
}
#endif

void ElPhQeToPhoebeApp::writeWannierCoupling(
    Context &context, Eigen::Tensor<std::complex<double>, 5> &gWannier,
    const int &numFilledWannier, const int &numSpin, const int &numModes,
    const int &numWannier, const Eigen::VectorXd &phDegeneracies,
    const Eigen::VectorXd &elDegeneracies,
    const Eigen::MatrixXd &phBravaisVectors,
    const Eigen::MatrixXd &elBravaisVectors, const Eigen::Vector3i &qMesh,
    const Eigen::Vector3i &kMesh) {

  if (mpi->mpiHead()) {
    std::cout << "\nStart writing el-ph coupling to file." << std::endl;
  }

#ifdef HDF5_AVAIL
  if (mpi->getSize() < 4) {
    // Note: this HDF5 had already been reported and being worked on.
    // It's beyond the purpose of Phoebe's project.
    Warning("HDF5 with <4 MPI process may crash (due to a "
            "library's bug),\nuse more MPI processes if that happens");
  }

  if (context.getHdf5ElPhFileFormat()==1) {
    writeElPhCouplingHDF5v1(context, gWannier, numFilledWannier, numSpin,
                               numModes, numWannier, phDegeneracies,
                               elDegeneracies, phBravaisVectors,
                               elBravaisVectors, qMesh, kMesh);
  } else {
    writeElPhCouplingHDF5v2(context, gWannier, numFilledWannier, numSpin,
                               numModes, numWannier, phDegeneracies,
                               elDegeneracies, phBravaisVectors,
                               elBravaisVectors, qMesh, kMesh);
  }
#else
  writeElPhCouplingNoHDF5(context, gWannier, numFilledWannier, numSpin,
                          numModes, numWannier, phDegeneracies,
                          elDegeneracies, phBravaisVectors,
                          elBravaisVectors, qMesh, constkMesh);
#endif

  if (mpi->mpiHead()) {
    std::cout << "Done writing el-ph coupling to file.\n"
              << std::endl;
  }
}
