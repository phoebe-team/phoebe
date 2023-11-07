#include <iomanip>
#include "electron_h0_wannier.h"
#include "constants.h"
#include "exceptions.h"

ElectronH0Wannier::ElectronH0Wannier(
    const Eigen::Matrix3d &directUnitCell_,
    const Eigen::Matrix<double, 3, Eigen::Dynamic> &bravaisVectors_,
    const Eigen::VectorXd &vectorsDegeneracies_,
    const Eigen::Tensor<std::complex<double>, 3> &h0R_,
    const Eigen::Tensor<std::complex<double>, 4> &rMatrix_)
    : particle(Particle::electron) {

  h0R = h0R_;
  rMatrix = rMatrix_;
  directUnitCell = directUnitCell_;
  bravaisVectors = bravaisVectors_;
  vectorsDegeneracies = vectorsDegeneracies_;

  if (h0R.dimension(1) != h0R.dimension(2)) {
    Error("WannierH0(): h0R should have dimensions (R,bands,bands)");
  }
  if (h0R.dimension(0) != bravaisVectors.cols()) {
    Error("WannierH0(): h0R and bravaisVectors not aligned");
  }
  if (vectorsDegeneracies.size() != bravaisVectors.cols()) {
    Error("WannierH0(): degeneracies not aligned with vectors");
  }

  if ((rMatrix.dimension(1) != h0R.dimension(0)) ||
      (rMatrix.dimension(2) != h0R.dimension(1)) ||
      (rMatrix.dimension(3) != h0R.dimension(2))) {
    Error("WannierH0(): h0R and rMatrix should be aligned");
  }

  if (rMatrix.dimension(0) != 3) {
    Error("WannierH0(): rMatrix should be a vector");
  }

  numWannier = int(h0R.dimension(1));
  numVectors = int(vectorsDegeneracies.size());

  // copy data to the GPU
  {
    Kokkos::resize(h0R_d, numWannier, numWannier, numVectors);
    Kokkos::resize(vectorsDegeneracies_d, numVectors);
    Kokkos::resize(bravaisVectors_d, numVectors, 3);

    auto h0R_h = create_mirror_view(h0R_d);
    auto vectorsDegeneracies_h = create_mirror_view(vectorsDegeneracies_d);
    auto bravaisVectors_h = create_mirror_view(bravaisVectors_d);

    for (int iR = 0; iR < numVectors; iR++) {
      for (int i = 0; i < numWannier; i++) {
        for (int j = 0; j < numWannier; j++) {
          h0R_h(i, j, iR) = h0R(iR, i, j);
        }
      }
      vectorsDegeneracies_h(iR) = vectorsDegeneracies(iR);
      for (int i = 0; i < 3; ++i) {
        bravaisVectors_h(iR, i) = bravaisVectors(i, iR);
      }
    }
    Kokkos::deep_copy(h0R_d, h0R_h);
    Kokkos::deep_copy(vectorsDegeneracies_d, vectorsDegeneracies_h);
    Kokkos::deep_copy(bravaisVectors_d, bravaisVectors_h);
    double memory = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(memory);
  }
}

// copy constructor
ElectronH0Wannier::ElectronH0Wannier(const ElectronH0Wannier &that)
    : particle(Particle::electron) {
  h0R = that.h0R;
  rMatrix = that.rMatrix;
  directUnitCell = that.directUnitCell;
  numWannier = that.numWannier;
  bravaisVectors = that.bravaisVectors;
  numVectors = that.numVectors;
  vectorsDegeneracies = that.vectorsDegeneracies;
  hasShiftedVectors = that.hasShiftedVectors;
  vectorsShifts = that.vectorsShifts;
  degeneracyShifts = that.degeneracyShifts;
  h0R_d = that.h0R_d;
  degeneracyShifts_d = that.degeneracyShifts_d;
  vectorsShifts_d = that.vectorsShifts_d;
  vectorsDegeneracies_d = that.vectorsDegeneracies_d;
  bravaisVectors_d = that.bravaisVectors_d;
  double memory = getDeviceMemoryUsage();
  kokkosDeviceMemory->addDeviceMemoryUsage(memory);
}

// copy assignment
ElectronH0Wannier &ElectronH0Wannier::operator=(const ElectronH0Wannier &that) {
  if (this != &that) {
    particle = that.particle;
    numVectors = that.numVectors;
    numWannier = that.numWannier;
    bravaisVectors = that.bravaisVectors;
    vectorsDegeneracies = that.vectorsDegeneracies;
    directUnitCell = that.directUnitCell;
    h0R = that.h0R;
    rMatrix = that.rMatrix;
    hasShiftedVectors = that.hasShiftedVectors;
    vectorsShifts = that.vectorsShifts;
    degeneracyShifts = that.degeneracyShifts;
    h0R_d = that.h0R_d;
    degeneracyShifts_d = that.degeneracyShifts_d;
    vectorsShifts_d = that.vectorsShifts_d;
    vectorsDegeneracies_d = that.vectorsDegeneracies_d;
    bravaisVectors_d = that.bravaisVectors_d;
    double memory = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(memory);
  }
  return *this;
}

ElectronH0Wannier::~ElectronH0Wannier() {
  double memory = getDeviceMemoryUsage(); // do this before the resize!
  kokkosDeviceMemory->removeDeviceMemoryUsage(memory);
  // Deallocate stuff from GPU
  // Eigen class attributes should be deallocated automatically
  Kokkos::resize(h0R_d, 0, 0, 0);
  Kokkos::resize(degeneracyShifts_d, 0, 0, 0);
  Kokkos::resize(vectorsShifts_d, 0, 0, 0, 0, 0);
  Kokkos::resize(vectorsDegeneracies_d, 0);
  Kokkos::resize(bravaisVectors_d, 0, 0);
}

Particle ElectronH0Wannier::getParticle() { return particle; }

int ElectronH0Wannier::getNumBands() { return numWannier; }

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
ElectronH0Wannier::diagonalize(Point &point) {
  Eigen::Vector3d k = point.getCoordinates(Points::cartesianCoordinates);

  auto tup = diagonalizeFromCoordinates(k);
  auto energies = std::get<0>(tup);
  auto eigenvectors = std::get<1>(tup);

  // note: the eigenvector matrix is the unitary transformation matrix U
  // from the Bloch to the Wannier gauge.

  return std::make_tuple(energies, eigenvectors);
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
ElectronH0Wannier::diagonalizeFromCoordinates(Eigen::Vector3d &k) {

  std::vector<Eigen::Vector3d> cartesianWavevectors;
  cartesianWavevectors.push_back(k);
  auto t = batchedDiagonalizeFromCoordinates(cartesianWavevectors);
  auto energies = std::get<0>(t)[0];
  auto eigenVectors = std::get<1>(t)[0];
  return std::make_tuple(energies, eigenVectors);
}

std::vector<Eigen::MatrixXcd> ElectronH0Wannier::batchedBuildHamiltonians(
    std::vector<Eigen::Vector3d>& cartesianWavevectors) {

  int numK = cartesianWavevectors.size();
  std::vector<Eigen::MatrixXcd> Hs(numK, Eigen::MatrixXcd::Zero(numWannier, numWannier));

  // fourier transform Hamiltonian, without or with shift
  if (!hasShiftedVectors) {
    Eigen::MatrixXcd phases(numVectors, numK);

    // precompute all phases
#pragma omp parallel for collapse(2)
    for (int iK = 0; iK < numK; ++iK) {
      for (int iR = 0; iR < numVectors; iR++) {
        double phase = cartesianWavevectors[iK].dot(bravaisVectors.col(iR));
        std::complex<double> phaseFactor = {cos(phase), sin(phase)};
        phases(iR, iK) = phaseFactor / vectorsDegeneracies(iR);
      }
    }

    // perform fourier transform
#pragma omp parallel for collapse(3)
    for (int iK = 0; iK < numK; ++iK) {
      for (int n = 0; n < numWannier; n++) {
        for (int m = 0; m < numWannier; m++) {
          for (int iR = 0; iR < numVectors; iR++) {
            Hs[iK](m, n) += phases(iR, iK) * h0R(iR, m, n);
          }
        }
      }
    }

  } else {
    // with shift

    for (int iK = 0; iK < numK; ++iK) {
      for (int iR = 0; iR < numVectors; iR++) {
        double phaseArg;
        std::complex<double> phase;
        for (int iw2 = 0; iw2 < numWannier; iw2++) {
          for (int iw1 = 0; iw1 < numWannier; iw1++) {
            for (int iDeg = 0; iDeg < degeneracyShifts(iw1, iw2, iR); ++iDeg) {
              phaseArg = 0.;
              for (int i : {0, 1, 2}) {
                phaseArg += cartesianWavevectors[iK](i) * vectorsShifts(i, iDeg, iw1, iw2, iR);
              }
              phase = {cos(phaseArg), sin(phaseArg)};
              phase /= vectorsDegeneracies(iR) * degeneracyShifts(iw1, iw2, iR);
              Hs[iK](iw1, iw2) += phase * h0R(iR, iw1, iw2);
            }
          }
        }
      }
    }
  }
  return Hs;
}

std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXcd>>
ElectronH0Wannier::batchedDiagonalizeFromCoordinates(std::vector<Eigen::Vector3d>& cartesianWavevectors) {

  auto Hs = batchedBuildHamiltonians(cartesianWavevectors);

  int numK = cartesianWavevectors.size();

  std::vector<Eigen::VectorXd> allEnergies(numK);
  std::vector<Eigen::MatrixXcd> allEigenvectors(numK);
#pragma omp parallel for
  for (int iK = 0; iK < numK; ++iK) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(Hs[iK]);
    allEnergies[iK] = eigenSolver.eigenvalues();
    allEigenvectors[iK] = eigenSolver.eigenvectors();
  }
  return std::make_tuple(allEnergies, allEigenvectors);

}

Eigen::Tensor<std::complex<double>, 3>
ElectronH0Wannier::diagonalizeVelocity(Point &point) {
  Eigen::Vector3d coordinates = point.getCoordinates(Points::cartesianCoordinates);
  return diagonalizeVelocityFromCoordinates(coordinates);
}

Eigen::Tensor<std::complex<double>, 3>
ElectronH0Wannier::diagonalizeVelocityFromCoordinates(
    Eigen::Vector3d &coordinates) {

  std::vector<Eigen::Vector3d> cartesianWavevectors;
  cartesianWavevectors.push_back(coordinates);
  auto t = batchedDiagonalizeWithVelocities(cartesianWavevectors);
  return std::get<2>(t)[0];
}

std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXcd>,
           std::vector<Eigen::Tensor<std::complex<double>,3>>>
ElectronH0Wannier::kokkosPopulate(const std::vector<Eigen::Vector3d>& cartesianCoordinates,
                                                    const bool &withVelocities,
                                                    const bool &withEigenvectors, const std::vector<int> &iks) {

  // TODO enforce hasEigenvectors here?
  (void) withEigenvectors;
  // could allow more kpoints to be on the GPU at the same time.
  // For now it's implemented like this because it's only ever called in cases that need
  // eigenvectors (phel_scattering.cpp)

  // cartesianCoordinates is the list of points which are local to this process
  size_t numPoints = cartesianCoordinates.size();

  std::vector<Eigen::VectorXd> allEnergies(numPoints, Eigen::VectorXd(numWannier));
  std::vector<Eigen::MatrixXcd> allEigenvectors(numPoints, Eigen::MatrixXcd(numWannier, numWannier));
  std::vector<Eigen::Tensor<std::complex<double>,3>> allVelocities(numPoints,
                                        Eigen::Tensor<std::complex<double>,3>(numWannier, numWannier, 3));

  // set up information about how kpoint batches will be moved onto host
  int batchSize = estimateBatchSize(withVelocities);
  std::vector<std::vector<int>> ikBatches = kokkosDeviceMemory->splitToBatches(iks, batchSize);

  // Note on indexing in this function:
  //   iik is local within the batch.
  //   ikBatch[iik] are the global kpoint indices
  // we need a value which indexes within the kpoints passed as the cartCoords argument
  // To do this, we keep a batch counter, that would be ik = iik + numFinishedPoints
  int numFinishedPoints = 0;

  // diagonalize batches of kpoints
  for (auto ikBatch : ikBatches) {

    int numBatchK= ikBatch.size();

    // set up kokkos view for the wavevectors
    DoubleView2D cartesianWavevectors_d("el_cartWav_d", numBatchK, 3);
    {
      auto cartesianWavevectors_h = Kokkos::create_mirror_view(cartesianWavevectors_d);
      for (int iik = 0; iik < numBatchK; ++iik) {

        // ik = index in cartCoords list, index of in iks, ikk is index in the batch
        int ik = iik + numFinishedPoints;
        for (int i = 0; i < 3; ++i) {
          cartesianWavevectors_h(iik, i) = cartesianCoordinates[ik](i);
        }
      }
      Kokkos::deep_copy(cartesianWavevectors_d, cartesianWavevectors_h);

      // now calculate the energies, eigenvectors, and velocities for the kpoints in this batch
      DoubleView2D batchEnergies_d;
      StridedComplexView3D batchEigenvectors_d;
      ComplexView4D batchVelocities_d;
      if (withVelocities) {
        auto t = kokkosBatchedDiagonalizeWithVelocities(cartesianWavevectors_d);
        batchEnergies_d = std::get<0>(t);
        batchEigenvectors_d = std::get<1>(t);
        batchVelocities_d = std::get<2>(t);
      } else {
        auto t = kokkosBatchedDiagonalizeFromCoordinates(cartesianWavevectors_d);
        batchEnergies_d = std::get<0>(t);
        batchEigenvectors_d = std::get<1>(t);
      }
      // no need to keep the wavevectors in memory after this
      Kokkos::realloc(cartesianWavevectors_d, 0, 0);

      auto batchEnergies_h = Kokkos::create_mirror_view(batchEnergies_d);
      auto batchEigenvectors_h = Kokkos::create_mirror_view(batchEigenvectors_d);

      Kokkos::deep_copy(batchEnergies_h, batchEnergies_d);
      Kokkos::deep_copy(batchEigenvectors_h, batchEigenvectors_d);
      for (int iik=0; iik<numBatchK; ++iik) {
        // ik = index in cartCoords list, index of in iks, ikk is index in the batch
        int ik = iik + numFinishedPoints;
        for (int ib1 = 0; ib1 < numWannier; ++ib1) {
          allEnergies[ik](ib1) = batchEnergies_h(iik, ib1);
          for (int ib2 = 0; ib2 < numWannier; ++ib2) {
            allEigenvectors[ik](ib1, ib2) = batchEigenvectors_h(iik, ib1, ib2);
          }
        }
      }

      if (withVelocities) {
        auto batchVelocities_h = Kokkos::create_mirror_view(batchVelocities_d);
        Kokkos::deep_copy(batchVelocities_h, batchVelocities_d);
        for (int iik=0; iik<numBatchK; ++iik) {
          // ik = index in cartCoords list, index of in iks, ikk is index in the batch
          int ik = iik + numFinishedPoints;
          for (int ib1 = 0; ib1 < numWannier; ++ib1) {
            for (int ib2 = 0; ib2 < numWannier; ++ib2) {
              for (int i = 0; i < 3; ++i) {
                allVelocities[ik](ib1, ib2, i) = batchVelocities_h(iik, ib1, ib2, i);
              }
            }
          }
        }
      }
      // clean up views which are now unused
      Kokkos::realloc(batchEnergies_d, 0, 0);
      batchEigenvectors_d = decltype(batchEigenvectors_d)();
      Kokkos::realloc(batchVelocities_d, 0, 0, 0, 0);
    }
    numFinishedPoints += numBatchK;
  }
  return std::make_tuple(allEnergies, allEigenvectors, allVelocities);
}

FullBandStructure ElectronH0Wannier::populate(Points &fullPoints,
                                              const bool &withVelocities,
                                              const bool &withEigenvectors,
                                              const bool isDistributed) {

  // generate a band structure to be filled and returned
  FullBandStructure fullBandStructure(numWannier, particle, withVelocities,
                                      withEigenvectors, fullPoints, isDistributed);

  // set up cartesian coords list from fullPoints
  // ask each process to diagonalize the local wavevectors for this bandstruct
  //
  // we index from the list of wavevector indices provided by
  // getWavevector indices because if FBS is distributed, it won't have every wavevector
  // from the points object on this process
  std::vector<int> iks = fullBandStructure.getLocalWavevectorIndices();
  int niks = iks.size();
  std::vector<Eigen::Vector3d> cartesianCoordinates(niks);

  #pragma omp parallel for
  for(int iik = 0; iik < niks; iik++) {
    int ik = iks[iik];
    Eigen::Vector3d kCart = fullPoints.getPointCoordinates(ik, Points::cartesianCoordinates);
    cartesianCoordinates[iik] = kCart;
  }

  // call kokkos populate to generate energies, velocities, and eigenvectors as necessary
  auto tup = kokkosPopulate(cartesianCoordinates, withVelocities, withEigenvectors, iks);

  // band structure quantities from kokkosPopulate, the outer std::vectors are indexed by
  // kpoints in cartesianCoordinates, Eigen components are indexed by bands
  std::vector<Eigen::VectorXd> localEnergies = std::get<0>(tup);
  std::vector<Eigen::MatrixXcd> localEigenvectors = std::get<1>(tup);
  std::vector<Eigen::Tensor<std::complex<double>,3>> localVelocities = std::get<2>(tup);

  // copy returned quantities into the band structure

  // if the bandstructure is distributed, it only has some fraction
  // of the kPoints. We loop over iks, which are the global
  // indices of the kpoints this process owns
#pragma omp parallel for
  for(int iik = 0; iik < niks; iik++) {

    // ik is the global index of the kpoint, iik is the local index
    int ik = iks[iik];
    Point point = fullBandStructure.getPoint(ik);

    Eigen::VectorXd ens(numWannier);
    for (int ib = 0; ib < numWannier; ++ib) {
      ens(ib) = localEnergies[iik](ib);
    }
    fullBandStructure.setEnergies(point, ens);

    if (withVelocities) {
      Eigen::Tensor<std::complex<double>,3> v(numWannier, numWannier, 3);
      for (int ib1 = 0; ib1 < numWannier; ++ib1) {
        for (int ib2 = 0; ib2 < numWannier; ++ib2) {
          for (int i = 0; i < 3; ++i) {
            v(ib1,ib2,i) = localVelocities[iik](ib1, ib2, i);
          }
        }
      }
      fullBandStructure.setVelocities(point, v);
    }
    if (withEigenvectors) {
      Eigen::MatrixXcd eigVec(numWannier, numWannier);
      for (int ib1 = 0; ib1 < numWannier; ++ib1) {
        for (int ib2 = 0; ib2 < numWannier; ++ib2) {
          eigVec(ib1, ib2) = localEigenvectors[iik](ib1, ib2);
        }
      }
      fullBandStructure.setEigenvectors(point, eigVec);
    }
  }
  return fullBandStructure;
}

// this is for when we want to run populate on a points list, without creating a new bandstructure object
// this might happen if we need energies for some non-uniform grid of points
// currently this only used by the phonon electron scattering calculation
std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXcd>,
           std::vector<Eigen::Tensor<std::complex<double>,3>>>
        ElectronH0Wannier::populate(const std::vector<Eigen::Vector3d>& cartesianCoordinates,
                                              const bool &withVelocities,
                                              const bool &withEigenvectors) {

  // kokkos populate requires an iterator of kpoint indices
  std::vector<int> iks(cartesianCoordinates.size());
  std::iota (std::begin(iks), std::end(iks), 0);

  // call kokkos populate and return the energies and eigenvectors for each chunk of kpts
  auto tup = kokkosPopulate(cartesianCoordinates, withVelocities, withEigenvectors, iks);

  // return vectors
  std::vector<Eigen::VectorXd> returnEnergies = std::get<0>(tup);
  std::vector<Eigen::MatrixXcd> returnEigenvectors = std::get<1>(tup);
  std::vector<Eigen::Tensor<std::complex<double>,3>> returnVelocities = std::get<2>(tup);

  return std::make_tuple(returnEnergies, returnEigenvectors, returnVelocities);
}

FullBandStructure ElectronH0Wannier::cpuPopulate(Points &fullPoints,
                                                 const bool &withVelocities,
                                                 const bool &withEigenvectors,
                                                 const bool isDistributed) {

  FullBandStructure fullBandStructure(numWannier, particle, withVelocities,
                                      withEigenvectors, fullPoints,
                                      isDistributed);

  // first prepare the list of wavevectors
  std::vector<int> iks = fullBandStructure.getLocalWavevectorIndices();
  int numK = iks.size();
  std::vector<Eigen::Vector3d> cartesianWavevectors(numK);
  #pragma omp parallel for
  for (int iik = 0; iik < numK; iik++) {
    int ik = iks[iik];
    WavevectorIndex ikIdx(ik);
    cartesianWavevectors[iik] = fullBandStructure.getWavevector(ikIdx);
  }

  // populate energies, eigenvectors, and velocities
  std::vector<Eigen::VectorXd> allEnergies;
  std::vector<Eigen::MatrixXcd> allEigenvectors;
  std::vector<Eigen::Tensor<std::complex<double>,3>> allVelocities;

  // call the function to diagonalize these wavevectors
  if (withVelocities) {
    auto t = batchedDiagonalizeWithVelocities(cartesianWavevectors);
    allEnergies = std::get<0>(t);
    allEigenvectors = std::get<1>(t);
    allVelocities = std::get<2>(t);
  } else {
    auto t = batchedDiagonalizeFromCoordinates(cartesianWavevectors);
    allEnergies = std::get<0>(t);
    allEigenvectors = std::get<1>(t);
  }

  // set the calculated results in the band structure object
  // then return it
#pragma omp parallel for
  for (int iik = 0; iik < numK; iik++) {
    int ik = iks[iik];
    Point point = fullBandStructure.getPoint(ik);
    auto tup = diagonalize(point);
    auto ens = allEnergies[iik];
    fullBandStructure.setEnergies(point, ens);
    if (withVelocities) {
      auto velocities = diagonalizeVelocity(point);
      fullBandStructure.setVelocities(point, velocities);
    }
    if (withEigenvectors) {
      auto eigenVectors = allEigenvectors[iik];
      fullBandStructure.setEigenvectors(point, eigenVectors);
    }
  }
  return fullBandStructure;
}

std::vector<Eigen::MatrixXcd>
ElectronH0Wannier::getBerryConnection(Point &point) {
  Eigen::Vector3d k = point.getCoordinates(Points::cartesianCoordinates);

  // first we diagonalize the hamiltonian
  auto tup = diagonalize(point);
  // auto ens = std::get<0>(tup);
  auto eigenvectors = std::get<1>(tup);

  // note: the eigenvector matrix is the unitary transformation matrix U
  // from the Bloch to the Wannier gauge.

  std::vector<Eigen::MatrixXcd> berryConnection;

  std::vector<std::complex<double>> phases(bravaisVectors.cols());
  for (int iR = 0; iR < bravaisVectors.cols(); iR++) {
    Eigen::Vector3d R = bravaisVectors.col(iR);
    double phase = k.dot(R);
    std::complex<double> phaseFactor = {cos(phase), sin(phase)};
    phases[iR] = phaseFactor / vectorsDegeneracies(iR);
  }

  for (int i : {0, 1, 2}) {
    // now construct the berryConnection in reciprocal space and Wannier gauge
    Eigen::MatrixXcd berryConnectionW = Eigen::MatrixXcd::Zero(numWannier, numWannier);
    for (int n = 0; n < numWannier; n++) {
      for (int m = 0; m < numWannier; m++) {
          for (int iR = 0; iR < bravaisVectors.cols(); iR++) {
          berryConnectionW(m, n) +=
              phases[iR] * rMatrix(iR, m, n, i);
        }
      }
    }
    Eigen::MatrixXcd thisBerryConnection(numWannier, numWannier);
    thisBerryConnection = eigenvectors.adjoint() * berryConnectionW * eigenvectors;
    berryConnection.push_back(thisBerryConnection);
  }
  return berryConnection;
}

void ElectronH0Wannier::addShiftedVectors(Eigen::Tensor<double,3> degeneracyShifts_,
                       Eigen::Tensor<double,5> vectorsShifts_) {
  // some validation
  if (degeneracyShifts_.dimension(0) != numWannier ||
      degeneracyShifts_.dimension(1) != numWannier ||
      degeneracyShifts_.dimension(2) != numVectors ) {
    Error("Inconsistent dimensions on degeneracyShifts");
  }
  if (vectorsShifts_.dimension(0) != 3 ||
      vectorsShifts_.dimension(1) != 8 ||
      vectorsShifts_.dimension(2) != numWannier ||
      vectorsShifts_.dimension(3) != numWannier ||
      vectorsShifts_.dimension(4) != numVectors) {
    Error("Inconsistent dimensions on vectorsShifts");
  }
  hasShiftedVectors = true;
  vectorsShifts = vectorsShifts_;
  degeneracyShifts = degeneracyShifts_;

  // note: for better performance, I shift all vectors by R,
  // so that in the Fourier transform we only work with one Eigen object
  for (int iR = 0; iR < numVectors; iR++) {
    for (int iw1 = 0; iw1 < numWannier; iw1++) {
      for (int iw2 = 0; iw2 < numWannier; iw2++) {
        for (int iDeg = 0; iDeg < degeneracyShifts(iw1, iw2, iR); ++iDeg) {
          Eigen::Vector3d r = bravaisVectors.col(iR);
          for (int i : {0, 1, 2}) {
            vectorsShifts(i, iDeg, iw1, iw2, iR) += r(i);
          }
        }
      }
    }
  }


  { // copy to GPU
    // Note: getDeviceMemory usage checks on the size of the *_d Views
    // so, we remove here the previously added vectors and add back later below
    double oldMemory = getDeviceMemoryUsage();
    kokkosDeviceMemory->removeDeviceMemoryUsage(oldMemory);

    Kokkos::resize(degeneracyShifts_d, numWannier, numWannier, numVectors);
    Kokkos::resize(vectorsShifts_d, 3, 8, numWannier, numWannier, numVectors);
    auto degeneracyShifts_h = create_mirror_view(degeneracyShifts_d);
    auto vectorsShifts_h = create_mirror_view(vectorsShifts_d);
    for (int iR = 0; iR < numVectors; iR++) {
      for (int i = 0; i < numWannier; i++) {
        for (int j = 0; j < numWannier; j++) {
          degeneracyShifts_h(i, j, iR) = degeneracyShifts(i, j, iR);
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 8; ++b) {
              vectorsShifts_h(a, b, i, j, iR) = vectorsShifts(a, b, i, j, iR);
            }
          }
        }
      }
    }
    Kokkos::deep_copy(degeneracyShifts_d, degeneracyShifts_h);
    Kokkos::deep_copy(vectorsShifts_d, vectorsShifts_h);

    double newMemory = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(newMemory);
  }
}

/**
 * Do diagonalization for a batch of k-points on the CPU.
 * Returns the energies (nk, nb), eigenvectors (nk, nb, nb)
 * and velocities (nk, nb, nb, 3) at each k-point.
 */
std::tuple<std::vector<Eigen::VectorXd>,
           std::vector<Eigen::MatrixXcd>,
           std::vector<Eigen::Tensor<std::complex<double>, 3>>>
ElectronH0Wannier::batchedDiagonalizeWithVelocities(
    std::vector<Eigen::Vector3d> cartesianCoordinates) {



  int numK = cartesianCoordinates.size();

  double delta = 1.0e-8;
  double threshold = 0.000001 / energyRyToEv; // = 1 micro-eV

  std::vector<Eigen::Vector3d> allVectors(numK * 7);
#pragma omp parallel for
  for (int iK = 0; iK < numK; ++iK) {
    Eigen::Vector3d coordinates = cartesianCoordinates[iK];
    allVectors[iK * 7] = coordinates;
    for (int i : {0, 1, 2}) {
      // define q+ and q- from finite differences.
      Eigen::Vector3d qPlus = coordinates;
      Eigen::Vector3d qMins = coordinates;
      qPlus(i) += delta;
      qMins(i) -= delta;
      allVectors[iK * 7 + i * 2 + 1] = qPlus; // 1, 3, 5, 8
      allVectors[iK * 7 + i * 2 + 2] = qMins;// 2, 4, 6, 9
    }
  }

  auto Hs = batchedBuildHamiltonians(allVectors);

  std::vector<Eigen::VectorXd> resultsEnergies(numK);
  std::vector<Eigen::MatrixXcd> resultsEigenvectors(numK);
  std::vector<Eigen::Tensor<std::complex<double>, 3>> resultsVelocities(numK);

  // find energies and eigenvectors at the desired wavevectors

#pragma omp parallel for
  for (int iK=0; iK<numK; ++iK) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(Hs[iK*7]);
    resultsEnergies[iK] = eigenSolver.eigenvalues();
    resultsEigenvectors[iK] = eigenSolver.eigenvectors();
  }

  // now build velocity operator

#pragma omp parallel for
  for (int iK=0; iK<numK; ++iK) {

    Eigen::Tensor<std::complex<double>, 3> velocity(numWannier, numWannier, 3);
    velocity.setZero();

    // if we are working at gamma, we set all velocities to zero.
    if (cartesianCoordinates[iK].norm() < 1.0e-6) {
      resultsVelocities[iK] = velocity;
      continue;
    }

    // now we compute the velocity operator, diagonalizing the expectation
    // value of the derivative of the dynamical matrix.
    // This works better than doing finite differences on the frequencies.
    for (int i : {0, 1, 2}) {
      // build the dynamical matrix at the two wavevectors
      // since we diagonalized it before, A = M.U.M*
      Eigen::MatrixXcd sqrtDPlus = Hs[iK * 7 + i * 2 + 1];
      Eigen::MatrixXcd sqrtDMins = Hs[iK * 7 + i * 2 + 2];

      // now we can build the velocity operator
      Eigen::MatrixXcd der = (sqrtDPlus - sqrtDMins) / (2. * delta);

      // and to be safe, we reimpose hermiticity
      der = 0.5 * (der + der.adjoint());

      // now we rotate in the basis of the eigenvectors at q.
      der = resultsEigenvectors[iK].adjoint() * der * resultsEigenvectors[iK];

      for (int ib1 = 0; ib1 < numWannier; ib1++) {
        for (int ib2 = 0; ib2 < numWannier; ib2++) {
          velocity(ib1, ib2, i) = der(ib1, ib2);
        }
      }
    }

    // turns out that the above algorithm has problems with degenerate bands
    // so, we diagonalize the velocity operator in the degenerate subspace,

    for (int ib = 0; ib < numWannier; ib++) {

      // first, we check if the band is degenerate, and the size of the
      // degenerate subspace
      int sizeSubspace = 1;
      for (int ib2 = ib + 1; ib2 < numWannier; ib2++) {
        // I consider bands degenerate if their frequencies are the same
        // within 0.0001 cm^-1
        if (abs(resultsEnergies[iK](ib) - resultsEnergies[iK](ib2)) > threshold) {
          break;
        }
        sizeSubspace += 1;
      }

      if (sizeSubspace > 1) {
        Eigen::MatrixXcd subMat(sizeSubspace, sizeSubspace);
        // we have to repeat for every direction
        for (int iCart : {0, 1, 2}) {

          // take the velocity matrix of the degenerate subspace
          for (int i = 0; i < sizeSubspace; i++) {
            for (int j = 0; j < sizeSubspace; j++) {
              subMat(i, j) = velocity(ib + i, ib + j, iCart);
            }
          }

          // reinforce hermiticity
          subMat = 0.5 * (subMat + subMat.adjoint());

          // diagonalize the subMatrix
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(subMat);
          const Eigen::MatrixXcd &newEigenvectors = eigenSolver.eigenvectors();

          // rotate the original matrix in the new basis
          // that diagonalizes the subspace.
          subMat = newEigenvectors.adjoint() * subMat * newEigenvectors;

          // reinforce hermiticity
          subMat = 0.5 * (subMat + subMat.adjoint());

          // substitute back
          for (int i = 0; i < sizeSubspace; i++) {
            for (int j = 0; j < sizeSubspace; j++) {
              velocity(ib + i, ib + j, iCart) = subMat(i, j);
            }
          }
        }
      }

      // we skip the bands in the subspace, since we corrected them already
      ib += sizeSubspace - 1;
    }
    resultsVelocities[iK] = velocity;
  }

  return std::make_tuple(resultsEnergies, resultsEigenvectors, resultsVelocities);
}

/**
 * Build Hamiltonians for a batch of k-points
 */
StridedComplexView3D ElectronH0Wannier::kokkosBatchedBuildBlochHamiltonian(
    const DoubleView2D &cartesianCoordinates) {

  // Kokkos quirkyness
  int numWannier = this->numWannier;
  int numVectors = this->numVectors;

  int numK = cartesianCoordinates.extent(0);

  // Col-major matrices stored contiguously
  Kokkos::LayoutStride Hlayout(
      numK, numWannier*numWannier,
      numWannier, 1,
      numWannier, numWannier
  );

  StridedComplexView3D hamiltonians("hamiltonians", Hlayout);
  Kokkos::complex<double> complexI(0.0, 1.0);

  auto bravaisVectors_d = this->bravaisVectors_d;
  auto vectorsDegeneracies_d = this->vectorsDegeneracies_d;
  auto h0R_d = this->h0R_d;

  // fourier transform the Hamiltonian
  if (!hasShiftedVectors) {
    ComplexView2D elPhases_d("elPhases_d", numK, numVectors);
    Kokkos::parallel_for(
        "el_phases", Range2D({0, 0}, {numK, numVectors}),
        KOKKOS_LAMBDA(int iK, int iR) {
          double arg = 0.0;
          for (int i = 0; i < 3; i++) {
            arg += cartesianCoordinates(iK, i) * bravaisVectors_d(iR, i);
          }
          elPhases_d(iK, iR) = exp(complexI * arg) / vectorsDegeneracies_d(iR);
        });

    Kokkos::parallel_for(
        "el_hamilton", Range3D({0, 0, 0}, {numK, numWannier, numWannier}),
        KOKKOS_LAMBDA(int iK, int m, int n) {
          Kokkos::complex<double> tmp(0.0);
          for (int iR = 0; iR < numVectors; iR++) {
            tmp += elPhases_d(iK, iR) * h0R_d(m, n, iR);
          }
          hamiltonians(iK, m, n) = tmp;
        });
    Kokkos::realloc(elPhases_d, 0, 0);

  } else {
    // with shifts
    auto vectorsShifts_d = this->vectorsShifts_d;
    auto degeneracyShifts_d = this->degeneracyShifts_d;

    Kokkos::parallel_for(
        "elHamiltonianShifted_d",
        Range3D({0, 0, 0}, {numK, numWannier, numWannier}),
        KOKKOS_LAMBDA(int iK, int iw1, int iw2) {
          Kokkos::complex<double> tmp(0.0);
          for (int iR = 0; iR < numVectors; iR++) {
            for (int iDeg = 0; iDeg < degeneracyShifts_d(iw1, iw2, iR); ++iDeg) {
              double arg = 0.;
              for (int i = 0; i < 3; ++i) {
                arg += cartesianCoordinates(iK, i) * vectorsShifts_d(i, iDeg, iw1, iw2, iR);
              }
              Kokkos::complex<double> phase = exp(complexI * arg)
                  / vectorsDegeneracies_d(iR) * degeneracyShifts_d(iw1, iw2, iR);
              tmp += h0R_d(iR, iw1, iw2) * phase;
            }
          }
          hamiltonians(iK, iw1, iw2) = tmp;
        });
  }

  return hamiltonians;
}

/**
 * Create and diagonalize Hamiltonians for a batch of k-points
 */
std::tuple<DoubleView2D, StridedComplexView3D> ElectronH0Wannier::kokkosBatchedDiagonalizeFromCoordinates(
    const DoubleView2D &cartesianCoordinates, const bool withMassScaling) {

  // currently not used, this supresses a warning
  (void) withMassScaling;

  int numWannier = this->numWannier; // Kokkos quirkiness

  // build Hamiltonians
  StridedComplexView3D blochHamiltonians =
      kokkosBatchedBuildBlochHamiltonian(cartesianCoordinates);

  // now the diagonalization.
  int numK = blochHamiltonians.extent(0);
  DoubleView2D allEnergies("energies_d", numK, numWannier);

  // perform diagonalization
  kokkosZHEEV(blochHamiltonians, allEnergies);

  // blochHamiltonians now contains eigenvectors
  // returns energies and eigenvectors
  return std::make_tuple(allEnergies, blochHamiltonians);
}

/**
 * Build and diagonalize Hamiltonians, with velocities
 * Returns the energies (nk, nb), eigenvectors (nk, nb, nb)
 * and velocities (nk, nb, nb, 3) at each k-point.
 * Note that for each k-point, the (nb, nb) eigenvector
 * matrix is column-major, as required by the cuSOLVER routine,
 * hence the StridedLayout.
 */
std::tuple<DoubleView2D, StridedComplexView3D, ComplexView4D>
ElectronH0Wannier::kokkosBatchedDiagonalizeWithVelocities(
    const DoubleView2D &cartesianCoordinates) {

  int numWannier = this->numWannier; // Kokkos quirkyness

  int numK = cartesianCoordinates.extent(0);

  double delta = 1.0e-8;
  double threshold = 0.000001 / energyRyToEv;// = 1 micro-eV

  // prepare all the wavevectors at which we need the hamiltonian
  DoubleView2D allVectors("wavevectors_k", numK * 7, 3);
  Kokkos::parallel_for(
      "elHamiltonianShifted_d", numK, KOKKOS_LAMBDA(int iK) {
        for (int i = 0; i < 3; ++i) {
          allVectors(iK * 7, i) = cartesianCoordinates(iK, i);
        }
        for (int iDir = 0; iDir < 3; ++iDir) {
          for (int i = 0; i < 3; ++i) {
            allVectors(iK * 7 + iDir * 2 + 1, i) = cartesianCoordinates(iK, i);
            allVectors(iK * 7 + iDir * 2 + 2, i) = cartesianCoordinates(iK, i);
            if (iDir == i) {
              allVectors(iK * 7 + iDir * 2 + 1, i) += delta;
              allVectors(iK * 7 + iDir * 2 + 2, i) -= delta;
            }
          }
        }
      });

  // Col-major matrices stored contiguously
  Kokkos::LayoutStride Hlayout(
      numK, numWannier*numWannier,
      numWannier, 1,
      numWannier, numWannier
  );

  // compute the electronic properties at all wavevectors
  StridedComplexView3D allHamiltonians = kokkosBatchedBuildBlochHamiltonian(allVectors);

  // save energies and eigenvectors to results
  DoubleView2D resultEnergies("energies", numK, numWannier);
  StridedComplexView3D resultEigenvectors("eigenvectors", Hlayout);
  ComplexView4D resultVelocities("velocities", numK, numWannier, numWannier, 3);

  // put the Hamiltonian matrix in resultEigenvectors
  Kokkos::parallel_for(
      "eigenvectors", Range3D({0, 0, 0}, {numK, numWannier, numWannier}),
      KOKKOS_LAMBDA(int iK, int m, int n) {
        resultEigenvectors(iK, m, n) = allHamiltonians(iK*7, m, n);
      });
  // now, diagonalize the H matrix in place
  kokkosZHEEV(resultEigenvectors, resultEnergies);

  // these are temporary "scratch" memory spaces
  ComplexView3D der("der", numK, numWannier, numWannier);
  ComplexView3D tmpV("tmpV", numK, numWannier, numWannier);

  for (int i = 0; i < 3; ++i) {

    // To build the velocity operator, we first compute the matrix
    // der = dH/dk = ( H(k+dk)-H(k-dk) ) / 2dk
    // we also reinforce the Hermiticity (better numerical properties)
    Kokkos::parallel_for(
        "der", Range3D({0, 0, 0}, {numK, numWannier, numWannier}),
        KOKKOS_LAMBDA(int iK, int m, int n) {
          auto HPlus = Kokkos::subview(allHamiltonians, iK * 7 + i * 2 + 1, Kokkos::ALL, Kokkos::ALL);
          auto HMins = Kokkos::subview(allHamiltonians, iK * 7 + i * 2 + 2, Kokkos::ALL, Kokkos::ALL);
          der(iK, m, n) = 0.25 / delta * ((HPlus(m, n) - HMins(m, n))
                 + Kokkos::conj(HPlus(n, m) - HMins(n, m)));
        });

    // Now we complete the Hellman-Feynman theorem
    // and compute the velocity as v = U(k)^* der * U(k)
    Kokkos::parallel_for(
        "tmpV", Range3D({0, 0, 0}, {numK, numWannier, numWannier}),
        KOKKOS_LAMBDA(int iK, int m, int n) {
          auto L = Kokkos::subview(resultEigenvectors, iK, Kokkos::ALL, Kokkos::ALL);
          auto R = Kokkos::subview(der, iK, Kokkos::ALL, Kokkos::ALL);
          auto A = Kokkos::subview(tmpV, iK, Kokkos::ALL, Kokkos::ALL);
          Kokkos::complex<double> tmp(0.,0.);
          for (int l = 0; l < numWannier; ++l) {
            tmp += Kokkos::conj(L(l,m)) * R(l, n);
          }
          A(m, n) = tmp;
        });

    Kokkos::parallel_for(
        "vel", Range3D({0, 0, 0}, {numK, numWannier, numWannier}),
        KOKKOS_LAMBDA(int iK, int m, int n) {
          double norm = 0.;
          for (int i=0; i<3; ++i) {
            norm += cartesianCoordinates(iK,i)*cartesianCoordinates(iK,i);
          }
          Kokkos::complex<double> tmp(0.,0.);
          if ( norm > 1.0e-6 ) {// skip the gamma point
            for (int l = 0; l < numWannier; ++l) {
              tmp += tmpV(iK, m, l) * resultEigenvectors(iK, l, n);
            }
          }
          resultVelocities(iK, m, n, i) = tmp;
        });
  }

  // deallocate the scratch
  Kokkos::resize(der, 0, 0, 0);
  Kokkos::resize(tmpV, 0, 0, 0);

  kokkosBatchedTreatDegenerateVelocities(cartesianCoordinates, resultEnergies,
                                         resultVelocities, threshold);

  return std::make_tuple(resultEnergies, resultEigenvectors, resultVelocities);
}

double ElectronH0Wannier::getDeviceMemoryUsage() {
  double memory = 16 * h0R_d.size() +
      8 * double(degeneracyShifts_d.size() + vectorsShifts_d.size() +
                 vectorsDegeneracies_d.size() + bravaisVectors_d.size());
  return memory;
}

// helper to figure out how many kpoints to put in a kpoint batch on host
int ElectronH0Wannier::estimateBatchSize(const bool& withVelocity) {

  double memoryAvailable = kokkosDeviceMemory->getAvailableMemory();
  std::complex<double> tmpC;

  int matrixSize = numWannier * numWannier;
  int transformSize = 0;
  if ( hasShiftedVectors) {
    transformSize = std::max(matrixSize, transformSize); // Fourier transform
  } else {
    transformSize = std::max(numVectors + matrixSize, transformSize); // Fourier transform
  }
  int diagonalizationSize = 2*matrixSize + numWannier; // diagonalization of H
  double memoryPerPoint = 0.;
  if (withVelocity) {
    memoryPerPoint += sizeof(tmpC)
        * std::max(diagonalizationSize*7,
                   transformSize*7 + 2*matrixSize + matrixSize*3);
  } else {
    memoryPerPoint += sizeof(tmpC) * std::max(diagonalizationSize, transformSize);
  }

  // we try to use 95% of the available memory (leave some buffer)
  int numBatches = int(memoryAvailable / memoryPerPoint * 0.95);
  if (numBatches < 1) {
    Error("Not enough memory available on device, reduce memory usage or run on "
        "a different device.");
  }

  return numBatches;
}

