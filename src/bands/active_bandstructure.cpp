#include "active_bandstructure.h"
#include "bandstructure.h"
#include "exceptions.h"
#include "mpiHelper.h"
#include "window.h"
#include "common_kokkos.h"

ActiveBandStructure::ActiveBandStructure(Particle &particle_, Points &points_)
    : particle(particle_), points(points_) {}

// copy constructor
ActiveBandStructure::ActiveBandStructure(const ActiveBandStructure &that)
    : particle(that.particle), points(that.points), energies(that.energies),
      velocities(that.velocities), eigenvectors(that.eigenvectors),
      hasEigenvectors(that.hasEigenvectors), numStates(that.numStates),
      numIrrStates(that.numIrrStates), numIrrPoints(that.numIrrPoints),
      numPoints(that.numPoints), numBands(that.numBands),
      numFullBands(that.numFullBands), windowMethod(that.windowMethod),
      auxBloch2Comb(that.auxBloch2Comb),
      cumulativeKbOffset(that.cumulativeKbOffset),
      bteAuxBloch2Comb(that.bteAuxBloch2Comb),
      bteCumulativeKbOffset(that.bteCumulativeKbOffset),
      cumulativeKbbOffset(that.cumulativeKbbOffset) {}

ActiveBandStructure &ActiveBandStructure::operator=(
    const ActiveBandStructure &that) { // assignment operator
  if (this != &that) {
    particle = that.particle;
    points = that.points;
    energies = that.energies;
    velocities = that.velocities;
    eigenvectors = that.eigenvectors;
    hasEigenvectors = that.hasEigenvectors;
    numStates = that.numStates;
    numIrrStates = that.numIrrStates;
    numIrrPoints = that.numIrrPoints;
    numPoints = that.numPoints;
    numBands = that.numBands;
    numFullBands = that.numFullBands;
    windowMethod = that.windowMethod;
    auxBloch2Comb = that.auxBloch2Comb;
    cumulativeKbOffset = that.cumulativeKbOffset;
    bteAuxBloch2Comb = that.bteAuxBloch2Comb;
    bteCumulativeKbOffset = that.bteCumulativeKbOffset;
    cumulativeKbbOffset = that.cumulativeKbbOffset;
  }
  return *this;
}

ActiveBandStructure::ActiveBandStructure(const Points &points_,
                                         HarmonicHamiltonian *h0,
                                         const bool &withEigenvectors,
                                         const bool &withVelocities)
    : particle(Particle(h0->getParticle().getParticleKind())), points(points_) {

  Kokkos::Profiling::pushRegion("ActiveBandStructure constructor");

  numPoints = points.getNumPoints();
  numFullBands = h0->getNumBands();
  numBands = Eigen::VectorXi::Zero(numPoints);
  for (int ik = 0; ik < numPoints; ik++) {
    numBands(ik) = numFullBands;
  }
  numStates = numFullBands * numPoints;
  hasEigenvectors = withEigenvectors;

  energies.resize(size_t(numPoints) * size_t(numFullBands), 0.);

  if (withVelocities) {
    try {
      size_t size = size_t(numPoints) * size_t(numFullBands) * size_t(numFullBands) * size_t(3);
      velocities.resize(size);
    } catch(std::bad_alloc &) {
      Error("Failed to allocate band structure velocities.\n"
        "You are likely running out of memory.");
    }
  }
  if (withEigenvectors) {
    try {
      size_t size = size_t(numPoints) * size_t(numFullBands) * size_t(numFullBands);
      eigenvectors.resize(size);
    } catch(std::bad_alloc &) {
      Error("Failed to allocate band structure eignevectors.\n"
        "You are likely running out of memory.");
    }
  }

  windowMethod = Window::nothing;
  buildIndices();

  std::vector<size_t> iks = mpi->divideWorkIter(numPoints);
  int niks = iks.size();

  DoubleView2D qs("qs", niks, 3);
  auto qs_h = Kokkos::create_mirror_view(qs);

  // store the full list of points for later use
#pragma omp parallel for
  for (size_t iik = 0; iik < size_t(niks); iik++) {
    size_t ik = iks[iik];
    Point point = points.getPoint(ik);
    Eigen::Vector3d q = point.getCoordinates(Points::cartesianCoordinates);
    for(int i = 0; i < 3; i++){
      qs_h(iik, i) = q(i);
    }
  }
  // copy the points to the GPU
  Kokkos::deep_copy(qs, qs_h);

  int approx_batch_size = h0->estimateBatchSize(false);

  // divide the q-points into batches, do all diagonalizations in parallel
  // with Kokkos for each batch
  Kokkos::Profiling::pushRegion("diagonalization loop");
  for(int start_iik = 0; start_iik < niks; start_iik += approx_batch_size){
    int stop_iik = std::min(niks, start_iik + approx_batch_size);

    Kokkos::Profiling::pushRegion("call diagonalization");
    auto tup = h0->kokkosBatchedDiagonalizeFromCoordinates(Kokkos::subview(qs, Kokkos::make_pair(start_iik, stop_iik), Kokkos::ALL));
    Kokkos::Profiling::popRegion();

    DoubleView2D energies_d = std::get<0>(tup);
    StridedComplexView3D eigenvectors_d = std::get<1>(tup);

    // copy the results to CPU
    auto energies_h = Kokkos::create_mirror_view(energies_d);
    auto eigenvectors_h = Kokkos::create_mirror_view(eigenvectors_d);
    Kokkos::deep_copy(energies_h, energies_d);
    Kokkos::deep_copy(eigenvectors_h, eigenvectors_d);

    // store the results in the old datastructures
    Kokkos::Profiling::pushRegion("store results");
#pragma omp parallel
    {
      int numBands = energies_h.extent(1);;
      Eigen::VectorXd energies(numBands);
      Eigen::MatrixXcd eigenvectors(numBands, numBands);
#pragma omp for
      for (size_t iik = 0; iik < size_t(stop_iik-start_iik); iik++) {
        size_t ik = iks[start_iik+iik];
        Point point = points.getPoint(ik);

        for(int i = 0; i < numBands; i++){
          energies(i) = energies_h(iik,i);
          if (withEigenvectors) {
            for(int j = 0; j < numBands; j++){
              eigenvectors(j,i) = eigenvectors_h(iik,j,i);
            }
          }
        }

        ActiveBandStructure::setEnergies(point, energies);
        if (withEigenvectors) {
          ActiveBandStructure::setEigenvectors(point, eigenvectors);
        }
      }
    }
    Kokkos::Profiling::popRegion();
  }
  Kokkos::Profiling::popRegion();

  if (withVelocities) {
    // the same again, but for velocities
    // TODO: I think this also returns the energies and velocities above, so we could avoid
    // the above calculation entirely (12.5% potential speedup)
    Kokkos::Profiling::pushRegion("bandstructure velocity loop");

    int approx_batch_size = h0->estimateBatchSize(true);

    // loop over points, do everything in parallel for all points in a batch
    for(int start_iik = 0; start_iik < niks; start_iik += approx_batch_size){
      int stop_iik = std::min(niks, start_iik + approx_batch_size);

      // compute batch of velocities
      auto tup = h0->kokkosBatchedDiagonalizeWithVelocities(Kokkos::subview(qs, Kokkos::make_pair(start_iik, stop_iik), Kokkos::ALL));
      ComplexView4D velocities_d = std::get<2>(tup);
      auto velocities_h = Kokkos::create_mirror_view(velocities_d);
      Kokkos::deep_copy(velocities_h, velocities_d);

      // store results in old datastructures
#pragma omp parallel
      {
        int numBands = velocities_h.extent(1);;
        Eigen::Tensor<std::complex<double>,3> velocities(numBands, numBands,3);
#pragma omp for
        for (size_t iik = 0; iik < size_t(stop_iik-start_iik); iik++) {
          size_t ik = iks[start_iik+iik];
          Point point = points.getPoint(ik);

          for(int k = 0; k < 3; k++){
            for(int i = 0; i < numBands; i++){
              for(int j = 0; j < numBands; j++){
                velocities(j,i,k) = velocities_h(iik,j,i,k);
              }
            }
          }
          ActiveBandStructure::setVelocities(point, velocities);
        }
      }
    }
    Kokkos::Profiling::popRegion();
  }
  mpi->allReduceSum(&energies);
  mpi->allReduceSum(&velocities);
  mpi->allReduceSum(&eigenvectors);
  Kokkos::Profiling::popRegion();
}

Particle ActiveBandStructure::getParticle() { return particle; }

Points ActiveBandStructure::getPoints() { return points; }

Point ActiveBandStructure::getPoint(const int &pointIndex) {
  return points.getPoint(pointIndex);
}

int ActiveBandStructure::getNumPoints(const bool &useFullGrid) {
  if (useFullGrid) {
    return points.getNumPoints();
  } else { // default
    return numPoints;
  }
}

int ActiveBandStructure::getNumBands() {
  if (windowMethod == Window::nothing) {
    return numFullBands;
  } else {
    Error("ActiveBandStructure doesn't have constant number of bands");
    return 0;
  }
}

int ActiveBandStructure::getNumBands(WavevectorIndex &ik) {
  return numBands(ik.get());
}

int ActiveBandStructure::hasWindow() { return windowMethod; }

bool ActiveBandStructure::getIsDistributed() { return false; }

int ActiveBandStructure::getIndex(const WavevectorIndex &ik,
                                  const BandIndex &ib) {
  return bloch2Comb(ik.get(), ib.get());
}

std::tuple<WavevectorIndex, BandIndex>
ActiveBandStructure::getIndex(const int &is) {
  auto tup = comb2Bloch(is);
  auto ik = std::get<0>(tup);
  auto ib = std::get<1>(tup);
  WavevectorIndex ikk(ik);
  BandIndex ibb(ib);
  return std::make_tuple(ikk, ibb);
}

std::tuple<WavevectorIndex, BandIndex>
ActiveBandStructure::getIndex(StateIndex &is) {
  int iss = is.get();
  return getIndex(iss);
}

int ActiveBandStructure::getNumStates() { return numStates; }

const double &ActiveBandStructure::getEnergy(StateIndex &is) {
  int stateIndex = is.get();
  if (energies.empty()) {
    Error("ActiveBandStructure energies haven't been populated");
  }
  return energies[stateIndex];
}

Eigen::VectorXd ActiveBandStructure::getEnergies(WavevectorIndex &ik) {
  int ikk = ik.get();
  int nb = numBands(ikk);
  Eigen::VectorXd x(nb);
  for (int ib = 0; ib < nb; ib++) {
    int ind = bloch2Comb(ikk, ib);
    x(ib) = energies[ind];
  }
  return x;
}

double ActiveBandStructure::getMaxEnergy() {
  if(getIsDistributed())
    Error("Developer error: getMaxEnergy not implemented when activeBS is distributed.");
  return *std::max_element(std::begin(energies), std::end(energies));
}

Eigen::Vector3d ActiveBandStructure::getGroupVelocity(StateIndex &is) {
  int stateIndex = is.get();
  if (velocities.empty()) {
    Error("ActiveBandStructure velocities haven't been populated");
  }
  auto tup = comb2Bloch(stateIndex);
  auto ik = std::get<0>(tup);
  auto ib = std::get<1>(tup);
  Eigen::Vector3d vel;
  vel(0) = velocities[velBloch2Comb(ik, ib, ib, 0)].real();
  vel(1) = velocities[velBloch2Comb(ik, ib, ib, 1)].real();
  vel(2) = velocities[velBloch2Comb(ik, ib, ib, 2)].real();
  return vel;
}

Eigen::MatrixXd ActiveBandStructure::getGroupVelocities(WavevectorIndex &ik) {
  int ikk = ik.get();
  int nb = numBands(ikk);
  Eigen::MatrixXd vel(nb, 3);
  for (int ib = 0; ib < nb; ib++) {
    for (int i : {0, 1, 2}) {
      vel(ib, i) = velocities[velBloch2Comb(ikk, ib, ib, i)].real();
    }
  }
  return vel;
}

Eigen::Tensor<std::complex<double>, 3>
ActiveBandStructure::getVelocities(WavevectorIndex &ik) {
  int ikk = ik.get();
  int nb = numBands(ikk);
  Eigen::Tensor<std::complex<double>, 3> vel(nb, nb, 3);
  for (int ib1 = 0; ib1 < nb; ib1++) {
    for (int ib2 = 0; ib2 < nb; ib2++) {
      for (int i : {0, 1, 2}) {
        vel(ib1, ib2, i) = velocities[velBloch2Comb(ikk, ib1, ib2, i)];
      }
    }
  }
  return vel;
}

Eigen::MatrixXcd ActiveBandStructure::getEigenvectors(WavevectorIndex &ik) {
  int ikk = ik.get();
  int nb = numBands(ikk);
  Eigen::MatrixXcd eigenVectors_(numFullBands, nb);
  eigenVectors_.setZero();
  for (int ib1 = 0; ib1 < numFullBands; ib1++) {
    for (int ib2 = 0; ib2 < nb; ib2++) {
      int ind = eigBloch2Comb(ikk, ib1, ib2);
      eigenVectors_(ib1, ib2) = eigenvectors[ind];
    }
  }
  return eigenVectors_;
}

Eigen::Tensor<std::complex<double>, 3>
ActiveBandStructure::getPhEigenvectors(WavevectorIndex &ik) {
  Eigen::MatrixXcd eigenMatrix = getEigenvectors(ik);
  int ikk = ik.get();
  int numAtoms = numFullBands / 3;
  Eigen::Tensor<std::complex<double>, 3> eigenS(3, numAtoms, numBands(ikk));
  for (int i = 0; i < numFullBands; i++) {
    auto tup = decompress2Indices(i, numAtoms, 3);
    auto iat = std::get<0>(tup);
    auto ic = std::get<1>(tup);
    for (int ib2 = 0; ib2 < numBands(ikk); ib2++) {
      eigenS(ic, iat, ib2) = eigenMatrix(i, ib2);
    }
  }
  return eigenS;
}

Eigen::Vector3d ActiveBandStructure::getWavevector(StateIndex &is) {
  auto tup = getIndex(is.get());
  WavevectorIndex ik = std::get<0>(tup);
  return getWavevector(ik);
}

Eigen::Vector3d ActiveBandStructure::getWavevector(WavevectorIndex &ik) {
  return points.getPointCoordinates(ik.get(), Points::cartesianCoordinates);
}

void ActiveBandStructure::setEnergies(Point &point,
                                      Eigen::VectorXd &energies_) {
  int ik = point.getIndex();
  for (int ib = 0; ib < energies_.size(); ib++) {
    int index = bloch2Comb(ik, ib);
    energies[index] = energies_(ib);
  }
}

void ActiveBandStructure::setEnergies(Point &point,
                                      std::vector<double> &energies_) {
  int ik = point.getIndex();
  for (int ib = 0; ib < int(energies_.size()); ib++) {
    int index = bloch2Comb(ik, ib);
    energies[index] = energies_[ib];
  }
}

void ActiveBandStructure::setEigenvectors(Point &point,
                                          Eigen::MatrixXcd &eigenvectors_) {
  int ik = point.getIndex();
  for (int i = 0; i < eigenvectors_.rows(); i++) {
    for (int j = 0; j < eigenvectors_.cols(); j++) {
      int index = eigBloch2Comb(ik, i, j);
      eigenvectors[index] = eigenvectors_(i, j);
    }
  }
}

void ActiveBandStructure::setVelocities(
    Point &point, Eigen::Tensor<std::complex<double>, 3> &velocities_) {
  int ik = point.getIndex();
  for (int ib1 = 0; ib1 < velocities_.dimension(0); ib1++) {
    for (int ib2 = 0; ib2 < velocities_.dimension(1); ib2++) {
      for (int j : {0, 1, 2}) {
        int index = velBloch2Comb(ik, ib1, ib2, j);
        velocities[index] = velocities_(ib1, ib2, j);
      }
    }
  }
}

int ActiveBandStructure::velBloch2Comb(const int &ik, const int &ib1,
                                       const int &ib2, const int &i) {
  return cumulativeKbbOffset(ik) + ib1 * numBands(ik) * 3 + ib2 * 3 + i;
}

int ActiveBandStructure::eigBloch2Comb(const int &ik, const int &ib1,
                                       const int &ib2) {
  return cumulativeKbOffset(ik) * numFullBands + ib1 * numBands(ik) + ib2;
}

int ActiveBandStructure::bloch2Comb(const int &ik, const int &ib) {
  return cumulativeKbOffset(ik) + ib;
}

std::tuple<int, int> ActiveBandStructure::comb2Bloch(const int &is) {
  return std::make_tuple(auxBloch2Comb(is, 0), auxBloch2Comb(is, 1));
}

int ActiveBandStructure::bteBloch2Comb(const int &ik, const int &ib) {
  return bteCumulativeKbOffset(ik) + ib;
}

std::tuple<int, int> ActiveBandStructure::bteComb2Bloch(const int &iBte) {
  return std::make_tuple(bteAuxBloch2Comb(iBte, 0), bteAuxBloch2Comb(iBte, 1));
}

void ActiveBandStructure::buildIndices() {
  auxBloch2Comb = Eigen::MatrixXi::Zero(numStates, 2);
  cumulativeKbOffset = Eigen::VectorXi::Zero(numPoints);
  cumulativeKbbOffset = Eigen::VectorXi::Zero(numPoints);

  for (int ik = 1; ik < numPoints; ik++) {
    cumulativeKbOffset(ik) = cumulativeKbOffset(ik - 1) + numBands(ik - 1);
    cumulativeKbbOffset(ik) =
        cumulativeKbbOffset(ik - 1) + 3 * numBands(ik - 1) * numBands(ik - 1);
  }

  int is = 0;
  for (int ik = 0; ik < numPoints; ik++) {
    for (int ib = 0; ib < numBands(ik); ib++) {
      auxBloch2Comb(is, 0) = ik;
      auxBloch2Comb(is, 1) = ib;
      is += 1;
    }
  }
}

void ActiveBandStructure::buildSymmetries() {

  Kokkos::Profiling::pushRegion("ABS.buildSymmetries");

  // ------------------
  // things to use in presence of symmetries
  {
    std::vector<Eigen::MatrixXd> allVelocities;
    for (int ik = 0; ik < getNumPoints(); ik++) {
      auto ikIdx = WavevectorIndex(ik);
      Eigen::MatrixXd v = getGroupVelocities(ikIdx);
      allVelocities.push_back(v);
    }
    points.setIrreduciblePoints(&allVelocities);
  }

  numIrrPoints = int(points.irrPointsIterator().size());
  numIrrStates = 0;
  for (int ik : points.irrPointsIterator()) {
    numIrrStates += numBands(ik);
  }

  bteAuxBloch2Comb = Eigen::MatrixXi::Zero(numIrrStates, 2);
  bteCumulativeKbOffset = Eigen::VectorXi::Zero(numIrrPoints);
  int is = 0;
  int ikOld = 0;
  for (int ik : points.irrPointsIterator()) {
    int ikIrr = points.asIrreducibleIndex(ik);
    if (ikIrr > 0) { // skip first iteration
      bteCumulativeKbOffset(ikIrr) =
          bteCumulativeKbOffset(ikIrr - 1) + numBands(ikOld);
    }
    for (int ib = 0; ib < numBands(ik); ib++) {
      bteAuxBloch2Comb(is, 0) = ikIrr;
      bteAuxBloch2Comb(is, 1) = ib;
      is++;
    }
    ikOld = ik;
  }
  Kokkos::Profiling::popRegion();
}

std::tuple<ActiveBandStructure, StatisticsSweep>
ActiveBandStructure::builder(Context &context, HarmonicHamiltonian &h0,
                             Points &points_, const bool &withEigenvectors,
                             const bool &withVelocities,
                             const bool &forceBuildAPP) {

  Particle particle = h0.getParticle();

  ActiveBandStructure activeBandStructure(particle, points_);

  // select a build method based on particle type
  // if it's an electron, we can't build on the fly for any reason.
  // must buildAPP (as post-processing), because we need to calculate chemical potential.
  // Phonons can be built APP.
  if (particle.isElectron() || forceBuildAPP) {

    StatisticsSweep s = activeBandStructure.buildAsPostprocessing(
        context, points_, h0, withEigenvectors, withVelocities);

    return std::make_tuple(activeBandStructure, s);

  }
  // but phonons are default built OTF.
  else { // if (particle.isPhonon())

    Eigen::VectorXd temperatures = context.getTemperatures();
    double temperatureMin = temperatures.minCoeff();
    double temperatureMax = temperatures.maxCoeff();

    Window window(context, particle, temperatureMin, temperatureMax);

    activeBandStructure.buildOnTheFly(window, points_, h0, withEigenvectors,
                                      withVelocities);

    StatisticsSweep statisticsSweep(context);
    return std::make_tuple(activeBandStructure, statisticsSweep);
  }
}

void ActiveBandStructure::buildOnTheFly(Window &window, Points points_,
                                        HarmonicHamiltonian &h0,
                                        const bool &withEigenvectors,
                                        const bool &withVelocities) {
  // this function proceeds in three logical blocks:
  // 1- we find out the list of "relevant" points
  // 2- initialize internal raw buffer for energies, velocities, eigenVectors
  // 3- populate the raw buffer

  // we have to build this in a way that works in parallel
  // ALGORITHM:
  // - loop over points. Diagonalize, and find if we want this k-point
  //   (while we are at it, we could save energies and the eigenvalues)
  // - find how many points each MPI rank has found
  // - communicate the indices
  // - loop again over wavevectors to compute energies and velocities

  Kokkos::Profiling::pushRegion("activeBandStructure::buildOnTheFly");
  numFullBands = 0; // save the unfiltered number of bands
  std::vector<int> myFilteredPoints;
  std::vector<std::vector<int>> myFilteredBands;

  Kokkos::Profiling::pushRegion("diagonalization loop");
  // iterate over mpi-parallelized wavevectors
  #pragma omp parallel
  {
  std::vector<int> filteredThreadPoints;
  std::vector<std::vector<int>> filteredThreadBands;

  std::vector<size_t> pointsIter = mpi->divideWorkIter(points_.getNumPoints());
  #pragma omp for nowait schedule(static)
  for (size_t iik = 0; iik < pointsIter.size(); iik++) {

    int ik = pointsIter[iik];

    Point point = points_.getPoint(ik);
    // diagonalize harmonic hamiltonian
    auto tup = h0.diagonalize(point);
    auto theseEnergies = std::get<0>(tup);
    auto theseEigenvectors = std::get<1>(tup);
    // ens is empty if no "relevant" energy is found.
    // bandsExtrema contains the lower and upper band index of "relevant"
    // bands at this point
    auto tup1 = window.apply(theseEnergies);
    auto ens = std::get<0>(tup1);
    auto bandsExtrema = std::get<1>(tup1);
    if (ens.empty()) { // nothing to do
      continue;
    } else { // save point index and "relevant" band indices
      filteredThreadPoints.push_back(ik);
      filteredThreadBands.push_back(bandsExtrema);
    }
  }
  // merge the vector collected by each thread
  int threadNum = 1;
  #ifdef OMP_AVAIL
  threadNum = omp_get_num_threads();
  #endif
  #pragma omp for schedule(static) ordered
  for(int i=0; i<threadNum; i++) {
    #pragma omp ordered
    {
    myFilteredPoints.insert(myFilteredPoints.end(),
        std::make_move_iterator(filteredThreadPoints.begin()),
        std::make_move_iterator(filteredThreadPoints.end()));

    myFilteredBands.insert(myFilteredBands.end(),
        std::make_move_iterator(filteredThreadBands.begin()),
        std::make_move_iterator(filteredThreadBands.end()));
    }
  }
  } // close OMP parallel region
  Kokkos::Profiling::popRegion();

  // this numBands is the full bands num, doesn't matter which point
  Point point = points_.getPoint(0);
  auto tup = h0.diagonalize(point);
  auto theseEnergies = std::get<0>(tup);
  numFullBands = int(theseEnergies.size());

  // now, we let each MPI process now how many points each process has found
  int myNumPts = int(myFilteredPoints.size());
  int mpiSize = mpi->getSize();

  // take the number of points of each process and fill
  // buffer receiveCounts with these values
  std::vector<int> receiveCounts(mpiSize);
  mpi->allGatherv(&myNumPts, &receiveCounts);

  // now we count the total number of wavevectors
  // by summing over receive counts
  numPoints = 0;
  for (int i = 0; i < mpi->getSize(); i++) {
    numPoints += receiveCounts[i];
  }

  // now we collect the wavevector indices
  // first we find the offset to compute global indices from local indices
  std::vector<int> displacements(mpiSize, 0);
  for (int i = 1; i < mpiSize; i++) {
    displacements[i] = displacements[i - 1] + receiveCounts[i - 1];
  }

  // collect all the indices in the filteredPoints vector
  Eigen::VectorXi filter(numPoints);
  filter.setZero();
  for (int i = 0; i < myNumPts; i++) {
    int index = i + displacements[mpi->getRank()];
    filter(index) = myFilteredPoints[i];
  }
  mpi->allReduceSum(&filter);

  // unfortunately, a vector<vector> isn't contiguous
  // let's use Eigen matrices
  Eigen::MatrixXi filteredBands(numPoints, 2);
  filteredBands.setZero();
  for (int i = 0; i < myNumPts; i++) {
    int index = i + displacements[mpi->getRank()];
    filteredBands(index, 0) = myFilteredBands[i][0];
    filteredBands(index, 1) = myFilteredBands[i][1];
  }
  mpi->allReduceSum(&filteredBands);

  //////////////// Done MPI recollection

  // numBands is a book-keeping of how many bands per point there are
  // this isn't a constant number.
  // Also, we look for the size of the arrays containing band structure.
  numBands = Eigen::VectorXi::Zero(numPoints);
  size_t numEnStates = 0;
  size_t numVelStates = 0;
  size_t numEigStates = 0;
  for (size_t ik = 0; ik < numPoints; ik++) {
    numBands(ik) = filteredBands(ik, 1) - filteredBands(ik, 0) + 1;
    numEnStates += numBands(ik);
    numVelStates += 3 * numBands(ik) * numBands(ik);
    numEigStates += numBands(ik) * numFullBands;
  }
  numStates = numEnStates;

  // initialize the raw data buffers of the activeBandStructure
  points = points_;
  points.setActiveLayer(filter);

  // construct the mapping from combined indices to Bloch indices
  buildIndices();

  energies.resize(numEnStates, 0.);
  if (withVelocities) {
    velocities.resize(numVelStates, complexZero);
  }
  if (withEigenvectors) {
    hasEigenvectors = true;
    eigenvectors.resize(numEigStates, complexZero);
  }

  windowMethod = window.getMethodUsed();

/////////////////

  std::vector<size_t> iks = mpi->divideWorkIter(numPoints);
  size_t niks = iks.size();
  Kokkos::Profiling::pushRegion("trimmed diagonalization loop");

// now we can loop over the trimmed list of points
#pragma omp parallel for default(none)                                         \
    shared(mpi, h0, window, filteredBands, withEigenvectors, withVelocities, iks, niks, Eigen::Dynamic)
  for (size_t iik = 0; iik < niks; iik++) {
    size_t ik = iks[iik];
    Point point = points.getPoint(ik);
    auto tup = h0.diagonalize(point);
    auto theseEnergies = std::get<0>(tup);
    auto theseEigenvectors = std::get<1>(tup);
    // eigenvectors(3,numAtoms,numBands)
    auto tup1 = window.apply(theseEnergies);
    auto ens = std::get<0>(tup1);
    auto bandsExtrema = std::get<1>(tup1);

    Eigen::VectorXd eigEns(numBands(ik));
    {
      int ibAct = 0;
      for (int ibFull = filteredBands(ik, 0); ibFull <= filteredBands(ik, 1);
           ibFull++) {
        eigEns(ibAct) = theseEnergies(ibFull);
        ibAct++;
      }
    }
    setEnergies(point, eigEns);

    if (withEigenvectors) {
      // we are reducing the basis size!
      // the first index has the size of the Hamiltonian
      // the second index has the size of the filtered bands
      Eigen::MatrixXcd theseEigenVectors_(numFullBands, numBands(ik));
      int ibAct = 0;
      for (int ibFull = filteredBands(ik, 0); ibFull <= filteredBands(ik, 1);
           ibFull++) {
        theseEigenVectors_.col(ibAct) = theseEigenvectors.col(ibFull);
        ibAct++;
      }
      setEigenvectors(point, theseEigenVectors_);
    }

    if (withVelocities) {
      // thisVelocity is a tensor of dimensions (ib, ib, 3)
      auto thisVelocity = h0.diagonalizeVelocity(point);

      // now we filter it
      Eigen::Tensor<std::complex<double>, 3> thisVelocities(numBands(ik),
                                                            numBands(ik), 3);
      int ib1New = 0;
      for (int ib1Old = filteredBands(ik, 0); ib1Old < filteredBands(ik, 1) + 1;
           ib1Old++) {
        int ib2New = 0;
        for (int ib2Old = filteredBands(ik, 0);
             ib2Old < filteredBands(ik, 1) + 1; ib2Old++) {
          for (int i = 0; i < 3; i++) {
            thisVelocities(ib1New, ib2New, i) = thisVelocity(ib1Old, ib2Old, i);
          }
          ib2New++;
        }
        ib1New++;
      }
      setVelocities(point, thisVelocities);
    }
  }
  Kokkos::Profiling::popRegion();
  mpi->allReduceSum(&energies);
  mpi->allReduceSum(&velocities);
  mpi->allReduceSum(&eigenvectors);

  buildSymmetries();
  Kokkos::Profiling::popRegion();
}

/** in this function, useful for electrons, we first compute the band structure
 * on a dense grid of wavevectors, then compute chemical potential/temperatures
 * and then filter it
 */
StatisticsSweep ActiveBandStructure::buildAsPostprocessing(
    Context &context, Points points_, HarmonicHamiltonian &h0,
    const bool &withEigenvectors, const bool &withVelocities) {

  Kokkos::Profiling::pushRegion("ActiveBandStructure::buildAsPostprocessing");
  bool tmpWithVel_ = false;
  bool tmpWithEig_ = true;
  bool tmpIsDistributed_ = true;

  // for now, we always generate a band structure which is distributed
  Kokkos::Profiling::pushRegion("h0.populate");
  FullBandStructure fullBandStructure =
      h0.populate(points_, tmpWithVel_, tmpWithEig_, tmpIsDistributed_);
  Kokkos::Profiling::popRegion();

  // ---------- establish mu and other statistics --------------- //
  // This will work even if fullBandStructure is distributed
  StatisticsSweep statisticsSweep(context, &fullBandStructure);

  // find min/max value of temperatures and chemical potentials
  int numCalculations = statisticsSweep.getNumCalculations();
  std::vector<double> chemPots;
  std::vector<double> temps;
  for (int i = 0; i < numCalculations; i++) {
    auto calcStat = statisticsSweep.getCalcStatistics(i);
    chemPots.push_back(calcStat.chemicalPotential);
    temps.push_back(calcStat.temperature);
  }

  double temperatureMin = *min_element(temps.begin(), temps.end());
  double temperatureMax = *max_element(temps.begin(), temps.end());
  double chemicalPotentialMin = *min_element(chemPots.begin(), chemPots.end());
  double chemicalPotentialMax = *max_element(chemPots.begin(), chemPots.end());

  // use statistics to determine the selection window for states
  Window window(context, particle, temperatureMin, temperatureMax,
                chemicalPotentialMin, chemicalPotentialMax);

  numFullBands = h0.getNumBands();
  std::vector<int> myFilteredPoints;
  std::vector<std::vector<int>> myFilteredBands;

  // ---------- select relevant bands and points  --------------- //
  // if all processes have the same points, divide up the points across
  // processes. As this band structure is already distributed, then we
  // can just perform this for the wavevectors belonging to each process's
  // part of the distributed band structure.
  //
  // If we for some reason wanted to revert to an un-distributed
  // fullBandStructure, we would need to replace parallelIter with:
  //     parallelIter = mpi->divideWorkIter(points.getNumPoints());
  // All else will function once the swap is made.

  Points pointsCopy = points_;
  points = pointsCopy; // first we copy, without the symmetries
  points_.setIrreduciblePoints();

  // Loop over the wavevectors belonging to each process
  std::vector<int> parallelIter = fullBandStructure.getWavevectorIndices();
  Kokkos::Profiling::pushRegion("filter points");

  // iterate over mpi-parallelized wavevectors
  #pragma omp parallel
  {
  std::vector<int> filteredThreadPoints;
  std::vector<std::vector<int>> filteredThreadBands;

  #pragma omp for nowait schedule(static)
  for (int iik = 0; iik < parallelIter.size(); iik++) {

    int ik = parallelIter[iik];

    auto ikIdx = WavevectorIndex(ik);
    //    Eigen::VectorXd theseEnergies =
    //    fullBandStructure.getEnergies(ikIndex);

    // Note: to respect symmetries, we want to make sure that the bands
    // filtered at point k are the same as the equivalent point.
    // this is slower (setIrreduciblePoints and re-diagonalization) but kind
    // of necessary to be consistent in using symmetries
    // also, since the band structure is distributed, we have to recompute
    // the quasiparticle energies, as they may not be available locally

    Eigen::Vector3d k = fullBandStructure.getWavevector(ikIdx);
    auto t = points_.getRotationToIrreducible(k, Points::cartesianCoordinates);
    int ikIrr = std::get<0>(t);
    Eigen::Vector3d kIrr =
        points_.getPointCoordinates(ikIrr, Points::cartesianCoordinates);
    auto t2 = h0.diagonalizeFromCoordinates(kIrr);
    Eigen::VectorXd theseEnergies = std::get<0>(t2);

    // ens is empty if no "relevant" energy is found.
    // bandsExtrema contains the lower and upper band index of "relevant"
    // bands at this point
    auto tup1 = window.apply(theseEnergies);
    auto ens = std::get<0>(tup1);
    auto bandsExtrema = std::get<1>(tup1);

    if (ens.empty()) { // nothing to do
      continue;
    } else {// save point index and "relevant" band indices
      filteredThreadPoints.push_back(ik);
      filteredThreadBands.push_back(bandsExtrema);
    }
  }

  // merge the vector collected by each thread
  int threadNum = 1;
  #ifdef OMP_AVAIL
  threadNum = omp_get_num_threads();
  #endif
  #pragma omp for schedule(static) ordered
  for(int i=0; i<threadNum; i++) {
    #pragma omp ordered
    {
    myFilteredPoints.insert(myFilteredPoints.end(),
        std::make_move_iterator(filteredThreadPoints.begin()),
        std::make_move_iterator(filteredThreadPoints.end()));

    myFilteredBands.insert(myFilteredBands.end(),
        std::make_move_iterator(filteredThreadBands.begin()),
        std::make_move_iterator(filteredThreadBands.end()));
    }
  }

  } // close OMP parallel region
  Kokkos::Profiling::popRegion();

  // the same for all points in full band structure
  numFullBands = fullBandStructure.getNumBands();

  // ---------- collect indices of relevant states  --------------- //
  // now that we've counted up the selected points and their
  // indices on each process, we need to reduce
  int myNumPts = int(myFilteredPoints.size());
  int mpiSize = mpi->getSize();

  // take the number of points of each process and fill
  // buffer receiveCounts with these values
  std::vector<int> receiveCounts(mpiSize);
  mpi->allGatherv(&myNumPts, &receiveCounts);

  // now we count the total number of wavevectors
  // by summing over receive counts
  numPoints = 0;
  for (int i = 0; i < mpi->getSize(); i++) {
    numPoints += receiveCounts[i];
  }

  // now we collect the wavevector indices
  // first we find the offset to compute global indices from local indices
  std::vector<int> displacements(mpiSize, 0);
  for (int i = 1; i < mpiSize; i++) {
    displacements[i] = displacements[i - 1] + receiveCounts[i - 1];
  }

  // collect all the indices in the filteredPoints vector
  Eigen::VectorXi filter(numPoints);
  filter.setZero();
  for (int i = 0; i < myNumPts; i++) {
    int index = i + displacements[mpi->getRank()];
    filter(index) = myFilteredPoints[i];
  }
  mpi->allReduceSum(&filter);

  // unfortunately, a vector<vector> isn't contiguous
  // let's use Eigen matrices
  Eigen::MatrixXi filteredBands(numPoints, 2);
  filteredBands.setZero();
  for (int i = 0; i < myNumPts; i++) {
    int index = i + displacements[mpi->getRank()];
    filteredBands(index, 0) = myFilteredBands[i][0];
    filteredBands(index, 1) = myFilteredBands[i][1];
  }
  mpi->allReduceSum(&filteredBands);

  //////////////// Done MPI recollection

  // ---------- count numBands and numStates  --------------- //
  // numBands is a book-keeping of how many bands per point there are
  // this isn't a constant number.
  // Also, we look for the size of the arrays containing band structure.
  numBands = Eigen::VectorXi::Zero(numPoints);
  size_t numEnStates = 0;
  size_t numVelStates = 0;
  size_t numEigStates = 0;
  for (int ik = 0; ik < numPoints; ik++) {
    numBands(ik) = filteredBands(ik, 1) - filteredBands(ik, 0) + 1;
    numEnStates += numBands(ik);
    numVelStates += 3 * numBands(ik) * numBands(ik);
    numEigStates += numBands(ik) * numFullBands;
  }
  numStates = numEnStates;

  // ---------- initialize internal data buffers --------------- //
  points.setActiveLayer(filter);

  // construct the mapping from combined indices to Bloch indices
  buildIndices();

  energies.resize(numEnStates, 0.);
  if (withVelocities) {
    velocities.resize(numVelStates, complexZero);
  }
  if (withEigenvectors) {
    hasEigenvectors = true;
    eigenvectors.resize(numEigStates, complexZero);
  }
  windowMethod = window.getMethodUsed();
  Kokkos::Profiling::pushRegion("collect energies and eigenvectors");

  // ----- collect ens, velocities, eigenVectors, at each localPt, then reduce
  // Now we can loop over the trimmed list of points.
  // To accommodate the case where FullBS is distributed,
  // we save the energies related to myFilteredPoints/Bands
  // and then allReduce or allGather those instead
  for (int i = 0; i < int(myFilteredPoints.size()); i++) {

    // index corresponding to index of wavevector in fullPoints
    auto ikIndex = WavevectorIndex(myFilteredPoints[i]);

    // index corresponding to wavevector in activePoints
    // as well as any array of length numActivePoints,
    // like numBands, filteredBands
    // ika = ikActive
    int ika = i + displacements[mpi->getRank()];
    Point point = points.getPoint(ika);

    // local ik, which corresponds to filteredPoints on this process
    Eigen::VectorXd theseEnergies = fullBandStructure.getEnergies(ikIndex);
    Eigen::MatrixXcd theseEigenvectors =
        fullBandStructure.getEigenvectors(ikIndex);

    // copy energies into internal storage
    Eigen::VectorXd eigEns(numBands(ika));
    {
      int ibAct = 0;
      for (int ibFull = filteredBands(ika, 0); ibFull <= filteredBands(ika, 1);
           ibFull++) {
        eigEns(ibAct) = theseEnergies(ibFull);
        ibAct++;
      }
    }
    setEnergies(point, eigEns);

    // copy eigenvectors into internal storage
    if (withEigenvectors) {
      // we are reducing the basis size!
      // the first index has the size of the Hamiltonian
      // the second index has the size of the filtered bands
      Eigen::MatrixXcd theseEigenVectors(numFullBands, numBands(ika));
      int ibAct = 0;
      for (int ibFull = filteredBands(ika, 0); ibFull <= filteredBands(ika, 1);
           ibFull++) {
        theseEigenVectors.col(ibAct) = theseEigenvectors.col(ibFull);
        ibAct++;
      }
      setEigenvectors(point, theseEigenVectors);
    }
  }
  // reduce over internal data buffers
  mpi->allReduceSum(&energies);
  if (withEigenvectors)
    mpi->allReduceSum(&eigenvectors);
  Kokkos::Profiling::popRegion();

  // compute velocities, store, reduce
  if (withVelocities) {
    Kokkos::Profiling::pushRegion("compute velocities");

// loop over the points available to this process
#pragma omp parallel for default(none)                                         \
    shared(myFilteredPoints, h0, displacements, mpi, filteredBands)
    for (int i = 0; i < int(myFilteredPoints.size()); i++) {

      // index corresponding to wavevector in points
      // as well as any array of length numActivePoints,
      // like numBands, filteredBands
      // ika = ikActive
      int ika = i + int(displacements[mpi->getRank()]);
      Point point = points.getPoint(ika);

      // thisVelocity is a tensor of dimensions (ib, ib, 3)
      auto thisVelocity = h0.diagonalizeVelocity(point);

      // now we filter it
      Eigen::Tensor<std::complex<double>, 3> thisVelocities(numBands(ika),
                                                            numBands(ika), 3);
      int ib1New = 0;
      for (int ib1Old = filteredBands(ika, 0);
           ib1Old < filteredBands(ika, 1) + 1; ib1Old++) {
        int ib2New = 0;
        for (int ib2Old = filteredBands(ika, 0);
             ib2Old < filteredBands(ika, 1) + 1; ib2Old++) {
          for (int ic = 0; ic < 3; ic++) {
            thisVelocities(ib1New, ib2New, ic) =
                thisVelocity(ib1Old, ib2Old, ic);
          }
          ib2New++;
        }
        ib1New++;
      }
      setVelocities(point, thisVelocities);
    }
    mpi->allReduceSum(&velocities);
    Kokkos::Profiling::popRegion();
  }
  buildSymmetries();
  Kokkos::Profiling::popRegion();
  return statisticsSweep;
}

std::vector<int> ActiveBandStructure::irrStateIterator() {
  std::vector<int> iter;
  for (int ik : points.irrPointsIterator()) {
    auto ikIdx = WavevectorIndex(ik);
    for (int ib = 0; ib < numBands(ik); ib++) {
      auto ibIdx = BandIndex(ib);
      int is = getIndex(ikIdx, ibIdx);
      iter.push_back(is);
    }
  }
  return iter;
}

std::vector<int> ActiveBandStructure::parallelIrrStateIterator() {
  auto v = irrStateIterator();
  //
  auto divs = mpi->divideWork(v.size());
  int start = divs[0];
  int stop = divs[1];
  //
  std::vector<int> iter(v.begin() + start, v.begin() + stop);
  return iter;
}

std::vector<int> ActiveBandStructure::irrPointsIterator() {
  return points.irrPointsIterator();
}

std::vector<int> ActiveBandStructure::parallelIrrPointsIterator() {
  return points.parallelIrrPointsIterator();
}

std::vector<Eigen::Matrix3d>
ActiveBandStructure::getRotationsStar(WavevectorIndex &ikIndex) {
  return points.getRotationsStar(ikIndex.get());
}

std::vector<Eigen::Matrix3d>
ActiveBandStructure::getRotationsStar(StateIndex &isIndex) {
  auto t = getIndex(isIndex);
  WavevectorIndex ikIndex = std::get<0>(t);
  return getRotationsStar(ikIndex);
}

BteIndex ActiveBandStructure::stateToBte(StateIndex &isIndex) {
  auto t = getIndex(isIndex);
  WavevectorIndex ikIdx = std::get<0>(t);
  BandIndex ibIdx = std::get<1>(t);
  // from k from 0 to N_k
  // to k from 0 to N_k_irreducible
  int ikBte = points.asIrreducibleIndex(ikIdx.get());
  if (ikBte < 0) {
    Error("stateToBte is used on a reducible point");
  }
  int iBte = bteBloch2Comb(ikBte, ibIdx.get());
  return BteIndex(iBte);
}

StateIndex ActiveBandStructure::bteToState(BteIndex &iBteIndex) {
  auto t = bteComb2Bloch(iBteIndex.get());
  int ikBte = std::get<0>(t);
  int ib = std::get<1>(t);
  int ik = points.asReducibleIndex(ikBte);
  int iss = getIndex(WavevectorIndex(ik), BandIndex(ib));
  return StateIndex(iss);
}

std::tuple<int, Eigen::Matrix3d>
ActiveBandStructure::getRotationToIrreducible(const Eigen::Vector3d &x,
                                              const int &basis) {
  return points.getRotationToIrreducible(x, basis);
}

int ActiveBandStructure::getPointIndex(
    const Eigen::Vector3d &crystalCoordinates, const bool &suppressError) {
  if (points.isPointStored(crystalCoordinates) == -1) {
    Error("Point not found in activeBandStructure, something wrong");
  }

  if (suppressError) {
    return points.isPointStored(crystalCoordinates);
  } else {
    return points.getIndex(crystalCoordinates);
  }
}

std::vector<int>
ActiveBandStructure::getReducibleStarFromIrreducible(const int &ik) {
  return points.getReducibleStarFromIrreducible(ik);
}
