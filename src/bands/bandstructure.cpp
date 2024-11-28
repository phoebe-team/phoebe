#include "bandstructure.h"

#include "exceptions.h"
#include "mpiHelper.h"
#include "particle.h"
#include "points.h"
#include "utilities.h"
#include "Matrix.h"
#include <set>

std::vector<size_t> BaseBandStructure::parallelStateIterator() {
    size_t numStates = getNumStates();
    return mpi->divideWorkIter(numStates);
}
//-----------------------------------------------------------------------------

FullBandStructure::FullBandStructure(int numBands_, Particle &particle_,
                                     bool withVelocities, bool withEigenvectors,
                                     Points &points_, bool isDistributed_)
    : particle(particle_), points(points_), isDistributed(isDistributed_) {

  numBands = numBands_;
  numAtoms = numBands_ / 3;
  numPoints = points.getNumPoints();
  hasVelocities = withVelocities;
  hasEigenvectors = withEigenvectors;

  // Initialize data structures depending on memory distribution.
  // If is distributed is true, numBlockCols is used to column/wavevector
  // distribute the internal matrices
  int numBlockCols = std::min((int)mpi->getSize(), numPoints);

  // this will cause a crash from BLACS
  if(mpi->getSize() > numPoints) {
    Error("Phoebe cannot run with more MPI processes than points. Increase mesh sampling \n"
        "or decrease number of processes.");
  }

  try {
    energies = Matrix<double>(numBands, numPoints, 1, numBlockCols, isDistributed);
  } catch(std::bad_alloc &) {
    Error("Failed to allocate band structure energies.\n"
        "You are likely out of memory.");
  }
  numLocalPoints = energies.localCols();
  if (hasVelocities) {
    try {
      velocities = Matrix<std::complex<double>>(
        numBands * numBands * 3, numPoints, 1, numBlockCols, isDistributed);
    } catch(std::bad_alloc &) {
      Error("Failed to allocate band structure velocities.\n"
        "You are likely out of memory.");
    }
  }
  if (hasEigenvectors) {
    try {
      if ( particle.isPhonon() ) {
        eigenvectors = Matrix<std::complex<double>>(
            3 * numAtoms * numBands, numPoints, 1, numBlockCols, isDistributed);
      } else {
        eigenvectors = Matrix<std::complex<double>>(
            numBands * numBands, numPoints, 1, numBlockCols, isDistributed);
      }
    } catch(std::bad_alloc &) {
      Error("Failed to allocate band structure eigenvectors.\n"
        "You are likely out of memory.");
    }
  }
}

// copy constructor
FullBandStructure::FullBandStructure(const FullBandStructure &that)
    : particle(that.particle), points(that.points) {
  isDistributed = that.isDistributed;
  hasEigenvectors = that.hasEigenvectors;
  hasVelocities = that.hasVelocities;
  energies = that.energies;
  velocities = that.velocities;
  eigenvectors = that.eigenvectors;
  numBands = that.numBands;
  numAtoms = that.numAtoms;
  numPoints = that.numPoints;
  numLocalPoints = that.numLocalPoints;
}

FullBandStructure &FullBandStructure::operator=(  // copy assignment
    const FullBandStructure &that) {
  if (this != &that) {
    particle = that.particle;
    points = that.points;
    isDistributed = that.isDistributed;
    hasEigenvectors = that.hasEigenvectors;
    hasVelocities = that.hasVelocities;
    energies = that.energies;
    velocities = that.velocities;
    eigenvectors = that.eigenvectors;
    numBands = that.numBands;
    numAtoms = that.numAtoms;
    numPoints = that.numPoints;
    numLocalPoints = that.numLocalPoints;
  }
  return *this;
}

Particle FullBandStructure::getParticle() { return particle; }

Points FullBandStructure::getPoints() { return points; }

Point FullBandStructure::getPoint(const int &pointIndex) {
  return points.getPoint(pointIndex);
}

int FullBandStructure::getNumPoints(const bool &useFullGrid) {
  if ( useFullGrid) {
    return points.getNumPoints();
  } else {
    return numLocalPoints;
  }
}

int FullBandStructure::getNumBands() { return numBands; }
int FullBandStructure::getNumBands(WavevectorIndex &ik) {
  (void) ik;
  return numBands;
}

int FullBandStructure::hasWindow() { return 0; }

bool FullBandStructure::getIsDistributed() { return isDistributed; }

bool FullBandStructure::getHasEigenvectors() { return hasEigenvectors; }

int FullBandStructure::getIndex(const WavevectorIndex &ik,
                                 const BandIndex &ib) {
  return ik.get() * numBands + ib.get();
}

std::tuple<WavevectorIndex, BandIndex> FullBandStructure::getIndex(
    const int &is) {
  int ik = is / numBands;
  int ib = is - ik * numBands;
  auto ikk = WavevectorIndex(ik);
  auto ibb = BandIndex(ib);
  return std::make_tuple(ikk, ibb);
}

std::tuple<WavevectorIndex, BandIndex> FullBandStructure::getIndex(
    StateIndex &is) {
  return getIndex(is.get());
}

int FullBandStructure::getNumStates() { return numBands * getNumPoints(); }

std::vector<int> FullBandStructure::getWavevectorIndices() {
  std::set<int> kPointsSet;
  for ( auto tup : energies.getAllLocalStates()) {
    // returns global indices for local index
    auto ik = std::get<1>(tup);
    kPointsSet.insert(ik);
  }
  std::vector<int> kPointsList(kPointsSet.begin(), kPointsSet.end());
  return kPointsList;
}

std::vector<std::tuple<WavevectorIndex,BandIndex>> FullBandStructure::getStateIndices() {
  auto allLocalStates = energies.getAllLocalStates();
  std::vector<std::tuple<WavevectorIndex, BandIndex>> indices;
  for ( auto t : allLocalStates ) {
    auto ib = BandIndex(std::get<0>(t));
    auto ik = WavevectorIndex(std::get<1>(t));
    auto p = std::make_tuple(ik, ib);
    indices.push_back(p);
  }
  return indices;
}

std::vector<int> FullBandStructure::getBandIndices() const {
  std::vector<int> bandsList;
  for(int ib = 0; ib < numBands; ib++) {
      bandsList.push_back(ib);
  }
  return bandsList;
}

const double &FullBandStructure::getEnergy(WavevectorIndex &ik, BandIndex &ib) {
  int ibb = ib.get();
  int ikk = ik.get();
  if (!energies.indicesAreLocal(ibb,ikk)) {
    Error("Cannot access a non-local energy.");
  }
  return energies(ibb, ikk);
}

const double &FullBandStructure::getEnergy(StateIndex &is) {
  int stateIndex = is.get();
  auto tup = decompress2Indices(stateIndex, numPoints, numBands);
  auto ik = std::get<0>(tup);
  auto ib = std::get<1>(tup);
  if (!energies.indicesAreLocal(ib,ik)) {
    Error("Cannot access a non-local energy.");
  }
  return energies(ib, ik);
}

Eigen::VectorXd FullBandStructure::getEnergies(WavevectorIndex &ik) {
  Eigen::VectorXd x(numBands);
  if (!energies.indicesAreLocal(0,ik.get())) {
    Error("Cannot access a non-local energy.");
  }
  for (int ib=0; ib<numBands; ib++) {
    x(ib) = energies(ib,ik.get());
  }
  return x;
}

double FullBandStructure::getMaxEnergy() {
  Error("Developer error: getMaxEnergy not implemented for fullbandstructure.");
  return 0;
}

Eigen::Vector3d FullBandStructure::getGroupVelocity(StateIndex &is) {
  int stateIndex = is.get();
  auto tup = decompress2Indices(stateIndex, numPoints, numBands);
  auto ik = std::get<0>(tup);
  auto ib = std::get<1>(tup);
  if (!velocities.indicesAreLocal(ib,ik)) { // note ib is smaller than nRows
    Error("Cannot access a non-local velocity.");
  }
  Eigen::Vector3d vel;
  for (int i : {0, 1, 2}) {
    int ind = compress3Indices(ib, ib, i, numBands, numBands, 3);
    vel(i) = velocities(ind, ik).real();
  }
  return vel;
}

Eigen::MatrixXd FullBandStructure::getGroupVelocities(WavevectorIndex &ik) {
  int ikk = ik.get();
  if (!velocities.indicesAreLocal(0,ikk)) {
    Error("Cannot access a non-local velocity.");
  }
  Eigen::MatrixXd vel(numBands,3);
  for (int ib=0; ib<numBands; ib++ ) {
    for (int i : {0, 1, 2}) {
      int ind = compress3Indices(ib, ib, i, numBands, numBands, 3);
      vel(ib,i) = velocities(ind, ikk).real();
    }
  }
  return vel;
}

Eigen::Tensor<std::complex<double>, 3> FullBandStructure::getVelocities(
    WavevectorIndex &ik) {
  int ikk = ik.get();
  if (!velocities.indicesAreLocal(0,ikk)) {
    Error("Cannot access a non-local velocity.");
  }
  Eigen::Tensor<std::complex<double>, 3> vel(numBands, numBands, 3);
  for (int ib1 = 0; ib1 < numBands; ib1++) {
    for (int ib2 = 0; ib2 < numBands; ib2++) {
      for (int i : {0, 1, 2}) {
        int ind = compress3Indices(ib1, ib2, i, numBands, numBands, 3);
        vel(ib1,ib2,i) = velocities(ind, ikk);
      }
    }
  }
  return vel;
}

Eigen::MatrixXcd FullBandStructure::getEigenvectors(WavevectorIndex &ik) {
  int ikk = ik.get();
  if (!eigenvectors.indicesAreLocal(0,ikk)) {
    Error("Cannot access a non-local eigenvector.");
  }

  Eigen::MatrixXcd eigenVectors_(numBands, numBands);
  eigenVectors_.setZero();
  for (int ib1 = 0; ib1 < numBands; ib1++) {
    for (int ib2 = 0; ib2 < numBands; ib2++) {
      int ind = compress2Indices(ib1, ib2, numBands, numBands);
      eigenVectors_(ib1, ib2) = eigenvectors(ind, ikk);
    }
  }
  return eigenVectors_;
}

Eigen::Tensor<std::complex<double>, 3> FullBandStructure::getPhEigenvectors(
    WavevectorIndex &ik) {
  int ikk = ik.get();
  if (!eigenvectors.indicesAreLocal(0,ikk)) {
    Error("Cannot access a non-local velocity.");
  }
  Eigen::Tensor<std::complex<double>, 3> eigenVectors_(3, numAtoms, numBands);
  for (int ib = 0; ib < numBands; ib++) {
    for (int ia = 0; ia < numAtoms; ia++) {
      for (int ic : {0, 1, 2}) {
        int ind = compress3Indices(ia, ic, ib, numAtoms, 3, numBands);
        eigenVectors_(ic, ia, ib) = eigenvectors(ind, ikk);
      }
    }
  }
  return eigenVectors_;
}

Eigen::Vector3d FullBandStructure::getWavevector(StateIndex &is) {
  auto tup = getIndex(is);
  WavevectorIndex ik = std::get<0>(tup);
  return getWavevector(ik);
}

Eigen::Vector3d FullBandStructure::getWavevector(WavevectorIndex &ik) {
  Eigen::Vector3d k =
      points.getPointCoordinates(ik.get(), Points::cartesianCoordinates);
  return points.bzToWs(k, Points::cartesianCoordinates);
}

void FullBandStructure::setEnergies(Eigen::Vector3d &coordinates,
                                    Eigen::VectorXd &energies_) {
  int ik = points.getIndex(coordinates);
  if (!energies.indicesAreLocal(0,ik)) {
    // col distributed, only need to check ik
    Error("Cannot access a non-local energy");
  }
  for (int ib = 0; ib < energies.localRows(); ib++) {
    energies(ib, ik) = energies_(ib);
  }
}

void FullBandStructure::setEnergies(Point &point, Eigen::VectorXd &energies_) {
  int ik = point.getIndex();
  if (!energies.indicesAreLocal(0,ik)) {
    // col distributed, only need to check ik
    Error("DeveloperError: Cannot access a non-local energy in setEnergies.");
  }
  for (int ib = 0; ib < energies.localRows(); ib++) {
    energies(ib, ik) = energies_(ib);
  }
}

void FullBandStructure::setVelocities(
    Point &point, Eigen::Tensor<std::complex<double>, 3> &velocities_) {

  if (!hasVelocities) {
    Error("FullBandStructure was initialized without velocities, cannot set velocities.");
  }
  
  // we convert from a tensor to a vector (how it's stored in memory)
  Eigen::VectorXcd tmpVelocities_(numBands * numBands * 3);
  for (int i = 0; i < numBands; i++) {
    for (int j = 0; j < numBands; j++) {
      for (int k = 0; k < 3; k++) {
        // Note: State must know this order of index compression
        int idx = compress3Indices(i, j, k, numBands, numBands, 3);
        tmpVelocities_(idx) = velocities_(i, j, k);
      }
    }
  }
  int ik = point.getIndex();
  if(ik >= numPoints || ik < 0) std::cout << "found a bad ik value " << ik << std::endl;
  if (!velocities.indicesAreLocal(0,ik)) {
    // col distributed, only need to check ik
    Error("DeveloperError: Cannot set a non-local velocity in distributed velocity vector.");
  }
  // here this isn't a band index, it's actually an index over all compressed band indices 
  for (int ib = 0; ib < velocities.localRows(); ib++) {
    velocities(ib, ik) = tmpVelocities_(ib);
  }
}

void FullBandStructure::setEigenvectors(Point &point,
                                        Eigen::MatrixXcd &eigenvectors_) {
  if (!hasEigenvectors) {
    Error("FullBandStructure was initialized without eigenVectors");
  }
  // we convert from a matrix to a vector (how it's stored in memory)
  Eigen::VectorXcd tmp(numBands * numBands);
  for (int i = 0; i < numBands; i++) {
    for (int j = 0; j < numBands; j++) {
      // Note: State must know this order of index compression
      int idx = compress2Indices(i, j, numBands, numBands);
      tmp(idx) = eigenvectors_(i, j);
    }
  }
  int ik = point.getIndex();
  if (!eigenvectors.indicesAreLocal(0,ik)) {
    // col distributed, only need to check ik
    Error("Cannot access a non-local eigenvector.");
  }
  for (int ib = 0; ib < eigenvectors.localRows(); ib++) {
    eigenvectors(ib, ik) = tmp(ib);
  }
}

Eigen::VectorXd FullBandStructure::getBandEnergies(int &bandIndex) {
  // note: here we use the getWavevectorIndices function because if the
  // energies are distributed, we need to use global k indices
  // when calling energies(ib,ik)
  Eigen::VectorXd bandEnergies(energies.localCols());
  std::vector<int> wavevectorIndices = getWavevectorIndices();
  for (int i = 0; i < energies.localCols(); i++) {
    int ik = wavevectorIndices[i];  // global wavevector index
    bandEnergies(i) = energies(bandIndex, ik);
  }
  return bandEnergies;
}

std::vector<Eigen::Matrix3d> FullBandStructure::getRotationsStar(
    WavevectorIndex &ikIndex) {
  return points.getRotationsStar(ikIndex.get());
}

std::vector<Eigen::Matrix3d> FullBandStructure::getRotationsStar(
    StateIndex &isIndex) {
  auto t = getIndex(isIndex);
  WavevectorIndex ikIndex = std::get<0>(t);
  return getRotationsStar(ikIndex);
}

std::tuple<int, Eigen::Matrix3d> FullBandStructure::getRotationToIrreducible(
    const Eigen::Vector3d &x, const int &basis) {
  return points.getRotationToIrreducible(x, basis);
}

BteIndex FullBandStructure::stateToBte(StateIndex &isIndex) {
  auto t = getIndex(isIndex);
  // ik is in [0,N_k]
  WavevectorIndex ikIdx = std::get<0>(t);
  BandIndex ibIdx = std::get<1>(t);
  // to k from 0 to N_k_irreducible
  // ik is in [0,N_kIrr]
  int ikBte = points.asIrreducibleIndex(ikIdx.get());
  if (ikBte<0){
    Error("stateToBte is used on a non-irreducible point");
  }
  auto ik2Idx = WavevectorIndex(ikBte);
  int iBte = getIndex(ik2Idx,ibIdx);
  auto iBteIdx = BteIndex(iBte);
  return iBteIdx;
}

StateIndex FullBandStructure::bteToState(BteIndex &iBteIndex) {
  int iBte = iBteIndex.get();
  auto t = getIndex(iBte);
  // find ikIrr in interval [0,N_kIrr]
  int ikIrr = std::get<0>(t).get();
  BandIndex ib = std::get<1>(t);
  // find ik in interval [0,N_k]
  int ik = points.asReducibleIndex(ikIrr);
  auto ikIdx = WavevectorIndex(ik);
  int iss = getIndex(ikIdx, ib);
  return StateIndex(iss);
}


std::vector<int> FullBandStructure::irrStateIterator() {
  std::vector<int> ikIter = points.irrPointsIterator();
  std::vector<int> iter;
  for (int ik : ikIter) {
    auto ikIdx = WavevectorIndex(ik);
    for (int ib=0; ib<numBands; ib++) {
      auto ibIdx = BandIndex(ib);
      int is = getIndex(ikIdx, ibIdx);
      iter.push_back(is);
    }
  }
  return iter;
}

std::vector<int> FullBandStructure::parallelIrrStateIterator() {
  auto v = irrStateIterator();
  //
  auto divs = mpi->divideWork(v.size());
  int start = divs[0];
  int stop = divs[1];
  //
  std::vector<int> iter(v.begin() + start, v.begin() + stop);
  return iter;
}

std::vector<int> FullBandStructure::irrPointsIterator() {
  return points.irrPointsIterator();
}

std::vector<int> FullBandStructure::parallelIrrPointsIterator() {
  return points.parallelIrrPointsIterator();
}

int FullBandStructure::getPointIndex(const Eigen::Vector3d &crystalCoordinates,
                   const bool &suppressError) {
  if (suppressError) {
    return points.isPointStored(crystalCoordinates);
  } else {
    return points.getIndex(crystalCoordinates);
  }
}

std::vector<int> FullBandStructure::getReducibleStarFromIrreducible(const int &ik) {
  return points.getReducibleStarFromIrreducible(ik);
}
