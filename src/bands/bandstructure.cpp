#include "bandstructure.h"

#include "exceptions.h"
#include "mpiHelper.h"
#include "particle.h"
#include "points.h"
#include "utilities.h"
#include "Matrix.h"
#include <iomanip>
#include <set>
#include <nlohmann/json.hpp>

std::vector<size_t> BaseBandStructure::parallelStateIterator() {
    size_t numStates = getNumStates();
    return mpi->divideWorkIter(numStates);
}

// this is very similar to the function which outputs the band
// structure info, we should make it generic
void BaseBandStructure::outputComponentsToJSON(const std::string &outFileName) {
                                        //const bool& symReduced) {

  if (!mpi->mpiHead()) return;

  std::string particleType;
  auto particle = getParticle();
  double energyConversion = energyRyToEv;
  std::string energyUnit = "eV";
  if (particle.isPhonon()) {
    particleType = "phonon";
    energyUnit = "meV";
    energyConversion *= 1000;
  } else {
    particleType = "electron";
  }

  // need to store as a vector format with dimensions
  // iCalc, ik. ib, iDim (where iState is unfolded into
  // ik, ib) for the velocities, no dim for energies
  std::vector<std::vector<std::vector<std::vector<double>>>> velocities;
  std::vector<std::vector<std::vector<double>>> energies;

  std::vector<int> pointsIterator = irrPointsIterator();
  // could use this to
  //if(symReduced) {
  //  pointsIterator = irrPointsIterator();
  //} else {
  //  pointsIterator = pointsIterator();
  //}

  // later we might also want to output population factors,
  // so we leave this here, but set numCalcs = 1,
  // as the plain band structure is only T = 0, etc
  int numCalculations = 1;
  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    //double temp = calcStatistics.temperature;
    //double chemPot = calcStatistics.chemicalPotential;
    //double doping = calcStatistics.doping;
    //temps.push_back(temp * temperatureAuToSi);
    //chemPots.push_back(chemPot * energyConversion);
    //dopings.push_back(doping);

    std::vector<std::vector<double>> wavevectorsE;
    std::vector<std::vector<std::vector<double>>> wavevectorsV;

    // loop over wavevectors
    for (int ik : pointsIterator) {

      auto ikIndex = WavevectorIndex(ik);

      std::vector<std::vector<double>> bandsV;
      std::vector<double> bandsE;

      // loop over bands here
      // get numBands at this point, in case it's an active band structure
      for (int ib = 0; ib < getNumBands(ikIndex); ib++) {

        auto ibIndex = BandIndex(ib);
        int is = getIndex(ikIndex, ibIndex);
        StateIndex isIdx(is);
        double ene = getEnergy(isIdx);
        bandsE.push_back(ene * energyConversion);
        Eigen::Vector3d vel = getGroupVelocity(isIdx);
        std::vector<double> tmp;
        for (int i = 0; i < 3; i++) {tmp.push_back(vel[i]*velocityRyToSi);}
        bandsV.push_back(tmp);
      }
      wavevectorsE.push_back(bandsE);
      wavevectorsV.push_back(bandsV);
    }
    energies.push_back(wavevectorsE);
    velocities.push_back(wavevectorsV);
  }

  //auto points = getPoints();
  // save the points in relevant coords to write to file
  std::vector<std::vector<double>> meshCoordinatesCart;
  std::vector<std::vector<double>> meshCoordinatesWSCart;
  std::vector<std::vector<double>> meshCoordinatesCrys;
  for (int ik : pointsIterator) {
    // save the wavevectors
    auto ikIndex = WavevectorIndex(ik);

    auto coordCart = getWavevector(ikIndex);
    auto refoldedCart = getPoints().bzToWs(coordCart,Points::cartesianCoordinates);
    meshCoordinatesCart.push_back({coordCart[0], coordCart[1], coordCart[2]});
    meshCoordinatesWSCart.push_back({refoldedCart[0], refoldedCart[1], refoldedCart[2]});
    auto coordCrys = getPoints().cartesianToCrystal(coordCart);
    meshCoordinatesCrys.push_back({coordCrys[0], coordCrys[1], coordCrys[2]});
  }

  // volume of the unit cell of the crystal
  double unitCellVolume = getPoints().getCrystal().getVolumeUnitCell();

  // output to json
  nlohmann::json output;
  output["unitCellVolume"] = unitCellVolume;
  output["volumeUnit"] = "Bohr^3";
  output["energies"] = energies;
  output["energyUnit"] = energyUnit;
  output["wavevectorsCartesian"] = meshCoordinatesCart;
  output["wavevectorsWignerSeitzCartesian"] = meshCoordinatesWSCart;
  output["wavevectorsCrystal"] = meshCoordinatesCrys;
  output["velocityCoordinatesType"] = "cartesian";
  output["velocityUnit"] = "m/s";
  output["velocities"] = velocities;
  output["distanceUnit"] = "Bohr";
  output["particleType"] = particleType;
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

void BaseBandStructure::printBandStructureStateInfo(const int& fullNumBands) {

  // print some info about how window and symmetries have reduced states
  if (mpi->mpiHead()) {

    bool useSym = getPoints().getCrystal().getNumSymmetries()>1;
    std::string particleName = "phonon";
    if(getParticle().isElectron()) particleName = "electron";

    // should be triggered by initial full bandstructure creation
    if(hasWindow() == 0) {
 
      std::cout << "Created " << particleName <<
      " band structure with " << getPoints().getNumPoints() <<
      " wavevector points and \n" << fullNumBands << " bands for a total of "
      << getPoints().getNumPoints()*fullNumBands << " states." << std::endl; 
    }
    // the next two blocks tell us about how when an ABS is used, state number is reduced
    if(hasWindow() != 0) {
      std::cout << "Window selection reduced " << particleName << " band structure from "
              << getPoints().getNumPoints() * fullNumBands << " to "
              << getNumStates() << " states."  << std::endl;
    }
    if(useSym) {
      std::cout << "Symmetries reduced " << particleName << " band structure from "
        << getNumStates() << " to " << irrStateIterator().size()
        << " states." << std::endl;
    }
  }
}

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
  int numBlockCols = int(std::min((size_t)mpi->getSize(), numPoints));

  // this will cause a crash from BLACS
  if(size_t(mpi->getSize()) > numPoints) {
    Error("Phoebe cannot run with more MPI processes than points. Increase mesh sampling \n"
        "or decrease number of processes.");
  }

  if (mpi->mpiHead()) { // print info on memory
    // count up the total memory use
    double x = numBands * numPoints; // number of energies, which will always be stored
    x *= 8; // size of double
    if(hasVelocities) {
      double xtemp = numBands * numBands * 3;
      xtemp *= numPoints;
      x += xtemp * 16; // complex double
    }
    if(hasEigenvectors) {
      if(particle.isPhonon()) {
        double xtemp = 3 * numAtoms * numBands;
        xtemp *= numPoints;
        x += xtemp * 16; // size of complex double
      } else {
        double xtemp = numAtoms * numBands;
        xtemp *= numPoints;
        x += xtemp * 16; // size of complex double
      }
    }
    x *= 1. / pow(1024,3);
    std::cout << std::setprecision(4);
    if(isDistributed) x *= 1./(1.*mpi->getSize());
    std::cout << "Allocating " << x << " GB (per MPI process) for the band structure." << std::endl;
  }

  try {
    energies = Matrix<double>(numBands, numPoints, 1, numBlockCols, isDistributed);
  } catch(std::bad_alloc& e) {
    Error("Failed to allocate band structure energies.\n"
        "You are likely out of memory.");
  }
  numLocalPoints = energies.localCols();

  if (hasVelocities) {
    try {
      velocities = Matrix<std::complex<double>>(
        numBands * numBands * 3, numPoints, 1, numBlockCols, isDistributed);
    } catch(std::bad_alloc& e) {
      Error("Failed to allocate band structure velocities.\n"
        "You are likely out of memory.");
    }
  }
  if (hasEigenvectors) {
    try {
      eigenvectors = Matrix<std::complex<double>>(numBands * numBands, numPoints, 1, numBlockCols, isDistributed);
    } catch(std::bad_alloc& e) {
      Error("Failed to allocate band structure eigenvectors.\n"
        "You are likely out of memory.");
    }
  }
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
int FullBandStructure::getNumBands([[maybe_unused]] WavevectorIndex &ik) {
  return numBands;
}
int FullBandStructure::getFullNumBands() { return numBands; }

int FullBandStructure::hasWindow() { return 0; }

bool FullBandStructure::getIsDistributed() { return isDistributed; }

bool FullBandStructure::getHasEigenvectors() { return hasEigenvectors; }

size_t FullBandStructure::getIndex(const WavevectorIndex &ik,
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

// if distributed, returns local kpt indices
std::vector<int> FullBandStructure::getLocalWavevectorIndices() {
  std::set<int> kPointsSet;
  for ( auto tup : energies.getAllLocalStates()) {
    // returns global indices for local index
    auto ik = std::get<1>(tup);
    kPointsSet.insert(ik);
  }
  std::vector<int> kPointsList(kPointsSet.begin(), kPointsSet.end());
  return kPointsList;
}

// if distributed, returns local state indices
// These states are local for the energies only! Velocities, etc cannot
// be accessed this way, would need another funciton
std::vector<std::tuple<WavevectorIndex,BandIndex>> FullBandStructure::getLocalEnergyStateIndices() {

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

// if distributed, returns local state indices
/*std::vector<size_t> FullBandStructure::getLocalStateIndices() {

  auto allLocalStates = energies.getAllLocalStates();
  std::vector<size_t> indices;
  for ( auto t : allLocalStates ) {
    auto ib = BandIndex(std::get<0>(t));
    auto ik = WavevectorIndex(std::get<1>(t));
    size_t is = getIndex(ik,ib);
    indices.push_back(is);
  }
  return indices;
}*/

std::vector<int> FullBandStructure::getLocalBandIndices() const {

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
    DeveloperError("Cannot access a non-local energy.");
  }
  return energies(ibb, ikk);
}

const double &FullBandStructure::getEnergy(StateIndex &is) {
  int stateIndex = is.get();
  auto tup = decompress2Indices(stateIndex, numPoints, numBands);
  auto ik = std::get<0>(tup);
  auto ib = std::get<1>(tup);
  if (!energies.indicesAreLocal(ib,ik)) {
    DeveloperError("Cannot access a non-local energy.");
  }
  return energies(ib, ik);
}

Eigen::VectorXd FullBandStructure::getEnergies(WavevectorIndex &ik) {
  Eigen::VectorXd x(numBands);
  if (!energies.indicesAreLocal(0,ik.get())) {
    DeveloperError("Cannot access a non-local energy.");
  }
  for (int ib=0; ib<numBands; ib++) {
    x(ib) = energies(ib,ik.get());
  }
  return x;
}

double FullBandStructure::getMaxEnergy() {
  DeveloperError("getMaxEnergy not implemented for fullbandstructure.");
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
    DeveloperError("Cannot access a non-local energy in setEnergies.");
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
  if (!velocities.indicesAreLocal(0,ik)) {
    // col distributed, only need to check ik
    DeveloperError("Cannot set a non-local velocity in distributed velocity vector.");
  }
  // here this isn't a band index, it's actually an index over all compressed band indices
  for (int ib = 0; ib < velocities.localRows(); ib++) {
    velocities(ib, ik) = tmpVelocities_(ib);
  }
}

void FullBandStructure::setEigenvectors(Point &point, Eigen::MatrixXcd &eigenvectors_) {

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
  std::vector<int> wavevectorIndices = getLocalWavevectorIndices();
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
  size_t iBte = getIndex(ik2Idx,ibIdx);
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

int FullBandStructure::getNumIrrStates() {
  return points.irrPointsIterator().size() * numBands;
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
