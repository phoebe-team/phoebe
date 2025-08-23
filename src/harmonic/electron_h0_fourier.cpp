#include "electron_h0_fourier.h"

#include "constants.h"
#include "exceptions.h"
#include "io.h"
#include "particle.h"
#include <cmath>

void ElectronH0Fourier::trimBands(Context &context, const double &minEn,
                                  const double &maxEn) {
  std::vector<int> retainBand;
  int newNumBands = 0;
  for (int ib = 0; ib < numBands; ib++) {
    bool keepBand = false;
    Eigen::VectorXd bandEnergies = coarseBandStructure.getBandEnergies(ib);
    for (auto x : bandEnergies) {
      if (x >= minEn && x <= maxEn) {
        keepBand = true;
      }
    }
    if (keepBand) {
      newNumBands += 1;
      retainBand.push_back(ib);
    }
  }

  numBands = newNumBands;
  Eigen::MatrixXcd newExpansionCoefficients(numBands, numPositionVectors);
  int newIb = 0;
  for (int ib : retainBand) {
    newExpansionCoefficients.row(newIb) = expansionCoefficients.row(ib);
    newIb += 1;
  }
  expansionCoefficients = newExpansionCoefficients;

  double n = context.getNumOccupiedStates();
  double spinFactor = 2.;
  if (context.getHasSpinOrbit()) {
    spinFactor = 1.;
  }
  n -= retainBand[0] * spinFactor;
  context.setNumOccupiedStates(n);
}

ElectronH0Fourier::ElectronH0Fourier(Crystal &crystal_, const Points& coarsePoints_,
                                     const FullBandStructure& coarseBandStructure_,
                                     double cutoff_)
    : crystal(crystal_), coarseBandStructure(coarseBandStructure_),
      coarsePoints(coarsePoints_), particle(Particle::electron) {

  numBands = coarseBandStructure.getNumBands();
  cutoff = cutoff_;
  numDataPoints = coarseBandStructure.getNumPoints();
  refWavevector = coarseBandStructure.getPoint(0).getCoordinates(
      Points::cartesianCoordinates);

  // to do the interpolation, we need the lattice vector basis:
  setPositionVectors();
  // now we look for the expansion coefficients that interpolates the bands
  // note that setPositionVectors must stay above this call

  LoopPrint loopPrint("setting up Fourier interpolation with total "
        + std::to_string(numBands) + " bands,", "bands/mpi process",
                      int(mpi->divideWorkIter(numBands).size()));

  expansionCoefficients.resize(numBands, numPositionVectors);
  expansionCoefficients.setZero();
  for (int iBand : mpi->divideWorkIter(numBands)) {
    loopPrint.update();
    Eigen::VectorXd energies = coarseBandStructure.getBandEnergies(iBand);
    expansionCoefficients.row(iBand) = getCoefficients(energies);
  }
  mpi->allReduceSum(&expansionCoefficients);
  loopPrint.close();
}

Particle ElectronH0Fourier::getParticle() { return particle; }

int ElectronH0Fourier::getNumBands() { return numBands; }

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
ElectronH0Fourier::diagonalize(Point &point) {
  Eigen::Vector3d coordinates =
      point.getCoordinates(Points::cartesianCoordinates);
  auto tup = diagonalizeFromCoordinates(coordinates);
  auto energies = std::get<0>(tup);
  auto eigenVectors = std::get<1>(tup);
  return std::make_tuple(energies, eigenVectors);
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
ElectronH0Fourier::diagonalizeFromCoordinates(Eigen::Vector3d &wavevector) {
  Eigen::MatrixXcd eigenVectors(1, 1);
  eigenVectors.setZero();
  Eigen::VectorXd energies(numBands);
  for (int ib = 0; ib < numBands; ib++) {
    energies(ib) = getEnergyFromCoordinates(wavevector, ib);
  }
  return std::make_tuple(energies, eigenVectors);
}

Eigen::Tensor<std::complex<double>, 3>
ElectronH0Fourier::diagonalizeVelocity(Point &point) {
  Eigen::Vector3d coordinates =
      point.getCoordinates(Points::cartesianCoordinates);
  return diagonalizeVelocityFromCoordinates(coordinates);
}

Eigen::Tensor<std::complex<double>, 3>
ElectronH0Fourier::diagonalizeVelocityFromCoordinates(
    Eigen::Vector3d &coordinates) {
  Eigen::Tensor<std::complex<double>, 3> velocity(numBands, numBands, 3);
  velocity.setZero();
  for (int ib = 0; ib < numBands; ib++) {
    Eigen::Vector3d v = getGroupVelocityFromCoordinates(coordinates, ib);
    for (int i : {0, 1, 2}) {
      velocity(ib, ib, i) = v(i);
    }
  }
  return velocity;
}

FullBandStructure ElectronH0Fourier::populate(Points &fullPoints,
                                              const bool &withVelocities,
                                              const bool &withEigenvectors,
                                              const bool isDistributed) {

  FullBandStructure fullBandStructure(numBands, particle, withVelocities,
                                      withEigenvectors, fullPoints,
                                      isDistributed);

  LoopPrint loopPrint("populating electronic band structure", "states",
                              fullBandStructure.getNumStates());

  // this is mpi parallel in wavevector indices
  std::vector<int> kIndices = fullBandStructure.getLocalWavevectorIndices();
  for (size_t i = 0; i<kIndices.size(); i++) {

    loopPrint.update();

    int ik = kIndices[i];
    Point point = fullBandStructure.getPoint(ik);
    auto tup = diagonalize(point);
    auto ens = std::get<0>(tup);
    fullBandStructure.setEnergies(point, ens);
    if (withVelocities) {
      auto velocities = diagonalizeVelocity(point);
      fullBandStructure.setVelocities(point, velocities);
    }
  }
  loopPrint.close();
  return fullBandStructure;
}

double
ElectronH0Fourier::getRoughnessFunction(const Eigen::Vector3d &position) const {
  double norm = position.norm();
  return pow(1. - coefficient1 * norm / minDistance, 2) +
         coefficient2 * pow(norm / minDistance, 6);
}

std::complex<double>
ElectronH0Fourier::getStarFunction(Eigen::Vector3d &wavevector, int &iR) {
  std::complex<double> phase =
      complexI * wavevector.dot(positionVectors.col(iR));
  std::complex<double> starFunction = exp(phase) / positionDegeneracies(iR);
  return starFunction;
}

Eigen::Vector3cd
ElectronH0Fourier::getDerivativeStarFunction(Eigen::Vector3d &wavevector,
                                             int &iR) {
  std::complex<double> phase =
      complexI * wavevector.dot(positionVectors.col(iR));
  Eigen::Vector3cd starFunctionDerivative = complexI * positionVectors.col(iR) *
                                            exp(phase) /
                                            positionDegeneracies(iR);
  return starFunctionDerivative;
}

void ElectronH0Fourier::setPositionVectors() {
  // load the info on the super cell that we will use for the interpolation
  Eigen::Matrix3d directUnitCell = crystal.getDirectUnitCell();
  auto tup = coarsePoints.getMesh();
  auto grid = std::get<0>(tup);

  // the cutoff specifies the grid on which lattice vectors are searched.
  // Given a lattice vector in integer coordinates R = (n0,n1,n2) in
  // integer coordinates (i.e. can be brought in cartesian coordinates as
  // R=n0*a0+n1*a1+n2*a2 where a* are the lattice vectors of the primitive
  // unit cell)
  // then, the coordinates of the lattice vector R vary over a range between
  // -Nk*cutoff and Nk*cutoff, where Nk is the coarse grid of wavevectors.
  // so, with this definition, cutoff is a positive integer.
  int searchSize0 = int(cutoff);
  int searchSize1 = int(cutoff);
  int searchSize2 = int(cutoff);
  // cutoff is an integer
  if (cutoff < 1.) {
    Error("Fourier cutoff is too small: set it >=1");
  }

  // distDim is the size of the
  int distDim = 1;
  distDim *= ((searchSize0 + 1) * 2 + 1);
  distDim *= ((searchSize1 + 1) * 2 + 1);
  distDim *= ((searchSize2 + 1) * 2 + 1);

  std::vector<Eigen::Vector3d> tmpVectors;
  std::vector<double> tmpDegeneracies;
  int tmpNumPoints = 0;

  // what are we doing:
  // the "n" specifies a grid of lattice vectors, that are (2*(cutoff+1)+1)^3
  // times bigger than the grid of wavevectors. a grid of lattice vectors
  // equal to the grid of wavevectors would be the bare minimum for the
  // interpolation to work, and wouldn't be enough for anything good.

  LoopPrint loopPrint("Fourier super cell vector search",
                      "as first cell dimension", searchSize0 * grid(0) * 2);

  // now, we loop over the vectors of super cell A.
  for (int n0 = -searchSize0 * grid(0); n0 <= searchSize0 * grid(0); n0++) {
    loopPrint.update();
    for (int n1 = -searchSize1 * grid(1); n1 <= searchSize1 * grid(1); n1++) {
      for (int n2 = -searchSize2 * grid(2); n2 <= searchSize2 * grid(2); n2++) {
        // loop over a super cell B of size ((searchSize+1)*2+1)^3
        // bigger than super cell A. We compute the distance between any
        // vector in super cell B w.r.t. the vector "n"

        std::vector<double> distances;
        for (int i0 = -searchSize0 - 1; i0 <= searchSize0 + 1; i0++) {
          for (int i1 = -searchSize1 - 1; i1 <= searchSize1 + 1; i1++) {
            for (int i2 = -searchSize2 - 1; i2 <= searchSize2 + 1; i2++) {
              Eigen::Vector3d dist;
              dist(0) = n0 - i0 * grid(0);
              dist(1) = n1 - i1 * grid(1);
              dist(2) = n2 - i2 * grid(2);
              // distances in cartesian space
              dist = directUnitCell * dist;
              double dist2 = dist.norm();
              distances.push_back(dist2);
            }
          }
        }

        // find the minimum distance out of all distances
        double distMin = distances[0];
        for (auto dist : distances) {
          if (dist < distMin) {
            distMin = dist;
          }
        }

        // the point at distDim/2 is the reference "n" vector
        // i.e. the one generated by i0=i1=i2=0
        // if it's the minimum vector, than it's in the
        // Wigner Seitz zone of super cell A.
        // Therefore we save this vector.
        if (abs(distances[distDim / 2] - distMin) < 1.0e-6) {
          tmpNumPoints += 1;

          // count its degeneracy
          double degeneracy = 0.;
          for (int i = 0; i < distDim; i++) {
            if (abs(distances[i] - distMin) < 1.0e-6) {
              degeneracy += 1.;
            }
          }
          tmpDegeneracies.push_back(degeneracy);

          Eigen::Vector3d thisVec;
          thisVec << n0, n1, n2;
          tmpVectors.push_back(thisVec);
        }
      }
    }
  }
  loopPrint.close();

  // now we store the list of these lattice vectors in the class members
  numPositionVectors = tmpNumPoints;
  positionDegeneracies = Eigen::VectorXd::Zero(numPositionVectors);
  positionVectors = Eigen::MatrixXd::Zero(3, numPositionVectors);
  int originIndex = 0; // to look for R=0 vector
  for (int iR = 0; iR < numPositionVectors; iR++) {
    auto thisVec = tmpVectors[iR];
    // we convert from crystal to cartesian coordinates
    positionVectors.col(iR) = directUnitCell * thisVec;
    positionDegeneracies(iR) = tmpDegeneracies[iR];
    //
    if (thisVec.norm() < 1.0e-6) {
      originIndex = iR;
    }
  }

  // It's essential we keep the origin vectors at the index iR = 0
  // so we swap it. The other vectors don't need to be in any order.
  double tmp1 = positionDegeneracies(originIndex);
  double tmp2 = positionDegeneracies(0);
  positionDegeneracies(0) = tmp1;
  positionDegeneracies(originIndex) = tmp2;
  Eigen::Vector3d tmpV1 = positionVectors.col(originIndex);
  Eigen::Vector3d tmpV2 = positionVectors.col(0);
  positionVectors.col(0) = tmpV1;
  positionVectors.col(originIndex) = tmpV2;

  // the interpolation schemes also requires to know the minimum norm
  // of the lattice vectors
  minDistance = directUnitCell.col(0).norm();       // this is a first guess
  for (int iR = 1; iR < numPositionVectors; iR++) { // exclude 0 vector!
    double thisNorm = positionVectors.col(iR).norm();
    if (thisNorm < minDistance) {
      minDistance = thisNorm;
    }
  }
}

Eigen::VectorXcd
ElectronH0Fourier::getLagrangeMultipliers(Eigen::VectorXd energies) {

  Eigen::VectorXcd deltaEnergies(numDataPoints - 1);
#pragma omp parallel for default(none)                                         \
    shared(numDataPoints, energies, deltaEnergies)
  for (int i = 1; i < numDataPoints; i++) {
    deltaEnergies(i - 1) = energies(i) - energies(0);
  }

  Eigen::MatrixXcd smk(numDataPoints, numPositionVectors);
#pragma omp parallel for collapse(2) default(none)                             \
    shared(numPositionVectors, numDataPoints, coarseBandStructure, smk)
  for (int iR = 0; iR < numPositionVectors; iR++) {
    for (int i = 0; i < numDataPoints; i++) {
      Eigen::Vector3d iWavevector =
          coarseBandStructure.getPoint(i).getCoordinates(
              Points::cartesianCoordinates);
      smk(i, iR) = getStarFunction(iWavevector, iR);
    }
  }

  Eigen::MatrixXcd h(numDataPoints - 1, numDataPoints - 1);
  h.setZero();

  std::vector<double> rhos(numPositionVectors);
  std::vector<std::complex<double>> smk0s(numPositionVectors);
#pragma omp parallel for default(none)                                         \
    shared(rhos, smk0s, numPositionVectors, refWavevector, positionVectors)
  for (int iR = 0; iR < numPositionVectors; iR++) {
    rhos[iR] = getRoughnessFunction(positionVectors.col(iR));
    smk0s[iR] = getStarFunction(refWavevector, iR);
  }

  // note: this is the "expensive" loop.
  for (int iR = 0; iR < numPositionVectors; iR++) {
#pragma omp parallel for collapse(2) default(none)                             \
    shared(numPositionVectors, smk, smk0s, h, rhos, iR)
    for (int i = 0; i < numDataPoints - 1; i++) {
      for (int j = 0; j < numDataPoints - 1; j++) {
        h(i, j) += (smk(i + 1, iR) - smk0s[iR]) *
                   std::conj(smk(j + 1, iR) - smk0s[iR]) / rhos[iR];
      }
    }
  }

  // solve h*multipliers = deltaEnergies
  Eigen::VectorXcd multipliers(numDataPoints - 1);
  multipliers = h.ldlt().solve(deltaEnergies);
  return multipliers;
}

Eigen::VectorXcd ElectronH0Fourier::getCoefficients(Eigen::VectorXd energies) {
  Eigen::VectorXcd multipliers = getLagrangeMultipliers(energies);
  Eigen::VectorXcd coefficients(numPositionVectors);
  coefficients.setZero();

#pragma omp parallel for default(none)                                         \
    shared(numPositionVectors, numDataPoints, refWavevector, multipliers,      \
           coefficients)
  for (int m = 1; m < numPositionVectors; m++) {
    std::complex<double> smk0 = getStarFunction(refWavevector, m);
    for (int i = 1; i < numDataPoints; i++) {
      Eigen::Vector3d wavevector =
          coarseBandStructure.getPoint(i).getCoordinates(
              Points::cartesianCoordinates);
      coefficients(m) +=
          multipliers(i - 1) * std::conj(getStarFunction(wavevector, m) - smk0);
    }
    Eigen::Vector3d position = positionVectors.col(m);
    coefficients(m) /= getRoughnessFunction(position);
  }

  // special case for R=0
  coefficients(0) = energies(0);
  for (int m = 1; m < numPositionVectors; m++) {
    coefficients(0) -= coefficients(m) * getStarFunction(refWavevector, m);
  }

  return coefficients;
}

double ElectronH0Fourier::getEnergyFromCoordinates(Eigen::Vector3d &wavevector,
                                              int &bandIndex) {
  double energy = 0.;
  for (int m = 0; m < numPositionVectors; m++) {
    std::complex<double> c =
        expansionCoefficients(bandIndex, m) * getStarFunction(wavevector, m);
    energy += c.real();
  }
  return energy;
}

Eigen::Vector3d
ElectronH0Fourier::getGroupVelocityFromCoordinates(Eigen::Vector3d &wavevector,
                                              int &bandIndex) {
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  for (int m = 0; m < numPositionVectors; m++) {
    Eigen::Vector3cd c = expansionCoefficients(bandIndex, m) *
                         getDerivativeStarFunction(wavevector, m);
    velocity += c.real();
  }
  return velocity;
}

StridedComplexView3D ElectronH0Fourier::kokkosBatchedBuildBlochHamiltonian(
    [[maybe_unused]] const DoubleView2D &cartesianCoordinates) {
  Error("Kokkos not implemented in ElectronH0Fourier");
  // dummy return to silence a warning
  StridedComplexView3D temp;
  return temp;
}

std::tuple<DoubleView2D, StridedComplexView3D, ComplexView4D>
ElectronH0Fourier::kokkosBatchedDiagonalizeWithVelocities(
    [[maybe_unused]] const DoubleView2D &cartesianCoordinates) {
  Error("Kokkos not implemented in ElectronH0Fourier");
  std::tuple<DoubleView2D, StridedComplexView3D, ComplexView4D> temp;
  return temp;
}

std::tuple<DoubleView2D, StridedComplexView3D>
ElectronH0Fourier::kokkosBatchedDiagonalizeFromCoordinates(
          [[maybe_unused]] const DoubleView2D &cartesianCoordinates,
          [[maybe_unused]] const bool withMassScaling) {
  Error("Kokkos not implemented in ElectronH0Fourier");
  std::tuple<DoubleView2D, StridedComplexView3D> temp;
  return temp;
}
