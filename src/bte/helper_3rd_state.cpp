#include "helper_3rd_state.h"
#include "constants.h"
#include "delta_function.h"
#include "mpiHelper.h"
#include <set>

Helper3rdState::Helper3rdState(BaseBandStructure &innerBandStructure_,
                               BaseBandStructure &outerBandStructure_,
                               Eigen::MatrixXd &outerBose_,
                               StatisticsSweep &statisticsSweep_,
                               const int &smearingType_, PhononH0& h0_)
    : innerBandStructure(innerBandStructure_),
      outerBandStructure(outerBandStructure_), outerBose(outerBose_),
      statisticsSweep(statisticsSweep_), smearingType(smearingType_), h0(h0_) {

  numCalculations = statisticsSweep.getNumCalculations();
  storedAllQ3 = false;
  // three conditions must be met to avoid recomputing q3
  // 1 - q1 and q2 mesh must be the same
  // 2 - the mesh is gamma-centered
  // 3 - the mesh is complete (if q1 and q2 are only around 0, q3 might be
  //     at the border)
  auto tup = outerBandStructure.getPoints().getMesh();
  auto offset = std::get<1>(tup);
  if ((&innerBandStructure == &outerBandStructure) && (offset.norm() == 0.) &&
      innerBandStructure.hasWindow() == 0) {
    storedAllQ3 = true;
    storedAllQ3Case = storedAllQ3Case1;

  } else if ((&innerBandStructure == &outerBandStructure) &&
             (offset.norm() == 0.) && innerBandStructure.hasWindow() != 0) {
    storedAllQ3 = true;
    storedAllQ3Case = storedAllQ3Case2;

    // in this case, we filtered some wavevectors out of inner and outer
    // band structures. We must find all q3 that conserve momentum

    // first, we build the full grid that the 3rd point would fall into
    auto innerPoints = innerBandStructure.getPoints();
    auto tup2 = innerPoints.getMesh();
    auto mesh2 = std::get<0>(tup2);
    auto offset2 = std::get<1>(tup2);
    fullPoints3 = std::make_unique<Points>(
        innerBandStructure.getPoints().getCrystal(), mesh2, offset2);

    // now, we loop over the pairs of wavevectors
    std::set<int> listOfIndexes;
    int numPoints = innerPoints.getNumPoints();
    std::vector<size_t> iq2s = mpi->divideWorkIter(numPoints);
    int niq2s = iq2s.size();
#pragma omp parallel
    {
      std::set<int> myIndexes;
#pragma omp for nowait
      for(int iiq2 = 0; iiq2 < niq2s; iiq2++){
        int iq2 = iq2s[iiq2];
        for (int iq1 = 0; iq1 < numPoints; iq1++) {
          Eigen::Vector3d q1Coordinates =
            innerPoints.getPointCoordinates(iq1, Points::crystalCoordinates);
          Eigen::Vector3d q2Coordinates =
            innerPoints.getPointCoordinates(iq2, Points::crystalCoordinates);

          Eigen::Vector3d q3PlusCoordinates = q1Coordinates + q2Coordinates;
          Eigen::Vector3d q3MinusCoordinates = q1Coordinates - q2Coordinates;

          int iq3Plus = fullPoints3->getIndex(q3PlusCoordinates);
          int iq3Minus = fullPoints3->getIndex(q3MinusCoordinates);

          myIndexes.insert(iq3Plus);
          myIndexes.insert(iq3Minus);
        }
      }
#pragma omp critical
      listOfIndexes.insert(myIndexes.begin(), myIndexes.end());
    }

    std::vector<int> localCounts(mpi->getSize(), 0);
    localCounts[mpi->getRank()] = int(listOfIndexes.size());
    mpi->allReduceSum(&localCounts);
    int totalCounts = 0;
    for (int iRank = 0; iRank < mpi->getSize(); iRank++) {
      totalCounts += localCounts[iRank];
    }

    // these are the offset to be used for MPI gather
    std::vector<int> displacements(mpi->getSize(), 0);
    for (int i = 1; i < mpi->getSize(); i++) {
      displacements[i] = displacements[i - 1] + localCounts[i - 1];
    }

    std::vector<int> globalListOfIndexes(totalCounts, 0);
    int i = 0;
    for (auto it : listOfIndexes) {
      int index = i + displacements[mpi->getRank()];
      i++; // after computing index!
      globalListOfIndexes[index] = it;
    }
    mpi->allReduceSum(&globalListOfIndexes);

    // now we must avoid duplicates between different MPI processes
    std::set<int> setOfIndexes(globalListOfIndexes.begin(), globalListOfIndexes.end());

    // create the filtered list of points
    Eigen::VectorXi filter(setOfIndexes.size());
    i = 0;
    for (int iq : setOfIndexes) {
      filter(i) = iq;
      i++;
    }
    Points activePoints3 = *fullPoints3;
    activePoints3.setActiveLayer(filter);

    // build band structure
    bool withEigenvectors = true;
    bool withVelocities = true;
    if(mpi->mpiHead()) {
      std::cout << "Constructing band structure of intermediate phonon states." << std::endl;
    }
    bandStructure3 = std::make_unique<ActiveBandStructure>(
        activePoints3, &h0, withEigenvectors, withVelocities);
  }
}

// auto [energies3Minus, nb3Minus, eigenVectors3Minus, v3Minus, bose3]
/** This function receives in input the cartesian coordinates of a vector,
 * and returns the harmonic info for that vector.
 * This is to be used for the third wavevector of the 3-phonon scattering.
 */
std::tuple<Eigen::Vector3d, Eigen::VectorXd, int, Eigen::MatrixXcd,
           Eigen::MatrixXd, Eigen::MatrixXd>
Helper3rdState::get(Point &point1, Point &point2, const int &thisCase) {
  Eigen::Vector3d q3;
  if (thisCase == casePlus) {
    q3 = point1.getCoordinates(Points::cartesianCoordinates) +
         point2.getCoordinates(Points::cartesianCoordinates);
  } else {
    q3 = point1.getCoordinates(Points::cartesianCoordinates) -
         point2.getCoordinates(Points::cartesianCoordinates);
  }

  if (storedAllQ3) {
    // if the meshes are the same (and gamma centered)
    // q3 will fall into the same grid, and it's easy to get

    // note: 3rdBandStructure might still be different from inner/outer bs.
    // so, we must use the points from 3rdBandStructure to get the values

    int iq3;
    int nb3;
    Eigen::VectorXd energies3;
    Eigen::MatrixXcd eigenVectors3;
    Eigen::MatrixXd v3s;
    Eigen::MatrixXd bose3Data;

    if (storedAllQ3Case == storedAllQ3Case1) { // we use innerBandStructure
      auto points = innerBandStructure.getPoints();
      Eigen::Vector3d crystalPoints = points.cartesianToCrystal(q3);
      iq3 = points.getIndex(crystalPoints);
      auto iq3Index = WavevectorIndex(iq3);
      energies3 = innerBandStructure.getEnergies(iq3Index);
      eigenVectors3 = innerBandStructure.getEigenvectors(iq3Index);

      if (smearingType == DeltaFunction::adaptiveGaussian || smearingType == DeltaFunction::symAdaptiveGaussian) {
        v3s = innerBandStructure.getGroupVelocities(iq3Index);
      }
      nb3 = int(energies3.size());
      bose3Data = Eigen::MatrixXd(numCalculations, nb3);
      for (int ib3 = 0; ib3 < nb3; ib3++) {
        int is3 =
            outerBandStructure.getIndex(WavevectorIndex(iq3), BandIndex(ib3));
        StateIndex is3Idx(is3);
        BteIndex iBte1Idx = outerBandStructure.stateToBte(is3Idx);
        int iBte3 = iBte1Idx.get();
        bose3Data.col(ib3) = outerBose.col(iBte3);
      }
    } else {
      auto points = bandStructure3->getPoints();
      Eigen::Vector3d crystalPoints = points.cartesianToCrystal(q3);
      iq3 = points.getIndex(crystalPoints);
      auto iq3Index = WavevectorIndex(iq3);
      energies3 = bandStructure3->getEnergies(iq3Index);
      eigenVectors3 = bandStructure3->getEigenvectors(iq3Index);
      if (smearingType == DeltaFunction::adaptiveGaussian || smearingType == DeltaFunction::symAdaptiveGaussian) {
        v3s = bandStructure3->getGroupVelocities(iq3Index);
      }
      nb3 = int(energies3.size());
      bose3Data = Eigen::MatrixXd(numCalculations, nb3);
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
        auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
        double temp = calcStatistics.temperature;
        // phonon chem potential is always zero
        double chemPot = 0;
        for (int ib3 = 0; ib3 < nb3; ib3++) {
          bose3Data(iCalc, ib3) =
              h0.getParticle().getPopulation(energies3(ib3), temp, chemPot);
        }
      }
    }

    return std::make_tuple(q3, energies3, nb3, eigenVectors3, v3s, bose3Data);

  } else {
    // otherwise, q3 doesn't fall into the same grid
    // and we must therefore compute it from the hamiltonian

    int iq1 = point1.getIndex();

    Eigen::VectorXd energies3;
    Eigen::MatrixXcd eigenVectors3;
    Eigen::MatrixXd v3s;
    Eigen::MatrixXd bose3Data;

    int iq1Counter = iq1 - cacheOffset;

    if (thisCase == casePlus) {
      energies3 = cachePlusEnergies[iq1Counter];
      eigenVectors3 = cachePlusEigenVectors[iq1Counter];
      v3s = cachePlusVelocity[iq1Counter];
      bose3Data = cachePlusBose[iq1Counter];
    } else {
      energies3 = cacheMinusEnergies[iq1Counter];
      eigenVectors3 = cacheMinusEigenVectors[iq1Counter];
      v3s = cacheMinusVelocity[iq1Counter];
      bose3Data = cacheMinusBose[iq1Counter];
    }

    int nb3 = int(energies3.size());
    return std::make_tuple(q3, energies3, nb3, eigenVectors3, v3s, bose3Data);
  }
}

void Helper3rdState::prepare(const std::vector<int> &q1Indexes, const int &iq2) {

  if (!storedAllQ3) {
    auto numPoints = int(q1Indexes.size());
    cacheOffset = q1Indexes[0];

    try {
      cachePlusEnergies.resize(numPoints);
      cachePlusEigenVectors.resize(numPoints);
      cachePlusBose.resize(numPoints);
      cachePlusVelocity.resize(numPoints);

      cacheMinusEnergies.resize(numPoints);
      cacheMinusEigenVectors.resize(numPoints);
      cacheMinusBose.resize(numPoints);
      cacheMinusVelocity.resize(numPoints);
    } catch (std::bad_alloc& e) {
      Error("Out of memory trying to allocate third phonon state info.");
    }

    Particle particle = h0.getParticle();

    int iq1Counter = -1;
    for (int iq1 : q1Indexes) {
      iq1Counter++;
      Eigen::Vector3d q1 = outerBandStructure.getPoint(iq1).getCoordinates(
          Points::cartesianCoordinates);
      Eigen::Vector3d q2 = innerBandStructure.getPoint(iq2).getCoordinates(
          Points::cartesianCoordinates);

      Eigen::Vector3d q3Plus = q1 + q2;
      Eigen::Vector3d q3Minus = q1 - q2;

      auto tup = h0.diagonalizeFromCoordinates(q3Plus);
      auto energies3Plus = std::get<0>(tup);
      auto eigenVectors3Plus = std::get<1>(tup);
      auto tup1 = h0.diagonalizeFromCoordinates(q3Minus);
      auto energies3Minus = std::get<0>(tup1);
      auto eigenVectors3Minus = std::get<1>(tup1);

      int nb3Plus = int(energies3Plus.size());
      int nb3Minus = int(energies3Minus.size());

      Eigen::MatrixXd bose3DataPlus(numCalculations, nb3Plus);
      Eigen::MatrixXd bose3DataMinus(numCalculations, nb3Minus);
      bose3DataPlus.setZero();
      bose3DataMinus.setZero();

      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
        double temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
        // phonon chem potential is zero
        double chemPot = 0; //statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
        for (int ib3 = 0; ib3 < nb3Plus; ib3++) {
          bose3DataPlus(iCalc, ib3) =
              particle.getPopulation(energies3Plus(ib3), temp, chemPot);
        }
        for (int ib3 = 0; ib3 < nb3Minus; ib3++) {
          bose3DataMinus(iCalc, ib3) =
              particle.getPopulation(energies3Minus(ib3), temp, chemPot);
        }
      }

      Eigen::MatrixXd v3sPlus(nb3Plus, 3);
      Eigen::MatrixXd v3sMinus(nb3Minus, 3);
      v3sPlus.setZero();
      v3sMinus.setZero();
      if (smearingType == DeltaFunction::adaptiveGaussian || smearingType == DeltaFunction::symAdaptiveGaussian) {
        Eigen::Tensor<std::complex<double>, 3> v3sTmpPlus =
            h0.diagonalizeVelocityFromCoordinates(q3Plus);
        Eigen::Tensor<std::complex<double>, 3> v3sTmpMinus =
            h0.diagonalizeVelocityFromCoordinates(q3Minus);

        // we only need the diagonal elements of the velocity operator
        // i.e. the group velocity
        for (int i : {0, 1, 2}) {
          for (int ib3 = 0; ib3 < nb3Plus; ib3++) {
            v3sPlus(ib3, i) = v3sTmpPlus(ib3, ib3, i).real();
          }
          for (int ib3 = 0; ib3 < nb3Minus; ib3++) {
            v3sMinus(ib3, i) = v3sTmpMinus(ib3, ib3, i).real();
          }
        }
      }

      cachePlusEnergies[iq1Counter] = energies3Plus;
      cachePlusEigenVectors[iq1Counter] = eigenVectors3Plus;
      cachePlusBose[iq1Counter] = bose3DataPlus;
      cachePlusVelocity[iq1Counter] = v3sPlus;

      cacheMinusEnergies[iq1Counter] = energies3Minus;
      cacheMinusEigenVectors[iq1Counter] = eigenVectors3Minus;
      cacheMinusBose[iq1Counter] = bose3DataMinus;
      cacheMinusVelocity[iq1Counter] = v3sMinus;
    }
  }
}

const int Helper3rdState::casePlus = 0;

const int Helper3rdState::caseMinus = 1;
