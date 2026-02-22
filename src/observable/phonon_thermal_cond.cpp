#include "phonon_thermal_cond.h"

#include "constants.h"
#include "mpiHelper.h"
#include <ctime>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include "transport_io.h"

PhononThermalConductivity::PhononThermalConductivity(
    Context &context_, StatisticsSweep &statisticsSweep_, Crystal &crystal_, BaseBandStructure &bandStructure_)
    : Observable(context_, statisticsSweep_, crystal_), bandStructure(bandStructure_) {

  tensordxd = Eigen::Tensor<double, 3>(numCalculations, dimensionality, dimensionality);
  tensordxd.setZero();

  // set up units for writing to file
  thCondUnits = "W /(m K)";
  if (dimensionality == 3) {
    thCondConversion = thConductivityAuToSi;
  } else if (dimensionality == 2) {
    // multiply by the height of the cell / thickness of the cell to convert 3D -> 2D.
    // Because the unit cell volume is already reduced for dimensionality,
    // we only need to divide by thickness.
    //double height = crystal.getDirectUnitCell()(2,2);
    thCondConversion = thConductivityAuToSi * (1. / context.getThickness());
  } else { // dim = 1
    Warning("1D conductivity should be manually adjusted for the cross-section of the material.");
    thCondConversion = thConductivityAuToSi;
  }
}

// copy constructor
PhononThermalConductivity::PhononThermalConductivity(
    const PhononThermalConductivity &that)
    : Observable(that), bandStructure(that.bandStructure) {}

// copy assignment
PhononThermalConductivity &
PhononThermalConductivity::operator=(const PhononThermalConductivity &that) {
  Observable::operator=(that);
  if (this != &that) {
    bandStructure = that.bandStructure;
  }
  return *this;
}

PhononThermalConductivity
PhononThermalConductivity::operator-(const PhononThermalConductivity &that) {
  PhononThermalConductivity newObservable(context, statisticsSweep, crystal,
                                          bandStructure);
  baseOperatorMinus(newObservable, that);
  return newObservable;
}

void PhononThermalConductivity::calcFromCanonicalPopulation(VectorBTE &f) {
  VectorBTE n = f;
  n.canonical2Population(); // n = bose (bose+1) f
  calcFromPopulation(n);
}

void PhononThermalConductivity::calcFromPopulation(VectorBTE &n) {

  double norm = 1. / context.getQMesh().prod() / crystal.getVolumeUnitCell(dimensionality);

  auto excludeIndices = n.excludeIndices;

  tensordxd.setZero();

  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();

  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tensordxd_k(tensordxd.data(), numCalculations, dimensionality, dimensionality);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft, Kokkos::HostSpace> scatter_tensordxd(tensordxd_k);
  Kokkos::parallel_for("phonon_thermal_cond", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, niss), [&] (int iis){

      auto tensorPrivate = scatter_tensordxd.access();

      int is = iss[iis];
      StateIndex isIdx(is);
      double en = bandStructure.getEnergy(isIdx);
      Eigen::Vector3d velIrr = bandStructure.getGroupVelocity(isIdx);

      int iBte = bandStructure.stateToBte(isIdx).get();

      // skip the acoustic phonons
      if (std::ranges::find(excludeIndices, iBte) != excludeIndices.end()) {
        return;
      }

      auto rots = bandStructure.getRotationsStar(isIdx);
      for (const Eigen::Matrix3d &rot : rots) {

        auto vel = rot * velIrr;

        for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {

          Eigen::Vector3d nRot;
          for (int i = 0; i < dimensionality; i++) {
            nRot(i) = n(iCalc, i, iBte);
          }
          nRot = rot * nRot;

          for (int j = 0; j < dimensionality; j++) {
            for (int i = 0; i < dimensionality; i++) {
              tensorPrivate(iCalc, i, j) += nRot(i) * vel(j) * en * norm;
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(tensordxd_k, scatter_tensordxd);

  // lastly, the states were distributed with MPI
  mpi->allReduceSum(&tensordxd);

  // we print the unsymmetrized tensor to output file
  //if(mpi->mpiHead()) {
  //  std::cout << "Pre-symmetrization thermal conductivity:\n" << std::endl;
  //  print();
  //}
  // symmetrize the thermal conductivity
  //symmetrize(tensordxd);
}

void PhononThermalConductivity::calcVariational(VectorBTE &af, VectorBTE &f,
                                                VectorBTE &b) {

  double norm = 1. / context.getQMesh().prod() /
                crystal.getVolumeUnitCell(dimensionality);
  auto excludeIndices = f.excludeIndices;

  int numCalculations = statisticsSweep.getNumCalculations();

  tensordxd.setConstant(0.);

  Eigen::Tensor<double, 3> y1 = tensordxd.constant(0.);
  Eigen::Tensor<double, 3> y2 = tensordxd.constant(0.);

  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();

  // TODO should these be dimensionality?
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> y1_k(y1.data(), numCalculations, dimensionality, dimensionality),
    y2_k(y2.data(), numCalculations, dimensionality, dimensionality);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft, Kokkos::HostSpace> scatter_y1(y1_k), scatter_y2(y2_k);
  Kokkos::parallel_for("variational thermal conductivity", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, niss), [&] (int iis){
      auto x1 = scatter_y1.access();
      auto x2 = scatter_y2.access();

      int is = iss[iis];
      // skip the acoustic phonons
      if (std::find(excludeIndices.begin(), excludeIndices.end(), is) != excludeIndices.end()) {
        return;
      }

      StateIndex isIndex(is);
      BteIndex iBteIndex = bandStructure.stateToBte(isIndex);
      int isBte = iBteIndex.get();
      auto rots = bandStructure.getRotationsStar(isIndex);

      for (const Eigen::Matrix3d &rot : rots) {

        for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

          auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
          double temp = calcStat.temperature;
          double norm2 = norm * temp * temp;

          Eigen::Vector3d fRot, afRot, bRot;
          for (int i = 0; i < dimensionality; i++) {
            fRot(i) = f(iCalc, i, isBte);
            afRot(i) = af(iCalc, i, isBte);
            bRot(i) = b(iCalc, i, isBte);
          }
          fRot = rot * fRot;
          afRot = rot * afRot;
          bRot = rot * bRot;

          for (int i = 0; i < dimensionality; i++) {
            for (int j = 0; j < dimensionality; j++) {
              x1(iCalc, i, j) += fRot(i) * afRot(j) * norm2;
              x2(iCalc, i, j) += fRot(i) * bRot(j) * norm2;
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(y1_k, scatter_y1);
  Kokkos::Experimental::contribute(y2_k, scatter_y2);
  mpi->allReduceSum(&y1);
  mpi->allReduceSum(&y2);

  tensordxd = 2 * y2 - y1;
  // we print the unsymmetrized tensor to output file
  //if(mpi->mpiHead()) {
  //  std::cout << "Unsymmetrized thermal conductivity:\n" << std::endl;
  //  print();
  //}
  // symmetrize the thermal conductivity
  //symmetrize(tensordxd);
}

// IO related functions =====================================================

void PhononThermalConductivity::print() {
  
  // In the phonon only case, sigma, mu, S, are not printed 
  // so using kappa multiple times is fine as a dummy variable. 
  printHelper(statisticsSweep, dimensionality, tensordxd, tensordxd, tensordxd, tensordxd); 
 
}

void PhononThermalConductivity::outputToJSON(const std::string &outFileName) {

  if (!mpi->mpiHead()) return;
  
  outputPhononThermalCondToJSON(outFileName, statisticsSweep, 
                              dimensionality, tensordxd); 
}

// TODO move me to transport_io 
void PhononThermalConductivity::print(const int &iter) {
  
  // slightly lazy trick, pass kappa twice, the second arg is meant to be sigma
  // or an empty container. It is not printed in the phonon only case, however, 
  // so this doesn't matter. 
  printHelper(iter, statisticsSweep, dimensionality, tensordxd, tensordxd); 
  
}

int PhononThermalConductivity::whichType() { return is2Tensor; }

Eigen::Tensor<double,3> PhononThermalConductivity::getThermalConductivity() {
  return tensordxd;
}
