#include "phonon_viscosity.h"
#include "constants.h"
#include "mpiHelper.h"
//#include <fstream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

PhononViscosity::PhononViscosity(Context &context_, StatisticsSweep &statisticsSweep_,
                                 Crystal &crystal_, BaseBandStructure &bandStructure_)
    : Observable(context_, statisticsSweep_, crystal_), bandStructure(bandStructure_) {

  tensordxdxdxd = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  tensordxdxdxd.setZero();

}

void PhononViscosity::calcRTA(VectorBTE &tau) {

  double norm = 1. / context.getQMesh().prod() /
                crystal.getVolumeUnitCell(dimensionality);

  auto particle = bandStructure.getParticle();
  tensordxdxdxd.setZero();

  auto excludeIndices = tau.excludeIndices;

  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();

  Kokkos::View<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tensordxdxdxd_k(tensordxdxdxd.data(), numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  Kokkos::Experimental::ScatterView<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace> scatter_tensordxdxdxd(tensordxdxdxd_k);
  Kokkos::parallel_for("phonon_viscosity", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, niss), [&] (int iis){

    auto tmpTensor = scatter_tensordxdxdxd.access();
    int is = iss[iis];
    auto isIdx = StateIndex(is);
    int iBte = bandStructure.stateToBte(isIdx).get();

    // skip the acoustic phonons
    if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte) != excludeIndices.end()) {
      return;
    }

    auto en = bandStructure.getEnergy(isIdx);
    if (en < phEnergyCutoff) { return; }
    auto velIrr = bandStructure.getGroupVelocity(isIdx);
    auto qIrr = bandStructure.getWavevector(isIdx);

    auto rotations = bandStructure.getRotationsStar(isIdx);
    for (const Eigen::Matrix3d& rotation : rotations) {

      Eigen::Vector3d q = rotation * qIrr;
      q = bandStructure.getPoints().bzToWs(q,Points::cartesianCoordinates);
      Eigen::Vector3d vel = rotation * velIrr;

      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

        auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
        double kBT = calcStat.temperature;
        double chemPot = 0; // always zero for phonons
        double boseP1 = particle.getPopPopPm1(en, kBT, chemPot);

        for (int i = 0; i < dimensionality; i++) {
          for (int j = 0; j < dimensionality; j++) {
            for (int k = 0; k < dimensionality; k++) {
              for (int l = 0; l < dimensionality; l++) {
                tmpTensor(iCalc, i, j, k, l) +=
                  q(i) * vel(j) * q(k) * vel(l) * boseP1 * tau(iCalc, 0, iBte) / kBT * norm;
              }
            }
          }
        }
      }
    }
  });
  Kokkos::Experimental::contribute(tensordxdxdxd_k, scatter_tensordxdxdxd);
  mpi->allReduceSum(&tensordxdxdxd);

}

void PhononViscosity::print() {

  std::string viscosityName = "Phonon";
  printViscosity(viscosityName,tensordxdxdxd, statisticsSweep, dimensionality);

}

void PhononViscosity::outputToJSON(const std::string& outFileName) {

  bool append = false; // it's a new file to write to
  std::string viscosityName = "phononViscosity";
  outputViscosityToJSON(outFileName, viscosityName,
                tensordxdxdxd, append, statisticsSweep, dimensionality);

}

int PhononViscosity::whichType() { return is4Tensor; }

