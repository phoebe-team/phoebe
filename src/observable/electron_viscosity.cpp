#include "electron_viscosity.h"
#include "constants.h"
#include "mpiHelper.h"
#include "viscosity_io.h"
#include <iomanip>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

ElectronViscosity::ElectronViscosity(Context &context_, StatisticsSweep &statisticsSweep_,
                                 Crystal &crystal_, BaseBandStructure &bandStructure_)
    : Observable(context_, statisticsSweep_, crystal_), bandStructure(bandStructure_) {

  tensordxdxdxd = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  tensordxdxdxd.setZero();

  // add a relevant spin factor
  spinFactor = 2.;
  if (context.getHasSpinOrbit()) {
    spinFactor = 1.;
  }

}

void ElectronViscosity::calcRTA(VectorBTE &tau) {

  Kokkos::Profiling::pushRegion("calcViscosityRTA");

  double Nk = context.getKMesh().prod();
  double norm = spinFactor / Nk / crystal.getVolumeUnitCell(dimensionality);
  auto particle = bandStructure.getParticle();
  tensordxdxdxd.setZero();
  //auto excludeIndices = tau.excludeIndices; // not used for electrons

  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();

  Kokkos::View<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tensordxdxdxd_k(tensordxdxdxd.data(), numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  Kokkos::Experimental::ScatterView<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace> scatter_tensordxdxdxd(tensordxdxdxd_k);

  Kokkos::parallel_for("electron_viscosity", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, niss), [&] (int iis){

    auto tmpTensor = scatter_tensordxdxdxd.access();
    int is = iss[iis];
    auto isIdx = StateIndex(is);
    int iBte = bandStructure.stateToBte(isIdx).get();

    auto en = bandStructure.getEnergy(isIdx);
    auto velIrr = bandStructure.getGroupVelocity(isIdx);
    auto kIrr = bandStructure.getWavevector(isIdx);

    auto rotations = bandStructure.getRotationsStar(isIdx);
    for (const Eigen::Matrix3d& rotation : rotations) {

      Eigen::Vector3d kPt = rotation * kIrr;
      kPt = bandStructure.getPoints().bzToWs(kPt,Points::cartesianCoordinates);
      Eigen::Vector3d vel = rotation * velIrr;

      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

        auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
        double kBT = calcStat.temperature;
        double chemPot = calcStat.chemicalPotential;
        double fermiP1 = particle.getPopPopPm1(en, kBT, chemPot);

        for (int i = 0; i < dimensionality; i++) {
          for (int j = 0; j < dimensionality; j++) {
            for (int k = 0; k < dimensionality; k++) {
              for (int l = 0; l < dimensionality; l++) {
                tmpTensor(iCalc, i, j, k, l) +=
                  kPt(i) * vel(j) * kPt(k) * vel(l) * fermiP1 * tau(iCalc, 0, iBte) / kBT * norm;
              }
            }
          }
        }
      }
    }
  });
  Kokkos::Experimental::contribute(tensordxdxdxd_k, scatter_tensordxdxdxd);

  Kokkos::Profiling::popRegion();

  mpi->allReduceSum(&tensordxdxdxd);
}

void ElectronViscosity::print() {

  std::string viscosityName = "Electron ";
  printViscosity(viscosityName, tensordxdxdxd, statisticsSweep, dimensionality);

}

void ElectronViscosity::outputToJSON(const std::string &outFileName) {

  bool append = false; // it's a new file to write to
  std::string viscosityName = "electronViscosity";
  outputViscosityToJSON(outFileName, viscosityName,
                tensordxdxdxd, append, statisticsSweep, dimensionality);

}

int ElectronViscosity::whichType() { return is4Tensor; }
