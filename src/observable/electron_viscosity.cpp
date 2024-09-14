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

void ElectronViscosity::calcFromRelaxons(Eigen::VectorXd &eigenvalues, ParallelMatrix<double> &eigenvectors) {

  Kokkos::Profiling::pushRegion("calcViscosityFromRelaxons");

  if (numCalculations > 1) {
    Error("Developer error: Relaxons electron viscosity cannot be calculated for more than one T or mu value.");
  }

  // NOTE: view phonon viscosity for notes about which equations are calculated here.

  // we decide to skip relaxon states
  // 1) there is a relaxon with zero (or epsilon) eigenvalue -> infinite tau
  // 2) there might be other states with infinite lifetimes, we skip them
  // 3) states which are alpha > numRelaxons, which were not calculated to
  //    save computational expense.

  // TODO pretty sure this should be used
  // add a relevant spin factor
  //double spinFactor = 2.;
  //if (context.getHasSpinOrbit()) { spinFactor = 1.; }

  double volume = crystal.getVolumeUnitCell(dimensionality);
  auto particle = bandStructure.getParticle();
  int numRelaxons = eigenvalues.size();
  double Nk = context.getKMesh().prod();
  size_t numStates = bandStructure.getNumStates();

  int iCalc = 0; // set to zero because of relaxons
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double kBT = calcStat.temperature;
  double chemPot = calcStat.chemicalPotential;

  // print info about the special eigenvectors
  // and save the indices that need to be skipped
  relaxonEigenvectorsCheck(eigenvectors, numRelaxons);

  size_t states = eigenvectors.size();
  LoopPrint loopPrint("Transforming relaxon populations","relaxons", eigenvectors.getAllLocalStates().size()); 

  // transform from the relaxon population basis to the electron population ------------
  Eigen::Tensor<double, 3> fRelaxons(3, 3, int(numStates));
  fRelaxons.setZero();

  // TODO the problem is this is a nonstandard all reduce!
  #pragma omp parallel for default(none) shared(bandStructure,particle,kBT,chemPot,alpha0, alpha_e, numRelaxons,eigenvalues,eigenvectors, loopPrint,fRelaxons)
  for (auto tup0 : eigenvectors.getAllLocalStates()) {

    loopPrint.update();

    int is = std::get<0>(tup0);
    int alpha = std::get<1>(tup0);
    if (eigenvalues(alpha) <= 0. || alpha >= numRelaxons) { continue; }
    if (alpha == alpha0 || alpha == alpha_e) continue; // skip the energy eigenvector

    // TODO this should be replaced somehow, it's super slow!
    StateIndex isIdx(is);
    Eigen::Vector3d kPt = bandStructure.getWavevector(isIdx);
    kPt = bandStructure.getPoints().bzToWs(kPt,Points::cartesianCoordinates);
   
    Eigen::Vector3d vel = bandStructure.getGroupVelocity(isIdx);
    double en = bandStructure.getEnergy(isIdx);
    double sqrtPop = sqrt(particle.getPopPopPm1(en, kBT, chemPot));

    // true sets a sqrt term
    for (int k = 0; k < dimensionality; k++) {
      for (int l = 0; l < dimensionality; l++) {
        #pragma omp critical 
        {
        fRelaxons(k, l, alpha) += kPt(k) * vel(l) * sqrtPop / kBT /
                                  eigenvalues(alpha) * eigenvectors(is, alpha);
        }
      }
    }
  }
  loopPrint.close();
  mpi->allReduceSum(&fRelaxons);

  // transform from relaxon to electron populations
  Eigen::Tensor<double, 3> f(3, 3, bandStructure.getNumStates());
  f.setZero();
  for (auto tup0 : eigenvectors.getAllLocalStates()) {

    int is = std::get<0>(tup0);
    int alpha = std::get<1>(tup0);
    if (eigenvalues(alpha) <= 0. || alpha >= numRelaxons) { continue; }
    if (alpha == alpha0 || alpha == alpha_e) continue; // skip the energy eigenvector

    for (int i : {0, 1, 2}) {
      for (int j : {0, 1, 2}) {
        f(i, j, is) += eigenvectors(is, alpha) * fRelaxons(i, j, alpha);
      }
    }
  }
  mpi->allReduceSum(&f);

  // calculate the final viscosity --------------------------
  double norm = 1. / volume / Nk;
  tensordxdxdxd.setZero();
  for (int is : bandStructure.parallelStateIterator()) {

    StateIndex isIdx(is);
    Eigen::Vector3d kPt = bandStructure.getWavevector(isIdx);
    kPt = bandStructure.getPoints().bzToWs(kPt,Points::cartesianCoordinates);
    Eigen::Vector3d vel = bandStructure.getGroupVelocity(isIdx);
    double en = bandStructure.getEnergy(isIdx);
    double ffm1 = particle.getPopPopPm1(en, kBT, chemPot);

    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        for (int k = 0; k < dimensionality; k++) {
          for (int l = 0; l < dimensionality; l++) {
            // note: the sqrt(pop) is to rescale the population from the symmetrized exact BTE
            tensordxdxdxd(iCalc, i, j, k, l) +=
                0.5 * ffm1 * norm * sqrt(ffm1) *
                (kPt(i) * vel(j) * f(k, l, is) + kPt(i) * vel(l) * f(k, j, is));
          }
        }
      }
    }
  }
  mpi->allReduceSum(&tensordxdxdxd);

  Kokkos::Profiling::popRegion();

}

void ElectronViscosity::relaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors, int& numRelaxons) {

  Kokkos::Profiling::pushRegion("electronRelaxonsEigenvectorsCheck");

  // sets alpha0 and alpha_e, the indices
  // of the special eigenvectors in the eigenvector list,
  // to be excluded in later calculations
  Particle particle = bandStructure.getParticle();
  genericRelaxonEigenvectorsCheck(eigenvectors, numRelaxons, particle,
                                 theta0, theta_e, alpha0, alpha_e);

  Kokkos::Profiling::popRegion();

}

// calculate special eigenvectors
void ElectronViscosity::calcSpecialEigenvectors() {

  genericCalcSpecialEigenvectors(bandStructure, statisticsSweep,
                          spinFactor, theta0, theta_e, phi, C, A);
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


void ElectronViscosity::outputRealSpaceToJSON(ScatteringMatrix& scatteringMatrix) {

  // call the function in viscosity io
  genericOutputRealSpaceToJSON(scatteringMatrix, bandStructure, statisticsSweep,
                                theta0, theta_e, phi, C, A, context);

}

int ElectronViscosity::whichType() { return is4Tensor; }
