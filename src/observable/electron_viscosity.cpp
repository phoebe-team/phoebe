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
}

void ElectronViscosity::calcRTA(VectorBTE &tau) {

  // add a relevant spin factor
  double spinFactor = 2.;
  if (context.getHasSpinOrbit()) {
    spinFactor = 1.;
  }

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
  mpi->allReduceSum(&tensordxdxdxd);
}


void ElectronViscosity::calcFromRelaxons(Eigen::VectorXd &eigenvalues, ParallelMatrix<double> &eigenvectors) {

  if (numCalculations > 1) {
    Error("Developer error: Relaxons electron viscosity cannot be calculated for more than one T.");
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
  int numStates = bandStructure.getNumStates();

  int iCalc = 0; // set to zero because of relaxons
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double kBT = calcStat.temperature;
  double chemPot = calcStat.chemicalPotential;

  // print info about the special eigenvectors
  // and save the indices that need to be skipped
  relaxonEigenvectorsCheck(eigenvectors, numRelaxons);

  //if(mpi->mpiHead()) std::cout << "about to calculate relaxon populations " << std::endl;

  // transform from the relaxon population basis to the electron population ------------
  Eigen::Tensor<double, 3> fRelaxons(3, 3, numStates);
  fRelaxons.setZero();
  for (auto tup0 : eigenvectors.getAllLocalStates()) {

    int is = std::get<0>(tup0);
    int alpha = std::get<1>(tup0);
    if (eigenvalues(alpha) <= 0. || alpha >= numRelaxons) { continue; }
    if (alpha == alpha0 || alpha == alpha_e) continue; // skip the energy eigenvector

    StateIndex isIdx(is);
    Eigen::Vector3d kPt = bandStructure.getWavevector(isIdx);
    kPt = bandStructure.getPoints().bzToWs(kPt,Points::cartesianCoordinates);
    Eigen::Vector3d vel = bandStructure.getGroupVelocity(isIdx);
    double en = bandStructure.getEnergy(isIdx);
    double pop = particle.getPopPopPm1(en, kBT, chemPot);
    // true sets a sqrt term
    for (int k = 0; k < dimensionality; k++) {
      for (int l = 0; l < dimensionality; l++) {
        fRelaxons(k, l, alpha) += kPt(k) * vel(l) * sqrt(pop) / kBT /
                                  eigenvalues(alpha) * eigenvectors(is, alpha);
      }
    }
  }
  mpi->allReduceSum(&fRelaxons);

  //if(mpi->mpiHead()) std::cout << "about to switch to relaxon populations " << std::endl;

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

 //if(mpi->mpiHead()) std::cout << "about to do final viscosity" << std::endl;

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

// TODO could merge this into one function with the phonon one
void ElectronViscosity::relaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors, int& numRelaxons) {

  // goal is to print and save the indices and scalar products of the special eigenvectors
  // add a relevant spin factor
  double spinFactor = 2.;
  if (context.getHasSpinOrbit()) { spinFactor = 1.; }

  double volume = crystal.getVolumeUnitCell(dimensionality);
  auto particle = bandStructure.getParticle();
  //int numRelaxons = eigenvalues.size();
  double Nk = context.getKMesh().prod();
  int numStates = bandStructure.getNumStates();

  int iCalc = 0; // set to zero because of relaxons
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double kBT = calcStat.temperature;
  double T = calcStat.temperature / kBoltzmannRy;
  double chemPot = calcStat.chemicalPotential;

  // calculate the special eigenvectors' product with eigenvectors ----------------
  // to report it's index and overlap + remove it from the calculation
  double C = 0; double U = 0;
  Eigen::VectorXd theta0(numStates);  theta0.setZero();
  Eigen::VectorXd theta_e(numStates); theta_e.setZero();
  for (int is : bandStructure.parallelStateIterator()) {

    auto isIdx = StateIndex(is);
    auto en = bandStructure.getEnergy(isIdx);
    double popM1 = particle.getPopPopPm1(en, kBT, chemPot);

    theta0(is) = sqrt(popM1) * (en - chemPot) * sqrt(spinFactor);
    theta_e(is) = sqrt(popM1) * sqrt(spinFactor);
    U += popM1;
    C += popM1 * (en - chemPot) * (en - chemPot);
  }
  mpi->allReduceSum(&theta0); mpi->allReduceSum(&theta_e);
  mpi->allReduceSum(&C); mpi->allReduceSum(&U);
  // apply normalizations
  C *= spinFactor / (volume * Nk * kBT * T);
  U *= spinFactor / (volume * Nk * kBT);
  theta_e *= 1./sqrt(kBT * U * Nk * volume);
  theta0 *= 1./sqrt(kBT * T * volume * Nk * C);

  // calculate the overlaps with special eigenvectors
  Eigen::VectorXd prodTheta0(numRelaxons); prodTheta0.setZero();
  Eigen::VectorXd prodThetae(numRelaxons); prodThetae.setZero();
  for (auto tup : eigenvectors.getAllLocalStates()) {

    auto is = std::get<0>(tup);
    auto gamma = std::get<1>(tup);
    prodTheta0(gamma) += eigenvectors(is,gamma) * theta0(is);
    prodThetae(gamma) += eigenvectors(is,gamma) * theta_e(is);

  }
  mpi->allReduceSum(&prodThetae); mpi->allReduceSum(&prodTheta0);

  // find the element with the maximum product
  prodTheta0 = prodTheta0.cwiseAbs();
  prodThetae = prodThetae.cwiseAbs();
  Eigen::Index maxCol0, idxAlpha0;
  Eigen::Index maxCol_e, idxAlpha_e;
  float maxTheta0 = prodTheta0.maxCoeff(&idxAlpha0, &maxCol0);
  float maxThetae = prodThetae.maxCoeff(&idxAlpha_e, &maxCol_e);

  if(mpi->mpiHead()) {
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << "Maximum scalar product theta_0.theta_alpha = " << maxTheta0 << " at index " << idxAlpha0 << "." << std::endl;
    std::cout << "First ten products with theta_0:";
    for(int gamma = 0; gamma < 10; gamma++) { std::cout << " " << prodTheta0(gamma); }
    std::cout << "\n\nMaximum scalar product theta_e.theta_alpha = " << maxThetae << " at index " << idxAlpha_e << "." << std::endl;
    std::cout << "First ten products with theta_e:";
    for(int gamma = 0; gamma < 10; gamma++) { std::cout << " " << prodThetae(gamma); }
    std::cout << std::endl;
  }

  // save these indices to the class objects
  // if they weren't really found, we leave these indices
  // as -1 so that no relaxons are skipped
  if(maxTheta0 >= 0.75) alpha0 = idxAlpha0;
  if(maxThetae >= 0.75) alpha_e = idxAlpha_e;
}

void ElectronViscosity::print() {

  std::string viscosityName = "Electron ";
  printViscosity(viscosityName, tensordxdxdxd, statisticsSweep, dimensionality);

}

void ElectronViscosity::outputToJSON(const std::string &outFileName) {

  bool append = false; // it's a new file to write to
  bool isPhonon = false;
  std::string viscosityName = "electronViscosity";
  outputViscosityToJSON(outFileName, viscosityName,
                tensordxdxdxd, isPhonon, append, statisticsSweep, dimensionality);

}

int ElectronViscosity::whichType() { return is4Tensor; }
