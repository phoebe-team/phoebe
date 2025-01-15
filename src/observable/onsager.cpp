#include "onsager.h"
#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include "onsager_utilities.h"
#include "particle.h"
#include <fstream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

OnsagerCoefficients::OnsagerCoefficients(StatisticsSweep &statisticsSweep_,
                                         Crystal &crystal_,
                                         BaseBandStructure &bandStructure_,
                                         Context &context_)
    : statisticsSweep(statisticsSweep_), crystal(crystal_),
      bandStructure(bandStructure_), context(context_) {

  // TODO : change this to use the context getSpinDegeneracyFactor 
  if (context.getHasSpinOrbit()) { spinFactor = 1.;
  } else { spinFactor = 2.; }

  dimensionality = crystal.getDimensionality();
  numCalculations = statisticsSweep.getNumCalculations();

  sigma.resize(numCalculations, dimensionality, dimensionality);
  seebeck.resize(numCalculations, dimensionality, dimensionality);
  kappa.resize(numCalculations, dimensionality, dimensionality);
  mobility.resize(numCalculations, dimensionality, dimensionality);
  sigma.setZero();
  seebeck.setZero();
  kappa.setZero();
  mobility.setZero();
  LEE.resize(numCalculations, dimensionality, dimensionality);
  LTE.resize(numCalculations, dimensionality, dimensionality);
  LET.resize(numCalculations, dimensionality, dimensionality);
  LTT.resize(numCalculations, dimensionality, dimensionality);
  LEE.setZero();
  LTE.setZero();
  LET.setZero();
  LTT.setZero();
}

void OnsagerCoefficients::calcFromEPA(
    VectorEPA &scatteringRates,
    Eigen::Tensor<double, 3> &energyProjVelocity, Eigen::VectorXd &energies) {

  Particle particle(Particle::electron);
  double factor = spinFactor / pow(twoPi, dimensionality);
  double energyStep = energies(1) - energies(0);

  LEE.setZero();
  LET.setZero();
  LTE.setZero();
  LTT.setZero();
  for (int iCalc = 0; iCalc < numCalculations; ++iCalc) {
    double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
    double temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
    for (int iBeta = 0; iBeta < dimensionality; ++iBeta) {
      for (int iAlpha = 0; iAlpha < dimensionality; ++iAlpha) {
        for (int iEnergy = 0; iEnergy < energies.size(); ++iEnergy) {

          double pop = particle.getPopPopPm1(energies(iEnergy), temp, chemPot);
          double en = energies(iEnergy) - chemPot;
          if (scatteringRates.data(iCalc, iEnergy) <= 1.0e-10 ||
              pop <= 1.0e-20) {
            continue;
          }

          double term = energyProjVelocity(iAlpha, iBeta, iEnergy) /
                        scatteringRates.data(iCalc, iEnergy) * factor * pop *
                        energyStep;

          LEE(iCalc, iAlpha, iBeta) += term / temp;
          LET(iCalc, iAlpha, iBeta) -= term * en / pow(temp, 2);
          LTE(iCalc, iAlpha, iBeta) -= term * en / temp;
          LTT(iCalc, iAlpha, iBeta) -= term * pow(en, 2) / pow(temp, 2);
        }
      }
    }
  }
  onsagerToTransportCoeffs(statisticsSweep, dimensionality,
                        LEE, LTE, LET, LTT, kappa, sigma, mobility, seebeck);
}

void OnsagerCoefficients::calcFromCanonicalPopulation(VectorBTE &fE,
                                                      VectorBTE &fT) {
  VectorBTE nE = fE;
  VectorBTE nT = fT;
  nE.canonical2Population(); // n = bose (bose+1) f
  nT.canonical2Population(); // n = bose (bose+1) f
  calcFromPopulation(nE, nT);
}

void OnsagerCoefficients::calcFromSymmetricPopulation(VectorBTE &nE, VectorBTE &nT) {

  VectorBTE nE2 = nE;
  VectorBTE nT2 = nT;

  Particle electron = bandStructure.getParticle();

  for (int is : bandStructure.parallelIrrStateIterator()) {
    StateIndex isIdx(is);
    double energy = bandStructure.getEnergy(isIdx);
    int iBte = bandStructure.stateToBte(isIdx).get();

    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      double temp = calcStat.temperature;
      double chemPot = calcStat.chemicalPotential;
      double x = sqrt(electron.getPopPopPm1(energy, temp, chemPot));
      for (int i : {0, 1, 2}) {
        nE2(iCalc, i, iBte) *= x;
        nT2(iCalc, i, iBte) *= x;
      }
    }
  }
  calcFromPopulation(nE2, nT2);
}

void OnsagerCoefficients::calcFromPopulation(VectorBTE &nE, VectorBTE &nT) {

  Kokkos::Profiling::pushRegion("calcOnsagerFromPopulation");

  double norm = spinFactor / context.getKMesh().prod() /
                crystal.getVolumeUnitCell(dimensionality);
  LEE.setZero();
  LET.setZero();
  LTE.setZero();
  LTT.setZero();

  auto points = bandStructure.getPoints();

  std::vector<int> states = bandStructure.parallelIrrStateIterator();
  int numStates = states.size();

  // kokkos version of the below code
  /*
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      LEE_k(LEE.data(), numCalculations, 3, 3);
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      LET_k(LET.data(), numCalculations, 3, 3);
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      LTE_k(LTE.data(), numCalculations, 3, 3);
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      LTT_k(LTT.data(), numCalculations, 3, 3);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace> scatter_LEE_k(LEE_k);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace> scatter_LET_k(LET_k);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace> scatter_LTE_k(LTE_k);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace> scatter_LTT_k(LTT_k);
  Kokkos::parallel_for("onsager_coefficients",
                       Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, numStates),
                       [&] (int iis) {

    auto LEE_private = scatter_LEE_k.access();
    auto LET_private = scatter_LET_k.access();
    auto LTE_private = scatter_LTE_k.access();
    auto LTT_private = scatter_LTT_k.access();

    int is = states[iis];

    StateIndex isIdx(is);
    double energy = bandStructure.getEnergy(isIdx);
    Eigen::Vector3d velIrr = bandStructure.getGroupVelocity(isIdx);
    int iBte = bandStructure.stateToBte(isIdx).get();
    auto rotations = bandStructure.getRotationsStar(isIdx);

    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      double en = energy - calcStat.chemicalPotential;

      for (const Eigen::Matrix3d& r : rotations) {
        Eigen::Vector3d thisNE = Eigen::Vector3d::Zero();
        Eigen::Vector3d thisNT = Eigen::Vector3d::Zero();
        for (int i : {0, 1, 2}) {
          thisNE(i) += nE(iCalc, i, iBte);
          thisNT(i) += nT(iCalc, i, iBte);
        }
        thisNE = r * thisNE;
        thisNT = r * thisNT;
        Eigen::Vector3d vel = r * velIrr;

        for (int i : {0, 1, 2}) {
          for (int j : {0, 1, 2}) {
            LEE_private(iCalc, i, j) += thisNE(i) * vel(j) * norm;
            LET_private(iCalc, i, j) += thisNT(i) * vel(j) * norm;
            LTE_private(iCalc, i, j) += thisNE(i) * vel(j) * en * norm;
            LTT_private(iCalc, i, j) += thisNT(i) * vel(j) * en * norm;
          }
        }
      }
    }
  });
  // safely sum the contribution to tensors from all different threads
  Kokkos::Experimental::contribute(LEE_k, scatter_LEE_k);
  Kokkos::Experimental::contribute(LET_k, scatter_LET_k);
  Kokkos::Experimental::contribute(LTE_k, scatter_LTE_k);
  Kokkos::Experimental::contribute(LTT_k, scatter_LTT_k);
*/

  for (int is : bandStructure.parallelIrrStateIterator()) {
    StateIndex isIdx(is);
    double energy = bandStructure.getEnergy(isIdx);
    Eigen::Vector3d velIrr = bandStructure.getGroupVelocity(isIdx);
    int iBte = bandStructure.stateToBte(isIdx).get();
    auto rotations = bandStructure.getRotationsStar(isIdx);

    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      double en = energy - calcStat.chemicalPotential;

      for (const Eigen::Matrix3d& r : rotations) {
        Eigen::Vector3d thisNE = Eigen::Vector3d::Zero();
        Eigen::Vector3d thisNT = Eigen::Vector3d::Zero();
        for (int i : {0, 1, 2}) {
          thisNE(i) += nE(iCalc, i, iBte);
          thisNT(i) += nT(iCalc, i, iBte);
        }
        thisNE = r * thisNE;
        thisNT = r * thisNT;
        Eigen::Vector3d vel = r * velIrr;

        for (int i : {0, 1, 2}) {
          for (int j : {0, 1, 2}) {
            LEE(iCalc, i, j) += thisNE(i) * vel(j) * norm;
            LET(iCalc, i, j) += thisNT(i) * vel(j) * norm;
            LTE(iCalc, i, j) += thisNE(i) * vel(j) * en * norm;
            LTT(iCalc, i, j) += thisNT(i) * vel(j) * en * norm;
          }
        }
      }
    }
  }

  // lastly, the states were distributed with MPI
  mpi->allReduceSum(&LEE);
  mpi->allReduceSum(&LTE);
  mpi->allReduceSum(&LET);
  mpi->allReduceSum(&LTT);

  onsagerToTransportCoeffs(statisticsSweep, dimensionality,
                        LEE, LTE, LET, LTT, kappa, sigma, mobility, seebeck);

  // TODO remove, this is for dev purposes
  writeIntegralContributions();

  Kokkos::Profiling::popRegion();
}

void OnsagerCoefficients::writeIntegralContributions() { 

  int numCalcs = statisticsSweep.getNumCalculations();
  auto particle = bandStructure.getParticle();

  std::vector<double> dfdeOnly(numCalcs);  // we want to plot mu v fd/de
  std::vector<double> dfdeEmu(numCalcs);  // we want to plot mu v fd/de
  std::vector<double> dfdeEmuSq(numCalcs);  // we want to plot mu v fd/de
  std::vector<double> gaussianDOS(numCalcs);  // regular way to calculate dos
  std::vector<double> chemPots;
  std::vector<double> temperatures;

  // set these up so it's easier to use OMP below
  for (int iCalc = 0; iCalc < numCalcs; iCalc++) {

    double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
    double temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
    chemPots.push_back(chemPot);
    temperatures.push_back(temp);
  }

  size_t Nk = bandStructure.getPoints().getNumPoints();
  double volume = crystal.getVolumeUnitCell(dimensionality); 
  size_t Nkcount = 0; 

  for (int iCalc = 0; iCalc < numCalcs; iCalc++) {
 
    double mu = chemPots[iCalc];	  
    double temp = temperatures[iCalc];	  

    for (int is : bandStructure.parallelIrrStateIterator()) {

      StateIndex isIdx(is);
      double energy = bandStructure.getEnergy(isIdx);
      double dfde = particle.getDnde(energy, temp, mu);
      auto rotations = bandStructure.getRotationsStar(isIdx); 

      // weight by the number of rotations which reduce to this point
      double contrib = dfde * -1.0;
      contrib *= rotations.size();
      dfdeOnly[iCalc] += contrib;
      dfdeEmu[iCalc] += dfde * (energy - mu) * rotations.size();  
      dfdeEmuSq[iCalc] += dfde * (energy - mu) * (energy - mu) * rotations.size();
      Nkcount +=  rotations.size();
    }
    dfdeOnly[iCalc] /= Nk;
    dfdeOnly[iCalc] /= volume;
  }
  mpi->allReduceSum(&Nkcount);
  mpi->allReduceSum(&dfdeOnly);
  mpi->allReduceSum(&dfdeEmu);
  mpi->allReduceSum(&dfdeEmuSq);

  // find g(Ef)
  std::vector<double> Nmu(numCalcs); 

  for (int iCalc = 0; iCalc < numCalcs; iCalc++) {
    for (int jCalc = 0; jCalc < iCalc; jCalc++) {
      Nmu[iCalc] += dfdeOnly[jCalc];
    }
    double deltaMu = (chemPots[iCalc] - chemPots[0]) / iCalc;
    Nmu[iCalc] *= volume * deltaMu; 
  }

  if(mpi->mpiHead()) { 
    // output to json
    nlohmann::json output;
    output["chemicalPotentials"] = chemPots;
    output["chemicalPotentialUnit"] = "eV";
    output["temperatures"] = temperatures; 
    output["temperatureUnit"] = "K";
    output["sigmaIntegrand"] = dfdeOnly; 
    output["kappaIntegrand"] = dfdeEmu; 
    output["sigmaSIntegrand"] = dfdeEmuSq;
    output["Nmu"] = Nmu; 
    output["particleType"] = "electron";
    std::ofstream o("onsager_integrands.json");
    o << std::setw(3) << output << std::endl;
    o.close();
  }
}

void OnsagerCoefficients::calcFromRelaxons(
    Eigen::VectorXd &eigenvalues, ParallelMatrix<double> &eigenvectors,
    ElScatteringMatrix &scatteringMatrix) {

  Kokkos::Profiling::pushRegion("calcFromRelaxons");

  int numEigenvalues = eigenvalues.size();
  int iCalc = 0;
  double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
  double temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
  auto particle = bandStructure.getParticle();

  VectorBTE nE(statisticsSweep, bandStructure, 3);
  VectorBTE nT(statisticsSweep, bandStructure, 3);

  if (!context.getUseSymmetries()) {

    VectorBTE fE(statisticsSweep, bandStructure, 3);
    VectorBTE fT(statisticsSweep, bandStructure, 3);

    for (auto tup0 : eigenvectors.getAllLocalStates()) {
      int is = std::get<0>(tup0);
      int alpha = std::get<1>(tup0);
      // if we only calculated some eigenvalues,
      // we should not include any alpha
      // values past that -- the memory in eigenvectors is
      // still allocated, however, it contains zeros or nonsense.
      // we also need to discard any negative states (warned about already)
      if (eigenvalues(alpha) <= 0. || alpha >= numEigenvalues) {
        continue;
      }
      auto isIndex = StateIndex(is);
      double en = bandStructure.getEnergy(isIndex);
      auto vel = bandStructure.getGroupVelocity(isIndex);
      for (int i : {0, 1, 2}) {
        fE(iCalc, i, alpha) += -particle.getDnde(en, temp, chemPot, true)
            * vel(i) * eigenvectors(is, alpha) / eigenvalues(alpha);
        fT(iCalc, i, alpha) += -particle.getDndt(en, temp, chemPot, true)
            * vel(i) * eigenvectors(is, alpha) / eigenvalues(alpha);
      }
    }
    mpi->allReduceSum(&fE.data);
    mpi->allReduceSum(&fT.data);

    for (auto tup0 : eigenvectors.getAllLocalStates()) {
      int is = std::get<0>(tup0);
      int alpha = std::get<1>(tup0);
      // discard negative and non-computed alpha values
      if (eigenvalues(alpha) <= 0. || alpha >= numEigenvalues) {
        continue;
      }
      for (int i : {0, 1, 2}) {
        nE(iCalc, i, is) += fE(iCalc, i, alpha) * eigenvectors(is, alpha);
        nT(iCalc, i, is) += fT(iCalc, i, alpha) * eigenvectors(is, alpha);
      }
    }
    mpi->allReduceSum(&nE.data);
    mpi->allReduceSum(&nT.data);

  } else { // with symmetries

    DeveloperError("Theoretically, relaxons with symmetries may not work.");
    Eigen::MatrixXd fE(3, eigenvectors.cols());
    Eigen::MatrixXd fT(3, eigenvectors.cols());
    fE.setZero();
    fT.setZero();
    for (auto tup0 : eigenvectors.getAllLocalStates()) {
      int iMat1 = std::get<0>(tup0);
      int alpha = std::get<1>(tup0);
      // discard negative and non-computed alpha values
      if (eigenvalues(alpha) <= 0. || alpha >= numEigenvalues) {
        continue;
      }
      auto tup1 = scatteringMatrix.getSMatrixIndex(iMat1);
      BteIndex iBteIndex = std::get<0>(tup1);
      CartIndex dimIndex = std::get<1>(tup1);
      int iDim = dimIndex.get();
      StateIndex isIndex = bandStructure.bteToState(iBteIndex);

      auto vel = bandStructure.getGroupVelocity(isIndex);
      double en = bandStructure.getEnergy(isIndex);
      double dndt = particle.getDndt(en, temp, chemPot);
      double dnde = particle.getDnde(en, temp, chemPot);

      if (eigenvalues(alpha) <= 0.) {
        continue;
      }
      fE(iDim, alpha) +=
          -sqrt(dnde) * vel(iDim) * eigenvectors(iMat1, alpha) / eigenvalues(alpha);
      fT(iDim, alpha) +=
          -sqrt(dndt) * vel(iDim) * eigenvectors(iMat1, alpha) / eigenvalues(alpha);
    }
    mpi->allReduceSum(&fT);
    mpi->allReduceSum(&fE);

    // back rotate to Bloch electron coordinates
    for (auto tup0 : eigenvectors.getAllLocalStates()) {
      int iMat1 = std::get<0>(tup0);
      int alpha = std::get<1>(tup0);
      // discard negative and non-computed alpha values
      if (eigenvalues(alpha) <= 0. || alpha >= numEigenvalues) {
        continue;
      }
      auto tup1 = scatteringMatrix.getSMatrixIndex(iMat1);
      BteIndex iBteIndex = std::get<0>(tup1);
      CartIndex dimIndex = std::get<1>(tup1);
      int iBte = iBteIndex.get();
      int iDim = dimIndex.get();
      nE(iCalc, iDim, iBte) += fE(iDim, alpha) * eigenvectors(iMat1, alpha);
      nT(iCalc, iDim, iBte) += fT(iDim, alpha) * eigenvectors(iMat1, alpha);
    }
    mpi->allReduceSum(&nE.data);
    mpi->allReduceSum(&nT.data);
  }
  Kokkos::Profiling::popRegion();
  calcFromSymmetricPopulation(nE, nT);
}

// quick print for iterative solver
void OnsagerCoefficients::print(const int &iter) {

  printHelper(iter, statisticsSweep, dimensionality, kappa, sigma);
}

// standard print
void OnsagerCoefficients::print() {

  printHelper(statisticsSweep, dimensionality,
                        kappa, sigma, mobility, seebeck);
}

void OnsagerCoefficients::outputToJSON(const std::string &outFileName) {

  outputCoeffsToJSON(outFileName, statisticsSweep, dimensionality,
                        kappa, sigma, mobility, seebeck);
}

Eigen::Tensor<double, 3> OnsagerCoefficients::getElectricalConductivity() {
  return sigma;
}

Eigen::Tensor<double, 3> OnsagerCoefficients::getThermalConductivity() {
  return kappa;
}

void OnsagerCoefficients::calcVariational(VectorBTE &afE, VectorBTE &afT,
                                          VectorBTE &fE, VectorBTE &fT,
                                          VectorBTE &bE, VectorBTE &bT,
                                          VectorBTE &scalingCG) {

  double norm = spinFactor / context.getKMesh().prod() /
      crystal.getVolumeUnitCell(dimensionality);
  (void) scalingCG;
  int numCalculations = statisticsSweep.getNumCalculations();

  sigma.setConstant(0.);
  kappa.setConstant(0.);

  Eigen::Tensor<double, 3> y1E = sigma.constant(0.);
  Eigen::Tensor<double, 3> y2E = kappa.constant(0.);
  Eigen::Tensor<double, 3> y1T = sigma.constant(0.);
  Eigen::Tensor<double, 3> y2T = kappa.constant(0.);

  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();

#pragma omp parallel
  {
    Eigen::Tensor<double, 3> x1E(numCalculations, 3, 3);
    Eigen::Tensor<double, 3> x2E(numCalculations, 3, 3);
    Eigen::Tensor<double, 3> x1T(numCalculations, 3, 3);
    Eigen::Tensor<double, 3> x2T(numCalculations, 3, 3);
    x1E.setConstant(0.);
    x2E.setConstant(0.);
    x1T.setConstant(0.);
    x2T.setConstant(0.);

#pragma omp for nowait
    for (int iis=0; iis<niss; ++iis) {
      int is = iss[iis];
      // skip the acoustic phonons
      if (std::find(fE.excludeIndices.begin(), fE.excludeIndices.end(),
                    is) != fE.excludeIndices.end()) {
        continue;
      }

      StateIndex isIndex(is);
      BteIndex iBteIndex = bandStructure.stateToBte(isIndex);
      int isBte = iBteIndex.get();
      auto rots = bandStructure.getRotationsStar(isIndex);

      for (const Eigen::Matrix3d &rot : rots) {

        for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

          auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
          double temp = calcStat.temperature;

          Eigen::Vector3d fRotE, afRotE, bRotE;
          Eigen::Vector3d fRotT, afRotT, bRotT;
          for (int i : {0, 1, 2}) {
            fRotE(i) = fE(iCalc, i, isBte);
            afRotE(i) = afE(iCalc, i, isBte);
            bRotE(i) = bE(iCalc, i, isBte);
            fRotT(i) = fT(iCalc, i, isBte);
            afRotT(i) = afT(iCalc, i, isBte);
            bRotT(i) = bT(iCalc, i, isBte);
          }
          fRotE = rot * fRotE;
          afRotE = rot * afRotE;
          bRotE = rot * bRotE;

          fRotT = rot * fRotT;
          afRotT = rot * afRotT;
          bRotT = rot * bRotT;

          for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
              x1E(iCalc, i, j) += fRotE(i) * afRotE(j) * norm * temp;
              x2E(iCalc, i, j) += fRotE(i) * bRotE(j) * norm * temp;
              x1T(iCalc, i, j) += fRotT(i) * afRotT(j) * norm * temp * temp;
              x2T(iCalc, i, j) += fRotT(i) * bRotT(j) * norm * temp * temp;
            }
          }
        }
      }
    }

#pragma omp critical
    for (int j = 0; j < dimensionality; j++) {
      for (int i = 0; i < dimensionality; i++) {
        for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
          y1E(iCalc, i, j) += x1E(iCalc, i, j);
          y2E(iCalc, i, j) += x2E(iCalc, i, j);
          y1T(iCalc, i, j) += x1T(iCalc, i, j);
          y2T(iCalc, i, j) += x2T(iCalc, i, j);
        }
      }
    }
  }
  mpi->allReduceSum(&y1E);
  mpi->allReduceSum(&y2E);
  mpi->allReduceSum(&y1T);
  mpi->allReduceSum(&y2T);
  sigma = 2. * y2E - y1E;
  kappa = 2. * y2T - y1T;

  if(context.getSymmetrizeBandStructure()) {
    // we print the unsymmetrized tensor to output file
    if(mpi->mpiHead()) {
      std::cout << "Unsymmetrized electronic transport properties:\n" << std::endl;
      print();
    }
    // symmetrize the conductivity 
    symmetrize(sigma);
    symmetrize(kappa);
    mpi->barrier();
  }
}

// TODO this should be a function of observable rather than of onsager, 
// however, somehow Onsager does not inherit from observable... 
void OnsagerCoefficients::symmetrize(Eigen::Tensor<double, 3>& allTransportCoeffs) {

  // get symmetry rotations of the crystal in cartesian coords
  // in case there's no symmetries, we need to trick Phoebe into
  // generating a crystal which uses them.
  bool useSyms = context.getUseSymmetries();
  context.setUseSymmetries(true);
  Crystal symCrystal(crystal);
  symCrystal.generateSymmetryInformation(context);
  auto symOps = symCrystal.getSymmetryOperations();
  context.setUseSymmetries(useSyms);

  auto invLVs = crystal.getDirectUnitCell().inverse();
  auto LVs = crystal.getDirectUnitCell();

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    Eigen::Matrix3d transportCoeffs;

    // copy the 3x3 matrix of a single calculation
    for (int j : {0, 1, 2}) {
      for (int i : {0, 1, 2}) {
        transportCoeffs(i,j) = allTransportCoeffs(iCalc,i,j);
      }
    }
    // to hold the symmetrized coeffs
    Eigen::Matrix3d symCoeffs;
    symCoeffs.setZero();

    for(SymmetryOperation symOp: symOps) {
      Eigen::Matrix3d rotation = symOp.rotation;
      rotation = LVs * rotation * invLVs; //convert to Cartesian
      Eigen::Matrix3d rotationTranspose = rotation.transpose();
      symCoeffs += rotationTranspose * transportCoeffs * rotation;
    }
    transportCoeffs = symCoeffs * (1. / symOps.size());

    // place them back into the full tensor
    for (int j : {0, 1, 2}) {
      for (int i : {0, 1, 2}) {
        allTransportCoeffs(iCalc,i,j) = transportCoeffs(i,j);
      }
    }
  }
}

