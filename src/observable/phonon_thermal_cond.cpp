#include "phonon_thermal_cond.h"

#include "constants.h"
#include "mpiHelper.h"
#include <ctime>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

PhononThermalConductivity::PhononThermalConductivity(
    Context &context_, StatisticsSweep &statisticsSweep_, Crystal &crystal_,
    BaseBandStructure &bandStructure_)
    : Observable(context_, statisticsSweep_, crystal_),
      bandStructure(bandStructure_) {
  tensordxd =
      Eigen::Tensor<double, 3>(numCalculations, dimensionality, dimensionality);
  tensordxd.setZero();
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

  double norm = 1. / context.getQMesh().prod() /
                crystal.getVolumeUnitCell(dimensionality);

  auto excludeIndices = n.excludeIndices;

  tensordxd.setZero();

  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();

  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tensordxd_k(tensordxd.data(), numCalculations, 3, 3);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft, Kokkos::HostSpace> scatter_tensordxd(tensordxd_k);
  Kokkos::parallel_for("phonon_thermal_cond", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, niss), [&] (int iis){

      auto tensorPrivate = scatter_tensordxd.access();

      int is = iss[iis];
      StateIndex isIdx(is);
      double en = bandStructure.getEnergy(isIdx);
      Eigen::Vector3d velIrr = bandStructure.getGroupVelocity(isIdx);

      int iBte = bandStructure.stateToBte(isIdx).get();

      // skip the acoustic phonons
      if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte) !=
          excludeIndices.end())
        return;

      auto rots = bandStructure.getRotationsStar(isIdx);
      for (const Eigen::Matrix3d &rot : rots) {

        auto vel = rot * velIrr;

        for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {

          Eigen::Vector3d nRot;
          for (int i : {0, 1, 2}) {
            nRot(i) = n(iCalc, i, iBte);
          }
          nRot = rot * nRot;

          for (int j : {0, 1, 2}) {
            for (int i : {0, 1, 2}) {
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
  if(mpi->mpiHead()) { 
    std::cout << "Pre-symmetrization thermal conductivity:\n" << std::endl;
    print();
  }
  // symmetrize the thermal conductivity 
  symmetrize(tensordxd);
}

void PhononThermalConductivity::calcVariational(VectorBTE &af, VectorBTE &f, VectorBTE &b) {

  double norm = 1. / context.getQMesh().prod() /
                crystal.getVolumeUnitCell(dimensionality);
  auto excludeIndices = f.excludeIndices;

  int numCalculations = statisticsSweep.getNumCalculations();

  tensordxd.setConstant(0.);

  Eigen::Tensor<double, 3> y1 = tensordxd.constant(0.);
  Eigen::Tensor<double, 3> y2 = tensordxd.constant(0.);

  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();

  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> y1_k(y1.data(), numCalculations, 3, 3),
    y2_k(y2.data(), numCalculations, 3, 3);
  Kokkos::Experimental::ScatterView<double***, Kokkos::LayoutLeft, Kokkos::HostSpace> scatter_y1(y1_k), scatter_y2(y2_k);
  Kokkos::parallel_for("variational thermal conductivity", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, niss), [&] (int iis){
      auto x1 = scatter_y1.access();
      auto x2 = scatter_y2.access();

      int is = iss[iis];
      // skip the acoustic phonons
      if (std::find(excludeIndices.begin(), excludeIndices.end(), is) !=
          excludeIndices.end()) {
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
          for (int i : {0, 1, 2}) {
            fRot(i) = f(iCalc, i, isBte);
            afRot(i) = af(iCalc, i, isBte);
            bRot(i) = b(iCalc, i, isBte);
          }
          fRot = rot * fRot;
          afRot = rot * afRot;
          bRot = rot * bRot;

          for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
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
  if(mpi->mpiHead()) {
    std::cout << "Unsymmetrized thermal conductivity:\n" << std::endl;
    print();
  }
  // symmetrize the thermal conductivity
  symmetrize(tensordxd);
}

// TODO this should be commented better to make it more understandable -- Jenny
void PhononThermalConductivity::calcFromRelaxons(
    Context &context, StatisticsSweep &statisticsSweep,
    ParallelMatrix<double> &eigenvectors, PhScatteringMatrix &scatteringMatrix,
    const Eigen::VectorXd &eigenvalues) {

  // Comments are Jenny attempting to trace a function written by Andrea
  // See Cepellotti 2016 PRX about relaxons
  // This function calculates the relaxon population delta n
  // defined in that paper, and then supplies it to the thermal conductivity
  // calculator.
  //
  // Here,
  // alpha = relaxons eigenvalue index
  // iMat1 = phonon state index

  int numEigenvalues = eigenvalues.size();
  auto particle = bandStructure.getParticle();

  int iCalc = 0; // relaxons only allows one calc in memory
  double temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
  double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
  VectorBTE population(statisticsSweep, bandStructure, 3);

  // if we only calculated some eigenvalues,
  // we should not include any alpha
  // values past that -- scalapack required the memory in eigenvectors to be allocated,
  // however, for alpha>numEigenvalues, it contains zeros or nonsense.
  // we also need to discard any negative states
  std::vector<std::tuple<int, int>> allLocalStates = eigenvectors.getAllLocalStates();
  std::vector<std::tuple<int, int>> localStates;
  for(auto state : allLocalStates) {
    int alpha = std::get<1>(state);
    if (eigenvalues(alpha) <= 0. || alpha >= numEigenvalues) {
      continue;
    }
    localStates.push_back(state);
  }
  int nlocalStates = localStates.size();

  // now, use these states to calcuate the populations
  // first, we calculate the relaxon populations f_alpha
  if (context.getUseSymmetries()) {

    // relaxons population f_alpha
    Eigen::VectorXd relaxonsPopulation(numEigenvalues);
    relaxonsPopulation.setZero();

#pragma omp parallel default(none)                                             \
    shared(relaxonsPopulation, temp, chemPot, scatteringMatrix, eigenvectors,       \
           eigenvalues, particle, localStates, nlocalStates)
    {
      // each omp process needs its own copy
      Eigen::VectorXd relaxonsPopPrivate(eigenvalues.size());
      relaxonsPopPrivate.setZero();

#pragma omp for nowait
      for(int ilocalState = 0; ilocalState < nlocalStates; ilocalState++){

        // calculate related state indices (need to use SMatrix for this when sym present)
        // scalapack required eigenvectors to be the same size as
        // the SMatrix, so this indexing works for eigenvectors, too.
        std::tuple<int,int> tup0 = localStates[ilocalState];
        int iMat1 = std::get<0>(tup0);
        int alpha = std::get<1>(tup0);
        auto tup1 = scatteringMatrix.getSMatrixIndex(iMat1);
        BteIndex iBteIdx = std::get<0>(tup1);
        CartIndex dimIdx = std::get<1>(tup1);
        StateIndex isIdx = bandStructure.bteToState(iBteIdx);
        int iDim = dimIdx.get();

        auto vel = bandStructure.getGroupVelocity(isIdx);
        double en = bandStructure.getEnergy(isIdx);
        double term = sqrt(particle.getPopPopPm1(en, temp, chemPot));
        double dndt = particle.getDndt(en, temp, chemPot);
        // Not sure which equation this comes from in 2016 PRX,
        // maybe should check 2020
        if (eigenvalues(alpha) > 0. && en > 0.) {
          //  sum(alpha, i) dn/dT * (1/sqrt(n(n+1)))
          //                        * v_i * theta_mu,alpha * tau_alpha
          relaxonsPopPrivate(alpha) += dndt / term * vel(iDim) *
                                  eigenvectors(iMat1, alpha) /
                                  eigenvalues(alpha);
        }
      }
#pragma omp critical
      for (int alpha = 0; alpha < eigenvalues.size(); alpha++) {
        relaxonsPopulation(alpha) += relaxonsPopPrivate(alpha);
      }
    }
    mpi->allReduceSum(&relaxonsPopulation);

    // back rotate to phonon coordinates

#pragma omp parallel default(none)                                             \
    shared(bandStructure, eigenvectors, relaxonsPopulation, population,             \
           scatteringMatrix, iCalc, localStates, nlocalStates)
    {
      Eigen::MatrixXd popPrivate(3, bandStructure.irrStateIterator().size());
      popPrivate.setZero();

#pragma omp for nowait
      for(int ilocalState = 0; ilocalState < nlocalStates; ilocalState++){
        std::tuple<int,int> tup0 = localStates[ilocalState];
        int iMat1 = std::get<0>(tup0);
        int alpha = std::get<1>(tup0);
        auto tup1 = scatteringMatrix.getSMatrixIndex(iMat1);
        BteIndex iBteIndex = std::get<0>(tup1);
        CartIndex dimIndex = std::get<1>(tup1);
        int iBte = iBteIndex.get();
        int iDim = dimIndex.get();
        popPrivate(iDim, iBte) +=
            eigenvectors(iMat1, alpha) * relaxonsPopulation(alpha);
      }

#pragma omp critical
      for (int is = 0; is < popPrivate.cols(); is++) {
        for (int iDim = 0; iDim < 3; iDim++) {
          population(iCalc, iDim, is) += popPrivate(iDim, is);
        }
      }
    }
    mpi->allReduceSum(&population.data);

  } else { // case without symmetries ------------------------------------------

    Eigen::MatrixXd relaxonsPopulation(eigenvalues.size(), 3);
    relaxonsPopulation.setZero();
#pragma omp parallel default(none)                                             \
    shared(eigenvectors, eigenvalues, temp, chemPot, particle, relaxonsPopulation, localStates, nlocalStates)
    {
      Eigen::MatrixXd relaxonsPopPrivate(eigenvalues.size(), 3);
      relaxonsPopPrivate.setZero();
#pragma omp for nowait
      for(int ilocalState = 0; ilocalState < nlocalStates; ilocalState++){
        std::tuple<int,int> tup0 = localStates[ilocalState];
        int is = std::get<0>(tup0);
        StateIndex isIdx(is);
        int alpha = std::get<1>(tup0);
        double en = bandStructure.getEnergy(isIdx);
        if (eigenvalues(alpha) > 0. && en > 0.) {
          auto vel = bandStructure.getGroupVelocity(isIdx);
          double term = sqrt(particle.getPopPopPm1(en, temp, chemPot));
          double dndt = particle.getDndt(en, temp, chemPot);
          for (int i = 0; i < 3; i++) {
            relaxonsPopPrivate(alpha, i) += dndt / term * vel(i) *
                                       eigenvectors(is, alpha) /
                                       eigenvalues(alpha);
          }
        }
      }
#pragma omp critical
      for (int alpha = 0; alpha < eigenvalues.size(); alpha++) {
        for (int i = 0; i < 3; i++) {
          relaxonsPopulation(alpha, i) += relaxonsPopPrivate(alpha, i);
        }
      }
    }
    mpi->allReduceSum(&relaxonsPopulation);

    // back rotate to phonon coordinates
#pragma omp parallel default(none)                                             \
    shared(eigenvectors, relaxonsPopulation, population, iCalc, localStates, nlocalStates)
    {
      Eigen::MatrixXd popPrivate(3, bandStructure.getNumStates());
      popPrivate.setZero();

#pragma omp for nowait
      for(int ilocalState = 0; ilocalState < nlocalStates; ilocalState++){
        std::tuple<int,int> tup0 = localStates[ilocalState];
        int is = std::get<0>(tup0);
        int alpha = std::get<1>(tup0);
        for (int i = 0; i < 3; i++) {
          popPrivate(i, is) += eigenvectors(is, alpha) * relaxonsPopulation(alpha, i);
        }
      }
#pragma omp critical
      for (int is = 0; is < bandStructure.getNumStates(); is++) {
        for (int i = 0; i < 3; i++) {
          population(iCalc, i, is) += popPrivate(i, is);
        }
      }
    }
    mpi->allReduceSum(&population.data);
  }

  // put back the rescaling factor
  std::vector<int> iss = bandStructure.irrStateIterator();
  int niss = iss.size();
#pragma omp parallel for default(none)                                         \
    shared(population, particle, temp, chemPot, iCalc, bandStructure, iss, niss)
  for(int iis = 0; iis < niss; iis++){
    int is = iss[iis];
    auto isIdx = StateIndex(is);
    double en = bandStructure.getEnergy(isIdx);
    int iBte = bandStructure.stateToBte(isIdx).get();
    if (en > 0.) {
      double term = particle.getPopPopPm1(en, temp, chemPot);
      for (int iDim = 0; iDim < 3; iDim++) {
        population(iCalc, iDim, iBte) *= sqrt(term);
      }
    }
  }
  // now calculate the thermal conductivity using the standard phonon population
  calcFromPopulation(population);
}

void PhononThermalConductivity::print() {
  if (!mpi->mpiHead())
    return;

  std::string units;
  if (dimensionality == 1) {
    units = "W m / K";
  } else if (dimensionality == 2) {
    units = "W / K";
  } else {
    units = "W / m / K";
  }

  std::cout << "\n";
  std::cout << "Thermal Conductivity (" << units << ")\n";

  for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;

    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "Temperature: " << temp * temperatureAuToSi << " (K)\n";
    std::cout.precision(5);
    for (int i = 0; i < dimensionality; i++) {
      std::cout << "  " << std::scientific;
      for (int j = 0; j < dimensionality; j++) {
        std::cout << " " << std::setw(13) << std::right;
        std::cout << tensordxd(iCalc, i, j) * thConductivityAuToSi;
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
}

void PhononThermalConductivity::outputToJSON(const std::string &outFileName) {
  if (!mpi->mpiHead())
    return;

  std::string units;
  if (dimensionality == 1) {
    units = "W m / K";
  } else if (dimensionality == 2) {
    units = "W / K";
  } else {
    units = "W /(m K)";
  }

  std::vector<double> temps;
  std::vector<std::vector<std::vector<double>>> conductivities;
  for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp * temperatureAuToSi);

    // store conductivity
    std::vector<std::vector<double>> rows;
    for (int i = 0; i < dimensionality; i++) {
      std::vector<double> cols;
      for (int j = 0; j < dimensionality; j++) {
        cols.push_back(tensordxd(iCalc, i, j) * thConductivityAuToSi);
      }
      rows.push_back(cols);
    }
    conductivities.push_back(rows);
  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["thermalConductivity"] = conductivities;
  output["temperatureUnit"] = "K";
  output["thermalConductivityUnit"] = units;
  output["particleType"] = "phonon";
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

void PhononThermalConductivity::print(const int &iter) {
  if (!mpi->mpiHead())
    return;

  // get the time
  time_t currentTime;
  currentTime = time(nullptr);
  // and format the time nicely
  char s[200];
  struct tm *p = localtime(&currentTime);
  strftime(s, 200, "%F, %T", p);

  std::cout << "Iteration: " << iter << " | " << s << "\n";
  for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "T = " << temp * temperatureAuToSi << ", k = ";
    std::cout.precision(5);
    for (int i = 0; i < dimensionality; i++) {
      std::cout << std::scientific;
      std::cout << tensordxd(iCalc, i, i) * thConductivityAuToSi << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

int PhononThermalConductivity::whichType() { return is2Tensor; }

Eigen::Tensor<double,3> PhononThermalConductivity::getThermalConductivity() {
  return tensordxd;
}
