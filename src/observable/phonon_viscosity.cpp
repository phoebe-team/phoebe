#include "phonon_viscosity.h"
#include "constants.h"
#include "mpiHelper.h"
#include "viscosity_io.h"
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

  // TODO for ALEX: call a function from viscosity_io.h here, to output
  // the ballistic viscosity. 

}

void PhononViscosity::calcFromRelaxons(Eigen::VectorXd &eigenvalues,
                                       ParallelMatrix<double> &eigenvectors) {

  // to simplify, here I do everything considering there is a single
  // temperature (due to memory constraints)
  if (numCalculations > 1) {
    Error("Developer error: Viscosity for relaxons only for 1 temperature.");
  }

  int numStates = bandStructure.getNumStates();
  int numRelaxons = eigenvalues.size();
  int iCalc = 0; // zero index, because we only run one for relaxons

  // search for the indices of the special eigenvectors and print info about them
  relaxonEigenvectorsCheck(eigenvectors, numRelaxons);

  // Code by Andrea, annotation by Jenny
  // Here we are calculating Eq. 9 from the PRX Simoncelli 2020
  //    mu_ijkl = (eta_ijkl + eta_ilkj)/2
  // where
  //   eta_ijkl =
  //    sqrt(A_i A_j) \sum_(alpha > 0) w_i,alpha^j * w_k,alpha^l * tau_alpha
  //
  // For this, we will need three quantities:
  //    A_i = specific momentum in direction i
  //        = (1/kBT*vol) \sum_nu n_nu (n_nu+1) (hbar q_i)^2
  //    w^j_i,alpha = velocity tensor
  //        = 1/vol * \sum_nu phi_nu^i v_nu^j theta_nu^alpha
  //    phi^i_nu = drift eigenvectors  (appendix eq A12)
  //             = zero eigenvectors linked to momentum conservation
  //             = sqrt(n(n+1)/kbT * A_i) * hbar * q_i
  //
  // And here the definitions are:
  //    alpha = relaxons eigenlabel index
  //    ijkl = cartesian direction indices
  //    nu  = phonon mode index
  //    n   = bose factor
  //    phi = special zero eigenvectors linked to momentum conservation
  //        = drift eigenvectors -- see appendix eq A12
  //    q   = wavevector
  //    theta = relaxons eigenvector
  //    tau = relaxons eigenvalues/relaxation times

  // NOTE: phi, theta0, A, and specific heat are calculated earlier
  // and stored ready to use here

  // calculate the first part of w^j_i,alpha
  Eigen::Tensor<double, 3> tmpDriftEigvecs(dimensionality, dimensionality, numStates);
  tmpDriftEigvecs.setZero();
  for (int is : bandStructure.parallelStateIterator()) {
    auto isIdx = StateIndex(is);
    auto v = bandStructure.getGroupVelocity(isIdx);
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        tmpDriftEigvecs(i, j, is) = phi(j, is) * v(i);
      }
    }
  }
  mpi->allReduceSum(&tmpDriftEigvecs);

  // now we're calculating w
  Eigen::Tensor<double, 3> w(dimensionality, dimensionality, numStates);
  w.setZero();
  for (int i = 0; i < dimensionality; i++) {
    for (int j = 0; j < dimensionality; j++) {
      // drift eigenvectors * v -- only have phonon state indices
      // and cartesian directions
      Eigen::VectorXd x(numStates);
      for (int is = 0; is < numStates; is++) {
        x(is) = tmpDriftEigvecs(i, j, is);
      }

      // w^j_i,alpha = sum_is1 phi*v*theta
      std::vector<double> x2(numRelaxons, 0.);
      for (auto tup : eigenvectors.getAllLocalStates()) {
        auto is1 = std::get<0>(tup);
        auto alpha = std::get<1>(tup);
        if(alpha >= numRelaxons) continue; // wasn't calculated
        x2[alpha] += x(is1) * eigenvectors(is1, alpha);
      }
      mpi->allReduceSum(&x2);

      // this normalization is needed to make the overall normalization work
      // given that scalapack normalizes the eigenvectors to theta*theta = 1
      for (int ialpha = 0; ialpha < numRelaxons; ialpha++) {
        w(i, j, ialpha) = x2[ialpha];// / ( sqrt(volume) * sqrt(context.getQMesh().prod()) );
      }
      // Andrea's note: in Eq. 9 of PRX, w is normalized by V*N_q
      // here however I normalize the eigenvectors differently:
      // \sum_state theta_s^2 = 1, instead of 1/VN_q \sum_state theta_s^2 = 1
    }
  }

  // Eq. 9, Simoncelli PRX (2020)
  tensordxdxdxd.setZero();
  // eigenvectors and values may only be calculated up to numRelaxons, < numStates
  std::vector<size_t> iss = mpi->divideWorkIter(numRelaxons);
  int niss = iss.size();

  Kokkos::View<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tensordxdxdxd_k(tensordxdxdxd.data(), numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  Kokkos::Experimental::ScatterView<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace> scatter_tensordxdxdxd(tensordxdxdxd_k);

  Kokkos::parallel_for("phonon_viscosity", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, niss), [&] (int iis){

      auto tmpTensor = scatter_tensordxdxdxd.access();
      int ialpha = iss[iis];
      if (eigenvalues(ialpha) <= 0.) { // avoid division by zero
        return; // return is here continue for kokkos
      }
      // discard the bose eigenvector contribution,
      // which has a divergent lifetime and should not be counted
      if(ialpha == alpha0) { return; }

      for (int i = 0; i < dimensionality; i++) {
        for (int j = 0; j < dimensionality; j++) {
          for (int k = 0; k < dimensionality; k++) {
            for (int l = 0; l < dimensionality; l++) {
            tmpTensor(iCalc, i, j, k, l) += 0.5 *
                (w(i, j, ialpha) * w(k, l, ialpha) + w(i, l, ialpha) * w(k, j, ialpha)) *
                    sqrt(A(i) * A(k)) / eigenvalues(ialpha);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(tensordxdxdxd_k, scatter_tensordxdxdxd);
  mpi->allReduceSum(&tensordxdxdxd);
}

// calculate special eigenvectors
void PhononViscosity::calcSpecialEigenvectors() {

  genericCalcSpecialEigenvectors(context, bandStructure, statisticsSweep,
                          spinFactor, theta0, theta_e, phi, C, A);

}

void PhononViscosity::relaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors,
                                                        int& numRelaxons) {

  // sets alpha0 and alpha_e, the indices
  // of the special eigenvectors in the eigenvector list,
  // to be excluded in later calculations
  Particle particle = bandStructure.getParticle();
  genericRelaxonEigenvectorsCheck(eigenvectors, numRelaxons, particle,
                                 theta0, theta_e, alpha0, alpha_e);
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

void PhononViscosity::outputRealSpaceToJSON(ScatteringMatrix& scatteringMatrix) {

  // we need a dummy variable for theta_e, as it doesn't matter for phonons
  Eigen::VectorXd theta_e(bandStructure.getNumStates());

  // call the function in viscosity io
  genericOutputRealSpaceToJSON(scatteringMatrix, bandStructure, statisticsSweep,
                                theta0, theta_e, phi, C, A, context);

}


