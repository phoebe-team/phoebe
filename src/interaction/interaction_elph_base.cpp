#include "interaction_elph_base.h"
#include "constants.h"
#include "impl/Kokkos_Profiling.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

// constructors =============================================================

InteractionElPhBase::InteractionElPhBase(Crystal &crystal_, PhononH0 *phononH0_)
    : crystal(crystal_), phononH0(phononH0_) {
  // determine if polar correction should be used
  usePolarCorrection = false;
  if (phononH0 != nullptr) {
    if (phononH0->getDielectricMatrix().squaredNorm() > 1.0e-10) {
      if (crystal.getNumSpecies() > 1) {
        usePolarCorrection = true;
      }
    }
  }
}

InteractionElPhBase::InteractionElPhBase(
    Crystal &crystal_, PhononH0 *phononH0_, int numPhBands_, int numElBands_,
    int numElBravaisVectors_, int numPhBravaisVectors_,
    const DoubleView2D &phBravaisVectors_k_,
    const DoubleView1D &phBravaisVectorsDegeneracies_k_,
    const DoubleView2D &elBravaisVectors_k_,
    const DoubleView1D &elBravaisVectorsDegeneracies_k_)
    : crystal(crystal_), phononH0(phononH0_), numPhBands(numPhBands_),
      numElBands(numElBands_), numElBravaisVectors(numElBravaisVectors_),
      numPhBravaisVectors(numPhBravaisVectors_),
      phBravaisVectors_k(phBravaisVectors_k_),
      phBravaisVectorsDegeneracies_k(phBravaisVectorsDegeneracies_k_),
      elBravaisVectors_k(elBravaisVectors_k_),
      elBravaisVectorsDegeneracies_k(elBravaisVectorsDegeneracies_k_) {
  // determine if polar correction should be used
  usePolarCorrection = false;
  if (phononH0 != nullptr) {
    if (phononH0->getDielectricMatrix().squaredNorm() > 1.0e-10) {
      if (crystal.getNumSpecies() > 1) {
        usePolarCorrection = true;
      }
    }
  }
}

// Helper methods ===========================================================

void InteractionElPhBase::requirePhononH0() const {
  if (phononH0 == nullptr) {
    throw std::runtime_error(
        "PhononH0 required for this operation but was not provided.");
  }
}

// Polar correction methods =================================================

Eigen::Tensor<std::complex<double>, 3>
InteractionElPhBase::getPolarCorrection(const Eigen::Vector3d &q3,
                                        const Eigen::MatrixXcd &ev1,
                                        const Eigen::MatrixXcd &ev2,
                                        const Eigen::MatrixXcd &ev3) {
  Eigen::VectorXcd x = polarCorrectionPart1(q3, ev3);
  return polarCorrectionPart2(ev1, ev2, x);
}

Eigen::VectorXcd
InteractionElPhBase::polarCorrectionPart1(const Eigen::Vector3d &q3,
                                          const Eigen::MatrixXcd &ev3) {
  requirePhononH0(); // ensures phononH0 is not null
  double volume = crystal.getVolumeUnitCell();
  Eigen::Matrix3d reciprocalUnitCell = crystal.getReciprocalUnitCell();
  Eigen::Matrix3d epsilon = phononH0->getDielectricMatrix();
  Eigen::Tensor<double, 3> bornCharges = phononH0->getBornCharges();
  Eigen::MatrixXd atomicPositions = crystal.getAtomicPositions();
  Eigen::Vector3i qCoarseMesh = phononH0->getCoarseGrid();

  return polarCorrectionPart1Static(q3, ev3, volume, reciprocalUnitCell,
                                    epsilon, bornCharges, atomicPositions,
                                    qCoarseMesh);
}

Eigen::Tensor<std::complex<double>, 3>
InteractionElPhBase::polarCorrectionPart2(const Eigen::MatrixXcd &ev1,
                                          const Eigen::MatrixXcd &ev2,
                                          const Eigen::VectorXcd &x) {
  return polarCorrectionPart2Static(ev1, ev2, x);
}

// Static Polar correction methods ==========================================

Eigen::Tensor<std::complex<double>, 3>
InteractionElPhBase::getPolarCorrectionStatic(
    const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
    const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3,
    const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
    const Eigen::Matrix3d &epsilon,
    const Eigen::Tensor<double, 3> &bornCharges,
    const Eigen::MatrixXd &atomicPositions,
    const Eigen::Vector3i &qCoarseMesh) {
  Eigen::VectorXcd x =
      polarCorrectionPart1Static(q3, ev3, volume, reciprocalUnitCell, epsilon,
                                 bornCharges, atomicPositions, qCoarseMesh);
  return polarCorrectionPart2Static(ev1, ev2, x);
}

Eigen::VectorXcd InteractionElPhBase::polarCorrectionPart1Static(
    const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3,
    const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
    const Eigen::Matrix3d &epsilon,
    const Eigen::Tensor<double, 3> &bornCharges,
    const Eigen::MatrixXd &atomicPositions,
    const Eigen::Vector3i &qCoarseMesh) {

  auto numAtoms = atomicPositions.rows();
  double gMax = 14.;
  double chargeSquare = 2.; // e^2 / (4 * pi * eps_0) in atomic units
  std::complex<double> factor = chargeSquare * fourPi / volume * complexI;

  // build a list of (q+G) vectors
  std::vector<Eigen::Vector3d> gVectors;
  for (int m1 = -qCoarseMesh(0); m1 <= qCoarseMesh(0); m1++) {
    for (int m2 = -qCoarseMesh(1); m2 <= qCoarseMesh(1); m2++) {
      for (int m3 = -qCoarseMesh(2); m3 <= qCoarseMesh(2); m3++) {
        Eigen::Vector3d gVector;
        gVector << m1, m2, m3;
        gVector = reciprocalUnitCell * gVector;
        gVector += q3;
        gVectors.push_back(gVector);
      }
    }
  }

  auto numPhBands = ev3.rows();
  Eigen::VectorXcd x = Eigen::VectorXcd::Zero(numPhBands);
  for (const auto &gVector : gVectors) {
    double qEq = gVector.transpose() * epsilon * gVector;
    if (qEq > 0. && qEq / 4. < gMax) {
      std::complex<double> factor2 = factor * exp(-qEq / 4.) / qEq;
      for (int iAt = 0; iAt < numAtoms; iAt++) {
        double arg = -gVector.dot(atomicPositions.row(iAt));
        std::complex<double> phase = {cos(arg), sin(arg)};
        std::complex<double> factor3 = factor2 * phase;
        for (int iPol : {0, 1, 2}) {
          double gqDotZ = gVector(0) * bornCharges(iAt, 0, iPol) +
                          gVector(1) * bornCharges(iAt, 1, iPol) +
                          gVector(2) * bornCharges(iAt, 2, iPol);
          int k = PhononH0::getIndexEigenvector(iAt, iPol, numAtoms);
          for (int ib3 = 0; ib3 < numPhBands; ib3++) {
            x(ib3) += factor3 * gqDotZ * ev3(k, ib3);
          }
        }
      }
    }
  }
  return x;
}

Eigen::Tensor<std::complex<double>, 3>
InteractionElPhBase::polarCorrectionPart2Static(const Eigen::MatrixXcd &ev1,
                                                const Eigen::MatrixXcd &ev2,
                                                const Eigen::VectorXcd &x) {
  // overlap = <U^+_{b2 k+q}|U_{b1 k}>
  Eigen::MatrixXcd overlap = ev2.adjoint() * ev1; // matrix size (nb2,nb1)
  overlap.transposeInPlace();                     // matrix size (nb1,nb2)

  int numPhBands = x.rows();
  Eigen::Tensor<std::complex<double>, 3> v(overlap.rows(), overlap.cols(),
                                           numPhBands);
  v.setZero();
  for (int ib3 = 0; ib3 < numPhBands; ib3++) {
    for (int i = 0; i < overlap.rows(); i++) {
      for (int j = 0; j < overlap.cols(); j++) {
        v(i, j, ib3) += x(ib3) * overlap(i, j);
      }
    }
  }
  return v;
}