#include "phonon_h0.h"

#include <cmath>
#include <complex>
#include <iostream>

#include "constants.h"
#include "eigen.h"
#include "exceptions.h"
#include "points.h"
#include "utilities.h"

PhononH0::PhononH0(Crystal &crystal, const Eigen::Matrix3d &dielectricMatrix_,
                   const Eigen::Tensor<double, 3> &bornCharges_,
                   Eigen::Tensor<double, 7> &forceConstants_,
                   const std::string &sumRule)
    : particle(Particle::phonon) {
  // in this section, we save as class properties a few variables
  // that are needed for the diagonalization of phonon frequencies

  Kokkos::Profiling::pushRegion("phononH0 constructor");

  directUnitCell = crystal.getDirectUnitCell();
  Eigen::Matrix3d reciprocalUnitCell = crystal.getReciprocalUnitCell();
  volumeUnitCell = crystal.getVolumeUnitCell();
  atomicSpecies = crystal.getAtomicSpecies();
  speciesMasses = crystal.getSpeciesMasses();
  atomicPositions = crystal.getAtomicPositions();
  dielectricMatrix = dielectricMatrix_;
  bornCharges = bornCharges_;

  if (dielectricMatrix.sum() > 0.) {
    hasDielectric = true; // defaulted to false in *.h file
  }

  qCoarseGrid(0) = int(forceConstants_.dimension(2));
  qCoarseGrid(1) = int(forceConstants_.dimension(3));
  qCoarseGrid(2) = int(forceConstants_.dimension(4));

  numAtoms = crystal.getNumAtoms();
  numBands = numAtoms * 3;
  dimensionality = crystal.getDimensionality();

  // now, I initialize an auxiliary set of vectors that are needed
  // for the diagonalization, which are precomputed once and for all.

  setAcousticSumRule(sumRule, forceConstants_);

  reorderDynamicalMatrix(directUnitCell, forceConstants_);

  if (hasDielectric) { // prebuild terms useful for long range corrections
    double cutoff = gMax * 4.;

    // Estimate of nr1x,nr2x,nr3x generating all vectors up to G^2 < geg
    // Only for dimensions where periodicity is present, e.g. if nr1=1
    // and nr2=1, then the G-vectors run along nr3 only.
    // (useful if system is in vacuum, e.g. 1D or 2D)

    int nr1x = 0;
    if (qCoarseGrid(0) > 1) {
      nr1x = (int)(sqrt(cutoff) / reciprocalUnitCell.col(0).norm()) + 1;
    }
    int nr2x = 0;
    if (qCoarseGrid(1) > 1) {
      nr2x = (int)(sqrt(cutoff) / reciprocalUnitCell.col(1).norm()) + 1;
    }
    int nr3x = 0;
    if (qCoarseGrid(2) > 1) {
      nr3x = (int)(sqrt(cutoff) / reciprocalUnitCell.col(2).norm()) + 1;
    }

    double norm;
    Eigen::Matrix3d reff;
    if(dimensionality == 2 && longRange2d) {
      norm = e2 * twoPi / volumeUnitCell;  // originally
      // (e^2 * 2\pi) / Area
      // fac = (sign * e2 * tpi) / (omega * bg(3, 3) / alat)
      reff = dielectricMatrix * 0.5 * directUnitCell(2,2);
    } else {
      norm = e2 * fourPi / volumeUnitCell;
    }

    int numG = (2*nr1x+1) * (2*nr2x+1) * (2*nr3x+1);
    gVectors.resize(3,numG);
    {
      int ig = 0;
      for (int m1 = -nr1x; m1 <= nr1x; m1++) {
        for (int m2 = -nr2x; m2 <= nr2x; m2++) {
          for (int m3 = -nr3x; m3 <= nr3x; m3++) {
            gVectors(0, ig) = double(m1) * reciprocalUnitCell(0, 0) + double(m2) * reciprocalUnitCell(0, 1) + double(m3) * reciprocalUnitCell(0, 2);
            gVectors(1, ig) = double(m1) * reciprocalUnitCell(1, 0) + double(m2) * reciprocalUnitCell(1, 1) + double(m3) * reciprocalUnitCell(1, 2);
            gVectors(2, ig) = double(m1) * reciprocalUnitCell(2, 0) + double(m2) * reciprocalUnitCell(2, 1) + double(m3) * reciprocalUnitCell(2, 2);
            ++ig;
          }
        }
      }
    }

    // TODO also add 2d option and move this to a separate function
    longRangeCorrection1.resize(3,3,numAtoms);
    longRangeCorrection1.setZero();
    for (int ig=0; ig<numG; ++ig) {
      Eigen::Vector3d g = gVectors.col(ig);

      // calculate the correction term for screening, Wc as in
      // https://doi.org/10.1021/acs.nanolett.7b01090
      double geg = 0;

      if(dimensionality == 2 && longRange2d) {
        if(g(0)*g(0) + g(1)*g(1) > 1.e-8) {
          geg = (g.transpose() * reff * g).value() / (g.norm() * g.norm());
        }
      } else {
        geg = (g.transpose() * dielectricMatrix * g).value();
      }

      if (0. < geg && geg < 4. * gMax) {
        double normG;

        if(dimensionality == 2 && longRange2d) {
          normG = norm * exp(-g.norm() * 0.25) / sqrt(g.norm()) * (1.0 + geg * sqrt(g.norm()));
        } else {
          normG = norm * exp(-geg * 0.25) / geg;
        }

        Eigen::MatrixXd gZ(3, numAtoms);
        for (int na = 0; na < numAtoms; na++) {
          for (int i : {0, 1, 2}) {
            gZ(i, na) = g(0) * bornCharges(na, 0, i) + g(1) * bornCharges(na, 1, i) + g(2) * bornCharges(na, 2, i);
          }
        }

        Eigen::MatrixXd fnAt = Eigen::MatrixXd::Zero(numAtoms, 3);
        for (int na = 0; na < numAtoms; na++) {
          for (int i : {0, 1, 2}) {
            for (int nb = 0; nb < numAtoms; nb++) {
              double arg =
                  (atomicPositions.row(na) - atomicPositions.row(nb)).dot(g);
              fnAt(na, i) += gZ(i, nb) * cos(arg);
            }
          }
        }
        for (int na = 0; na < numAtoms; na++) {
          for (int j : {0, 1, 2}) {
            for (int i : {0, 1, 2}) {
              longRangeCorrection1(i, j, na) += -normG * gZ(i, na) * fnAt(na, j);
            }
          }
        }
      }
    }
  }

  // copy to GPU
  {
    int numG = gVectors.cols();
    Kokkos::resize(atomicMasses_d, numBands);
    Kokkos::resize(bravaisVectors_d, numBravaisVectors, 3);
    Kokkos::resize(weights_d, numBravaisVectors);
    Kokkos::resize(mat2R_d, numBands, numBands, numBravaisVectors);

    auto atomicMasses_h = create_mirror_view(atomicMasses_d);
    auto bravaisVectors_h = create_mirror_view(bravaisVectors_d);
    auto weights_h = create_mirror_view(weights_d);
    auto mat2R_h = create_mirror_view(mat2R_d);

    for (int iR = 0; iR < numBravaisVectors; iR++) {
      for (int i = 0; i < 3; i++) {
        for (int iAt=0; iAt<numAtoms; ++iAt) {
          for (int j = 0; j < 3; j++) {
            for (int jAt = 0; jAt < numAtoms; ++jAt) {
              mat2R_h(iAt*3+i, jAt*3+j, iR) = mat2R(i, j, iAt, jAt, iR);
            }
          }
        }
      }
    }
    for (int iAt=0; iAt<numAtoms; ++iAt) {
      int iType = atomicSpecies(iAt);
      for (int i = 0; i < 3; i++) {
        atomicMasses_h(iAt * 3 + i) = speciesMasses(iType);
      }
    }
    for (int iR=0; iR<numBravaisVectors; ++iR) {
      weights_h(iR) = weights(iR);
      for (int i = 0; i < 3; i++) {
        bravaisVectors_h(iR, i) = bravaisVectors(i, iR);
      }
    }
    Kokkos::deep_copy(atomicMasses_d, atomicMasses_h);
    Kokkos::deep_copy(bravaisVectors_d, bravaisVectors_h);
    Kokkos::deep_copy(weights_d, weights_h);
    Kokkos::deep_copy(mat2R_d, mat2R_h);
    if (hasDielectric) {
      Kokkos::resize(longRangeCorrection1_d, 3, 3, numAtoms);
      Kokkos::resize(gVectors_d, numG, 3);
      Kokkos::resize(dielectricMatrix_d, 3, 3);
      Kokkos::resize(bornCharges_d, 3, 3, numAtoms);
      Kokkos::resize(atomicPositions_d, numAtoms, 3);
      auto longRangeCorrection1_h = create_mirror_view(longRangeCorrection1_d);
      auto gVectors_h = create_mirror_view(gVectors_d);
      auto dielectricMatrix_h = create_mirror_view(dielectricMatrix_d);
      auto bornCharges_h = create_mirror_view(bornCharges_d);
      auto atomicPositions_h = create_mirror_view(atomicPositions_d);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          dielectricMatrix_h(i, j) = dielectricMatrix(i, j);
        }
      }
      for (int iAt=0; iAt<numAtoms; ++iAt) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            bornCharges_h(j, i, iAt) = bornCharges(iAt, i, j);
            longRangeCorrection1_h(i, j, iAt) = longRangeCorrection1(i, j, iAt);
          }
        }
      }
      for (int iG=0; iG<numG; ++iG) {
        for (int i = 0; i < 3; i++) {
          gVectors_h(iG,i) = gVectors(i,iG);
        }
      }
      for (int iAt=0; iAt<numAtoms; ++iAt) {
        for (int i = 0; i < 3; i++) {
          atomicPositions_h(iAt, i) = atomicPositions(iAt, i);
        }
      }
      Kokkos::deep_copy(longRangeCorrection1_d, longRangeCorrection1_h);
      Kokkos::deep_copy(gVectors_d, gVectors_h);
      Kokkos::deep_copy(dielectricMatrix_d, dielectricMatrix_h);
      Kokkos::deep_copy(bornCharges_d, bornCharges_h);
      Kokkos::deep_copy(atomicPositions_d, atomicPositions_h);
    }
    double mem = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(mem);
  }
  Kokkos::Profiling::popRegion();
}

// copy constructor
PhononH0::PhononH0(const PhononH0 &that)
    : particle(that.particle), hasDielectric(that.hasDielectric),
      numAtoms(that.numAtoms), numBands(that.numBands),
      volumeUnitCell(that.volumeUnitCell), atomicSpecies(that.atomicSpecies),
      speciesMasses(that.speciesMasses), atomicPositions(that.atomicPositions),
      dielectricMatrix(that.dielectricMatrix), bornCharges(that.bornCharges),
      qCoarseGrid(that.qCoarseGrid),
      numBravaisVectors(that.numBravaisVectors),
      bravaisVectors(that.bravaisVectors), weights(that.weights),
      mat2R(that.mat2R), gVectors(that.gVectors),
      longRangeCorrection1(that.longRangeCorrection1),
      atomicMasses_d(that.atomicMasses_d),
      longRangeCorrection1_d(that.longRangeCorrection1_d),
      gVectors_d(that.gVectors_d),
      dielectricMatrix_d(that.dielectricMatrix_d),
      bornCharges_d(that.bornCharges_d),
      atomicPositions_d(that.atomicPositions_d),
      bravaisVectors_d(that.bravaisVectors_d),
      weights_d(that.weights_d),
      mat2R_d(that.mat2R_d) {
  double memory = getDeviceMemoryUsage();
  kokkosDeviceMemory->addDeviceMemoryUsage(memory);
}

// copy assignment
PhononH0 &PhononH0::operator=(const PhononH0 &that) {
  if (this != &that) {
    particle = that.particle;
    hasDielectric = that.hasDielectric;
    numAtoms = that.numAtoms;
    numBands = that.numBands;
    volumeUnitCell = that.volumeUnitCell;
    atomicSpecies = that.atomicSpecies;
    speciesMasses = that.speciesMasses;
    atomicPositions = that.atomicPositions;
    dielectricMatrix = that.dielectricMatrix;
    bornCharges = that.bornCharges;
    qCoarseGrid = that.qCoarseGrid;
    numBravaisVectors = that.numBravaisVectors;
    bravaisVectors = that.bravaisVectors;
    weights = that.weights;
    mat2R = that.mat2R;
    gVectors = that.gVectors;
    longRangeCorrection1 = that.longRangeCorrection1;

    atomicMasses_d = that.atomicMasses_d;
    longRangeCorrection1_d = that.longRangeCorrection1_d;
    gVectors_d = that.gVectors_d;
    dielectricMatrix_d = that.dielectricMatrix_d;
    bornCharges_d = that.bornCharges_d;
    atomicPositions_d = that.atomicPositions_d;
    bravaisVectors_d = that.bravaisVectors_d;
    weights_d = that.weights_d;
    mat2R_d = that.mat2R_d;
    double memory = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(memory);
  }
  return *this;
}

PhononH0::~PhononH0() {
  double mem = getDeviceMemoryUsage();
  kokkosDeviceMemory->removeDeviceMemoryUsage(mem);
  Kokkos::realloc(atomicMasses_d, 0);
  Kokkos::realloc(longRangeCorrection1_d, 0, 0, 0);
  Kokkos::realloc(gVectors_d, 0, 0);
  Kokkos::realloc(dielectricMatrix_d, 0, 0);
  Kokkos::realloc(bornCharges_d, 0, 0, 0);
  Kokkos::realloc(atomicPositions_d, 0, 0);
  Kokkos::realloc(bravaisVectors_d, 0, 0);
  Kokkos::realloc(weights_d, 0);
  Kokkos::realloc(mat2R_d, 0, 0, 0);
}

int PhononH0::getNumBands() { return numBands; }

Particle PhononH0::getParticle() { return particle; }

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
PhononH0::diagonalize(Point &point) {
  Eigen::Vector3d q = point.getCoordinates(Points::cartesianCoordinates);
  bool withMassScaling = true;
  auto tup = diagonalizeFromCoordinates(q, withMassScaling);
  auto energies = std::get<0>(tup);
  auto eigenvectors = std::get<1>(tup);
  return std::make_tuple(energies, eigenvectors);
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
PhononH0::diagonalizeFromCoordinates(Eigen::Vector3d &q) {
  bool withMassScaling = true;
  return diagonalizeFromCoordinates(q, withMassScaling);
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
PhononH0::diagonalizeFromCoordinates(Eigen::Vector3d &q,
                                     const bool &withMassScaling) {
  // to be executed at every q-point to get phonon frequencies and wavevectors

  Eigen::Tensor<std::complex<double>, 4> dyn(3, 3, numAtoms, numAtoms);
  dyn.setZero();

  // first, the short range term, which is just a Fourier transform
  shortRangeTerm(dyn, q);

  // then the long range term, which uses some convergence
  // tricks by X. Gonze et al.
  if (hasDielectric) {
    addLongRangeTerm(dyn, q);
  }

  // once everything is ready, here we scale by masses and diagonalize

  auto tup = dynDiagonalize(dyn);
  auto energies = std::get<0>(tup);
  auto eigenvectors = std::get<1>(tup);

  if (withMassScaling) {
    // we normalize with the mass.
    // In this way, the Eigenvector matrix U, doesn't satisfy (U^+) * U = I
    // but instead (U^+) * M * U = I, where M is the mass matrix
    // (M is diagonal with atomic masses on the diagonal)
    // (this should give us the eigendisplacements)
    for (int iBand = 0; iBand < numBands; iBand++) {
      for (int iAt = 0; iAt < numAtoms; iAt++) {
        int iType = atomicSpecies(iAt);
        for (int iPol : {0, 1, 2}) {
          int ind = getIndexEigenvector(iAt, iPol);
          eigenvectors(ind, iBand) /= sqrt(speciesMasses(iType));
        }
      }
    }
  }

  return std::make_tuple(energies, eigenvectors);
}

FullBandStructure PhononH0::populate(Points &points,
                                     const bool &withVelocities,
                                     const bool &withEigenvectors,
                                     const bool isDistributed) {
  return kokkosPopulate(points, withVelocities, withEigenvectors, isDistributed);
}

FullBandStructure PhononH0::cpuPopulate(Points &points, bool &withVelocities,
                                        bool &withEigenvectors,
                                        bool isDistributed) {
  FullBandStructure fullBandStructure(numBands, particle, withVelocities,
                                      withEigenvectors, points, isDistributed);

  std::vector<int> ikIndices = fullBandStructure.getWavevectorIndices();
  int numIkIndices = ikIndices.size();
#pragma omp parallel for
  for (int iik = 0; iik<numIkIndices; ++iik) {
    int ik = ikIndices[iik];

    Point point = fullBandStructure.getPoint(ik);

    auto tup = diagonalize(point);
    auto ens = std::get<0>(tup);
    auto eigenVectors = std::get<1>(tup);
    fullBandStructure.setEnergies(point, ens);

    if (withVelocities) {
      auto velocities = diagonalizeVelocity(point);
      fullBandStructure.setVelocities(point, velocities);
    }
    if (withEigenvectors) {
      fullBandStructure.setEigenvectors(point, eigenVectors);
    }
  }
  return fullBandStructure;
}

Eigen::Tensor<double, 5> PhononH0::wsInit(const Eigen::MatrixXd &unitCell,
                                          const Eigen::Matrix3d &directUnitCell,
                                          const int& nr1Big,
                                          const int& nr2Big,
                                          const int& nr3Big) {
  const int numMax = 2;
  int index = 0;
  const int numMaxRWS = 200;

  Eigen::MatrixXd tmpResult(3, numMaxRWS);

  for (int ir = -numMax; ir <= numMax; ir++) {
    for (int jr = -numMax; jr <= numMax; jr++) {
      for (int kr = -numMax; kr <= numMax; kr++) {
        for (int i : {0, 1, 2}) {
          tmpResult(i, index) =
              unitCell(i, 0) * ir + unitCell(i, 1) * jr + unitCell(i, 2) * kr;
        }

        if (tmpResult.col(index).squaredNorm() > 1.0e-6) {
          index += 1;
        }
        if (index > numMaxRWS) {
          Error("WSInit > numMaxRWS");
        }
      }
    }
  }
  int numRWS = index;
  Eigen::MatrixXd rws(3, numRWS);
  for (int i = 0; i < numRWS; i++) {
    rws.col(i) = tmpResult.col(i);
  }

  // now, I also prepare the wsCache, which is used to accelerate
  // the shortRange() calculation

  Eigen::Tensor<double, 5> wsCache(2 * nr3Big + 1, 2 * nr2Big + 1,
                                   2 * nr1Big + 1, numAtoms, numAtoms);
  wsCache.setZero();

  for (int na = 0; na < numAtoms; na++) {
    for (int nb = 0; nb < numAtoms; nb++) {
      double total_weight = 0.;

      // sum over r vectors in the super cell - very safe range!

      for (int n1 = -nr1Big; n1 <= nr1Big; n1++) {
        int n1ForCache = n1 + nr1Big;
        for (int n2 = -nr2Big; n2 <= nr2Big; n2++) {
          int n2ForCache = n2 + nr2Big;
          for (int n3 = -nr3Big; n3 <= nr3Big; n3++) {
            int n3ForCache = n3 + nr3Big;

            Eigen::Vector3d r_ws;
            for (int i : {0, 1, 2}) {
              // note that this cell is different from above
              r_ws(i) = double(n1) * directUnitCell(i, 0) +
                        double(n2) * directUnitCell(i, 1) +
                        double(n3) * directUnitCell(i, 2);
              r_ws(i) += atomicPositions(na, i) - atomicPositions(nb, i);
            }

            double x = wsWeight(r_ws, rws);
            wsCache(n3ForCache, n2ForCache, n1ForCache, nb, na) = x;
            total_weight += x;
          }
        }
      }

      if (abs(total_weight - qCoarseGrid(0) * qCoarseGrid(1) * qCoarseGrid(2)) > 1.0e-8) {
        Error("DeveloperError: wrong total_weight, weight: "
                + std::to_string(total_weight) + " qMeshProd: " + std::to_string(qCoarseGrid.prod()) );
      }
    }
  }
  return wsCache;
}

double PhononH0::wsWeight(const Eigen::VectorXd &r, const Eigen::MatrixXd &rws) {
  // wsWeight() assigns this weight:
  // - if a point is inside the Wigner-Seitz cell:    weight=1
  // - if a point is outside the WS cell:             weight=0
  // - if a point q is on the border of the WS cell, it finds the number N
  //   of translationally equivalent point q+G  (where G is a lattice vector)
  //   that are also on the border of the cell. Then: weight = 1/N
  //
  // I.e. if a point is on the surface of the WS cell of a cubic lattice
  // it will have weight 1/2; on the vertex of the WS it would be 1/8;
  // the K point of an hexagonal lattice has weight 1/3 and so on.

  // rws: contains the list of nearest neighbor atoms
  // r: the position of the reference point
  // rws.cols(): number of nearest neighbors

  int numREq = 1;

  for (int ir = 0; ir < rws.cols(); ir++) {
    double rrt = r.dot(rws.col(ir));
    double ck = rrt - rws.col(ir).squaredNorm() / 2.;
    if (ck > 1.0e-5) {
      return 0.;
    }
    if (abs(ck) <= 1.0e-5) {
      numREq += 1;
    }
  }
  double x = 1. / (double)numREq;
  return x;
}

void PhononH0::addLongRangeTerm(Eigen::Tensor<std::complex<double>, 4> &dyn,
                                const Eigen::VectorXd &q) {
  // this subroutine is the analogous of rgd_blk in QE
  // compute the rigid-ion (long-range) term for q
  // The long-range term used here, to be added to or subtracted from the
  // dynamical matrices, is exactly the same of the formula introduced in:
  // X. Gonze et al, PRB 50. 13035 (1994) . Only the G-space term is
  // implemented: the Ewald parameter alpha must be large enough to
  // have negligible r-space contribution

  //   complex(DP) :: dyn(3,3,numAtoms,numAtoms) ! dynamical matrix
  //   real(DP) &
  //        q(3),           &! q-vector

  // alpha, implicitly set to 1, is the Ewald parameter,
  // geg is an estimate of G^2 such that the G-space sum is convergent for alpha
  // very rough estimate: geg/4/alpha > gMax = 14
  // (exp (-14) = 10^-6)

  Kokkos::Profiling::pushRegion("PhononH0::addLongRangeTerm");

/*
       IF (loto_2d) THEN
          geg = g1**2 + g2**2 + g3**2
          grg = 0.0d0
          IF (g1**2 + g2**2 > 1d-8) THEN
            grg = g1 * reff(1, 1) * g1 + g1 * reff(1, 2) * g2 + g2 * reff(2, 1) * g1 + g2 * reff(2, 2) * g2
            grg = grg / (g1**2 + g2**2)
          ENDIF
        ELSE
          geg = (g1 * (epsil(1, 1) * g1 + epsil(1, 2) * g2 + epsil(1, 3) * g3) + &
                 g2 * (epsil(2, 1) * g1 + epsil(2, 2) * g2 + epsil(2, 3) * g3) + &
                 g3 * (epsil(3, 1) * g1 + epsil(3, 2) * g2 + epsil(3, 3) * g3))
        ENDIF
        !
        IF (geg > 0.0d0 .AND. geg / (alph * 4.0d0) < gmax) THEN
          !
          IF (loto_2d) THEN
            facgd = fac * (tpi / alat) * EXP(-geg / (alph * 4.0d0)) / (SQRT(geg) * (1.0 + grg * SQRT(geg)))
          ELSE
            facgd = fac * EXP(-geg / (alph * 4.0d0)) / geg
          ENDIF
          !
          DO nb = 1, nat
            zbg(:) = g1 * zeu(1, :, nb) + g2 * zeu(2, :, nb) + g3 * zeu(3, :, nb)
            DO na = 1, nat
              zag(:) = g1 * zeu(1, :, na) + g2 * zeu(2, :, na) + g3 * zeu(3, :, na)
              arg = 2.d0 * pi * (g1 * (tau(1, na) - tau(1 ,nb)) + &
                                 g2 * (tau(2, na) - tau(2, nb)) + &
                                 g3 * (tau(3, na) - tau(3, nb)) )
              facg = facgd * CMPLX(COS(arg), SIN(arg), KIND=DP)
              !
              DO j = 1, 3
                DO i = 1, 3
                  dyn(i, j, na, nb) = dyn(i, j, na, nb) + facg * zag(i) * zbg(j)
                ENDDO ! i
              ENDDO ! j
            ENDDO ! na
          ENDDO ! nb
 */




  for (int na = 0; na < numAtoms; na++) {
    for (int j : {0, 1, 2}) {
      for (int i : {0, 1, 2}) {
        dyn(i, j, na, na) += longRangeCorrection1(i, j, na);
      }
    }
  }

  double norm;
  Eigen::Matrix3d reff;
  if(dimensionality == 2 && longRange2d) {
    norm = e2 * twoPi / volumeUnitCell;  // originally
    // (e^2 * 2\pi) / Area
    // fac = (sign * e2 * tpi) / (omega * bg(3, 3) / alat)
    reff = dielectricMatrix * 0.5 * directUnitCell(2,2);
  } else {
    norm = e2 * fourPi / volumeUnitCell;
  }

  for (int ig=0; ig<gVectors.cols(); ++ig) {
    Eigen::Vector3d gq = gVectors.col(ig) + q;

      double geg = 0;
      if(dimensionality == 2 && longRange2d) {
        if(gq(0)*gq(0) + gq(1)*gq(1) > 1.e-8) {
          geg = (gq.transpose() * reff * gq).value() / (gq.norm() * gq.norm());
        }
      } else {
        geg = (gq.transpose() * dielectricMatrix * gq).value();
      }

    if (geg > 0. && geg < 4. * gMax) {
      double normG;
      if(dimensionality == 2 && longRange2d) { // TODO check if norm is squared
        normG = norm * exp(-gq.norm() * 0.25) / sqrt(gq.norm()) * (1.0 + geg * sqrt(gq.norm()));
      } else {
        normG = norm * exp(-geg * 0.25) / geg;
      }

      Eigen::MatrixXd gqZ(3, numAtoms);
      for (int i : {0, 1, 2}) {
        for (int nb = 0; nb < numAtoms; nb++) {
          gqZ(i, nb) = gq(0) * bornCharges(nb, 0, i) +
                       gq(1) * bornCharges(nb, 1, i) +
                       gq(2) * bornCharges(nb, 2, i);
        }
      }

      Eigen::MatrixXcd phases(numAtoms, numAtoms);
      for (int nb = 0; nb < numAtoms; nb++) {
        for (int na = 0; na < numAtoms; na++) {
          double arg =
              (atomicPositions.row(na) - atomicPositions.row(nb)).dot(gq);
          phases(na, nb) = {cos(arg), sin(arg)};
        }
      }

      for (int nb = 0; nb < numAtoms; nb++) {
        for (int na = 0; na < numAtoms; na++) {
          for (int j : {0, 1, 2}) {
            for (int i : {0, 1, 2}) {
              dyn(i, j, na, nb) += normG * phases(na, nb) * std::complex<double>(gqZ(i, na) * gqZ(j, nb),0.);
            }
          }
        }
      }
    }
  }
  Kokkos::Profiling::popRegion();
}

void PhononH0::reorderDynamicalMatrix(const Eigen::Matrix3d& directUnitCell,
                                      const Eigen::Tensor<double, 7>& forceConstants) {

  Kokkos::Profiling::pushRegion("PhononH0::reorderDynamicalMatrix");

  // this part can actually be expensive to execute, so we compute it once
  // at the beginning

  Eigen::MatrixXd directUnitCellSup(3, 3);
  directUnitCellSup.col(0) = directUnitCell.col(0) * qCoarseGrid(0);
  directUnitCellSup.col(1) = directUnitCell.col(1) * qCoarseGrid(1);
  directUnitCellSup.col(2) = directUnitCell.col(2) * qCoarseGrid(2);

  int nr1Big = 2 * qCoarseGrid(0);
  int nr2Big = 2 * qCoarseGrid(1);
  int nr3Big = 2 * qCoarseGrid(2);

  // start by generating the weights for the Fourier transform
  auto wsCache = wsInit(directUnitCellSup, directUnitCell, nr1Big, nr2Big, nr3Big);

  // we compute the total number of bravais lattice vectors

  numBravaisVectors = 0;
  for (int n3 = -nr3Big; n3 <= nr3Big; n3++) {
    int n3ForCache = n3 + nr3Big;
    for (int n2 = -nr2Big; n2 <= nr2Big; n2++) {
      int n2ForCache = n2 + nr2Big;
      for (int n1 = -nr1Big; n1 <= nr1Big; n1++) {
        int n1ForCache = n1 + nr1Big;
        for (int nb = 0; nb < numAtoms; nb++) {
          for (int na = 0; na < numAtoms; na++) {
            if (wsCache(n3ForCache, n2ForCache, n1ForCache, nb, na) > 0.) {
              numBravaisVectors += 1;
            }
          }
        }
      }
    }
  }

  // next, we reorder the dynamical matrix along the bravais lattice vectors

  bravaisVectors = Eigen::MatrixXd::Zero(3, numBravaisVectors);
  weights = Eigen::VectorXd::Zero(numBravaisVectors);
  mat2R.resize(3, 3, numAtoms, numAtoms, numBravaisVectors);
  mat2R.setZero();

  int iR = 0;
  for (int n3 = -nr3Big; n3 <= nr3Big; n3++) {
    int n3ForCache = n3 + nr3Big;
    for (int n2 = -nr2Big; n2 <= nr2Big; n2++) {
      int n2ForCache = n2 + nr2Big;
      for (int n1 = -nr1Big; n1 <= nr1Big; n1++) {
        int n1ForCache = n1 + nr1Big;
        for (int nb = 0; nb < numAtoms; nb++) {
          for (int na = 0; na < numAtoms; na++) {
            double weight = wsCache(n3ForCache, n2ForCache, n1ForCache, nb, na);
            if (weight > 0.) {
              weights(iR) = weight;

              Eigen::Vector3d r;
              for (int i : {0, 1, 2}) {
                r(i) = n1 * directUnitCell(i, 0) + n2 * directUnitCell(i, 1) +
                       n3 * directUnitCell(i, 2);
              }
              bravaisVectors.col(iR) = r;

              int m1 = mod((n1 + 1), qCoarseGrid(0));
              if (m1 <= 0) {
                m1 += qCoarseGrid(0);
              }
              int m2 = mod((n2 + 1), qCoarseGrid(1));
              if (m2 <= 0) {
                m2 += qCoarseGrid(1);
              }
              int m3 = mod((n3 + 1), qCoarseGrid(2));
              if (m3 <= 0) {
                m3 += qCoarseGrid(2);
              }
              m1 += -1;
              m2 += -1;
              m3 += -1;

              for (int j : {0, 1, 2}) {
                for (int i : {0, 1, 2}) {
                  mat2R(i, j, na, nb, iR) +=
                      forceConstants(i, j, m1, m2, m3, na, nb);
                }
              }
              iR += 1;
            }
          }
        }
      }
    }
  }

  // wsCache.resize(0, 0, 0, 0, 0);
  // forceConstants.resize(0, 0, 0, 0, 0, 0, 0);
  Kokkos::Profiling::popRegion();
}

void PhononH0::shortRangeTerm(Eigen::Tensor<std::complex<double>, 4> &dyn,
                              const Eigen::VectorXd &q) {
  // calculates the dynamical matrix at q from the (short-range part of the)
  // force constants21, by doing the Fourier transform of the force constants

  Kokkos::Profiling::pushRegion("phononH0.shortRangeTerm");

  std::vector<std::complex<double>> phases(numBravaisVectors);
  for (int iR = 0; iR < numBravaisVectors; iR++) {
    Eigen::Vector3d r = bravaisVectors.col(iR);
    double arg = q.dot(r);
    phases[iR] = exp(-complexI * arg); // {cos(arg), -sin(arg)};
    //printf("old = %.16e %.16e\n", phases[iR].real(), phases[iR].imag());
    //for(int i = 0; i < 3; i++)
    //  printf("old = %.16e\n", r[i]);
  }

  for (int iR = 0; iR < numBravaisVectors; iR++) {
    for (int nb = 0; nb < numAtoms; nb++) {
      for (int na = 0; na < numAtoms; na++) {
        for (int j : {0, 1, 2}) {
          for (int i : {0, 1, 2}) {
            dyn(i, j, na, nb) +=
                mat2R(i, j, na, nb, iR) * phases[iR] * weights(iR);
          }
        }
      }
    }
  }
    //for (int nb = 0; nb < numAtoms; nb++) {
    //  for (int na = 0; na < numAtoms; na++) {
    //    for (int j : {0, 1, 2}) {
    //      for (int i : {0, 1, 2}) {
    //        printf("old = %.16e %.16e\n", dyn(i,j,na,nb).real(), dyn(i,j,na,nb).imag());
    //      }
    //    }
    //  }
    //}
  Kokkos::Profiling::popRegion();
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXcd>
PhononH0::dynDiagonalize(Eigen::Tensor<std::complex<double>, 4> &dyn) {
  // diagonalise the dynamical matrix
  // On input:  speciesMasses = masses, in amu
  // On output: w2 = energies, z = displacements

  Kokkos::Profiling::pushRegion("PhononH0::dynDiagonalize");

  // fill the two-indices dynamical matrix

  Eigen::MatrixXcd dyn2Tmp(numBands, numBands);
  for (int jat = 0; jat < numAtoms; jat++) {
    for (int iat = 0; iat < numAtoms; iat++) {
      for (int j : {0, 1, 2}) {
        for (int i : {0, 1, 2}) {
          dyn2Tmp(iat * 3 + i, jat * 3 + j) = dyn(i, j, iat, jat);
        }
      }
    }
  }

  // impose hermiticity

  Eigen::MatrixXcd dyn2(numBands, numBands);
  dyn2 = dyn2Tmp + dyn2Tmp.adjoint();
  dyn2 *= 0.5;

  //  divide by the square root of masses

  for (int jat = 0; jat < numAtoms; jat++) {
    int jType = atomicSpecies(jat);
    for (int j : {0, 1, 2}) {
      for (int iat = 0; iat < numAtoms; iat++) {
        int iType = atomicSpecies(iat);
        for (int i : {0, 1, 2}) {
          dyn2(iat * 3 + i, jat * 3 + j) /=
              sqrt(speciesMasses(iType) * speciesMasses(jType));
        }
      }
    }
  }
  //for(int i = 0; i < 3*numAtoms; i++){
  //  for(int j = 0; j < 3*numAtoms; j++){
  //    auto x = dyn2(i,j);
  //    printf("old = %.16e %.16e\n", x.real(), x.imag());
  //  }
  //}

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(dyn2);
  Eigen::VectorXd w2 = eigenSolver.eigenvalues();

  Eigen::VectorXd energies(numBands);
  for (int i = 0; i < numBands; i++) {
    if (w2(i) < 0) {
      energies(i) = -sqrt(-w2(i));
    } else {
      energies(i) = sqrt(w2(i));
    }
    //printf("old = %.16e\n", energies(i));
  }
  Eigen::MatrixXcd eigenvectors = eigenSolver.eigenvectors();
  //eigenvectors = 3*dyn2; // TODO: undo

  //for(int i = 0; i < 3*numAtoms; i++){
  //  for(int j = 0; j < 3*numAtoms; j++){
  //    auto x = eigenvectors(i,j);
  //    printf("old = %.16e %.16e\n", x.real(), x.imag());
  //  }
  //}
  Kokkos::Profiling::popRegion();
  return std::make_tuple(energies, eigenvectors);
}

Eigen::Tensor<std::complex<double>, 3>
PhononH0::diagonalizeVelocity(Point &point) {
  Eigen::Vector3d coordinates =
      point.getCoordinates(Points::cartesianCoordinates);
  return diagonalizeVelocityFromCoordinates(coordinates);
}

Eigen::Tensor<std::complex<double>, 3>
PhononH0::diagonalizeVelocityFromCoordinates(Eigen::Vector3d &coordinates) {

  Kokkos::Profiling::pushRegion("PhononH0::diagonalizeVelocityFromCoordinates");

  Eigen::Tensor<std::complex<double>, 3> velocity(numBands, numBands, 3);
  velocity.setZero();

  bool withMassScaling = false;
    //for(int i = 0; i < 3; i++){
    //  printf("old = %.16e\n", coordinates(i));
    //}

  // get the eigenvectors and the energies of the q-point
  auto tup = diagonalizeFromCoordinates(coordinates, withMassScaling);
  auto energies = std::get<0>(tup);
  auto eigenvectors = std::get<1>(tup);

  //for(int i = 0; i < numBands; i++) printf("old = %.16e\n", energies(i));
  //for(int i = 0; i < numBands; i++)
  //  for(int j = 0; j < numBands; j++)
  //    printf("old = %.16e %.16e\n", eigenvectors(i,j).real(), eigenvectors(i,j).imag());

  // now we compute the velocity operator, diagonalizing the expectation
  // value of the derivative of the dynamical matrix.
  // This works better than doing finite differences on the frequencies.
  double deltaQ = 1.0e-8;
  for (int i : {0, 1, 2}) {
    // define q+ and q- from finite differences.
    Eigen::Vector3d qPlus = coordinates;
    Eigen::Vector3d qMinus = coordinates;
    qPlus(i) += deltaQ;
    qMinus(i) -= deltaQ;

    //for(int i = 0; i < 3; i++){
    //  printf("old = %.16e %.16e\n", qPlus(i), qMinus(i));
    //}

    // diagonalize the dynamical matrix at q+ and q-
    auto tup2 = diagonalizeFromCoordinates(qPlus, withMassScaling);
    auto enPlus = std::get<0>(tup2);
    auto eigPlus = std::get<1>(tup2);
    auto tup1 = diagonalizeFromCoordinates(qMinus, withMassScaling);
    auto enMinus = std::get<0>(tup1);
    auto eigMinus = std::get<1>(tup1);
    //for(int i = 0; i < numBands; i++) printf("old = %.16e\n", enPlus(i));
    //for(int i = 0; i < numBands; i++) printf("old = %.16e\n", enMinus(i));
    //for(int i = 0; i < numBands; i++)
    //  for(int j = 0; j < numBands; j++)
    //    printf("old = %.16e %.16e\n", eigPlus(i,j).real(), eigPlus(i,j).imag());
    //for(int i = 0; i < numBands; i++)
    //  for(int j = 0; j < numBands; j++)
    //    printf("old = %.16e %.16e\n", eigMinus(i,j).real(), eigMinus(i,j).imag());

    // build diagonal matrices with frequencies
    Eigen::MatrixXd enPlusMat(numBands, numBands);
    Eigen::MatrixXd enMinusMat(numBands, numBands);
    enPlusMat.setZero();
    enMinusMat.setZero();
    enPlusMat.diagonal() << enPlus;
    enMinusMat.diagonal() << enMinus;

    // build the dynamical matrix at the two wavevectors
    // since we diagonalized it before, A = M.U.M*
    Eigen::MatrixXcd sqrtDPlus(numBands, numBands);
    sqrtDPlus = eigPlus * enPlusMat * eigPlus.adjoint();
    Eigen::MatrixXcd sqrtDMinus(numBands, numBands);
    sqrtDMinus = eigMinus * enMinusMat * eigMinus.adjoint();

    // now we can build the velocity operator
    Eigen::MatrixXcd der(numBands, numBands);
    der = (sqrtDPlus - sqrtDMinus) / (2. * deltaQ);

    //for(int i = 0; i < numBands; i++){
    //  for(int j = 0; j < numBands; j++){
    //    auto x = der(i,j);
    //    printf("old = %.16e %.16e\n", x.real(), x.imag());
    //  }
    //}
    //for(int i = 0; i < numBands; i++){
    //  for(int j = 0; j < numBands; j++){
    //    auto x = sqrtDPlus(i,j);
    //    printf("old = %.16e %.16e\n", x.real(), x.imag());
    //  }
    //}
    //for(int i = 0; i < numBands; i++){
    //  for(int j = 0; j < numBands; j++){
    //    auto x = sqrtDMinus(i,j);
    //    printf("old = %.16e %.16e\n", x.real(), x.imag());
    //  }
    //}

    // and to be safe, we reimpose hermiticity
    der = 0.5 * (der + der.adjoint());

    // now we rotate in the basis of the eigenvectors at q.
    der = eigenvectors.adjoint() * der * eigenvectors;

    for (int ib2 = 0; ib2 < numBands; ib2++) {
      for (int ib1 = 0; ib1 < numBands; ib1++) {
        velocity(ib1, ib2, i) = der(ib1, ib2);
        //auto x = velocity(ib1,ib2,i);
        //printf("old = %.16e %.16e\n", x.real(), x.imag());
      }
    }
  }

  // turns out that the above algorithm has problems with degenerate bands
  // so, we diagonalize the velocity operator in the degenerate subspace,

  for (int ib = 0; ib < numBands; ib++) {
    // first, we check if the band is degenerate, and the size of the
    // degenerate subspace
    int sizeSubspace = 1;
    for (int ib2 = ib + 1; ib2 < numBands; ib2++) {
      // I consider bands degenerate if their frequencies are the same
      // within 0.0001 cm^-1
      if (abs(energies(ib) - energies(ib2)) > 0.0001 / ryToCmm1) {
        break;
      }
      sizeSubspace += 1;
    }

    if (sizeSubspace > 1) {
      Eigen::MatrixXcd subMat(sizeSubspace, sizeSubspace);
      // we have to repeat for every direction
      for (int iCart : {0, 1, 2}) {
        // take the velocity matrix of the degenerate subspace
        for (int j = 0; j < sizeSubspace; j++) {
          for (int i = 0; i < sizeSubspace; i++) {
            subMat(i, j) = velocity(ib + i, ib + j, iCart);
          }
        }

        // reinforce hermiticity
        subMat = 0.5 * (subMat + subMat.adjoint());

        // diagonalize the subMatrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(subMat);
        Eigen::MatrixXcd newEigenVectors = eigenSolver.eigenvectors();
        //newEigenVectors = 3*subMat; // TODO: undo

        // rotate the original matrix in the new basis
        // that diagonalizes the subspace.
        subMat = newEigenVectors.adjoint() * subMat * newEigenVectors;

        // reinforce hermiticity
        subMat = 0.5 * (subMat + subMat.adjoint());

        // substitute back
        for (int j = 0; j < sizeSubspace; j++) {
          for (int i = 0; i < sizeSubspace; i++) {
            velocity(ib + i, ib + j, iCart) = subMat(i, j);
          }
        }
      }
    }

    // we skip the bands in the subspace, since we corrected them already
    ib += sizeSubspace - 1;
  }
  // if we are working at gamma, we set all velocities to zero.
  if (coordinates.norm() < 1.0e-6) {
    velocity.setZero();
  }
  //for(int i = 0; i < 3; i++){
  //  for (int ib2 = 0; ib2 < numBands; ib2++) {
  //    for (int ib1 = 0; ib1 < numBands; ib1++) {
  //      auto x = velocity(ib1,ib2,i);
  //      printf("old = %.16e %.16e\n", x.real(), x.imag());
  //    }
  //  }
  //}
  Kokkos::Profiling::popRegion();
  return velocity;
}

Eigen::Vector3i PhononH0::getCoarseGrid() { return qCoarseGrid; }

Eigen::Matrix3d PhononH0::getDielectricMatrix() { return dielectricMatrix; }

Eigen::Tensor<double, 3> PhononH0::getBornCharges() { return bornCharges; }

int PhononH0::getIndexEigenvector(const int &iAt, const int &iPol) const {
  return compress2Indices(iAt, iPol, numAtoms, 3);
}

int PhononH0::getIndexEigenvector(const int &iAt, const int &iPol,
                                  const int &nAtoms) {
  return compress2Indices(iAt, iPol, nAtoms, 3);
}

