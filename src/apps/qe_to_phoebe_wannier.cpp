#include "bandstructure.h"
#include "eigen.h"
#include "elph_qe_to_phoebe_app.h"
#include "interaction_elph.h"
#include "io.h"
#include "qe_input_parser.h"
#include "utilities.h"
#include <algorithm>
#include <exception>
#include <sstream>
#include <string>
#include <Kokkos_Core.hpp>

Eigen::Tensor<std::complex<double>, 5>
ElPhQeToPhoebeApp::BlochToWannierEfficient(
    Context &context, const Eigen::MatrixXd &energies,
    const Eigen::MatrixXd &kGridFull, const int &numIrrQPoints,
    const int &numQEBands, const Eigen::MatrixXd &elBravaisVectors,
    const Eigen::MatrixXd &phBravaisVectors,
    const Eigen::Tensor<std::complex<double>, 3> &uMatrices, Points &kPoints,
    Points &qPoints, Crystal &crystal, PhononH0 &phononH0) {

  if (mpi->getSize() > numIrrQPoints) {
    Error("We cannot run the Bloch2Wannier transformation with more "
          "MPI processes than available prefix.phoebe.*.dat files");
  }

  int numModes = crystal.getNumAtoms() * 3;
  int numBands = int(uMatrices.dimension(0));
  int numWannier = int(uMatrices.dimension(1));
  int numKPoints = kPoints.getNumPoints();
  int numQPoints = qPoints.getNumPoints();
  int numElBravaisVectors = int(elBravaisVectors.cols());
  int numPhBravaisVectors = int(phBravaisVectors.cols());

  std::string wannierPrefix = context.getWannier90Prefix();
  int bandsOffset = computeOffset(energies, wannierPrefix);

  Eigen::VectorXi ikMap(numKPoints);
#pragma omp parallel for default(none)                                         \
    shared(numKPoints, kGridFull, kPoints, ikMap, Eigen::Dynamic)
  for (int ikOld = 0; ikOld < numKPoints; ikOld++) {
    Eigen::Vector3d kOld = kGridFull.col(ikOld);
    int ikNew = kPoints.getIndex(kOld);
    ikMap(ikOld) = ikNew;
  }
  std::string phoebePrefixQE = context.getQuantumEspressoPrefix();

  std::array<Eigen::Index, 5> zeros;
  for (auto &s : zeros) {
    s = 0;
  }

  bool usePolarCorrection = false;
  Eigen::Matrix3d dielectricMatrix = phononH0.getDielectricMatrix();
  if (dielectricMatrix.squaredNorm() > 1.0e-10) { // i.e. if dielectricMatrix
    // wasn't computed
    if (crystal.getNumSpecies() > 1) { // otherwise polar correction = 0
      usePolarCorrection = true;
    }
  }
  if (usePolarCorrection && mpi->mpiHead()) {
    std::cout << "Polar correction" << std::endl;
  }

  auto localElIndices = mpi->divideWorkIter(numElBravaisVectors);
  int numLocalElIndices = localElIndices.size();
  int localElIndicesOffset = localElIndices[0];

  Eigen::Tensor<std::complex<double>, 5> gWannierPara(
      numWannier, numWannier, numModes, numPhBravaisVectors, numLocalElIndices);
  gWannierPara.setZero();

  // decide the number of loops over IrrQPoints
  // note: in every loop, there is a call to mpi->allReduceSum()
  // therefore, we must make sure that every process enters the loop,
  // even if it doesn't need to read a file.
  // (e.g. analyzing 4 irrQPoints with 3 MPI processes has load imbalance)
  auto localIrrPoints = int(mpi->divideWorkIter(numIrrQPoints).size());
  int loopSize = localIrrPoints;
  mpi->allReduceMax(&loopSize);
  auto pointsIterator = mpi->divideWorkIter(numIrrQPoints);

  Eigen::MatrixXcd phPhases;
  Eigen::Tensor<std::complex<double>, 5> gWannierTmp;

  LoopPrint loopPrint("Wannier transform of coupling", "irreducible q-points",
                      loopSize);
  // for (int iqIrr : mpi->divideWorkIter(numIrrQPoints)) {
  for (int iLoop = 0; iLoop < loopSize; iLoop++) {
    loopPrint.update(false);

    int iqIrr = -1;
    if (iLoop < int(pointsIterator.size())) {
      iqIrr = pointsIterator[iLoop];
    }

    if (iqIrr >= 0) {

      auto t = readChunkGFromQE(iqIrr, context, kPoints, numModes, numQEBands,
                                ikMap);
      auto &gStarTmp = std::get<0>(t);
      auto phononEigenvectorsStar = std::get<1>(t);
      auto qStar = std::get<3>(t);
      int numQStar = int(qStar.cols());

      // Note: the tensor read from file contains
      // gFull(ib1, ib2, nu, ik, iq)
      // = < k+q,ib1 |  dV_{q,nu}  |  k,ib2  >
      // where k and q run over the full mesh.
      // ikMap takes care of the fact that k-points in QE have a different
      // order then phoebe.

      // reorder the q/k indices
      Eigen::Tensor<std::complex<double>, 5> gStar(numBands, numBands, numModes,
                                                   numKPoints, numQStar);
      gStar.setZero();
      for (int iqStar = 0; iqStar < numQStar; iqStar++) {
        for (int ik = 0; ik < numKPoints; ik++) {
          for (int nu = 0; nu < numModes; nu++) {
            for (int ib2 = 0; ib2 < numWannier; ib2++) {
              for (int ib1 = 0; ib1 < numWannier; ib1++) {
                gStar(ib1, ib2, nu, ik, iqStar) =
                    gStarTmp(bandsOffset + ib1, bandsOffset + ib2, nu,
                             ik, iqStar);
              }
            }
          }
        }
      }
      gStarTmp.resize(0,0,0,0,0);

      if (usePolarCorrection) {
        // we need to subtract the polar correction
        // this contribution will be reinstated during the interpolation
        auto volume = crystal.getVolumeUnitCell();
        auto reciprocalUnitCell = crystal.getReciprocalUnitCell();
        auto bornCharges = phononH0.getBornCharges();
        auto atomicPositions = crystal.getAtomicPositions();
        auto qCoarseMesh = phononH0.getCoarseGrid();
	auto dimensionality = crystal.getDimensionality();

        for (int iq = 0; iq < numQStar; iq++) {
          Eigen::Vector3d qCrystal = qStar.col(iq);
          Eigen::Vector3d q = qPoints.crystalToCartesian(qCrystal);
          if (q.norm() > 1.0e-8) {

            Eigen::MatrixXcd ev3(numModes, numModes);
            for (int j = 0; j < numModes; j++) {
              for (int i = 0; i < numModes; i++) {
                ev3(i, j) = phononEigenvectorsStar(i, j, iq);
              }
            }

            for (int ik = 0; ik < numKPoints; ik++) {
              Eigen::Vector3d k =
                  kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);

              // Coordinates and index of k+q point
              Eigen::Vector3d kq = k + q;
              Eigen::Vector3d kqCrystal = kPoints.cartesianToCrystal(kq);
              int ikq = kPoints.getIndex(kqCrystal);

              // gather eigenvectors
              // uMatrices has size (numBands, numWannier, numKPoints)
              Eigen::MatrixXcd ev1(numBands, numWannier);
              Eigen::MatrixXcd ev2(numBands, numWannier);
              for (int j = 0; j < numWannier; j++) {
                for (int i = 0; i < numBands; i++) {
                  ev1(i, j) = uMatrices(i, j, ik);
                  ev2(i, j) = uMatrices(i, j, ikq);
                }
              }
              ev1 = ev1.adjoint(); // (Wannier,bands)
              ev2 = ev2.adjoint(); // (Wannier,bands)

              auto v = InteractionElPhWan::getPolarCorrectionStatic(
                  q, ev1, ev2, ev3, volume, reciprocalUnitCell,
                  dielectricMatrix, bornCharges, atomicPositions, qCoarseMesh, dimensionality);
              for (int nu = 0; nu < numModes; nu++) {
                for (int j = 0; j < numBands; j++) {
                  for (int i = 0; i < numBands; i++) {
                    // note the swapped indices
                    // since gStar has indices on k+q, k bands in this order
                    // and v has indices on k, k+q in this other order
                    gStar(i, j, nu, ik, iq) -= v(j, i, nu);
                  }
                }
              }
            }
          }
        }
      }

      Eigen::Tensor<std::complex<double>, 5> gFullTmp(
          numWannier, numWannier, numModes, numKPoints, numQStar);
      gFullTmp.setZero();

      for (int iq = 0; iq < numQStar; iq++) {
        Eigen::Vector3d qCrystal = qStar.col(iq);
        Eigen::Vector3d q = qPoints.crystalToCartesian(qCrystal);
        for (int ik = 0; ik < numKPoints; ik++) {
          Eigen::Vector3d k =
              kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);

          // Coordinates and index of k+q point
          Eigen::Vector3d kq = k + q;
          Eigen::Vector3d kqCrystal = kPoints.cartesianToCrystal(kq);
          int ikq = kPoints.getIndex(kqCrystal);

          // First we transform from the Bloch to Wannier Gauge

          // u has size (numBands, numWannier, numKPoints)
          Eigen::MatrixXcd uK(numBands, numWannier);
          Eigen::MatrixXcd uKq(numBands, numWannier);
          for (int j = 0; j < numWannier; j++) {
            for (int i = 0; i < numBands; i++) {
              uK(i, j) = uMatrices(i, j, ik);
              uKq(i, j) = uMatrices(i, j, ikq);
            }
          }
          Eigen::MatrixXcd uKDagger(numWannier, numBands);
          uKDagger = uK.adjoint();

#pragma omp parallel default(none) shared(numModes, numWannier, numBands, uKq, \
                                          gStar, gFullTmp, ik, iq, uKDagger)
          {
            Eigen::Tensor<std::complex<double>, 3> tmp(numWannier, numBands,
                                                       numModes);
            tmp.setZero();
#pragma omp for nowait collapse(4)
            for (int nu = 0; nu < numModes; nu++) {
              for (int i = 0; i < numWannier; i++) {
                for (int j = 0; j < numBands; j++) {
                  for (int l = 0; l < numBands; l++) {
                    // ukq has size(numWannier, numBands)
                    // gFull has size numBands, numBands, ...
                    tmp(i, j, nu) += uKq(l, i) * gStar(l, j, nu, ik, iq);
                  }
                }
              }
            }
            Eigen::Tensor<std::complex<double>, 3> tmp2(numWannier, numWannier,
                                                        numModes);
            tmp2.setZero();
#pragma omp for nowait collapse(4)
            for (int nu = 0; nu < numModes; nu++) {
              for (int i = 0; i < numWannier; i++) {
                for (int j = 0; j < numWannier; j++) {
                  for (int l = 0; l < numBands; l++) {
                    tmp2(i, j, nu) += tmp(i, l, nu) * uKDagger(j, l);
                  }
                }
              }
            }

#pragma omp critical
            for (int nu = 0; nu < numModes; nu++) {
              for (int i = 0; i < numWannier; i++) {
                for (int j = 0; j < numWannier; j++) {
                  gFullTmp(i, j, nu, ik, iq) += tmp2(i, j, nu);
                }
              }
            }
          }
        } // ik
      }   // iq
      gStar.reshape(zeros);

      // Fourier transform on the electronic coordinates
      Eigen::Tensor<std::complex<double>, 5> gMixed(
          numWannier, numWannier, numModes, numElBravaisVectors, numQStar);
      gMixed.setZero();

      {
        Eigen::MatrixXcd phases(numKPoints, numElBravaisVectors);
        phases.setZero();
#pragma omp parallel for
        for (int ik = 0; ik < numKPoints; ik++) {
          Eigen::Vector3d k =
              kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);
          for (int iR = 0; iR < numElBravaisVectors; iR++) {
            double arg = k.dot(elBravaisVectors.col(iR));
            phases(ik, iR) = exp(-complexI * arg) / double(numKPoints);
          }
        }

        for (int iq = 0; iq < numQStar; iq++) {
#pragma omp parallel default(none)                                             \
    shared(iq, gFullTmp, phases, numElBravaisVectors, numKPoints, numModes,    \
           numWannier, gMixed)
          {
            Eigen::Tensor<std::complex<double>, 4> tmp(
                numWannier, numWannier, numModes, numElBravaisVectors);
            tmp.setZero();
#pragma omp for nowait
            for (int iR = 0; iR < numElBravaisVectors; iR++) {
              for (int ik = 0; ik < numKPoints; ik++) {
                for (int nu = 0; nu < numModes; nu++) {
                  for (int j = 0; j < numWannier; j++) {
                    for (int i = 0; i < numWannier; i++) {
                      tmp(i, j, nu, iR) +=
                          gFullTmp(i, j, nu, ik, iq) * phases(ik, iR);
                    }
                  }
                }
              }
            }
#pragma omp critical
            for (int iR = 0; iR < numElBravaisVectors; iR++) {
              for (int nu = 0; nu < numModes; nu++) {
                for (int j = 0; j < numWannier; j++) {
                  for (int i = 0; i < numWannier; i++) {
                    gMixed(i, j, nu, iR, iq) += tmp(i, j, nu, iR);
                  }
                }
              }
            }
          }
        } // iq
      }
      gFullTmp.reshape(zeros);

      gWannierTmp.resize(numWannier, numWannier, numModes, numElBravaisVectors,
                         numQStar);
      gWannierTmp.setZero();
      {
        Eigen::Tensor<std::complex<double>, 3> uQM1s(numModes, numModes,
                                                     numQStar);
        uQM1s.setZero();
        for (int iq = 0; iq < numQStar; iq++) {
          Eigen::MatrixXcd uQ(numModes, numModes);
          for (int nu2 = 0; nu2 < numModes; nu2++) {
            for (int nu = 0; nu < numModes; nu++) {
              uQ(nu, nu2) = phononEigenvectorsStar(nu, nu2, iq);
            }
          }
          auto uQM1 = uQ.inverse();
          for (int nu2 = 0; nu2 < numModes; nu2++) {
            for (int nu = 0; nu < numModes; nu++) {
              uQM1s(nu, nu2, iq) = uQM1(nu, nu2);
            }
          }
          // this isn't equal to the adjoint, due to mass renormalization
          // should be parallelized with OMP already
        }
        for (int iq = 0; iq < numQStar; iq++) {
          for (int nu = 0; nu < numModes; nu++) {
            for (int nu2 = 0; nu2 < numModes; nu2++) {
#pragma omp parallel for collapse(3) default(none) shared(                     \
    numElBravaisVectors, numWannier, gMixed, uQM1s, iq, nu, nu2, gWannierTmp)
              for (int irE = 0; irE < numElBravaisVectors; irE++) {
                for (int i = 0; i < numWannier; i++) {
                  for (int j = 0; j < numWannier; j++) {
                    gWannierTmp(i, j, nu, irE, iq) +=
                        gMixed(i, j, nu2, irE, iq) * uQM1s(nu2, nu, iq);
                  }
                }
              }
            }
          }
        }
      }
      gMixed.reshape(zeros);

      phPhases.resize(numPhBravaisVectors, numQStar);
      phPhases.setZero();
#pragma omp parallel for
      for (int iq = 0; iq < numQStar; iq++) {
        Eigen::Vector3d qCrystal = qStar.col(iq);
        Eigen::Vector3d q = qPoints.crystalToCartesian(qCrystal);
        for (int irP = 0; irP < numPhBravaisVectors; irP++) {
          double arg = q.dot(phBravaisVectors.col(irP));
          phPhases(irP, iq) = exp(-complexI * arg) / double(numQPoints);
        }
      }

    } // if iqIrr != -1

    // Create a vector with the indices from 0, 1, ..., numElBravaisVectors-1
    std::vector<int> v(numElBravaisVectors);
    std::iota(std::begin(v), std::end(v), 0);
    // split it into chunks
    int chunkSize = numElBravaisVectors / mpi->getSize();
    auto chunks = splitVector(v, chunkSize);

    for (std::vector<int> chunk : chunks) {
      Eigen::Tensor<std::complex<double>, 5> tmp(numWannier, numWannier,
                                                 numModes, numPhBravaisVectors,
                                                 int(chunk.size()));
      tmp.setZero();

      if (iqIrr >= 0) { // if the MPI process actually is computing something
        for (int irE : chunk) {
#pragma omp parallel for collapse(4) default(none)                             \
    shared(numPhBravaisVectors, phPhases, numModes, numWannier, gWannierTmp,   \
           tmp, irE, chunk)
          for (int irP = 0; irP < numPhBravaisVectors; irP++) {
            for (int nu = 0; nu < numModes; nu++) {
              for (int j = 0; j < numWannier; j++) {
                for (int i = 0; i < numWannier; i++) {
                  for (int iq = 0; iq < phPhases.cols(); iq++) {
                    tmp(i, j, nu, irP, irE - chunk[0]) +=
                        phPhases(irP, iq) * gWannierTmp(i, j, nu, irE, iq);
                  }
                }
              }
            }
          }
        }
      }

      // now, all MPI receives `tmp`
      mpi->allReduceSum(&tmp);

      for (int irE : chunk) {
        if (std::find(localElIndices.begin(), localElIndices.end(), irE) !=
            localElIndices.end()) {
          int irELocal = irE - localElIndicesOffset;
          for (int irP = 0; irP < numPhBravaisVectors; irP++) {
            for (int nu = 0; nu < numModes; nu++) {
              for (int j = 0; j < numWannier; j++) {
                for (int i = 0; i < numWannier; i++) {
                  gWannierPara(i, j, nu, irP, irELocal) +=
                      tmp(i, j, nu, irP, irE - chunk[0]);
                }
              }
            }
          }
        }
      }
    }
  } // loop on files
  loopPrint.close();

  if (mpi->mpiHead()) {
    std::cout << "Done Wannier-transform of g\n" << std::endl;
  }

  return gWannierPara;
}

Eigen::Tensor<std::complex<double>, 5> ElPhQeToPhoebeApp::blochToWannier(
    const Eigen::MatrixXd &elBravaisVectors,
    const Eigen::MatrixXd &phBravaisVectors,
    Eigen::Tensor<std::complex<double>, 5> &gFull,
    const Eigen::Tensor<std::complex<double>, 3> &uMatrices,
    const Eigen::Tensor<std::complex<double>, 3> &phEigenvectors,
    Points &kPoints, Points &qPoints, Crystal &crystal, PhononH0 &phononH0) {

  if (mpi->mpiHead()) {
    std::cout << "Start Wannier-transform of g" << std::endl;
  }

  int numBands = int(gFull.dimension(0)); // # of entangled bands
  int numModes = int(gFull.dimension(2));
  int numKPoints = int(gFull.dimension(3));
  int numQPoints = int(gFull.dimension(4));
  int numElBravaisVectors = int(elBravaisVectors.cols());
  int numPhBravaisVectors = int(phBravaisVectors.cols());
  int numWannier = int(uMatrices.dimension(1));

  Eigen::array<Eigen::Index, 5> zeros({0,0,0,0,0});

  bool usePolarCorrection = false;
  Eigen::Matrix3d dielectricMatrix = phononH0.getDielectricMatrix();
  if (dielectricMatrix.squaredNorm() > 1.0e-10) { // i.e. if dielectricMatrix
                                                  // wasn't computed
    if (crystal.getNumSpecies() > 1) { // otherwise polar correction = 0
      usePolarCorrection = true;
    }
  }

  if (usePolarCorrection) {
    if (mpi->mpiHead()) {
      std::cout << "Polar correction" << std::endl;
    }

    // we need to subtract the polar correction
    // this contribution will be reinstated during the interpolation
    auto volume = crystal.getVolumeUnitCell();
    auto dimensionality = crystal.getDimensionality();
    auto reciprocalUnitCell = crystal.getReciprocalUnitCell();
    auto bornCharges = phononH0.getBornCharges();
    auto atomicPositions = crystal.getAtomicPositions();
    auto qCoarseMesh = phononH0.getCoarseGrid();

    for (int iq = 0; iq < numQPoints; iq++) {
      Eigen::Vector3d q =
          qPoints.getPointCoordinates(iq, Points::cartesianCoordinates);
      if (q.norm() > 1.0e-8) {

        Eigen::MatrixXcd ev3(numModes, numModes);
        for (int j = 0; j < numModes; j++) {
          for (int i = 0; i < numModes; i++) {
            ev3(i, j) = phEigenvectors(i, j, iq);
          }
        }

        for (int ik = 0; ik < numKPoints; ik++) {
          Eigen::Vector3d k =
              kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);

          // Coordinates and index of k+q point
          Eigen::Vector3d kq = k + q;
          Eigen::Vector3d kqCrystal = kPoints.cartesianToCrystal(kq);
          int ikq = kPoints.getIndex(kqCrystal);

          // gather eigenvectors
          Eigen::MatrixXcd ev1(numBands, numWannier);
          Eigen::MatrixXcd ev2(numBands, numWannier);
          for (int j = 0; j < numBands; j++) {
            for (int i = 0; i < numBands; i++) {
              ev1(i, j) = uMatrices(i, j, ik);
              ev2(i, j) = uMatrices(i, j, ikq);
            }
          }
          ev1 = ev1.adjoint();
          ev2 = ev2.adjoint();

          auto v = InteractionElPhWan::getPolarCorrectionStatic(
              q, ev1, ev2, ev3, volume, reciprocalUnitCell, dielectricMatrix,
              bornCharges, atomicPositions, qCoarseMesh, dimensionality);
          for (int nu = 0; nu < numModes; nu++) {
            for (int j = 0; j < numBands; j++) {
              for (int i = 0; i < numBands; i++) {
                gFull(i, j, nu, ik, iq) -= v(i, j, nu);
              }
            }
          }
        }
      }
    }
  }

  if (mpi->mpiHead()) {
    std::cout << "Wannier rotation" << std::endl;
  }

  Eigen::Tensor<std::complex<double>, 5> gFullTmp(
      numWannier, numWannier, numModes, numKPoints, numQPoints);
  gFullTmp.setZero();

  for (size_t iq : mpi->divideWorkIter(numQPoints)) {
    Eigen::Vector3d q =
        qPoints.getPointCoordinates(iq, Points::cartesianCoordinates);
    for (int ik = 0; ik < numKPoints; ik++) {
      Eigen::Vector3d k =
          kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);

      // Coordinates and index of k+q point
      Eigen::Vector3d kq = k + q;
      Eigen::Vector3d kqCrystal = kPoints.cartesianToCrystal(kq);
      int ikq = kPoints.getIndex(kqCrystal);

      // First we transform from the Bloch to Wannier Gauge

      // u has size (numBands, numWannier, numKPoints)
      Eigen::MatrixXcd uK(numBands, numWannier);
      Eigen::MatrixXcd uKq(numBands, numWannier);
      for (int i = 0; i < numBands; i++) {
        for (int j = 0; j < numWannier; j++) {
          uK(i, j) = uMatrices(i, j, ik);
          uKq(i, j) = uMatrices(i, j, ikq);
        }
      }
      Eigen::MatrixXcd uKDagger(numWannier, numBands);
      uKDagger = uK.adjoint();

#pragma omp parallel default(none) shared(numModes, numWannier, numBands, uKq, \
                                          gFull, gFullTmp, ik, iq, uKDagger)
      {
        Eigen::Tensor<std::complex<double>, 3> tmp(numWannier, numBands,
                                                   numModes);
        tmp.setZero();
#pragma omp for nowait collapse(4)
        for (int nu = 0; nu < numModes; nu++) {
          for (int i = 0; i < numWannier; i++) {
            for (int j = 0; j < numBands; j++) {
              for (int l = 0; l < numBands; l++) {
                // ukq has size(numWannier, numBands)
                // gFull has size numBands, numBands, ...
                tmp(i, j, nu) += uKq(l, i) * gFull(l, j, nu, ik, int(iq));
              }
            }
          }
        }
        Eigen::Tensor<std::complex<double>, 3> tmp2(numWannier, numWannier,
                                                    numModes);
        tmp2.setZero();
#pragma omp for nowait collapse(4)
        for (int nu = 0; nu < numModes; nu++) {
          for (int i = 0; i < numWannier; i++) {
            for (int j = 0; j < numWannier; j++) {
              for (int l = 0; l < numBands; l++) {
                tmp2(i, j, nu) += tmp(i, l, nu) * uKDagger(j, l);
              }
            }
          }
        }

#pragma omp critical
        for (int nu = 0; nu < numModes; nu++) {
          for (int i = 0; i < numWannier; i++) {
            for (int j = 0; j < numWannier; j++) {
              gFullTmp(i, j, nu, ik, int(iq)) += tmp2(i, j, nu);
            }
          }
        }
      }
    } // ik
  }   // iq
  mpi->allReduceSum(&gFullTmp);
  gFull.reshape(zeros);

  if (mpi->mpiHead()) {
    std::cout << "Electronic Fourier Transform" << std::endl;
  }

  // Fourier transform on the electronic coordinates
  Eigen::Tensor<std::complex<double>, 5> gMixed(
      numWannier, numWannier, numModes, numElBravaisVectors, numQPoints);
  gMixed.setZero();

  {
    Eigen::MatrixXcd phases(numKPoints, numElBravaisVectors);
    phases.setZero();
    std::vector<size_t> iks = mpi->divideWorkIter(numKPoints);
    size_t niks = iks.size();
#pragma omp parallel for
    for (size_t iik = 0; iik < niks; iik++) {
      size_t ik = iks[iik];
      Eigen::Vector3d k =
          kPoints.getPointCoordinates(ik, Points::cartesianCoordinates);
      for (int iR = 0; iR < numElBravaisVectors; iR++) {
        double arg = k.dot(elBravaisVectors.col(iR));
        phases(ik, iR) = exp(-complexI * arg) / double(numKPoints);
      }
    }
    mpi->allReduceSum(&phases);

    for (auto iq : mpi->divideWorkIter(numQPoints)) {
#pragma omp parallel default(none)                                             \
    shared(iq, gFullTmp, phases, numElBravaisVectors, numKPoints, numModes,    \
           numWannier, gMixed)
      {
        Eigen::Tensor<std::complex<double>, 4> tmp(
            numWannier, numWannier, numModes, numElBravaisVectors);
        tmp.setZero();
#pragma omp for nowait
        for (int iR = 0; iR < numElBravaisVectors; iR++) {
          for (int ik = 0; ik < numKPoints; ik++) {
            for (int nu = 0; nu < numModes; nu++) {
              for (int j = 0; j < numWannier; j++) {
                for (int i = 0; i < numWannier; i++) {
                  tmp(i, j, nu, iR) +=
                      gFullTmp(i, j, nu, ik, int(iq)) * phases(ik, iR);
                }
              }
            }
          }
        }
#pragma omp critical
        for (int iR = 0; iR < numElBravaisVectors; iR++) {
          for (int nu = 0; nu < numModes; nu++) {
            for (int j = 0; j < numWannier; j++) {
              for (int i = 0; i < numWannier; i++) {
                gMixed(i, j, nu, iR, int(iq)) += tmp(i, j, nu, iR);
              }
            }
          }
        }
      }
    } // iq
    mpi->allReduceSum(&gMixed);
  }
  gFullTmp.reshape(zeros);

  if (mpi->mpiHead()) {
    std::cout << "Phonon rotation" << std::endl;
  }

  Eigen::Tensor<std::complex<double>, 5> gWannierTmp(
      numWannier, numWannier, numModes, numElBravaisVectors, numQPoints);
  gWannierTmp.setZero();
  {
    Eigen::Tensor<std::complex<double>, 3> uQM1s(numModes, numModes,
                                                 numQPoints);
    uQM1s.setZero();
    for (auto iq : mpi->divideWorkIter(numQPoints)) {
      Eigen::MatrixXcd uQ(numModes, numModes);
      for (int nu2 = 0; nu2 < numModes; nu2++) {
        for (int nu = 0; nu < numModes; nu++) {
          uQ(nu, nu2) = phEigenvectors(nu, nu2, int(iq));
        }
      }
      auto uQM1 = uQ.inverse();
      for (int nu2 = 0; nu2 < numModes; nu2++) {
        for (int nu = 0; nu < numModes; nu++) {
          uQM1s(nu, nu2, int(iq)) = uQM1(nu, nu2);
        }
      }
      // this isn't equal to the adjoint, due to mass renormalization
      // should be parallelized with OMP already
    }
    mpi->allReduceSum(&uQM1s);
    for (auto iq : mpi->divideWorkIter(numQPoints)) {
      for (int nu = 0; nu < numModes; nu++) {
        for (int nu2 = 0; nu2 < numModes; nu2++) {
#pragma omp parallel for collapse(3) default(none) shared(                     \
    numElBravaisVectors, numWannier, gMixed, uQM1s, iq, nu, nu2, gWannierTmp)
          for (int irE = 0; irE < numElBravaisVectors; irE++) {
            for (int i = 0; i < numWannier; i++) {
              for (int j = 0; j < numWannier; j++) {
                gWannierTmp(i, j, nu, irE, int(iq)) +=
                    gMixed(i, j, nu2, irE, int(iq)) * uQM1s(nu2, nu, int(iq));
              }
            }
          }
        }
      }
    }
    mpi->allReduceSum(&gWannierTmp);
  }
  gMixed.reshape(zeros);

  if (mpi->mpiHead()) {
    std::cout << "Phonon Fourier Transform" << std::endl;
  }

  Eigen::Tensor<std::complex<double>, 5> gWannier(numWannier, numWannier,
                                                  numModes, numPhBravaisVectors,
                                                  numElBravaisVectors);
  gWannier.setZero();
  {
    Eigen::MatrixXcd phases(numPhBravaisVectors, numQPoints);
    phases.setZero();
    std::vector<size_t> iqs = mpi->divideWorkIter(numQPoints);
    size_t niqs = iqs.size();
#pragma omp parallel for
    for (size_t iiq = 0; iiq < niqs; iiq++) {
      size_t iq = iqs[iiq];
      Eigen::Vector3d q =
          qPoints.getPointCoordinates(iq, Points::cartesianCoordinates);
      for (int irP = 0; irP < numPhBravaisVectors; irP++) {
        double arg = q.dot(phBravaisVectors.col(irP));
        phases(irP, iq) = exp(-complexI * arg) / double(numQPoints);
      }
    }
    mpi->allReduceSum(&phases);

    for (auto irE : mpi->divideWorkIter(numElBravaisVectors)) {
#pragma omp parallel default(none)                                             \
    shared(numQPoints, numPhBravaisVectors, numModes, numWannier, phases,      \
           gWannier, irE, gWannierTmp)
      {
        Eigen::Tensor<std::complex<double>, 4> tmp(
            numWannier, numWannier, numModes, numPhBravaisVectors);
        tmp.setZero();
#pragma omp for nowait
        for (int iq = 0; iq < numQPoints; iq++) {
          for (int irP = 0; irP < numPhBravaisVectors; irP++) {
            for (int nu = 0; nu < numModes; nu++) {
              for (int j = 0; j < numWannier; j++) {
                for (int i = 0; i < numWannier; i++) {
                  tmp(i, j, nu, irP) +=
                      phases(irP, iq) * gWannierTmp(i, j, nu, int(irE), iq);
                }
              }
            }
          }
        }
#pragma omp critical
        for (int irP = 0; irP < numPhBravaisVectors; irP++) {
          for (int nu = 0; nu < numModes; nu++) {
            for (int i = 0; i < numWannier; i++) {
              for (int j = 0; j < numWannier; j++) {
                gWannier(i, j, nu, irP, int(irE)) += tmp(i, j, nu, irP);
              }
            }
          }
        }
      }
    }
    mpi->allReduceSum(&gWannier);
  }
  gWannierTmp.reshape(zeros);

  if (mpi->mpiHead()) {
    std::cout << "Done Wannier-transform of g\n" << std::endl;
  }
  mpi->barrier();

  return gWannier;
}

Eigen::Tensor<std::complex<double>, 3>
ElPhQeToPhoebeApp::setupRotationMatrices(const std::string &wannierPrefix,
                                         Points &fullPoints) {
  std::string line;

  if (wannierPrefix.empty()) {
    Error("Must provide an input H0 file name");
  }

  std::string fileName = wannierPrefix + "_u.mat";

  // open input file
  std::ifstream infile(fileName);
  if (not infile.is_open()) {
    Error("U-matrix file not found");
  }

  // Title line
  std::getline(infile, line);

  int numPoints, numWannier, tmpI;
  infile >> numPoints >> numWannier >> tmpI;

  assert(numPoints == fullPoints.getNumPoints());

  Eigen::Tensor<std::complex<double>, 3> uMatrix(numWannier, numWannier,
                                                 numPoints);
  uMatrix.setZero();

  for (int ik = 0; ik < numPoints; ik++) {
    // empty line
    std::getline(infile, line);

    Eigen::Vector3d thisK; // vector in crystal coordinates
    infile >> thisK(0) >> thisK(1) >> thisK(2);

    int ikk = fullPoints.getIndex(thisK);

    double re, im;
    for (int j = 0; j < numWannier; j++) {
      for (int i = 0; i < numWannier; i++) {
        infile >> re >> im;
        uMatrix(i, j, ikk) = {re, im};
      }
    }
  }
  infile.close();

  // ---------------------------------------------------------------------

  // Now we get the disentanglement matrix

  std::string fileNameDis = wannierPrefix + "_u_dis.mat";

  // open input file
  std::ifstream infileDis(fileNameDis);
  if (not infileDis.is_open()) {
    // if the disentanglement file is not found
    // we assume there's no disentanglement and quit the function
    return uMatrix;
  } // else, we parse the file

  // Title line
  std::getline(infileDis, line);

  int numWannier2, numBands;
  infileDis >> numPoints >> numWannier2 >> numBands;

  assert(numWannier2 == numWannier);
  assert(numBands >= numWannier);

  Eigen::Tensor<std::complex<double>, 3> uMatrixDis(numBands, numWannier,
                                                    numPoints);
  uMatrixDis.setZero();

  for (int ik = 0; ik < numPoints; ik++) {
    // empty line
    std::getline(infileDis, line);

    double x, y, z;
    infileDis >> x >> y >> z;
    Eigen::Vector3d thisK;
    thisK << x, y, z; // vector in crystal coordinates

    int ikk = fullPoints.getIndex(thisK);

    double re, im;
    for (int j = 0; j < numWannier; j++) {
      for (int i = 0; i < numBands; i++) {
        infileDis >> re >> im;
        uMatrixDis(i, j, ikk) = {re, im};
      }
    }
  }
  infileDis.close();

  // Now I multiply the two rotation matrices

  Eigen::Tensor<std::complex<double>, 3> u(numBands, numWannier, numPoints);
  u.setZero();
  for (int ik = 0; ik < numPoints; ik++) {

    Eigen::MatrixXcd a(numBands, numWannier);
    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < numWannier; j++) {
        a(i, j) = uMatrixDis(i, j, ik);
      }
    }

    Eigen::MatrixXcd b(numWannier, numWannier);
    for (int i = 0; i < numWannier; i++) {
      for (int j = 0; j < numWannier; j++) {
        b(i, j) = uMatrix(i, j, ik);
      }
    }

    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < numWannier; j++) {
        for (int k = 0; k < numWannier; k++) {
          u(i, j, ik) += b(k, j) * a(i, k);
        }
      }
    }
  } // ik
  return u;
}

int ElPhQeToPhoebeApp::computeOffset(const Eigen::MatrixXd &energies,
                                     const std::string &wannierPrefix) {
  Eigen::VectorXd energiesQEAtZero = energies.col(0); // k = 0

  { // check the first point in Wannier90 is gamma
    std::string fileName = wannierPrefix + ".nnkp";
    std::ifstream infile(fileName);
    if (not infile.is_open()) {
      Error("Wannier90 *.nnkp file not found");
    }
    std::string line;
    for (int i = 0; i < 18; i++) {
      std::getline(infile, line); // skip the first 18 lines
    }
    double kx, ky, kz;
    infile >> kx >> ky >> kz;
    if (kx * kx + ky * ky + kz * kz > 1.0e-5) {
      Error("Expecting first coarse grid k-point in Wannier90 to be gamma");
    }
  }

  // read .eig file to get energies

  std::vector<double> energiesWannierAtZero;
  {
    std::string eigFileName = wannierPrefix + ".eig";
    std::ifstream eigenFile(eigFileName);
    if (not eigenFile.is_open()) {
      Error("Wannier90 *.eig file not found");
    }
    int ib, ik;
    double x;
    while (eigenFile >> ib >> ik >> x) {
      if (ik > 1) {
        break;
      }
      x /= energyRyToEv;
      energiesWannierAtZero.push_back(x);
    }
  }

  int numBandsWannier = int(energiesWannierAtZero.size());
  int numFull = int(energiesQEAtZero.size());

  // we find the offset by comparing the energy differences
  // the offset which minimizes energy differences is the chosen one
  int possibleValues = numFull - numBandsWannier;
  Eigen::VectorXd difference(possibleValues);
  difference.setZero();
  for (int i = 0; i < possibleValues; i++) {
    for (int ib = 0; ib < numBandsWannier; ib++) {
      difference(i) +=
          pow(energiesQEAtZero(ib + i) - energiesWannierAtZero[ib], 2);
    }
  }

  // offset = index of min difference
  int offset = -1;
  for (int i = 0; i < possibleValues; i++) {
    if (difference(i) == difference.minCoeff()) {
      offset = i + 1;
      break;
    }
  }

  if (possibleValues == 0) { // there cannot be an offset, then offset is 0
    offset = 0;
  }

  if (offset == -1) {
    Error("Bands offset not found");
  }

  return offset;
}

void ElPhQeToPhoebeApp::postProcessingWannier(
    Context &context, Crystal &crystal, PhononH0 &phononH0, Points &kPoints,
    Points &qPoints, int numQEBands, int numModes, int numIrrQPoints,
    int numElectrons, int numSpin, const Eigen::MatrixXd &energies,
    const Eigen::MatrixXd &kGridFull, const Eigen::Vector3i &kMesh,
    const Eigen::Vector3i &qMesh, bool runTests) {
  if (mpi->mpiHead()) {
    std::cout << "Starting Wannier post-processing\n" << std::endl;
  }

  std::string wannierPrefix = context.getWannier90Prefix();

  auto t2 = QEParser::parseElHarmonicWannier(context, &crystal);
  auto electronH0 = std::get<1>(t2);
  int numWannier = int(electronH0.getNumBands());

  //----------------------------------------------------------------------------
  // read Wannier90 rotation matrices

  Eigen::Tensor<std::complex<double>, 3> uMatrices;
  // uMatrices has size (numBands, numWannier, numKPoints)
  uMatrices = setupRotationMatrices(wannierPrefix, kPoints);
  auto numBands = int(uMatrices.dimension(0));  // number of entangled bands
  assert(numWannier == uMatrices.dimension(1)); // number of entangled bands

  //----------------------------------------------------------------------------
  // figure out the number of occupied Wannier states

  int bandsOffset = computeOffset(energies, wannierPrefix);
  // Note: Quantum-ESPRESSO defines numElectrons as the number of occupied
  // states in the valence manifold (it doesn't tell us about core electrons,
  // which we probably can only infer from the atomic numbers)
  if (numSpin == 2) {
    Error("Spin is not currently supported");
  }
  // the factor 2 is the spin degeneracy factor. Change if spin support added
  int numFilledWannier = numElectrons - bandsOffset * 2;
  // note how we only allow numFilledWannier to be an integer
  // it can be an even or odd number, so be careful if dividing it by 2

  //----------------------------------------------------------------------------

  // Find the lattice vectors for the Fourier transforms

  auto t3 = crystal.buildWignerSeitzVectors(kMesh);
  Eigen::MatrixXd elBravaisVectors = std::get<0>(t3);
  Eigen::VectorXd elDegeneracies = std::get<1>(t3);

  auto t4 = crystal.buildWignerSeitzVectors(qMesh);
  Eigen::MatrixXd phBravaisVectors = std::get<0>(t4);
  Eigen::VectorXd phDegeneracies = std::get<1>(t4);

  //----------------------------------------------------------------------------

  Eigen::Tensor<std::complex<double>, 5> gWannier;
  if (!context.getDistributedElPhCoupling()) {
    // read coupling from file
    auto t5 = readGFromQEFile(context, numModes, numBands, numWannier, kPoints,
                              qPoints, kGridFull, numIrrQPoints, numQEBands,
                              energies);
    auto gFull = std::get<0>(t5);          // (nBands,nBands,nModes,numK,numQ)
    auto phEigenvectors = std::get<1>(t5); // (numModes,numModes,numQPoints)
    auto phEnergies = std::get<2>(t5);     // (numModes,numQPoints)
    gWannier =
        blochToWannier(elBravaisVectors, phBravaisVectors, gFull, uMatrices,
                       phEigenvectors, kPoints, qPoints, crystal, phononH0);
  } else {
    gWannier =
        BlochToWannierEfficient(context, energies, kGridFull, numIrrQPoints,
                                numQEBands, elBravaisVectors, phBravaisVectors,
                                uMatrices, kPoints, qPoints, crystal, phononH0);
  }
  writeWannierCoupling(context, gWannier, numFilledWannier, numSpin, numModes,
                       numWannier, phDegeneracies, elDegeneracies,
                       phBravaisVectors, elBravaisVectors, qMesh, kMesh);

  //--------------------------------------------------------------------------

  if (runTests) {
    auto t6 = readGFromQEFile(context, numModes, numBands, numWannier, kPoints,
                              qPoints, kGridFull, numIrrQPoints, numQEBands,
                              energies);
    auto gFull = std::get<0>(t6);          // (nBands,nBands,nModes,numK,numQ)
    auto phEigenvectors = std::get<1>(t6); // (numModes,numModes,numQPoints)
    auto phEnergies = std::get<2>(t6);     // (numModes,numQPoints)

    testElectronicTransform(kPoints, wannierPrefix, elBravaisVectors, uMatrices,
                            elDegeneracies, electronH0);

    testPhononTransform(crystal, phononH0, qPoints, phEigenvectors,
                        phBravaisVectors, phDegeneracies, phEnergies);

    testBackTransform(context, phononH0, kPoints, qPoints, electronH0, crystal,
                      gFull);
  }
}
