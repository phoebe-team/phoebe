#include "bandstructure.h"
#include "ifc3_parser.h"
#include "ph_scattering.h"
#include "points.h"
#include "qe_input_parser.h"
#include <fstream>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <complex>

TEST(FullBandStructureTest, BandStructureStorage) {
  // in this test, we check whether we are storing data inside the
  // band structure object correctly and consistently

  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  // Number of atoms
  int numAtoms = crystal.getNumAtoms();
  // Number of bands
  int numBands = 3 * numAtoms;

  //------------end setup-----------//

  Eigen::Vector3i qMesh;
  qMesh << 7, 7, 7;
  Points points(crystal, qMesh);

  bool withVelocities = true;
  bool withEigenvectors = true;
  FullBandStructure bandStructure =
      phononH0.populate(points, withVelocities, withEigenvectors);

  int ik = 7;
  auto ikIndex = WavevectorIndex(ik);
  Point point = bandStructure.getPoint(ik);
  // set up phonon band quantities from direct diag
  auto tup1 = phononH0.diagonalize(point);
  auto ensT = std::get<0>(tup1);
  auto eigenVectorsT = std::get<1>(tup1);
  auto velocitiesT = phononH0.diagonalizeVelocity(point);

  auto ens = bandStructure.getEnergies(ikIndex);
  Eigen::Tensor<std::complex<double>, 3> eigenVectors =
      bandStructure.getPhEigenvectors(ikIndex);
  auto velocities = bandStructure.getVelocities(ikIndex);

  // now we check the difference between direct diag and
  // bandstructure calculation
  double x1 = (ens - ensT).norm();
  ASSERT_NEAR(x1, 0.,1e-14);

  {
    double c1 = 0.;
    double norm = 0.0, normT=0.0;
    for (int ib1 = 0; ib1 < numBands; ib1++) {
      for (int ib2 = 0; ib2 < numBands; ib2++) {
        for (int ic = 0; ic < 3; ic++) {
          c1 += std::norm(velocitiesT(ib1, ib2, ic) - velocities(ib1, ib2, ic));
          norm += std::norm(velocities(ib1, ib2, ic));
          normT += std::norm(velocitiesT(ib1, ib2, ic));
        }
      }
    }
#ifdef KOKKOS_ENABLE_CUDA
    ASSERT_NEAR(std::abs(norm-normT)/norm, 0.0, 1e-7);
#else
    ASSERT_NEAR(c1, 0., 1.e-16);
#endif
  }

  {
    double norm = 0.0, normT=0.0;
    double c2 = 0.;
    for (int i = 0; i < numBands; i++) {
      auto tup2 = decompress2Indices(i, numAtoms, 3);
      auto iat = std::get<0>(tup2);
      auto ic = std::get<1>(tup2);
      for (int j = 0; j < numBands; j++) {
        c2 += std::norm(eigenVectorsT(i, j) - eigenVectors(ic, iat, j));
          norm += std::norm(eigenVectors(ic, iat, j));
          normT += std::norm(eigenVectorsT(i, j));
      }
    }
#ifdef KOKKOS_ENABLE_CUDA
    ASSERT_NEAR(std::abs(norm-normT)/norm, 0.0, 1e-15);
#else
    ASSERT_NEAR(c2, 0., 1.e-16);
#endif
  }

  // now we check that we get the same eigenvectors in the two different
  // shapes
  auto k = point.getCoordinates(Points::cartesianCoordinates);
  auto tup2 = phononH0.diagonalizeFromCoordinates(k);
  auto ensC = std::get<0>(tup2);
  auto eigenVectorsC = std::get<1>(tup2);
  x1 = (ens - ensC).norm();
  ASSERT_NEAR(x1/ens.norm(), 0.0, 1e-14);

  Eigen::MatrixXcd eigenVectorsT2 = bandStructure.getEigenvectors(ikIndex);
  {
    double c2 = 0.;
    double norm = 0.0, normT=0.0;
    for (int ib1 = 0; ib1 < numBands; ib1++) {
      for (int ib2 = 0; ib2 < numBands; ib2++) {
        c2 += std::norm(eigenVectorsT2(ib1, ib2) - eigenVectorsC(ib1, ib2));
        norm += std::norm(eigenVectorsT2(ib1, ib2));
        normT += std::norm(eigenVectorsC(ib1, ib2));
      }
    }
#ifdef KOKKOS_ENABLE_CUDA
    ASSERT_NEAR(std::abs(norm-normT)/norm, 0.0, 1e-15);
#else
    ASSERT_NEAR(c2, 0., 1e-16);
#endif
  }

  // we check what happens if we set eigenvectors as a matrix
  Eigen::MatrixXcd eigenVectorsC2 = bandStructure.getEigenvectors(ikIndex);
  {
    double c2 = 0.;
    double norm = 0.0, normT=0.0;
    for (int ib1 = 0; ib1 < numBands; ib1++) {
      for (int ib2 = 0; ib2 < numBands; ib2++) {
        c2 += std::norm(eigenVectorsC2(ib1, ib2) - eigenVectorsC(ib1, ib2));
        norm += std::norm(eigenVectorsC2(ib1, ib2));
        normT += std::norm(eigenVectorsC(ib1, ib2));
      }
    }
#ifdef KOKKOS_ENABLE_CUDA
    ASSERT_NEAR(std::abs(norm-normT)/norm, 0.0, 1e-15);
#else
    ASSERT_NEAR(c2, 0., 1e-16);
#endif
  }

  // make a comparison between tensor and matrix
  {
    double c2 = 0.;
    double norm = 0.0, normT=0.0;
    for (int iBand = 0; iBand < numBands; iBand++) {
      for (int iat = 0; iat < numAtoms; iat++) {
        for (int iPol = 0; iPol < 3; iPol++) {
          auto ind = compress2Indices(iat, iPol, numAtoms, 3);
          c2 += std::norm(eigenVectors(iPol, iat, iBand) - eigenVectorsC(ind, iBand));
          norm += std::norm(eigenVectors(iPol, iat, iBand));
          normT += std::norm(eigenVectorsC(ind, iBand));
        }
      }
    }
#ifdef KOKKOS_ENABLE_CUDA
    ASSERT_NEAR(std::abs(norm-normT)/norm, 0.0, 1e-15);
#else
    ASSERT_NEAR(c2, 0., 1e-16);
#endif
  }
}
