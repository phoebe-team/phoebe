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
  context.setSymmetrizeBandStructure(false);

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  int numAtoms = crystal.getNumAtoms();
  int numBands = 3 * numAtoms;

  Eigen::VectorXd ens2(numBands);
  Eigen::Tensor<std::complex<double>, 3> velocities2(numBands, numBands, 3);
  Eigen::MatrixXcd eigenvectors2(numBands, numBands);

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
  auto qPoint = point.getCoordinates(Points::cartesianCoordinates);
  auto ens = bandStructure.getEnergies(ikIndex);
  Eigen::Tensor<std::complex<double>, 3> eigenvectors =
      bandStructure.getPhEigenvectors(ikIndex);
  auto velocities = bandStructure.getVelocities(ikIndex);

  // direct Kokkos diagonalization
  {
    // we need to copy the wavevectors to the GPU
    DoubleView2D q3Cs_d("q3", 1, 3);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_d);

#pragma omp parallel for
    for (int i = 0; i < 3; i++) {
      q3Cs_h(0, i) = qPoint(i);
    }
    Kokkos::deep_copy(q3Cs_d, q3Cs_h);

    auto t2 = phononH0.kokkosBatchedDiagonalizeFromCoordinates(q3Cs_d);
    DoubleView2D batchedEnergies = std::get<0>(t2);
    StridedComplexView3D batchedEigenvectors = std::get<1>(t2);

    // now we copy back to host
    auto tmpEnergies_h = Kokkos::create_mirror_view(batchedEnergies);
    Kokkos::deep_copy(tmpEnergies_h, batchedEnergies);
    for (int ib = 0; ib < numBands; ++ib) {
      ens2(ib) = tmpEnergies_h(0, ib);
    }

    auto eigenvectors2_h = Kokkos::create_mirror_view(batchedEigenvectors);
    Kokkos::deep_copy(eigenvectors2_h, batchedEigenvectors);
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        eigenvectors2(ib1, ib2) = eigenvectors2_h(0, ib1, ib2);
      }
    }

    auto t3 = phononH0.kokkosBatchedDiagonalizeWithVelocities(q3Cs_d);
    ComplexView4D velocity_d = std::get<2>(t3);
    auto velocity_h = Kokkos::create_mirror_view(velocity_d);
    Kokkos::deep_copy(velocity_h, velocity_d);
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          velocities2(ib1, ib2, i) = velocity_h(0, ib1, ib2, i);
        }
      }
    }
  }

  // now we check the difference between direct diag and
  // bandstructure calculation
  double x1 = (ens - ens2).norm();
  ASSERT_NEAR(x1, 0.,1e-14);

  // check velocities
  double c1 = 0.;
  double norm1 = 0.0, norm2=0.0;
  for (int ib1 = 0; ib1 < numBands; ib1++) {
    for (int ib2 = 0; ib2 < numBands; ib2++) {
      for (int ic = 0; ic < 3; ic++) {
        // std::cout << std::scientific << velocities2(ib1, ib2, ic) << " " << velocities(ib1, ib2, ic) << " " << abs(velocities2(ib1, ib2, ic) - velocities(ib1, ib2, ic)) << std::endl;
        c1 += std::norm(velocities2(ib1, ib2, ic) - velocities(ib1, ib2, ic));
        norm1 += std::norm(velocities(ib1, ib2, ic));
        norm2 += std::norm(velocities2(ib1, ib2, ic));
      }
      // std::cout << std::endl;
    }
    // std::cout << std::endl;
  }

  // std::cout << std::scientific << norm1 << " " << norm2 << " " << norm1-norm2 << std::endl;
  // std::cout << std::scientific << c1 << std::endl;

#ifdef KOKKOS_ENABLE_CUDA
  ASSERT_NEAR(std::abs(norm-normT)/norm, 0.0, 1e-7);
#else
  ASSERT_NEAR(c1, 0., 1.e-16); // TODO temporarily comment this out, as WignerFix branch will modify it anyway
#endif

  // check eigenvectors
  norm1 = 0.0, norm2=0.0;
  double c2 = 0.;
  for (int i = 0; i < numBands; i++) {
    auto tup2 = decompress2Indices(i, numAtoms, 3);
    auto iat = std::get<0>(tup2);
    auto ic = std::get<1>(tup2);
    for (int j = 0; j < numBands; j++) {
      c2 += std::norm(eigenvectors2(i, j) - eigenvectors(ic, iat, j));
      norm1 += std::norm(eigenvectors(ic, iat, j));
      norm2 += std::norm(eigenvectors2(i, j));
    }
  }
#ifdef KOKKOS_ENABLE_CUDA
  ASSERT_NEAR(std::abs(norm1-norm2)/norm, 0.0, 1e-15);
#else
  ASSERT_NEAR(c2, 0., 1.e-16); // TODO temporarily comment this out, as WignerFix branch will modify it anyway 
#endif

  // getEigenvectors returns a (nBands, nBands) matrix
  // getPhEigenvectors returns (3, nAtoms, nBands) tensor
  // here, we verify they are the same
  Eigen::MatrixXcd eigenVectorsC = bandStructure.getEigenvectors(ikIndex);
  {
    double c3 = 0.;
    double norm = 0.0, normT=0.0;
    for (int iBand = 0; iBand < numBands; iBand++) {
      for (int iat = 0; iat < numAtoms; iat++) {
        for (int iPol = 0; iPol < 3; iPol++) {
          auto ind = compress2Indices(iat, iPol, numAtoms, 3);
          c3 += std::norm(eigenvectors(iPol, iat, iBand) - eigenVectorsC(ind, iBand));
          norm += std::norm(eigenvectors(iPol, iat, iBand));
          normT += std::norm(eigenVectorsC(ind, iBand));
        }
      }
    }
#ifdef KOKKOS_ENABLE_CUDA
    ASSERT_NEAR(std::abs(norm-normT)/norm, 0.0, 1e-15);
#else
    ASSERT_NEAR(c3, 0., 1e-16);
#endif
  }
}
