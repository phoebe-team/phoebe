#include "points.h"
#include "qe_input_parser.h"
#include "gtest/gtest.h"
#include "test_utils.h"

/** Here I estimate the mass at the top of the valence band of silicon
 */
TEST(Kokkos, Wannier1) {
  Context context;
  context.setElectronH0Name("../test/data/666_si_tb.dat");

  // setup crystal input
  Eigen::MatrixXd atomicPositions(2, 3);
  atomicPositions.row(0) << 0., 0., 0.;
  atomicPositions.row(1) << 1.34940, 1.34940, 1.34940;
  Eigen::VectorXi atomicSpecies(2);
  atomicSpecies(0) = 0;
  atomicSpecies(1) = 0;
  std::vector<std::string> speciesNames;
  speciesNames.emplace_back("Si");
  context.setInputAtomicPositions(atomicPositions);
  context.setInputAtomicSpecies(atomicSpecies);
  context.setInputSpeciesNames(speciesNames);

  // read electron hamiltonian
  auto tup = QEParser::parseElHarmonicWannier(context);
  auto crystal = std::get<0>(tup);
  auto electronH0 = std::get<1>(tup);

  int numBands = electronH0.getNumBands();

  // pick a point to be diagonalized
  // Gamma is a special case, where we set v=0
  Eigen::Vector3d k1 = {0., 0., 0.};

  // first, we diagonalize on the CPU
  auto tup1 = electronH0.diagonalizeFromCoordinates(k1);
  auto ens1 = std::get<0>(tup1);
  auto eigenvectors1 = std::get<1>(tup1);

  Eigen::Tensor<std::complex<double>, 3> velocity1 =
      electronH0.diagonalizeVelocityFromCoordinates(k1);

  // now we add the code on Kokkos
  Eigen::VectorXd ens2(electronH0.getNumBands());
  Eigen::Tensor<std::complex<double>, 3> velocity2(numBands, numBands, 3);
  Eigen::MatrixXcd eigenvectors2(numBands, numBands);
  {
    int numK = 1;

    // we need to copy the wavevectors to the GPU
    DoubleView2D q3Cs_d("q3", numK, 3);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_d);

#pragma omp parallel for
    for (int ik = 0; ik < numK; ik++) {
      for (int i = 0; i < 3; i++) {
        q3Cs_h(ik, i) = k1(i);
      }
    }
    Kokkos::deep_copy(q3Cs_d, q3Cs_h);

    auto t2 = electronH0.kokkosBatchedDiagonalizeFromCoordinates(q3Cs_d);
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

    auto t3 = electronH0.kokkosBatchedDiagonalizeWithVelocities(q3Cs_d);
    ComplexView4D velocity_d = std::get<2>(t3);
    auto velocity_h = Kokkos::create_mirror_view(velocity_d);
    Kokkos::deep_copy(velocity_h, velocity_d);
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          velocity2(ib1, ib2, i) = velocity_h(0, ib1, ib2, i);
        }
      }
    }
  }

  // the two masses should be similar
  ASSERT_NEAR((ens1 - ens2).norm(), 0., 0.00001);

  //ASSERT_NEAR((eigenvectors1 - eigenvectors2).norm(), 0., 0.0001);
  EXPECT_NEAR((mat_vec_mat_adj(eigenvectors1, ens1, numBands)
               - mat_vec_mat_adj(eigenvectors2, ens2, numBands)).norm(), 0, 1e-4);

  {
    double norm = 0.;
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          norm += abs(velocity1(ib1, ib2, i) - velocity2(ib1, ib2, i));
        }
      }
    }
    ASSERT_NEAR(norm, 0., 0.00001);
  }

  {
    // make sure it's 0
    double norm = 0.;
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          norm += abs(velocity2(ib1, ib2, i));
        }
      }
    }
    ASSERT_NEAR(norm, 0., 0.00001);
  }

}

TEST(Kokkos, Wannier2) {
  Context context;
  context.setElectronH0Name("../test/data/666_si_tb.dat");

  // setup crystal input
  Eigen::MatrixXd atomicPositions(2, 3);
  atomicPositions.row(0) << 0., 0., 0.;
  atomicPositions.row(1) << 1.34940, 1.34940, 1.34940;
  Eigen::VectorXi atomicSpecies(2);
  atomicSpecies(0) = 0;
  atomicSpecies(1) = 0;
  std::vector<std::string> speciesNames;
  speciesNames.emplace_back("Si");
  context.setInputAtomicPositions(atomicPositions);
  context.setInputAtomicSpecies(atomicSpecies);
  context.setInputSpeciesNames(speciesNames);

  // read electron hamiltonian
  auto tup = QEParser::parseElHarmonicWannier(context);
  auto crystal = std::get<0>(tup);
  auto electronH0 = std::get<1>(tup);

  int numBands = electronH0.getNumBands();

  // pick a point to be diagonalized
  Eigen::Vector3d k1 = {0.1,0.23,-0.17};

  // first, we diagonalize on the CPU
  auto tup1 = electronH0.diagonalizeFromCoordinates(k1);
  auto ens1 = std::get<0>(tup1);
  auto eigenvectors1 = std::get<1>(tup1);



  Eigen::Tensor<std::complex<double>, 3> velocity1 =
      electronH0.diagonalizeVelocityFromCoordinates(k1);

  // now we add the code on Kokkos
  Eigen::VectorXd ens2(electronH0.getNumBands());
  Eigen::Tensor<std::complex<double>, 3> velocity2(numBands, numBands, 3);
  Eigen::MatrixXcd eigenvectors2(numBands, numBands);
  {
    int numK = 1;

    // we need to copy the wavevectors to the GPU
    DoubleView2D q3Cs_d("q3", numK, 3);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_d);

#pragma omp parallel for
    for (int ik = 0; ik < numK; ik++) {
      for (int i = 0; i < 3; i++) {
        q3Cs_h(ik, i) = k1(i);
      }
    }
    Kokkos::deep_copy(q3Cs_d, q3Cs_h);

    auto t2 = electronH0.kokkosBatchedDiagonalizeFromCoordinates(q3Cs_d);
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

    auto t3 = electronH0.kokkosBatchedDiagonalizeWithVelocities(q3Cs_d);
    ComplexView4D velocity_d = std::get<2>(t3);
    auto velocity_h = Kokkos::create_mirror_view(velocity_d);
    Kokkos::deep_copy(velocity_h, velocity_d);
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          velocity2(ib1, ib2, i) = velocity_h(0, ib1, ib2, i);
        }
      }
    }
  }

  // the two masses should be similar
  ASSERT_NEAR((ens1 - ens2).norm()/ens1.norm(), 0., 1e-12);

  //ASSERT_NEAR((eigenvectors1 - eigenvectors2).norm(), 0., 0.0001);
  auto evres1 = mat_vec_mat_adj(eigenvectors1, ens1, numBands);
  auto evres2 = mat_vec_mat_adj(eigenvectors2, ens2, numBands);
  EXPECT_NEAR((evres1 - evres2).norm()/evres2.norm(), 0, 1e-14);

  // TODO in WTE, both diagonals and off-diagonals matter
  // should norm run over all elements, not just diagonal?
  // see PhononH0 test for example
  double norm = 0., norm1=0.;
  for (int ib1 = 0; ib1 < numBands; ++ib1) {
    for (int ib2 = 0; ib2 < numBands; ++ib2) {
      for (int i = 0; i < 3; ++i) {
        norm += abs(velocity1(ib1, ib1, i).real() - velocity2(ib1, ib1, i).real());
        norm1 += abs(velocity1(ib1, ib1, i).real());
      }
    }
  }
  ASSERT_NEAR(norm/norm1, 0., 0.00001);
}

TEST(Kokkos, Wannier3) {
  Context context;
  context.setElectronH0Name("../test/data/666_si_tb.dat");

  // setup crystal input
  Eigen::MatrixXd atomicPositions(2, 3);
  atomicPositions.row(0) << 0., 0., 0.;
  atomicPositions.row(1) << 1.34940, 1.34940, 1.34940;
  Eigen::VectorXi atomicSpecies(2);
  atomicSpecies(0) = 0;
  atomicSpecies(1) = 0;
  std::vector<std::string> speciesNames;
  speciesNames.emplace_back("Si");
  context.setInputAtomicPositions(atomicPositions);
  context.setInputAtomicSpecies(atomicSpecies);
  context.setInputSpeciesNames(speciesNames);

  // read electron hamiltonian
  auto tup = QEParser::parseElHarmonicWannier(context);
  auto crystal = std::get<0>(tup);
  auto electronH0 = std::get<1>(tup);

  int numBands = electronH0.getNumBands();

  // pick a point to be diagonalized
  // Eigen::Vector3d k1 = {0.1,0.23,-0.17};
  // auto b1 = crystal.getReciprocalUnitCell().col(0);
  Eigen::Vector3d k1 = crystal.getReciprocalUnitCell().col(0) / 2.;

  // first, we diagonalize on the CPU
  auto tup1 = electronH0.diagonalizeFromCoordinates(k1);
  auto ens1 = std::get<0>(tup1);
  auto eigenvectors1 = std::get<1>(tup1);

  Eigen::Tensor<std::complex<double>, 3> velocity1 =
      electronH0.diagonalizeVelocityFromCoordinates(k1);

  // now we add the code on Kokkos
  Eigen::VectorXd ens2(electronH0.getNumBands());
  Eigen::Tensor<std::complex<double>, 3> velocity2(numBands, numBands, 3);
  Eigen::MatrixXcd eigenvectors2(numBands, numBands);
  {
    int numK = 1;

    // we need to copy the wavevectors to the GPU
    DoubleView2D q3Cs_d("q3", numK, 3);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_d);

#pragma omp parallel for
    for (int ik = 0; ik < numK; ik++) {
      for (int i = 0; i < 3; i++) {
        q3Cs_h(ik, i) = k1(i);
      }
    }
    Kokkos::deep_copy(q3Cs_d, q3Cs_h);

    auto t2 = electronH0.kokkosBatchedDiagonalizeFromCoordinates(q3Cs_d);
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

    auto t3 = electronH0.kokkosBatchedDiagonalizeWithVelocities(q3Cs_d);
    ComplexView4D velocity_d = std::get<2>(t3);
    auto velocity_h = Kokkos::create_mirror_view(velocity_d);
    Kokkos::deep_copy(velocity_h, velocity_d);
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          velocity2(ib1, ib2, i) = velocity_h(0, ib1, ib2, i);
        }
      }
    }
  }

  // the two masses should be similar
  ASSERT_NEAR((ens1 - ens2).norm(), 0., 0.00001);

  //ASSERT_NEAR((eigenvectors1 - eigenvectors2).norm(), 0., 0.0001);
  auto evres1 = mat_vec_mat_adj(eigenvectors1, ens1, numBands);
  auto evres2 = mat_vec_mat_adj(eigenvectors2, ens2, numBands);
  EXPECT_NEAR((evres1 - evres2).norm()/evres2.norm(), 0, 1e-14);

  // TODO in WTE, both diagonals and off-diagonals matter
  // should norm run over all elements, not just diagonal?
  // see PhononH0 test for example
  {
    double norm = 0.;
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          norm += abs(velocity1(ib1, ib1, i).real() - velocity2(ib1, ib1, i).real());
        }
      }
    }
    ASSERT_NEAR(norm, 0., 0.00001);
  }
}

TEST(Kokkos, PhononH0) {
  Context context;

  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setPhFC3FileName("../test/data/FORCE_CONSTANTS_3RD");
  context.setSumRuleFC2("simple");

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  int numBands = phononH0.getNumBands();

  // pick a point to be diagonalized
  Eigen::Vector3d q1 = {0.1,0.23,-0.17};

  // first, we diagonalize on the CPU
  auto tup1 = phononH0.diagonalizeFromCoordinates(q1, true);
  auto ens1 = std::get<0>(tup1);
  auto eigenvectors1 = std::get<1>(tup1);



  Eigen::Tensor<std::complex<double>, 3> velocity1 =
      phononH0.diagonalizeVelocityFromCoordinates(q1);

  // now we add the code on Kokkos
  Eigen::VectorXd ens2(numBands);
  Eigen::Tensor<std::complex<double>, 3> velocity2(numBands, numBands, 3);
  Eigen::MatrixXcd eigenvectors2(numBands, numBands);
  {
    int numK = 1;

    // we need to copy the wavevectors to the GPU
    DoubleView2D q3Cs_d("q3", numK, 3);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_d);

#pragma omp parallel for
    for (int ik = 0; ik < numK; ik++) {
      for (int i = 0; i < 3; i++) {
        q3Cs_h(ik, i) = q1(i);
      }
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
          velocity2(ib1, ib2, i) = velocity_h(0, ib1, ib2, i);
        }
      }
    }
  }

  // the two masses should be similar
  EXPECT_NEAR((ens1 - ens2).norm(), 0., 0.00001);

  //EXPECT_NEAR((eigenvectors1 - eigenvectors2).norm(), 0., 0.0001);
  auto evres1 = mat_vec_mat_adj(eigenvectors1, ens1, numBands);
  auto evres2 = mat_vec_mat_adj(eigenvectors2, ens2, numBands);
  EXPECT_NEAR((evres1 - evres2).norm()/evres2.norm(), 0, 1e-14);

  /**
   * Check that CPU and Kokkos velocities match
   * Note that I'm summing norms of differences of complex numbers
   * So even a phase mismatch between velocities will make the test fail
   * I am not sure if the phase here is physically significant,
   * but currently the test passes as written
   */
  double norm = 0.;
  for (int ib1 = 0; ib1 < numBands; ++ib1) {
    for (int ib2 = 0; ib2 < numBands; ++ib2) {
      for (int i = 0; i < 3; ++i) {
        // std::cout << abs(velocity1(ib1, ib2, i)) << " " << abs(velocity2(ib1, ib2, i)) << std::endl;
        norm += abs(velocity1(ib1, ib2, i) - velocity2(ib1, ib2, i));
        // norm += abs(abs(velocity1(ib1, ib2, i)) - abs(velocity2(ib1, ib2, i)));
      }
      // std::cout << std::endl;
    }
    // std::cout << std::endl;
  }
  // std::cout << norm << std::endl;
  EXPECT_NEAR(norm, 0.0, 1e-9);
}
