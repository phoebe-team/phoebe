#include "gtest/gtest.h"
#include <iomanip>
#include "points.h"
#include "constants.h"
#include "phonopy_input_parser.h"

#include <fstream>
#include <nlohmann/json.hpp>

// check that the CPU velocity operator matches reference Si data
TEST (WTE, CPUVelocityOperator) {
  // read the JSON reference file
  std::ifstream f("./data/silicon_velocity_operator_reference.json");
  json data = json::parse(f);
  std::vector<double> qPointJson = data["qPoint"];
  std::vector<double> energiesJson = data["energies"];
  std::vector<std::vector<double>> sqmod_vop_x_json = data["|v_x|^2"];
  std::vector<std::vector<double>> sqmod_vop_y_json = data["|v_y|^2"];
  std::vector<std::vector<double>> sqmod_vop_z_json = data["|v_z|^2"];

  ASSERT_STREQ(data["qPointUnit"], "crystal") << "`qPoint` must be in crystal units and `qPointUnit` must be specified";
  ASSERT_STREQ(data["energiesUnit"], "meV") << "`energies` must be in meV and `energiesUnit` must be specified";
  ASSERT_STREQ(data["velocitySqModUnit"], "(m/s)^2") << "`|v_i|^2` must be in (m/s)^2 and `velocitySqModUnit` must be specified";

  Context context;
  context.setPhFC2FileName("./data/phono3py/fc2.hdf5");
  context.setPhonopyDispFileName("./data/phono3py/phono3py_disp.yaml");

  auto [crystal,phononH0] = PhonopyParser::parsePhHarmonic(context);

  // check the velocity operator at given point
  Eigen::Vector3d qPoint(qPointJson);
  // std::cout << "qPoint [crystal coords]: " << qPoint.transpose() << std::endl;
  qPoint = crystal.crystalToCartesian(qPoint);
  
  auto [energies, eigenvecs] = phononH0.diagonalizeFromCoordinates(qPoint);
  auto v = phononH0.diagonalizeVelocityFromCoordinates(qPoint);

  // take out the velocity operator, both on- and off-diagonal els
  // only checking magnitude and not phase of complex entries
  // hence calculate square modulus of each el, and convert to SI
  int numBands = energies.size();
  std::vector<Eigen::MatrixXd> sqmod_vop(3);
  for (int ib = 0; ib < numBands; ib++) {
    for (int i : {0, 1, 2}) {
      sqmod_vop[i].resize(numBands,numBands);
      for (int ib2 = 0; ib2 < numBands; ib2++) {
        sqmod_vop[i](ib,ib2) = (v(ib,ib2,i) * v(ib2,ib,i)).real() * pow(velocityRyToSi, 2);
      }
    }
  }

// check energies are equal
for (int unsigned ib = 0; ib < numBands; ib++) {
  // std::cout << "Energies [meV]: ";
  // std::cout << std::scientific << std::setprecision(4) << energies[i] << "\n" <<std::endl;
  EXPECT_NEAR(abs((energies[ib]-energiesJson[ib])/energiesJson[ib]), 0, 0.001);
}

// check velocities are equal
for (int unsigned ib = 0; ib < numBands; ib++) {
  // std::cout << "Velocity square modulus [(m/s)^2]: ";
  // std::cout << std::scientific << std::setprecision(4) << sqmod_vop[i] << "\n" <<std::endl;
  for (int unsigned ib2 = 0; ib2 < numBands; ib++) {
    EXPECT_NEAR(abs((sqmod_vop[0](ib,ib2) - sqmod_vop_x_json[ib][ib2])/sqmod_vop_x_json[ib][ib2]), 0, 0.001);
    EXPECT_NEAR(abs((sqmod_vop[1](ib,ib2) - sqmod_vop_y_json[ib][ib2])/sqmod_vop_y_json[ib][ib2]), 0, 0.001);
    EXPECT_NEAR(abs((sqmod_vop[2](ib,ib2) - sqmod_vop_z_json[ib][ib2])/sqmod_vop_z_json[ib][ib2]), 0, 0.001);
  }
}

}

// check that the Kokkos velocity operator matches reference Si data
TEST (WTE, KokkosVelocityOperator) {
  // read the JSON reference file
  std::ifstream f("./data/silicon_velocity_operator_reference.json");
  json data = json::parse(f);
  std::vector<double> qPointJson = data["qPoint"];
  std::vector<double> energiesJson = data["energies"];
  std::vector<std::vector<double>> sqmod_vop_x_json = data["|v_x|^2"];
  std::vector<std::vector<double>> sqmod_vop_y_json = data["|v_y|^2"];
  std::vector<std::vector<double>> sqmod_vop_z_json = data["|v_z|^2"];

  ASSERT_STREQ(data["qPointUnit"], "crystal") << "`qPoint` must be in crystal units and `qPointUnit` must be specified";
  ASSERT_STREQ(data["energiesUnit"], "meV") << "`energies` must be in meV and `energiesUnit` must be specified";
  ASSERT_STREQ(data["velocitySqModUnit"], "(m/s)^2") << "`|v_i|^2` must be in (m/s)^2 and `velocitySqModUnit` must be specified";

  Context context;
  context.setPhFC2FileName("./data/phono3py/fc2.hdf5");
  context.setPhonopyDispFileName("./data/phono3py/phono3py_disp.yaml");

  auto [crystal, phononH0] = PhonopyParser::parsePhHarmonic(context);

  // check the velocity operator at given point
  Eigen::Vector3d qPoint(qPointJson);
  // std::cout << "qPoint [crystal coords]: " << qPoint.transpose() << std::endl;
  qPoint = crystal.crystalToCartesian(qPoint);
  
  Eigen::VectorXd energies(numBands);
  Eigen::Tensor<std::complex<double>, 3> velocity(numBands, numBands, 3);
  Eigen::MatrixXcd eigenvectors(numBands, numBands);
  {
    int numK = 1;

    // we need to copy the wavevectors to the GPU
    DoubleView2D qPoint_d("qPoint", numK, 3);
    auto qPoint_h = Kokkos::create_mirror_view(qPoint_d);

#pragma omp parallel for
    for (int ik = 0; ik < numK; ik++) {
      for (int i = 0; i < 3; i++) {
        qPoint_h(ik, i) = qPoint(i);
      }
    }
    Kokkos::deep_copy(qPoint_d, qPoint_h);

    auto t2 = phononH0.kokkosBatchedDiagonalizeFromCoordinates(qPoint_d);
    DoubleView2D batchedEnergies = std::get<0>(t2);
    StridedComplexView3D batchedEigenvectors = std::get<1>(t2);

    // now we copy back to host
    auto energies_h = Kokkos::create_mirror_view(batchedEnergies);
    Kokkos::deep_copy(energies_h, batchedEnergies);
    for (int ib = 0; ib < numBands; ++ib) {
      energies(ib) = energies_h(0, ib);
    }

    auto eigenvectors_h = Kokkos::create_mirror_view(batchedEigenvectors);
    Kokkos::deep_copy(eigenvectors_h, batchedEigenvectors);
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        eigenvectors(ib1, ib2) = eigenvectors_h(0, ib1, ib2);
      }
    }

    auto t3 = phononH0.kokkosBatchedDiagonalizeWithVelocities(qPoint_d);
    ComplexView4D velocity_d = std::get<2>(t3);
    auto velocity_h = Kokkos::create_mirror_view(velocity_d);
    Kokkos::deep_copy(velocity_h, velocity_d);
    for (int ib1 = 0; ib1 < numBands; ++ib1) {
      for (int ib2 = 0; ib2 < numBands; ++ib2) {
        for (int i = 0; i < 3; ++i) {
          velocity(ib1, ib2, i) = velocity_h(0, ib1, ib2, i);
        }
      }
    }
  }

// check energies are equal
for (int unsigned ib = 0; ib < numBands; ib++) {
  // std::cout << "Energies [meV]: ";
  // std::cout << std::scientific << std::setprecision(4) << energies[i] << "\n" <<std::endl;
  EXPECT_NEAR(abs((energies(ib)-energiesJson[ib])/energiesJson[ib]), 0, 0.001);
}

// check velocities are equal
for (int unsigned ib = 0; ib < numBands; ib++) {
  for (int unsigned ib2 = 0; ib2 < numBands; ib++) {
    EXPECT_NEAR(abs((velocity(ib, ib2, 0) - sqmod_vop_x_json[ib][ib2])/sqmod_vop_x_json[ib][ib2]), 0, 0.001);
    EXPECT_NEAR(abs((velocity(ib, ib2, 1) - sqmod_vop_y_json[ib][ib2])/sqmod_vop_y_json[ib][ib2]), 0, 0.001);
    EXPECT_NEAR(abs((velocity(ib, ib2, 2) - sqmod_vop_z_json[ib][ib2])/sqmod_vop_z_json[ib][ib2]), 0, 0.001);
  }
}

}
