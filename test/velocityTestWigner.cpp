#include "gtest/gtest.h"
#include <iomanip>
#include "points.h"
#include "constants.h"
#include "phonopy_input_parser.h"

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * check that the velocity operator matches reference Si data
 * this reference data was verified against msimoncelli/phono3py-wte repo
 * we only test CPU implementation here, consistency between CPU and
 * Kokkos implementations is tested in the Kokkos.PhononH0 test 
 */
TEST (PhononH0, VelocityOperator) {
  // read the JSON reference file
  std::ifstream f("../test/data/silicon_velocity_operator_reference.json");
  json data = json::parse(f);
  std::vector<double> qPointJson = data["qPoint"];
  std::vector<double> energiesJson = data["energies"];
  std::vector<std::vector<double>> sqmod_vop_x_json = data["|v_x|^2"];
  std::vector<std::vector<double>> sqmod_vop_y_json = data["|v_y|^2"];
  std::vector<std::vector<double>> sqmod_vop_z_json = data["|v_z|^2"];

  Context context;
  context.setPhFC2FileName("../test/data/phono3py/fc2.hdf5");
  context.setPhonopyDispFileName("../test/data/phono3py/phono3py_disp.yaml");

  auto [crystal,phononH0] = PhonopyParser::parsePhHarmonic(context);

  // check the velocity operator at given point
  Eigen::Vector3d qPoint({qPointJson[0], qPointJson[1], qPointJson[2]});
  // std::cout << "qPoint [crystal coords]: " << qPoint.transpose() << std::endl;
  qPoint = crystal.crystalToCartesian(qPoint);
  
  auto [energies, eigenvecs] = phononH0.diagonalizeFromCoordinates(qPoint);
  auto v = phononH0.diagonalizeVelocityFromCoordinates(qPoint);
  energies = energies * energyRyToEv * 1000; // convert to meV

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
for (int ib = 0; ib < numBands; ib++) {
  // std::cout << "Energies [meV]: ";
  // std::cout << std::scientific << std::setprecision(4) << energies[ib] << "\n" <<std::endl;
  // std::cout << std::scientific << std::setprecision(4) << energiesJson[ib] << "\n" <<std::endl;
  EXPECT_NEAR(abs((energies[ib]-energiesJson[ib])/energiesJson[ib]), 0, 0.001);
}

// check velocities are equal
for (int ib = 0; ib < numBands; ib++) {
  // std::cout << "Velocity square modulus [(m/s)^2]: ";
  // std::cout << std::scientific << std::setprecision(4) << sqmod_vop[i] << "\n" <<std::endl;
  for (int ib2 = 0; ib2 < numBands; ib2++) {
    EXPECT_NEAR(abs((sqmod_vop[0](ib,ib2) - sqmod_vop_x_json[ib][ib2])/sqmod_vop_x_json[ib][ib2]), 0, 0.001);
    EXPECT_NEAR(abs((sqmod_vop[1](ib,ib2) - sqmod_vop_y_json[ib][ib2])/sqmod_vop_y_json[ib][ib2]), 0, 0.001);
    EXPECT_NEAR(abs((sqmod_vop[2](ib,ib2) - sqmod_vop_z_json[ib][ib2])/sqmod_vop_z_json[ib][ib2]), 0, 0.001);
  }
}

}
