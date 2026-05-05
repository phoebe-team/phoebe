#include "gtest/gtest.h"
#include <iomanip>
#include "points.h"
#include "constants.h"
#include "phonopy_input_parser.h"

TEST (WTE, VelocityOperator) {

  Context context;
  context.setPhFC2FileName("../example/velocities/fc2.hdf5");
  context.setPhonopyDispFileName("../example/velocities/phonopy.yaml");
  //context.setSumRuleFC2("none");

  auto [crystal,phononH0] = PhonopyParser::parsePhHarmonic(context);

  // check the velocity on a certain point

  // qpoint in crystal coordinates
  Eigen::Vector3d qPoint({0.1, 0.22, 0.33});
  //Eigen::Vector3d qPoint({0.5, 0.0, 0.5});

  qPoint = crystal.crystalToCartesian(qPoint);
  std::cout << qPoint.transpose() << std::endl;
  auto [energies, eigenvecs] = phononH0.diagonalizeFromCoordinates(qPoint);
  auto v = phononH0.diagonalizeVelocityFromCoordinates(qPoint);

  // take out the group velocity
  int numBands = energies.size();
  Eigen::MatrixXd groupV(3, numBands);
  std::vector<Eigen::MatrixXcd> vop_dir(3);
  for (int ib = 0; ib < numBands; ib++) {
    for (int i : {0, 1, 2}) {
      groupV(i, ib) = v(ib, ib, i).real();

      vop_dir[i].resize(numBands,numBands);
      for (int ib2 = 0; ib2 < numBands; ib2++) {
        vop_dir[i](ib,ib2) = v(ib,ib2,i) * v(ib2,ib,i); // * 8.03961136006;
      }
    }
  }

// this is printing the kokkos_hamiltonian generated velocity operator
for (int i : {0, 1, 2}) {
  std::cout << std::scientific << vop_dir[i]*velocityRyToSi << "\n" <<std::endl;
}
// print the energies
std::cout <<std::fixed << std::setprecision(4) << energies.transpose() * energyRyToEv * 1000. << std::endl;
