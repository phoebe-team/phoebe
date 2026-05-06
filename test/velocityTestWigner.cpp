#include "gtest/gtest.h"
#include <iomanip>
#include "points.h"
#include "constants.h"
#include "phonopy_input_parser.h"

TEST (WTE, VelocityOperator) {

  Context context;
  // Si
  // context.setPhFC2FileName("./data/phono3py/fc2.hdf5");
  // context.setPhonopyDispFileName("./data/phono3py/phono3py_disp.yaml");
  // Cu
  context.setPhFC2FileName("./data/phono3py/copper-fc2.hdf5");
  context.setPhonopyDispFileName("./data/phono3py/copper-phonopy.yaml");
  //context.setSumRuleFC2("none");

  auto [crystal,phononH0] = PhonopyParser::parsePhHarmonic(context);

  // check the velocity on a certain point

  // qpoint in crystal coordinates
  Eigen::Vector3d qPoint({0.1, 0.22, 0.33});
  // Eigen::Vector3d qPoint({0.02, 0., 0.});
  // Eigen::Vector3d qPoint({0.01, 0.022, 0.033});
  //Eigen::Vector3d qPoint({0.5, 0.0, 0.5});

  std::cout << "qPoint [crystal coords]: " << qPoint.transpose() << std::endl;
  qPoint = crystal.crystalToCartesian(qPoint);
  auto [energies, eigenvecs] = phononH0.diagonalizeFromCoordinates(qPoint);
  auto v = phononH0.diagonalizeVelocityFromCoordinates(qPoint);

  // take out the group velocity
  int numBands = energies.size();
  Eigen::MatrixXd groupV(3, numBands);
  std::vector<Eigen::MatrixXd> vop_dir(3);
  for (int ib = 0; ib < numBands; ib++) {
    for (int i : {0, 1, 2}) {
      groupV(i, ib) = v(ib, ib, i).real();

      vop_dir[i].resize(numBands,numBands);
      for (int ib2 = 0; ib2 < numBands; ib2++) {
        vop_dir[i](ib,ib2) = (v(ib,ib2,i) * v(ib2,ib,i)).real(); // * 8.03961136006;
      }
    }
  }

// this is printing the kokkos_hamiltonian generated velocity operator
for (int i : {0, 1, 2}) {
  std::cout << std::scientific << std::setprecision(4) << vop_dir[i] * pow(velocityRyToSi, 2) << "\n" <<std::endl;
  // std::cout << std::scientific << vop_dir[i]/vop_dir[0](0,0) << "\n" <<std::endl;
}
// print the energies
std::cout << "Energies [meV]: ";
std::cout <<std::fixed << std::setprecision(4) << energies.transpose() * energyRyToEv * 1000. << std::endl;

}
