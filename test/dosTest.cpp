#include "points.h"
#include "qe_input_parser.h"
#include "delta_function.h"
#include <gtest/gtest.h>

/** In this test I check for the implementation of the tetrahedron method.
 * If everything's fine, we expect (\int dos(e) de) = numBands
 * Note: in the test, I find 5.98 instead of 6.
 * It gets better with smaller deltaEnergy and more points (but slower!)
 */
TEST(TetrahedronTest, Normalization) {
  // set up a phononH0
  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setSumRuleFC2("simple");

  context.setUseSymmetries(true);

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto h0 = std::get<1>(tup);

  // setup parameters for active band structure creation
  Eigen::Vector3i qMesh;
  qMesh << 15, 15, 15;
  Points points(crystal, qMesh);
  bool withVelocities = false;
  bool withEigenvectors = false;
  auto fullBandStructure = h0.populate(points, withVelocities,
                                   withEigenvectors,false);

  // Form tetrahedra and fill them with eigenvalues
  TetrahedronDeltaFunction tetrahedra(fullBandStructure);

  double minEnergy = 0.;
  double maxEnergy = 0.1 / energyRyToEv;
  double deltaEnergy = 0.0004 / energyRyToEv;
  int numEnergies = int((maxEnergy - minEnergy) / deltaEnergy) + 1;

  std::vector<double> energies(numEnergies);
  for (int i = 0; i < numEnergies; i++) {
    energies[i] = i * deltaEnergy + minEnergy;
  }

  // DOS should integrate to the number of bands
  {
    double x = 0.;
    for (int i = 0; i < numEnergies; i++) {
      x += tetrahedra.getDOS(energies[i]) * deltaEnergy;
    }
    ASSERT_NEAR(x, h0.getNumBands(), 0.03);
  }

  // now we try to use symmetries
  points.setIrreduciblePoints();
  double x = 0.;
  for (int i = 0; i < numEnergies; i++) {
    x += tetrahedra.getDOS(energies[i]) * deltaEnergy;
  }
  ASSERT_NEAR(x,h0.getNumBands(),0.025);

}
