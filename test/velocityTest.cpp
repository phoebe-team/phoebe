#include "gtest/gtest.h"
#include "points.h"
#include "qe_input_parser.h"

TEST (PhononH0, Velocity) {
//int main() {
  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setSumRuleFC2("simple");

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  // now, let's create a fine mesh

  Eigen::Vector3i qMesh;
  qMesh << 40, 40, 40;
  Points points(crystal, qMesh);

  // pick a point close to gamma and get energies/velocities
  int iq = 1;
  auto qPoint = points.getPoint(iq);
  auto tup1 = phononH0.diagonalize(qPoint);
  auto energies = std::get<0>(tup1);
  auto v = phononH0.diagonalizeVelocity(qPoint);

  // take out the group velocity
  int numBands = energies.size();
  Eigen::MatrixXd groupV(3, numBands);
  for (int ib = 0; ib < numBands; ib++) {
    for (int i : {0, 1, 2}) {
      groupV(i, ib) = v(ib, ib, i).real();
    }
  }

  // select acoustic modes close to gamma
  auto v0 = groupV.col(0);
  auto v1 = groupV.col(1);
  auto v2 = groupV.col(2);
  auto qCoordinates = qPoint.getCoordinates(Points::cartesianCoordinates);

  // for these three acoustic modes, check velocity is parallel to wavevector
  ASSERT_NEAR(abs(v0.dot(qCoordinates)) / qCoordinates.norm() / v0.norm(), 1., 0.0425);
  ASSERT_NEAR(abs(v1.dot(qCoordinates)) / qCoordinates.norm() / v1.norm(), 1., 0.0425);
  ASSERT_NEAR(abs(v2.dot(qCoordinates)) / qCoordinates.norm() / v2.norm(), 1., 0.0425);

  // for silicon, the velocity is around 2200 m/s
  double c1 = abs(v0.minCoeff()) * velocityRyToSi;
  double c2 = abs(v1.minCoeff()) * velocityRyToSi;
  double c3 = abs(v2.minCoeff()) * velocityRyToSi;
  double speedOfSound = std::min(c1,c2);
  speedOfSound = std::min(speedOfSound,c3);
  ASSERT_NEAR(speedOfSound, 2200., 200.);

  // for another sanity check
  // we can also verify that, for acoustic phonons in silicon close to gamma,
  // the velocity is approximately (energies/q)we can approximate the velocity

  double err0 = abs(energies(0) - v0.dot(qCoordinates)) / v0.norm();
  double err1 = abs(energies(1) - v0.dot(qCoordinates)) / v0.norm();
  double err2 = abs(energies(2) - v0.dot(qCoordinates)) / v0.norm();
  // we allow a 4% error, (anisotropies...)
  ASSERT_NEAR(err0, 0., 0.0425);
  ASSERT_NEAR(err1, 0., 0.0425);
  ASSERT_NEAR(err2, 0., 0.0425);
}

/** Here I estimate the mass at the top of the valence band of silicon
 */
TEST (WannierH0, Velocity) {
  Context context;
  context.setElectronH0Name("../test/data/666_si_tb.dat");

  // setup crystal input
  Eigen::MatrixXd atomicPositions(2,3);
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

  // gamma point
  Eigen::Vector3d k1;
  k1.setZero();

  // build a point close to gamma point
  double deltaK = 0.00025;
  Eigen::Vector3d k2;
  k2 << deltaK, 0., 0.;
  auto v2 = electronH0.diagonalizeVelocityFromCoordinates(k2);

  auto tup1 = electronH0.diagonalizeFromCoordinates(k1);
  auto ens1 = std::get<0>(tup1);
  auto tup2 = electronH0.diagonalizeFromCoordinates(k2);
  auto ens2 = std::get<0>(tup2);

  // we hard code the index of the top of the valence band
  int ib = 3;

  // compute the mass
  // 1) as second derivative of energy
  // 2) as first derivative of velocity
  double mass1 = 2. * (ens2(ib)-ens1(ib)) / deltaK / deltaK;
  double mass2 = 2. * v2(ib,ib,0).real() / deltaK;

  // the two masses should be similar
  ASSERT_NEAR(abs((mass1-mass2)/mass1), 0., 0.02);

  // we also fix their value in this test
  ASSERT_NEAR(abs(mass1), 686.779, 0.02);
}

/** Here I check the velocity of the WannierH0 class by comparing it with the
 * velocity estimated by finite differences in the electronic energy.
 */
TEST (WannierH0, Velocity2) {
  Context context;
  context.setElectronH0Name("../test/data/666_si_tb.dat");

  // setup crystal input
  Eigen::MatrixXd atomicPositions(2,3);
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

  // pick a k-point
  Eigen::Vector3d k1;
  k1 << 0.1, 0.2, 0.3;

  // compute velocity analytically at k1
  auto v1 = electronH0.diagonalizeVelocityFromCoordinates(k1);

  for (int iCart : {0, 1, 2}) {

    double deltaK = 0.0001;

    Eigen::Vector3d k1PlusDelta;
    k1PlusDelta = k1;
    k1PlusDelta(iCart) += deltaK;

    auto tup1 = electronH0.diagonalizeFromCoordinates(k1);
    auto ens1 = std::get<0>(tup1);
    auto tup2 = electronH0.diagonalizeFromCoordinates(k1PlusDelta);
    auto ens2 = std::get<0>(tup2);

    // we hard code the index of the top of the valence band
    int ib = 0;

    double x = (ens2(ib) - ens1(ib)) / deltaK;

    double error = ( x - v1(ib, ib, iCart).real() ) / v1(ib, ib, iCart).real();

    ASSERT_NEAR(error, 0., 0.001);

  }
}
