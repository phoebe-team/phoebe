#include "bandstructure.h"
#include "ifc3_parser.h"
#include "ph_scattering.h"
#include "points.h"
#include "parser.h"
#include "gtest/gtest.h"
#include <fstream>

TEST(Interaction3Ph, Coupling3Ph000) {
  // here we test the 3-ph matrix element (squared)
  // against results produced by ShengBTE.
  // in particular, we compute the couplings at q1=0, q2=0, q3=0

  // Note: Eigenvectors are complex: they can get a phase
  //       the phase doesn't change the coupling
  // Note: Eigenvectors can be degenerate! Hence, it's not straightforward to
  //       compare the coupling with ShengBTE or other codes, since the
  //       coupling at fixed mode indices might be different. However, the
  //       SUM of the coupling matrix elements within a degenerate subspace
  //       is the correct invariant quantity to be compared.
  // In this test, specifically, we use the three-fold degenerate modes at
  // gamma, and we compare the sum of the interaction within the optical
  // manifold

  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setPhFC3FileName("../test/data/FORCE_CONSTANTS_3RD");
  context.setSumRuleFC2("simple");

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  auto coupling3Ph = IFC3Parser::parse(context, crystal);

  // Number of atoms
  int numAtoms = crystal.getNumAtoms();
  // Number of bands
  int numBands = phononH0.getNumBands();

  ASSERT_EQ(numBands / 3, numAtoms);

  Eigen::Vector3d q1, q2, q3;
  q1.setZero();
  q2.setZero();
  q3.setZero();

  auto tup1 = phononH0.diagonalizeFromCoordinates(q1);
  auto energies1 = std::get<0>(tup1);
  auto ev1 = std::get<1>(tup1);
  auto tup2 = phononH0.diagonalizeFromCoordinates(q2);
  auto energies2 = std::get<0>(tup2);
  auto ev2 = std::get<1>(tup2);
  auto tup3 = phononH0.diagonalizeFromCoordinates(q3);
  auto energies3 = std::get<0>(tup3);
  auto ev3 = std::get<1>(tup3);

  std::vector<Eigen::Vector3d> q1s_e(1);
  Eigen::Vector3d q2_e;
  std::vector<Eigen::MatrixXcd> ev1s_e(1);
  Eigen::MatrixXcd ev2_e;
  std::vector<Eigen::MatrixXcd> ev3Pluss_e(1);
  std::vector<Eigen::MatrixXcd> ev3Minss_e(1);
  std::vector<int> nb1s_e(1);
  int nb2 = energies2.size();
  std::vector<int> nb3Pluss_e(1);
  std::vector<int> nb3Minss_e(1);

  q1s_e[0] = q1;
  nb1s_e[0] = energies1.size();
  nb3Pluss_e[0] = energies3.size();
  nb3Minss_e[0] = energies3.size();
  ev1s_e[0] = ev1;
  ev3Pluss_e[0] = ev3;
  ev3Minss_e[0] = ev3;
  ev2_e = ev2;
  q2_e = q2;

  coupling3Ph.cacheD3(q2_e);
  auto tup4 = coupling3Ph.getCouplingsSquared(q1s_e, q2_e, ev1s_e, ev2_e,
                                              ev3Pluss_e, ev3Minss_e, nb1s_e,
                                              nb2, nb3Pluss_e, nb3Minss_e);
  auto couplingPlus = std::get<0>(tup4)[0];
  auto couplingMins = std::get<1>(tup4)[0];

  for (int i = 0; i < numBands; i++) {
    for (int j = 0; j < numBands; j++) {
      for (int k = 0; k < numBands; k++) {
        ASSERT_NEAR(couplingPlus(i, j, k), couplingMins(i, j, k),1e-40);
      }
    }
  }

  // we load reference data

  Eigen::Tensor<double, 3> referenceCoupling(numBands, numBands, numBands);
  referenceCoupling.setZero();
  {
    std::ifstream tfile("../test/data/reference3Ph000");
    double x1, x2;
    int i_, j_, k_;
    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < numBands; j++) {
        for (int k = 0; k < numBands; k++) {
          tfile >> i_ >> j_ >> k_ >> x1 >> x2;
          referenceCoupling(i, j, k) = x1;
        }
      }
    }
  }

  // note that one cannot check for exact equality, due to state degeneracies
  double x1, x2;
  x1 = 0.;
  x2 = 0.;
  for (int i = 3; i < numBands; i++) {
    for (int j = 3; j < numBands; j++) {
      for (int k = 3; k < numBands; k++) {
        x2 += couplingPlus(i, j, k);
        x1 += referenceCoupling(i, j, k);
      }
    }
  }
  double relativeError = abs((x1 - x2) / x1);

  ASSERT_NEAR(relativeError, 0., 1.0e-4);
}

TEST(Interaction3Ph, Coupling3Ph210) {
  // here we test the 3-ph matrix element (squared)
  // against results produced by ShengBTE.
  // in particular, we compute the couplings at q1=0, q2=0, q3=0

  // Note: Eigenvectors are complex: they can get a phase
  //       the phase doesn't change the coupling
  // Note: Eigenvectors can be degenerate! Hence, it's not straightforward to
  //       compare the coupling with ShengBTE or other codes, since the
  //       coupling at fixed mode indices might be different. However, the
  //       SUM of the coupling matrix elements within a degenerate subspace
  //       is the correct invariant quantity to be compared.
  // In this test, specifically, we use the three-fold degenerate modes at
  // gamma, and we compare the sum of the interaction within the optical
  // manifold

  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setPhFC3FileName("../test/data/FORCE_CONSTANTS_3RD");
  context.setSumRuleFC2("simple");

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  auto coupling3Ph = IFC3Parser::parseFromShengBTE(context, crystal);

  // Number of atoms
  int numAtoms = crystal.getNumAtoms();
  // Number of bands
  int numBands = 3 * numAtoms;

  auto atomicMasses = crystal.getAtomicMasses();

  // Form a triplet to test vertex calculator
  int iq1 = 10;
  int iq2 = 210;
  int iq3 = 200;
  auto iq1Index = WavevectorIndex(iq1);
  auto iq2Index = WavevectorIndex(iq2);
  auto iq3Index = WavevectorIndex(iq3);

  Eigen::Vector3i qMesh;
  qMesh << 20, 20, 20;
  Points points(crystal, qMesh);
  auto p1 = points.getPoint(iq1);
  auto p2 = points.getPoint(iq2);
  auto p3 = points.getPoint(iq3);

  auto tup1 = phononH0.diagonalize(p1);
  auto energies1 = std::get<0>(tup1);
  auto evm1 = std::get<1>(tup1);
  auto tup2 = phononH0.diagonalize(p2);
  auto energies2 = std::get<0>(tup2);
  auto evm2 = std::get<1>(tup2);
  auto tup3 = phononH0.diagonalize(p3);
  auto energies3 = std::get<0>(tup3);
  auto evm3 = std::get<1>(tup3);
  auto q1 = p1.getCoordinates(Points::cartesianCoordinates);
  auto q2 = p2.getCoordinates(Points::cartesianCoordinates);
  //  auto q3 = p3.getCoordinates(Points::cartesianCoordinates);

  // note: the reference was generated without the normalization by energies
  // so we set them to one.
  Eigen::VectorXd energies(numBands);
  energies.setConstant(1.);

  int nb1 = energies.size();
  int nb2 = energies.size();

  std::vector<Eigen::Vector3d> q1s_e(1);
  Eigen::Vector3d q2_e;
  std::vector<Eigen::MatrixXcd> ev1s_e(1);
  Eigen::MatrixXcd ev2_e;
  std::vector<Eigen::MatrixXcd> ev3Pluss_e(1);
  std::vector<Eigen::MatrixXcd> ev3Minss_e(1);
  std::vector<int> nb1s_e(1);
  std::vector<int> nb3Pluss_e(1);
  std::vector<int> nb3Minss_e(1);

  q1s_e[0] = q1;
  nb1s_e[0] = nb1;
  nb3Pluss_e[0] = energies.size();
  nb3Minss_e[0] = energies.size();
  ev1s_e[0] = evm1;
  ev3Pluss_e[0] = evm3;
  ev3Minss_e[0] = evm3;
  ev2_e = evm2;
  q2_e = q2;

  coupling3Ph.cacheD3(q2_e);
  auto tup4 = coupling3Ph.getCouplingsSquared(q1s_e, q2_e, ev1s_e, ev2_e,
                                              ev3Pluss_e, ev3Minss_e, nb1s_e,
                                              nb2, nb3Pluss_e, nb3Minss_e);
  auto couplingPlus = std::get<0>(tup4)[0];
  auto couplingMins = std::get<1>(tup4)[0];
  ScatteringMatrix::symmetrizeCoupling(
      couplingPlus, energies1, energies2, energies3
  );
  ScatteringMatrix::symmetrizeCoupling(
      couplingMins, energies1, energies2, energies3
  );

  //  for (int i = 0; i < numBands; i++) {
  //    for (int j = 0; j < numBands; j++) {
  //      for (int k = 0; k < numBands; k++) {
  //        std::cout << std::setprecision(8);
  //        std::cout << i << " " << j << " " << k << " " <<
  //        std::setprecision(8) <<
  //                couplingPlus(i, j, k) << " " << couplingMins(i, j, k) <<
  //                "\n";
  //      }
  //    }
  //  }

  Eigen::Tensor<double, 3> referenceCoupling(numBands, numBands, numBands);
  referenceCoupling.setZero();
  {
    std::ifstream tfile("../test/data/reference3Ph210");
    int i_, j_, k_;
    double x1, x2;
    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < numBands; j++) {
        for (int k = 0; k < numBands; k++) {
          tfile >> i_ >> j_ >> k_ >> x1 >> x2;
          referenceCoupling(i, j, k) = x1;
        }
      }
    }
  }

  // now we compare the difference

  double x1, x2, x3;
  x1 = 0.;
  x2 = 0.;
  x3 = 0.;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < numBands; j++) {
      for (int k = 0; k < numBands; k++) {
        x1 += couplingPlus(i, j, k);
        x2 += couplingMins(i, j, k);
        x3 += referenceCoupling(i, j, k);
      }
    }
  }

  double relativeErrorP = abs((x1 - x3) / x1);
  double relativeErrorM = abs((x1 - x3) / x1);

  ASSERT_NEAR(relativeErrorP, 0., 1.0e-3);
  ASSERT_NEAR(relativeErrorM, 0., 1.0e-3);

  // now we test the same configuration, but we use State<FullPoints>
  // rather than DetachedState, to generate the coupling

  bool withVelocities = true;
  bool withEigenvectors = true;
  FullBandStructure bandStructure =
      phononH0.populate(points, withVelocities, withEigenvectors);

  p1 = bandStructure.getPoint(iq1);
  p2 = bandStructure.getPoint(iq2);
  auto p3PlusTest = p1 + p2;
  auto p3MinsTest = p1 - p2;
  auto p3Plus = bandStructure.getPoint(iq3);
  auto p3Mins = bandStructure.getPoint(iq3);

  // check that the sum of Point works
  ASSERT_EQ((p3PlusTest.getCoordinates(Points::cartesianCoordinates) -
             p3Plus.getCoordinates(Points::cartesianCoordinates))
                .norm(),
            0.);
  ASSERT_EQ((p3MinsTest.getCoordinates(Points::cartesianCoordinates) -
             p3Mins.getCoordinates(Points::cartesianCoordinates))
                .norm(),
            0.);

  auto en1 = bandStructure.getEnergies(iq1Index);
  auto en2 = bandStructure.getEnergies(iq2Index);
  auto en3Plus = bandStructure.getEnergies(iq3Index);
  auto en3Mins = bandStructure.getEnergies(iq3Index);

  nb1 = en1.size();
  nb2 = en2.size();
  q2 = bandStructure.getWavevector(iq2Index);

  q1s_e[0] = bandStructure.getWavevector(iq1Index);
  nb1s_e[0] = nb1;
  nb3Pluss_e[0] = en3Plus.size();
  nb3Minss_e[0] = en3Mins.size();
  ev1s_e[0] = bandStructure.getEigenvectors(iq1Index);
  ev2_e = bandStructure.getEigenvectors(iq2Index);
  ev3Pluss_e[0] = bandStructure.getEigenvectors(iq3Index);
  ev3Minss_e[0] = bandStructure.getEigenvectors(iq3Index);
  q2_e = q2;
  auto tup6 = coupling3Ph.getCouplingsSquared(q1s_e, q2_e, ev1s_e, ev2_e,
                                              ev3Pluss_e, ev3Minss_e, nb1s_e,
                                              nb2, nb3Pluss_e, nb3Minss_e);
  auto couplingPlus2 = std::get<0>(tup6)[0];
  auto couplingMins2 = std::get<1>(tup6)[0];

  ScatteringMatrix::symmetrizeCoupling(
      couplingPlus2, en1, en2, en3Plus
  );
  ScatteringMatrix::symmetrizeCoupling(
      couplingMins2, en1, en2, en3Mins
  );

  x1 = 0.;
  x2 = 0.;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < numBands; j++) {
      for (int k = 0; k < numBands; k++) {
        x1 += pow(couplingPlus(i, j, k) - couplingPlus2(i, j, k), 2);
        x2 += pow(couplingMins(i, j, k) - couplingMins2(i, j, k), 2);
      }
    }
  }
  ASSERT_NEAR(x1, 0., 1.0e-40);
  ASSERT_NEAR(x2, 0., 1.0e-40);
}

TEST(Interaction3Ph, Coupling3Ph000_p3py) {
  // here we test the 3-ph matrix element (squared)
  // against results produced by phono3py.
  // in particular, we compute the couplings at q1=0, q2=0, q3=0

  // Note: Eigenvectors are complex: they can get a phase
  //       the phase doesn't change the coupling
  // Note: Eigenvectors can be degenerate! Hence, it's not straightforward to
  //       compare the coupling with ShengBTE or other codes, since the
  //       coupling at fixed mode indices might be different. However, the
  //       SUM of the coupling matrix elements within a degenerate subspace
  //       is the correct invariant quantity to be compared.
  // In this test, specifically, we use the three-fold degenerate modes at
  // gamma, and we compare the sum of the interaction within the optical
  // manifold

  Context context;
  context.setPhFC2FileName("../test/data/phono3py/fc2.hdf5");
  context.setPhFC3FileName("../test/data/phono3py/fc3.hdf5");
  context.setPhonopyDispFileName("../test/data/phono3py/phono3py_disp.yaml");
  context.setSumRuleFC2("simple");

  auto tup = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  auto coupling3Ph = IFC3Parser::parse(context, crystal);

  // Number of atoms
  int numAtoms = crystal.getNumAtoms();
  // Number of bands
  int numBands = phononH0.getNumBands();

  ASSERT_EQ(numBands / 3, numAtoms);

  Eigen::Vector3d q1, q2, q3;
  q1.setZero();
  q2.setZero();
  q3.setZero();

  auto tup1 = phononH0.diagonalizeFromCoordinates(q1);
  auto energies1 = std::get<0>(tup1);
  auto ev1 = std::get<1>(tup1);
  auto tup2 = phononH0.diagonalizeFromCoordinates(q2);
  auto energies2 = std::get<0>(tup2);
  auto ev2 = std::get<1>(tup2);
  auto tup3 = phononH0.diagonalizeFromCoordinates(q3);
  auto energies3 = std::get<0>(tup3);
  auto ev3 = std::get<1>(tup3);

  std::vector<Eigen::Vector3d> q1s_e(1);
  Eigen::Vector3d q2_e;
  std::vector<Eigen::MatrixXcd> ev1s_e(1);
  Eigen::MatrixXcd ev2_e;
  std::vector<Eigen::MatrixXcd> ev3Pluss_e(1);
  std::vector<Eigen::MatrixXcd> ev3Minss_e(1);
  std::vector<int> nb1s_e(1);
  int nb2 = energies2.size();
  std::vector<int> nb3Pluss_e(1);
  std::vector<int> nb3Minss_e(1);

  q1s_e[0] = q1;
  nb1s_e[0] = energies1.size();
  nb3Pluss_e[0] = energies3.size();
  nb3Minss_e[0] = energies3.size();
  ev1s_e[0] = ev1;
  ev3Pluss_e[0] = ev3;
  ev3Minss_e[0] = ev3;
  ev2_e = ev2;
  q2_e = q2;

  coupling3Ph.cacheD3(q2_e);
  auto tup4 = coupling3Ph.getCouplingsSquared(q1s_e, q2_e, ev1s_e, ev2_e,
                                              ev3Pluss_e, ev3Minss_e, nb1s_e,
                                              nb2, nb3Pluss_e, nb3Minss_e);
  auto couplingPlus = std::get<0>(tup4)[0];
  auto couplingMins = std::get<1>(tup4)[0];

  for (int i = 0; i < numBands; i++) {
    for (int j = 0; j < numBands; j++) {
      for (int k = 0; k < numBands; k++) {
        ASSERT_EQ(couplingPlus(i, j, k), couplingMins(i, j, k));
      }
    }
  }

  // we load reference data
  Eigen::Tensor<double, 3> referenceCoupling(numBands, numBands, numBands);
  referenceCoupling.setZero();
  {
    std::ifstream tfile("../test/data/phono3py/reference3Ph000_p3py");
    double x1;
    int i_, j_, k_;
    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < numBands; j++) {
        for (int k = 0; k < numBands; k++) {
          tfile >> i_ >> j_ >> k_ >> x1;
          referenceCoupling(i, j, k) = x1;
        }
      }
    }
  }

  // note that one cannot check for exact equality, due to state degeneracies
  double x1, x2;
  x1 = 0.;
  x2 = 0.;
  for (int i = 3; i < numBands; i++) {
    for (int j = 3; j < numBands; j++) {
      for (int k = 3; k < numBands; k++) {
        x2 += couplingPlus(i, j, k);
        x1 += referenceCoupling(i, j, k);
      }
    }
  }
  double relativeError = abs((x1 - x2) / x1);

  ASSERT_NEAR(relativeError, 0., 1.0e-4);
}
