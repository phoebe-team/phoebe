#include "active_bandstructure.h"
#include "ifc3_parser.h"
#include "points.h"
#include "qe_input_parser.h"
#include <fstream>
#include <gtest/gtest.h>
#include "test_utils.h"

/* This test checks that the active band structure construction
 * for a phonon Hamiltonian is generated the same way regardless
 * of if it was made via buildOnTheFly or buildAsPostProcessing.
 *
 * We use the phonon case, because the two functions should behave
 * the same when they are done using the same chemical potential,
 * and for phonons, of course, the chemical potential is zero.
 */
TEST(ActiveBandStructureTest, BandStructureStorage) {

  std::vector<Context> testContexts;

  // set up a phononH0
  Context context1;
  context1.setPhFC2FileName("../test/data/444_silicon.fc");
  context1.setWindowType("energy");
  Eigen::Vector2d x2;
  x2 << 0, 0.004;
  context1.setWindowEnergyLimit(x2);
  Eigen::VectorXd x3(1);
  x3(0) = 300. / temperatureAuToSi;
  context1.setTemperatures(x3);

  testContexts.push_back(context1);

  Context context2;
  context2.setPhFC2FileName("../test/data/444_silicon.fc");
  context2.setWindowType("population");
  context2.setWindowPopulationLimit(0.5e-8);
  context2.setTemperatures(x3);
  testContexts.push_back(context2);

  for (Context context : testContexts) {
    context.setUseSymmetries(true);

    auto tup = QEParser::parsePhHarmonic(context);
    auto crystal = std::get<0>(tup);
    auto phH0 = std::get<1>(tup);

    // Number of atoms
    int numAtoms = crystal.getNumAtoms();

    // setup parameters for active band structure creation
    Eigen::Vector3i qMesh;
    qMesh << 10, 10, 10;
    Points points(crystal, qMesh);
    bool withVelocities = true;
    bool withEigenvectors = true;

    // create two active band structures for phonons, built with different
    // methods
    //  call OTF or APP based on builder(..., forceBuildAsAPP)
    auto bsTup1 = ActiveBandStructure::builder(
        context, phH0, points, withEigenvectors, withVelocities, true);
    ActiveBandStructure absAPP = std::get<0>(bsTup1);

    auto bsTup2 = ActiveBandStructure::builder(
        context, phH0, points, withEigenvectors, withVelocities, false);
    ActiveBandStructure absOTF = std::get<0>(bsTup2);

    // TEST check that they selected the same number of states
    EXPECT_EQ(absOTF.getNumPoints(), absAPP.getNumPoints());
    //printf("NUMPOINTS = %d\n", absOTF.getNumPoints());

    // TEST check that the number of bands are the same
    int sumBands = 0;
    for (int point = 0; point < absOTF.getNumPoints(); point++) {
      auto ikTemp = WavevectorIndex(point);
      sumBands += absOTF.getNumBands(ikTemp) - absAPP.getNumBands(ikTemp);
    }
    EXPECT_EQ(sumBands, 0);

    // pick a wavevector to check energies, eigenvectors, velocities
    int ik = 7;
    auto ikIndex = WavevectorIndex(ik);

    // grab the points associated with this wavevector
    // TODO might be nice to check that they reference the same point
    Point pointOTF = absOTF.getPoint(ik);
    // Point pointAPP = absAPP.getPoint(ik);

    // get the number of bands at this point
    double nbOTF = absOTF.getNumBands(ikIndex);
    double nbAPP = absAPP.getNumBands(ikIndex);

    // generate values directly from the hamiltonian
    auto tup1 = phH0.diagonalize(pointOTF);
    auto ensT = std::get<0>(tup1);
    auto eigenVectorsT = std::get<1>(tup1);
    auto velocitiesT = phH0.diagonalizeVelocity(pointOTF);

    // get the values stored in each active band structure
    auto ensOTF = absOTF.getEnergies(ikIndex);
    auto ensAPP = absAPP.getEnergies(ikIndex);

    Eigen::Tensor<std::complex<double>, 3> eigenVectorsOTF =
        absOTF.getPhEigenvectors(ikIndex);
    Eigen::Tensor<std::complex<double>, 3> eigenVectorsAPP =
        absAPP.getPhEigenvectors(ikIndex);

    auto velocitiesOTF = absOTF.getVelocities(ikIndex);
    auto velocitiesAPP = absAPP.getVelocities(ikIndex);

    // TEST check OTF built band structure -----------------------

    // check the energies
    double otfEns = (ensT - ensOTF).norm();
    EXPECT_NEAR(otfEns, 0.,1e-16);

    // check the velocities
    std::complex<double> otfVelocities = complexZero;
    for (int ib1 = 0; ib1 < nbOTF; ib1++) {
      for (int ib2 = 0; ib2 < nbOTF; ib2++) {
        for (int ic = 0; ic < 3; ic++) {
          otfVelocities +=
              pow(velocitiesT(ib1, ib2, ic) - velocitiesOTF(ib1, ib2, ic), 2);
        }
      }
    }
    EXPECT_NEAR(otfVelocities.real(), 0, 1e-3);
    EXPECT_NEAR(otfVelocities.imag(), 0, 1e-3);

    Eigen::MatrixXcd resultT = mat_vec_mat_adj(eigenVectorsT, ensT, nbOTF);
    Eigen::MatrixXcd resultOTF = mat_vec_mat_adj(ev3Dto2D(eigenVectorsOTF), ensOTF, nbOTF);
    Eigen::MatrixXcd resultAPP = mat_vec_mat_adj(ev3Dto2D(eigenVectorsAPP), ensAPP, nbAPP);

    // TODO: Disabled because everything seems fine and eigenvectors have freedom
    //EXPECT_EQ((resultAPP-resultOTF).norm()/resultAPP.norm(), 0.0);
    //EXPECT_EQ((resultT-resultOTF).norm()/resultOTF.norm(), 0.0);
    //EXPECT_EQ((resultT-resultAPP).norm()/resultOTF.norm(), 0.0);

    // TEST check APP built band structure -----------------------

    // check the energies
    double appEns = (ensT - ensAPP).norm();
    EXPECT_NEAR(appEns, 0., 1.e-16);

    // check the velocities
    std::complex<double> appVelocities = complexZero;
    for (int ib1 = 0; ib1 < nbAPP; ib1++) {
      for (int ib2 = 0; ib2 < nbAPP; ib2++) {
        for (int ic = 0; ic < 3; ic++) {
          appVelocities +=
              pow(velocitiesT(ib1, ib2, ic) - velocitiesAPP(ib1, ib2, ic), 2);
        }
      }
    }
    EXPECT_NEAR(appVelocities.real(), 0, 1e-3);
    EXPECT_NEAR(appVelocities.imag(), 0, 1e-3);


    // check the eigenvectors
    {
      std::complex<double> appEigenVectors = complexZero;
      for (int i = 0; i < nbAPP; i++) {
        auto tup2 = decompress2Indices(i, numAtoms, 3);
        auto iat = std::get<0>(tup2);
        auto ic = std::get<1>(tup2);
        for (int j = 0; j < nbAPP; j++) {
          appEigenVectors +=
              pow(eigenVectorsT(i, j) - eigenVectorsAPP(ic, iat, j), 2);
        }
      }
      // I deactivate this test.
      // In fact, APP computes eigenvectors folded in the WS zone,
      // while OTF computes on the 1st BZ. So, energies are the same
      // but eigenvectors can have different phases
      // EXPECT_EQ(appEigenVectors, complexZero);
    }
  }
}

/** In the test above, no state was filtered
 * Here we are testing that we can successfully throw away some states
 */
TEST(ActiveBandStructureTest, WindowFilter) {
  // set up input parameters
  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setWindowType("population");
  context.setWindowPopulationLimit(1.0e-2);
  Eigen::VectorXd x3(1);
  x3(0) = 50. / temperatureAuToSi;
  context.setTemperatures(x3);
  Eigen::Vector3i qMesh;
  qMesh << 10, 10, 10;

  // read the phonon H0
  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phH0 = std::get<1>(tup);

  // setup parameters for active band structure creation
  Points points(crystal, qMesh);

  bool withVelocities = true;
  bool withEigenvectors = true;

  // create two active band structures for phonons, built with different methods
  //  call OTF or APP based on builder(..., forceBuildAsAPP)
  auto bsTup1 = ActiveBandStructure::builder(
      context, phH0, points, withEigenvectors, withVelocities, true);
  ActiveBandStructure absAPP = std::get<0>(bsTup1);
  auto bsTup2 = ActiveBandStructure::builder(
      context, phH0, points, withEigenvectors, withVelocities, false);
  ActiveBandStructure absOTF = std::get<0>(bsTup2);

  // TEST check that they selected the same number of states
  EXPECT_EQ(absOTF.getNumPoints(), absAPP.getNumPoints());
  EXPECT_LT(absOTF.getNumPoints(), qMesh(0) * qMesh(1) * qMesh(2));

  // TEST check that the number of bands are the same
  int sumBands = 0;
  for (int point = 0; point < absOTF.getNumPoints(); point++) {
    auto ikTemp = WavevectorIndex(point);
    sumBands += absOTF.getNumBands(ikTemp) - absAPP.getNumBands(ikTemp);
  }
  EXPECT_EQ(sumBands, 0);

  // pick a wavevector to check energies, eigenvectors, velocities
  int ik = 12;
  auto ikIndex = WavevectorIndex(ik);
  double nb = absOTF.getNumBands(ikIndex);

  // get the values stored in each active band structure
  auto ensOTF = absOTF.getEnergies(ikIndex);
  auto ensAPP = absAPP.getEnergies(ikIndex);

  Eigen::Tensor<std::complex<double>, 3> eigenVectorsOTF =
      absOTF.getPhEigenvectors(ikIndex);
  Eigen::Tensor<std::complex<double>, 3> eigenVectorsAPP =
      absAPP.getPhEigenvectors(ikIndex);

  auto velocitiesOTF = absOTF.getVelocities(ikIndex);
  auto velocitiesAPP = absAPP.getVelocities(ikIndex);

  // check the energies
  double ens = (ensOTF - ensAPP).norm();
  EXPECT_NEAR(ens/ensOTF.norm(), 0., 1e-14);

  // check the velocities
  std::complex<double> velocities = complexZero;
  for (int ib1 = 0; ib1 < nb; ib1++) {
    for (int ib2 = 0; ib2 < nb; ib2++) {
      for (int ic = 0; ic < 3; ic++) {
        velocities +=
            pow(velocitiesOTF(ib1, ib2, ic) - velocitiesAPP(ib1, ib2, ic), 2);
      }
    }
  }
  EXPECT_EQ(velocities, complexZero);

  // check the eigenvectors
  //double eigenVectors = 0.0, totnorm = 0.0;
  //for (int i = 0; i < nb; i++) {
  //  auto tup2 = decompress2Indices(i, numAtoms, 3);
  //  auto iat = std::get<0>(tup2);
  //  auto ic = std::get<1>(tup2);
  //  for (int j = 0; j < nb; j++) {
  //    eigenVectors +=
  //        std::abs(eigenVectorsOTF(ic, iat, j) - eigenVectorsAPP(ic, iat, j));
  //    totnorm += std::abs(eigenVectorsOTF(ic,iat,j));
  //  }
  //}
  //EXPECT_NEAR(eigenVectors, 0.0, 1e-14);

  Eigen::MatrixXcd resultOTF = mat_vec_mat_adj(ev3Dto2D(eigenVectorsOTF), ensOTF, nb);
  Eigen::MatrixXcd resultAPP = mat_vec_mat_adj(ev3Dto2D(eigenVectorsAPP), ensAPP, nb);
  //resultOTF = ev3Dto2D(eigenVectorsOTF)*ensOTF.asDiagonal()*ev3Dto2D(eigenVectorsOTF).adjoint();
  EXPECT_NEAR((resultAPP-resultOTF).norm()/resultAPP.norm(), 0.0, 1e-14);
}
