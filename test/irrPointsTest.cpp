#include "points.h"
#include "gtest/gtest.h"

TEST(IrrPointsTest, Symmetries) {
  Eigen::Matrix3d directUnitCell;
  directUnitCell.col(0) << -5.1, 0., 5.1;
  directUnitCell.col(1) << 0., 5.1, 5.1;
  directUnitCell.col(2) << -5.1, 5.1, 0.;
  Eigen::MatrixXd atomicPositions(2, 3);
  atomicPositions.row(0) << 0., 0., 0.;
  atomicPositions.row(1) << 2.55, 2.55, 2.55;
  Eigen::VectorXi atomicSpecies(2);
  atomicSpecies << 0, 0;
  std::vector<std::string> speciesNames;
  speciesNames.emplace_back("Si");
  Eigen::VectorXd speciesMasses(1);
  speciesMasses(0) = 28.086;

  Context context;
  context.setUseSymmetries(true);

  // no born charges for the test
  Eigen::Tensor<double, 3> bornCharges(2, 3, 3);
  bornCharges.setZero();

  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses, bornCharges);

  ASSERT_EQ(crystal.getNumSymmetries(), 48);

  //-----------------

  Eigen::Vector3i mesh;
  mesh << 4, 4, 4;
  Points points(crystal, mesh);
  points.setIrreduciblePoints();

  //-----------------------------------

  // I hard code that I expect 8 irreducible points
  int numIrrPoints = points.irrPointsIterator().size();
  ASSERT_EQ(numIrrPoints, 8);

  int numFullPoints = 0;
  for (int ik : points.irrPointsIterator()) {
    numFullPoints += points.getRotationsStar(ik).size();
  }
  ASSERT_EQ(numFullPoints, points.getNumPoints());

  // here I check the symmetries matrices
  // loop over irreducible points, unfold the star, and check that we can
  // reconstruct the whole list of points of the full grid
  int counter = 0;
  std::vector<int> allIndices;
  for (int ikIrr : points.irrPointsIterator()) {
    auto kIrr = points.getPointCoordinates(ikIrr, Points::cartesianCoordinates);

    int ikIrrAsRed = points.asIrreducibleIndex(ikIrr);
    ASSERT_EQ(ikIrrAsRed, counter);
    counter++;

    auto rots = points.getRotationsStar(ikIrr);
    for (const auto &s : rots) {
      auto kRedCart = s * kIrr; // in cartesian coordinates
      auto kRedCrystal = points.cartesianToCrystal(kRedCart);
      int oldIndex = points.getIndex(kRedCrystal);
      allIndices.push_back(oldIndex);

      auto t = points.getRotationToIrreducible(kRedCart,
                                               Points::cartesianCoordinates);
      int ik2 = std::get<0>(t);
      ASSERT_EQ(ik2, ikIrr);

      Eigen::Matrix3d rot = std::get<1>(t);
      // this rotation should map the point to the irreducible one
      Eigen::Vector3d kIrr2 = rot * kRedCart;

      ASSERT_NEAR((rot.inverse() - s).squaredNorm(), 0., 0.0001);

      Eigen::Vector3d x1 = points.cartesianToCrystal(kIrr2).transpose();
      Eigen::Vector3d x2 = points.cartesianToCrystal(kIrr).transpose();
      ASSERT_NEAR((x1 - x2).squaredNorm(), 0., 0.0001);
    }
  }
  int uniqueCount =
      std::unique(allIndices.begin(), allIndices.end()) - allIndices.begin();
  ASSERT_EQ(mesh.prod(), uniqueCount);
}





TEST(IrrPointsTest, SymmetriesGaN) {
  Eigen::Matrix3d directUnitCell;
  directUnitCell.col(0) <<  3.150162, 0.000000, 0.000000;
  directUnitCell.col(1) << -1.575081, 2.728120, 0.000000;
  directUnitCell.col(2) <<  0.000000, 0.000000, 5.133369;

  Eigen::MatrixXd atomicPositions(4, 3);
  atomicPositions.row(0) << 1.57508, 0.90937, 2.56271;
  atomicPositions.row(1) << 0.00000, 1.81875, 5.12940;
  atomicPositions.row(2) << 0.00000, 1.81875, 1.92899;
  atomicPositions.row(3) << 1.57508, 0.90937, 4.49567;

  Eigen::VectorXi atomicSpecies(4);
  atomicSpecies(0) = 0;
  atomicSpecies(1) = 0;
  atomicSpecies(2) = 1;
  atomicSpecies(3) = 1;

  std::vector<std::string> speciesNames;
  speciesNames.emplace_back("Ga");
  speciesNames.emplace_back("N");

  Eigen::VectorXd speciesMasses(2);
  speciesMasses(0) = 69.723;
  speciesMasses(1) = 14.0067;

  // no born charges for the test
  Eigen::Tensor<double, 3> bornCharges(4, 3, 3);
  bornCharges.setZero();

  Context context;
  context.setUseSymmetries(true);

  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses, bornCharges);

  ASSERT_EQ(crystal.getNumSymmetries(), 12);

  //-----------------

  Eigen::Vector3i mesh;
  mesh << 4, 4, 3;
  Points points(crystal, mesh);
  points.setIrreduciblePoints();

  //-----------------------------------

  // I hard code that I expect 8 irreducible points
  int numIrrPoints = points.irrPointsIterator().size();
  ASSERT_EQ(numIrrPoints, 12);
  // note: using inversion symmetry, we could have 8 points


  int numFullPoints = 0;
  for (int ik : points.irrPointsIterator()) {
    numFullPoints += points.getRotationsStar(ik).size();
  }
  ASSERT_EQ(numFullPoints, points.getNumPoints());

  // here I check the symmetries matrices
  // loop over irreducible points, unfold the star, and check that we can
  // reconstruct the whole list of points of the full grid
  int counter = 0;
  std::vector<int> allIndices;
  for (int ikIrr : points.irrPointsIterator()) {
    auto kIrr = points.getPointCoordinates(ikIrr, Points::cartesianCoordinates);

    int ikIrrAsRed = points.asIrreducibleIndex(ikIrr);
    ASSERT_EQ(ikIrrAsRed, counter);
    counter++;

    auto rots = points.getRotationsStar(ikIrr);
    for (const auto &s : rots) {
      auto kRedCart = s * kIrr; // in cartesian coordinates
      auto kRedCrystal = points.cartesianToCrystal(kRedCart);
      int oldIndex = points.getIndex(kRedCrystal);
      allIndices.push_back(oldIndex);

      auto t = points.getRotationToIrreducible(kRedCart,
                                               Points::cartesianCoordinates);
      int ik2 = std::get<0>(t);
      ASSERT_EQ(ik2, ikIrr);

      Eigen::Matrix3d rot = std::get<1>(t);
      // this rotation should map the point to the irreducible one
      Eigen::Vector3d kIrr2 = rot * kRedCart;

      ASSERT_NEAR((rot.inverse() - s).squaredNorm(), 0., 0.0001);

      Eigen::Vector3d x1 = points.cartesianToCrystal(kIrr2).transpose();
      Eigen::Vector3d x2 = points.cartesianToCrystal(kIrr).transpose();
      ASSERT_NEAR((x1 - x2).squaredNorm(), 0., 0.0001);
    }
  }
  int uniqueCount =
      std::unique(allIndices.begin(), allIndices.end()) - allIndices.begin();
  ASSERT_EQ(mesh.prod(), uniqueCount);
}

