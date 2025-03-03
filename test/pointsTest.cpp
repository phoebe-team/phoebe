#include "points.h"
#include "gtest/gtest.h"

TEST(PointsTest, PointsHandling) {
  Eigen::Matrix3d directUnitCell;
  directUnitCell.row(0) << -5.1, 0., 5.1;
  directUnitCell.row(1) << 0., 5.1, 5.1;
  directUnitCell.row(2) << -5.1, 5.1, 0.;
  Eigen::MatrixXd atomicPositions(2, 3);
  atomicPositions.row(0) << 0., 0., 0.;
  atomicPositions.row(1) << 2.55, 2.55, 2.55;
  Eigen::VectorXi atomicSpecies(2);
  atomicSpecies << 0, 0;
  std::vector<std::string> speciesNames;
  speciesNames.emplace_back("Si");
  Eigen::VectorXd speciesMasses(1);
  speciesMasses(0) = 28.086;

  // no born charges for the test
  Eigen::Tensor<double, 3> bornCharges(2, 3, 3);
  bornCharges.setZero();
  Eigen::Matrix3d dielectricMatrix;
  dielectricMatrix.setZero();

  Context context;

  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses, bornCharges, dielectricMatrix);

  Eigen::Vector3i mesh;
  mesh << 4, 4, 4;
  Points points(crystal, mesh);

  //-----------------------------------
  // check mesh is what I set initially

  auto tup = points.getMesh();
  auto mesh_ = std::get<0>(tup);
  EXPECT_EQ((mesh - mesh_).norm(), 0.);

  //----------------------
  // check point inversion

  auto p1 = points.getPoint(4);
  // find the index of the inverted point
  int i4 = points.getIndex(-p1.getCoordinates(Points::crystalCoordinates));
  //	int i4 = points.getIndexInverted(4);
  auto p2 = points.getPoint(i4);
  auto p3 = p1 + p2;
  EXPECT_EQ(p3.getCoordinates(Points::cartesianCoordinates).norm(), 0.);

  //-----------------------
  // check point inversion

  mesh << 2, 2, 2;
  points = Points(crystal, mesh);
  int iq = 7;
  p1 = points.getPoint(iq);
  //	int iqr = points.getIndexInverted(iq);
  int iqr = points.getIndex(-p1.getCoordinates(Points::crystalCoordinates));
  p2 = points.getPoint(iqr);
  p3 = p1 + p2;

  EXPECT_EQ(p3.getCoordinates().norm(), 0.);
}
