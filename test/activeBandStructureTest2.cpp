#include "active_bandstructure.h"
#include "points.h"
#include "qe_input_parser.h"
#include <fstream>
#include <gtest/gtest.h>

/* This test checks that the active band structure construction
 * for a phonon Hamiltonian is generated the same way regardless
 * of if it was made via buildOnTheFly or buildAsPostProcessing.
 *
 * We use the phonon case, because the two functions should behave
 * the same when they are done using the same chemical potential,
 * and for phonons, of course, the chemical potential is zero.
 */
TEST(ABS, Symmetries) {

  // set up a phononH0
  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setWindowType("energy");
  Eigen::Vector2d x2;
  x2 << 0, 0.0042;
  context.setWindowEnergyLimit(x2);
  Eigen::VectorXd x3(1);
  x3(0) = 300. / temperatureAuToSi;
  context.setTemperatures(x3);
  context.setUseSymmetries(true);

  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phH0 = std::get<1>(tup);

  // setup parameters for active band structure creation
  Eigen::Vector3i qMesh;
  qMesh << 3, 3, 3;
  // note: the 2,2,2 test wasn't useful, because the velocity = 0 in all points
  // (2x2x2 only samples center + border of Brillouin zone)

  Points points(crystal, qMesh);
  bool withVelocities = true;
  bool withEigenvectors = true;

  FullBandStructure fbs =
      phH0.populate(points, withVelocities, withEigenvectors);

  auto bsTup2 = ActiveBandStructure::builder(context, phH0, points,
                                             withEigenvectors, withVelocities);
  ActiveBandStructure abs = std::get<0>(bsTup2);

  std::vector<Eigen::MatrixXd> allVelocities;
  std::vector<Eigen::VectorXd> allEnergies;
  for (long ik = 0; ik < fbs.getNumPoints(); ik++) {
    auto ikIdx = WavevectorIndex(ik);
    Eigen::MatrixXd v = fbs.getGroupVelocities(ikIdx);
    allVelocities.push_back(v);
    Eigen::VectorXd e = fbs.getEnergies(ikIdx);
    allEnergies.push_back(e);
  }

  points.setIrreduciblePoints(&allVelocities, &allEnergies);
  // we expect 3 irreducible points out of 8 reducible
  EXPECT_EQ(points.irrPointsIterator().size(), 4);
  EXPECT_EQ(points.getNumPoints(), qMesh.prod());
  // gamma is discarded by abs
  EXPECT_EQ(abs.irrPointsIterator().size(), 3);

  EXPECT_EQ(abs.getNumPoints(), 26);

  // check tha the first two irr (non gamma) points are 1 and 4,
  // where we expect the four irr points to be 0, 1, 4, 5.
  int count = 0;
  for (int ik : abs.irrPointsIterator()) {
    auto ikIdx = WavevectorIndex(ik);
    Eigen::Vector3d kAbs = abs.getWavevector(ikIdx);
    if (count == 0) {
      int idx = points.irrPointsIterator()[1];
      EXPECT_EQ(idx, 1);
      Eigen::Vector3d diff =
          kAbs - points.getPointCoordinates(idx, Points::cartesianCoordinates);
      EXPECT_EQ(diff.norm(), 0.);
    }
    if (count == 1) {
      int idx = points.irrPointsIterator()[2];
      EXPECT_EQ(idx, 4);
      Eigen::Vector3d diff =
          kAbs - points.getPointCoordinates(idx, Points::cartesianCoordinates);
      EXPECT_EQ(diff.norm(), 0.);
    }
    count++;
  }

  for (int ik : abs.irrPointsIterator()) {
    auto ikIdx = WavevectorIndex(ik);
    auto ens = abs.getEnergies(ikIdx);
//    if (ik == 0) {
//      EXPECT_EQ(ens.size(), 4);
//    } else if (ik == 2) {
//      EXPECT_EQ(ens.size(), 6);
//    }

    auto qIrr = abs.getWavevector(ikIdx);

    auto rotations = abs.getRotationsStar(ikIdx);
    for (const auto &r : rotations) {
      Eigen::Vector3d q = r * qIrr;

      Eigen::Vector3d qCrystal = abs.getPoints().cartesianToCrystal(q);
      int ikFull = points.getIndex(qCrystal);
      Eigen::Vector3d k1 =
          points.getPointCoordinates(ikFull, Points::cartesianCoordinates);

      k1 = points.bzToWs(k1, Points::cartesianCoordinates);
      q = points.bzToWs(q, Points::cartesianCoordinates);
      auto diff = (k1 - q).norm();
      EXPECT_NEAR(diff, 0., 1e-14);

      WavevectorIndex ikFullIdx(ikFull);
      Eigen::VectorXd enAbs = abs.getEnergies(ikIdx);
      Eigen::VectorXd enFbs = fbs.getEnergies(ikFullIdx);

      double diff2 = 0.;
      for (int i = 0; i < enAbs.size(); i++) {
        // note: I have an ad-hoc assumption on energies in fbs being from
        // 0 to a cutoff band, which is only true in this particular test
        diff2 += pow(enAbs(i) - enFbs(i), 2);
      }
      EXPECT_NEAR(diff2, 0., 1.e-12);
    }
  }

  // Here I check that I can reconstruct the full list of points from
  // the irreducible points
  {
    std::vector<long> allKs;
    for (long ik : abs.irrPointsIterator()) {
      WavevectorIndex ikIdx(ik);
      Eigen::Vector3d kIrr = abs.getWavevector(ikIdx);
      auto rotations = abs.getRotationsStar(ikIdx);
      for (const auto &rot : rotations) {
        Eigen::Vector3d kRot = rot * kIrr;
        kRot = points.cartesianToCrystal(kRot);
        long ikFull = points.getIndex(kRot);
        allKs.push_back(ikFull);
      }
    }
    long count2 =
        std::distance(allKs.begin(), std::unique(allKs.begin(), allKs.end()));
    EXPECT_EQ(count2, abs.getNumPoints());
  }

  // let's build a matrix containing the band degeneracy
  std::vector<std::vector<int>> bandDegeneracies;
  for (int ik=0; ik<points.getNumPoints(); ++ik) {
    int numBands = fbs.getNumBands();
    WavevectorIndex ikIdx(ik);
    Eigen::VectorXd energies = fbs.getEnergies(ikIdx);
    std::vector<int> kDeg(numBands, 1);
    for (int ib = 0; ib < numBands; ++ib) {
      // first, we check if the band is degenerate, and the size of the
      // degenerate subspace
      int degDegree = 1;
      for (int ib2 = ib + 1; ib2 < numBands; ib2++) {
        // I consider bands degenerate if their energies are the same
        double rel = (energies(ib) - energies(ib2)) / energies(ib);
        if ( std::abs( rel ) > 1.0e-3) {
          break;
        }
        degDegree += 1;
      }
      for (int ib2=0; ib2<degDegree; ++ib2 ) {
        kDeg[ib+ib2] = degDegree;
      }
      // we skip the bands in the subspace, since we corrected them already
      ib += degDegree - 1;
    }
    bandDegeneracies.push_back(kDeg);
  }

  // here we check that the rotated q-point and velocities are similar
  // to those of the FullBandStructure without symmetries

  for (int is : abs.irrStateIterator()) {
    auto isIdx = StateIndex(is);
    WavevectorIndex ikIdx = std::get<0>(abs.getIndex(isIdx));
    BandIndex ibIdx = std::get<1>(abs.getIndex(isIdx));
    Eigen::Vector3d vIrr = abs.getGroupVelocity(isIdx);
    Eigen::Vector3d qIrr = abs.getWavevector(isIdx);
    auto rotations = abs.getRotationsStar(ikIdx);

    for (const Eigen::Matrix3d &r : rotations) {
      Eigen::Vector3d v = r * vIrr;
      Eigen::Vector3d q = r * qIrr;

      int ikFull = fbs.getPoints().getIndex(points.cartesianToCrystal(q));
      auto ikFullIdx = WavevectorIndex(ikFull);

      StateIndex isFullIdx(fbs.getIndex(ikFullIdx, ibIdx));
      Eigen::Vector3d v2 = fbs.getGroupVelocity(isFullIdx);
      Eigen::Vector3d q2 = fbs.getWavevector(ikFullIdx);

      q = points.bzToWs(q, Points::cartesianCoordinates);
      q2 = points.bzToWs(q2, Points::cartesianCoordinates);

      double diff = (q - q2).squaredNorm();
      EXPECT_NEAR(diff, 0., 1.0e-6);

      for (int i : {0, 1, 2}) {

    	double diff2 = std::abs(v(i) - v2(i)) * velocityRyToSi;
        EXPECT_NEAR(diff2, 0., 0.0015); // there's variation on this due to degeneracies
      }

      double en1 = fbs.getEnergy(isFullIdx);
      double en2 = abs.getEnergy(isIdx);
      ASSERT_TRUE( std::abs((en1-en2)/en1) < 1e-3);

      // if velocities are degenerate, can't test equality (easily)
      if ( bandDegeneracies[ikFull][ibIdx.get()] > 1 ) continue;
      if (v.norm() < 1e-5) continue;
      double diff2 = std::abs( (v - v2).norm() / v.norm());
      EXPECT_NEAR(diff2, 0., 0.001);
    }
  }

  // The group velocity has odd parity over the Brillouin zone
  // its integral should be zero
  {
    Eigen::Vector3d x = Eigen::Vector3d::Zero();
    for (int is : abs.irrStateIterator()) {
      auto isIdx = StateIndex(is);
      WavevectorIndex ikIdx = std::get<0>(abs.getIndex(isIdx));
      Eigen::Vector3d vIrr = abs.getGroupVelocity(isIdx);
      auto rotations = abs.getRotationsStar(ikIdx);
      for (const Eigen::Matrix3d &r : rotations) {
        Eigen::Vector3d v = r * vIrr;
        x += v;
      }
    }
    for (int i : {0, 1, 2}) {
      EXPECT_NEAR(x(i), 0., 1.0e-3);
    }
  }

  // now we test getRotationToIrreducible, which is used in the scattering
  // it's used to map a reducible point to a irreducible one
  for (long ik = 0; ik < abs.getNumPoints(); ik++) {
    WavevectorIndex ikIdx(ik);
    Eigen::Vector3d k = abs.getWavevector(ikIdx);

    auto t = abs.getRotationToIrreducible(k, Points::cartesianCoordinates);
    long ikIrr = std::get<0>(t);
    Eigen::Matrix3d rot = std::get<1>(t);

    WavevectorIndex ikIrrIdx(ikIrr);
    Eigen::Vector3d kIrr = abs.getWavevector(ikIrrIdx);
    Eigen::Vector3d k2 = rot * k;

    k2 = points.bzToWs(k2, Points::cartesianCoordinates);
    kIrr = points.bzToWs(kIrr, Points::cartesianCoordinates);

    double diff = (k2 - kIrr).squaredNorm();
    EXPECT_NEAR(diff, 0., 1.0e-6);
  }
}
