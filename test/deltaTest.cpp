#include "bandstructure.h"
#include "ifc3_parser.h"
#include "ph_scattering.h"
#include "points.h"
#include "parser.h"
#include "gtest/gtest.h"
#include <fstream>

TEST(deltaFunctions, testAllDeltas) {

  // Here we test the different kinds of delta functions which can be used in Phoebe
  // Because it's quicker, we test it using phonons

  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setPhFC3FileName("../test/data/FORCE_CONSTANTS_3RD");
  context.setSumRuleFC2("simple");

  // basic hamiltonian properties
  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);
  int numBands = phononH0.getNumBands();

  // setup parameters for active band structure creation
  Eigen::Vector3i qMesh;
  qMesh << 5, 5, 5;
  Points points(crystal, qMesh);
  bool withVelocities = true;
  bool withEigenvectors = true;

  // create an undistributed full bandstructure
  FullBandStructure phononBandStructure = phononH0.populate(points, withVelocities, withEigenvectors);

  // calculate energies and velocities at three different points
  // defined by conservation of momentum
  int iq1 = 5;
  int iq2 = 7;

//int numPoints = points.getNumPoints();

//for( int iq1 = 0; iq1 < numPoints; iq1++) {
//for( int iq2 = 0; iq2 < numPoints; iq2++) {

  WavevectorIndex iq1Idx(iq1);
  WavevectorIndex iq2Idx(iq2);

  Eigen::Vector3d q1 = phononBandStructure.getWavevector(iq1Idx);
  Eigen::Vector3d q2 = phononBandStructure.getWavevector(iq2Idx);
  Eigen::Vector3d q3 = q2 - q1;
  q3 =  phononBandStructure.getPoints().cartesianToCrystal(q3);
  int iq3 = phononBandStructure.getPoints().getIndex(q3);
  WavevectorIndex iq3Idx(iq3);

//  for(int ib1 = 0; ib1 < numBands; ib1++)  {
//    for(int ib2 = 0; ib2 < numBands; ib2++)  {
//      for(int ib3 = 0; ib3 < numBands; ib3++)  {

// TODO move this into unmodded phoebe and see if adaptive still gives zero...

  int ib1 = 5;
  int ib2 = 5;
  int ib3 = 5;

  BandIndex ib1Idx(ib1);
  BandIndex ib2Idx(ib2);
  BandIndex ib3Idx(ib3);
  StateIndex is1 = StateIndex(phononBandStructure.getIndex(iq1Idx,ib1Idx));
  StateIndex is2 = StateIndex(phononBandStructure.getIndex(iq2Idx,ib2Idx));
  StateIndex is3 = StateIndex(phononBandStructure.getIndex(iq3Idx,ib3Idx));

  double en1 = phononBandStructure.getEnergy(is1);
  double en2 = phononBandStructure.getEnergy(is2);
  double en3 = phononBandStructure.getEnergy(is3);
  Eigen::Vector3d v1 = phononBandStructure.getGroupVelocity(is1);
  Eigen::Vector3d v2 = phononBandStructure.getGroupVelocity(is2);
  Eigen::Vector3d v3 = phononBandStructure.getGroupVelocity(is3);

  double enDiff = en1 + en2 - en3;
  //std::cout << "endiff bands " << enDiff << " " << ib1 << " " << ib2 << " " << ib3 << std::endl;
  //std::cout << "qs is en " << iq1 << " " << iq2 << " " << iq3 << " " << is1.get() << " " << is2.get() << " " << is3.get() << " " << en1 << " " << en2 << " " << en3 << std::endl;

//  context.setSmearingWidth(0.01);
//  GaussianDeltaFunction smearing1(context);
//  double delta = smearing1.getSmearing(enDiff);
//  std::cout << "gaussian " << delta << std::endl;
  //ASSERT_EQ(delta, );

  AdaptiveGaussianDeltaFunction smearing2(phononBandStructure);
  Eigen::Vector3d vdiff = v2 - v3;
  double delta = smearing2.getSmearing(enDiff, vdiff);
//  if(delta > 0) {
//  std::cout << "adaptive gaussian " << delta;
//  std::cout << "qs is en " << iq1 << " " << iq2 << " " << iq3 << " " << is1.get() << " " << is2.get() << " " << is3.get() << " " << en1 << " " << en2 << " " << en3 << std::endl;

  SymAdaptiveGaussianDeltaFunction smearing3(phononBandStructure);
  delta = smearing3.getSmearing(enDiff, v1, v2, v3);
  //std::cout << " sym " << delta << std::endl;
  //ASSERT_EQ(delta, );
  //TetrahedronDeltaFunction smearing4(phononBandStructure);
  //delta = smearing4.getSmearing(en3 - en1, is2);
  //ASSERT_EQ(delta, );
  //std::cout << "tetrahedron " << delta << std::endl;
//      }
//break;
//    }
//  }
//} // wavevector
//}

}

