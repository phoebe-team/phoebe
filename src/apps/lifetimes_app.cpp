#include "lifetimes_app.h"
#include "bands_app.h"
#include "bandstructure.h"
#include "context.h"
#include "el_scattering_matrix.h"
#include "exceptions.h"
#include "ifc3_parser.h"
#include "points.h"
#include "ph_scattering_matrix.h"
#include "parser.h"

void ElectronLifetimesApp::run(Context &context) {
  context.setScatteringMatrixInMemory(false);

  auto t2 = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(t2);
  auto phononH0 = std::get<1>(t2);

  auto t1 = Parser::parseElHarmonicWannier(context, &crystal);
  auto crystalEl = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);

  // load the el-ph coupling
  // Note: this file contains the number of electrons
  // which is needed to understand where to place the fermi level
  auto couplingElPh = InteractionElPhWan::parse(context, crystal, phononH0);

  // set k and q point meshes and paths
  Points pathKPoints(crystal, context.getPathExtrema(),
                     context.getDeltaPath());
  auto kMesh = context.getKMesh();
  Points fullKPoints(crystal, kMesh);

  //----------------------------------------------------------------------------

  bool withVelocities = true;
  bool withEigenvectors = true;
  FullBandStructure fullBandStructure =
      electronH0.populate(fullKPoints, withVelocities, withEigenvectors);
  FullBandStructure pathBandStructure =
      electronH0.populate(pathKPoints, withVelocities, withEigenvectors);

  StatisticsSweep statisticsSweep(context, &fullBandStructure);

  //----------------------------------------------------------------------------
  
  // build/initialize the scattering matrix and the smearing
  ElScatteringMatrix scatteringMatrix(context, statisticsSweep,
                                      fullBandStructure, pathBandStructure,
                                      phononH0);
  scatteringMatrix.setup();

  scatteringMatrix.outputToJSON("path_el_relaxation_times.json");
  outputBandsToJSON(pathBandStructure, context, pathKPoints,
                    "path_el_bandstructure.json");

  mpi->barrier();
}

void PhononLifetimesApp::run(Context &context) {
  context.setScatteringMatrixInMemory(false);

  auto t2 = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(t2);
  auto phononH0 = std::get<1>(t2);

  // set k and q point meshes and paths
  Points pathPoints(crystal, context.getPathExtrema(),
                    context.getDeltaPath());
  Points fullPoints(crystal, context.getQMesh());

  //----------------------------------------------------------------------------

  bool withVelocities = true;
  bool withEigenvectors = true;
  FullBandStructure fullBandStructure =
      phononH0.populate(fullPoints, withVelocities, withEigenvectors);
  FullBandStructure pathBandStructure =
      phononH0.populate(pathPoints, withVelocities, withEigenvectors);

  StatisticsSweep statisticsSweep(context);

  //----------------------------------------------------------------------------

  // build/initialize the scattering matrix and the smearing
  PhScatteringMatrix scatteringMatrix(context, statisticsSweep,
                                      fullBandStructure, pathBandStructure,
                                      &phononH0);
  scatteringMatrix.setup();

  scatteringMatrix.outputToJSON("path_ph_relaxation_times.json");
  outputBandsToJSON(pathBandStructure, context, pathPoints,
                    "path_ph_bandstructure.json");
  mpi->barrier();
}

void ElectronLifetimesApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getPhFC2FileName(), "phFC2FileName");
  throwErrorIfUnset(context.getPathExtrema(), "points path extrema");
  throwErrorIfUnset(context.getElphFileName(), "elphFileName");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  throwErrorIfUnset(context.getSmearingMethod(), "smearingMethod");
  if (context.getSmearingMethod() == DeltaFunction::gaussian) {
    throwErrorIfUnset(context.getSmearingWidth(), "smearingWidth");
  }
  if (context.getDopings().size() == 0 &&
      context.getChemicalPotentials().size() == 0) {
    Error("Either chemical potentials or dopings must be set");
  }
}

void PhononLifetimesApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getPhFC2FileName(), "phFC2FileName");
  throwWarningIfUnset(context.getSumRuleFC2(), "sumRuleFC2");
  //throwErrorIfUnset(context.getPhFC3FileName(), "phFC3FileName");
  throwErrorIfUnset(context.getPathExtrema(), "points path extrema");
  throwErrorIfUnset(context.getQMesh(), "qMesh");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  throwErrorIfUnset(context.getSmearingMethod(), "smearingMethod");
  if (context.getSmearingMethod() == DeltaFunction::gaussian) {
    throwErrorIfUnset(context.getSmearingWidth(), "smearingWidth");
  }
}
