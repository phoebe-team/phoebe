#include "coupled_bte_app.h"
#include "bandstructure.h"
#include "context.h"
#include "drift.h"
#include "exceptions.h"
#include "ifc3_parser.h"
#include "coupled_observables.h"
#include "parser.h"
#include "coupled_scattering_matrix.h"
#include "points.h"
#include "specific_heat.h"
#include <iomanip>

void CoupledTransportApp::run(Context &context) {

  // there are four major possible contributions to this application
  // electron-phonon, phonon-phonon, phonon-electron, and electron-electron
  // Here we set these booleans based on if their relevant files are set
  // If elph files are available, we automatically calculate both phel and elph

  //bool useElElInteraction = !context.getElElFileName().empty(); // not implemented
  bool useElPhInteraction = !context.getElphFileName().empty();
  bool usePhPhInteraction = !context.getPhFC3FileName().empty();

  bool useCRTA = !std::isnan(context.getConstantRelaxationTime());
  if(useCRTA) { // this isn't a possibility here
    Error("CRTA is not an option for the coupled BTE app!");
  }

  if(!usePhPhInteraction && !useElPhInteraction) {
    Error("To run the coupled BTE app supply a ph-ph or el-ph file!");
  }

  // TODO add a warning message if anything but the population window is used or somehow allow separate windows?
  // TODO warning about not using adaptive smearing if gaussian is specified?

  // set up the electron and phonon hamiltonians from file --------------------------
  // Read the necessary input files
  auto tup = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  auto t1 = Parser::parseElHarmonicWannier(context, &crystal);
  auto crystalEl = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);
  // TODO maybe add a check that these crystals are the same

  // TODO if we are struggling here for memory, we will want to add elph or phph contributions one
  // at a time, and subsequently release the relevant interaction objects from memory to free space
  // This shouldn't be too hard -- instead of calling builder, we should be able to structure
  // things so it's possible to add contribution separately

  // read in phonon-phonon coupling -------------------------------------
  // TODO does this also checks that the crystal is the same one read in for ph FC2?
  Interaction3Ph coupling3Ph = IFC3Parser::parse(context, crystal);

  // read in electron-phonon coupling ---------------------------------
  // TODO does this also checks that the crystal is the same one read in for ph FC2?
  InteractionElPhWan couplingElPh = InteractionElPhWan::parse(context, crystal, &phononH0);

  // Set up phonon bandstructure information ---------------------------------------------
  if (mpi->mpiHead()) {
    std::cout << "\nComputing phonon band structure." << std::endl;
  }

  Points qPoints(crystal, context.getQMesh());
  auto tup1 = ActiveBandStructure::builder(context, phononH0, qPoints);
  auto phBandStructure = std::get<0>(tup1);
  auto phStatisticsSweep = std::get<1>(tup1);

  // print some info about state number reduction
  if (mpi->mpiHead()) {
    if(phBandStructure.hasWindow() != 0) {
        std::cout << "Window selection reduced phonon band structure from "
                << qPoints.getNumPoints() * phononH0.getNumBands() << " to "
                << phBandStructure.getNumStates() << " states."  << std::endl;
    }
    if(context.getUseSymmetries()) {
      std::cout << "Symmetries reduced phonon band structure from "
          << phBandStructure.getNumStates() << " to "
          << phBandStructure.irrStateIterator().size() << " states." << std::endl;
    }
    std::cout << "Done computing phonon band structure.\n" << std::endl;
  }

  // Set up electron bandstructure information ---------------------------------------------
  if (mpi->mpiHead()) {
    std::cout << "\nComputing electronic band structure." << std::endl;
  }

  Points kPoints(crystal, context.getKMesh());
  auto t3 = ActiveBandStructure::builder(context, electronH0, kPoints);
  auto elBandStructure = std::get<0>(t3);
  auto elStatisticsSweep = std::get<1>(t3);

  // stop the code if someone tries to run it with more than one value of (mu, T)
  if(elStatisticsSweep.getNumChemicalPotentials() != 1) {
      Error("Can't run coupled BTE solve with more than one chemical potential or temperature "
        "at a time, as this would take up far too much memory at once!");
  }

  // print some info about how window and symmetries have reduced el bands
  if (mpi->mpiHead()) {
    if(elBandStructure.hasWindow() != 0) {
        std::cout << "Window selection reduced electronic band structure from "
                << kPoints.getNumPoints() * electronH0.getNumBands() << " to "
                << elBandStructure.getNumStates() << " states."  << std::endl;
    }
    if(context.getUseSymmetries()) {
      std::cout << "Symmetries reduced electronic band structure from "
        << elBandStructure.getNumStates() << " to "
        << elBandStructure.irrStateIterator().size() << " states." << std::endl;
    }
    std::cout << "Done computing electronic band structure.\n" << std::endl;
  }

/*
  if(mpi->mpiHead()) {
    std::cout << "Checking for Cartesian wraparound issues." << std::endl;
    int pointsCount = 0;
    for (int is : elBandStructure.irrPointsIterator()) {
      auto isIdx = WavevectorIndex(is);
      auto q = elBandStructure.getWavevector(isIdx);
      q = elBandStructure.getPoints().bzToWs(q,Points::cartesianCoordinates);
      Eigen::Vector3d qminus = -1*q;
      qminus = elBandStructure.getPoints().cartesianToCrystal(qminus);
      int qmidx = elBandStructure.getPoints().getIndex(qminus);
      if(qmidx == -1) std::cout << is << " point " << q.transpose() << " not found." << std::endl;
      Eigen::Vector3d qminusLookup = elBandStructure.getPoints().getPointCoordinates(qmidx,Points::cartesianCoordinates);
      qminusLookup = elBandStructure.getPoints().bzToWs(qminusLookup,Points::cartesianCoordinates);
      if(abs((q+qminusLookup).norm()) > 1e-8) {
        pointsCount++;
        //std::cout << "Points in Cartesian: " << q.transpose() << " | " << qminusLookup.transpose() << " | " << (q+qminusLookup).norm() << std::endl;
        //q = elBandStructure.getPoints().cartesianToCrystal(q);
        //if(abs((q+qminusLookup).norm()) > 1e-8) std::cout << "points crystal " << q.transpose() << " | " << qminus.transpose() << " | " << (q+qminus).norm() << std::endl;
      }
    }
    if(pointsCount != 0) std::cout << "Found " << pointsCount << " points that do not wraparound properly.\n" << std::endl;
  }
*/
  // Construct the full C matrix
  // the dimensions of this matrix are (numElStates + numPhStates, numElStates + numPhStates)
  // we provide the el stat sweep because we need the one that has information about the
  // electronic chemical potential (the ph one just has mu = 0)
  CoupledScatteringMatrix scatteringMatrix(context, elStatisticsSweep,
                                        phBandStructure, elBandStructure,
                                        &coupling3Ph, &couplingElPh,
                                        &electronH0, &phononH0);
  scatteringMatrix.setup();   // adds in all the scattering rates

  // BTE Solvers -----------------------------------------------------
  // For now, in this case, we just run the relaxons solver.
  // We could also output RTA and other info if we wanted to.

  std::vector<std::string> solverBTE = context.getSolverBTE();

  bool doIterative = false;
  bool doVariational = false;
  bool doRelaxons = false;
  for (const auto &s : solverBTE) {
    if (s.compare("iterative") == 0) {   doIterative = true; }
    if (s.compare("variational") == 0) {  doVariational = true; }
    if (s.compare("relaxons") == 0) {    doRelaxons = true; }
  }

  // here we do validation of the input, to check for consistency
  if (doRelaxons && !context.getScatteringMatrixInMemory()) {
    Error("Relaxons require matrix kept in memory");
  }
  if (doRelaxons && context.getUseSymmetries()) {
    Error("Relaxon solver only works without symmetries");
    // Note: this is a problem of the theory I suppose
    // because the scattering matrix may not be anymore symmetric
    // or may require some modifications to make it work
    // that we didn't have yet thought of
  }
  if (context.getScatteringMatrixInMemory() && elStatisticsSweep.getNumCalculations() != 1) {
    Error("If scattering matrix is kept in memory, only one "
          "temperature/chemical potential is allowed in a run");
  }

  if (context.getScatteringMatrixInMemory() && !context.getUseSymmetries()) {
    if (doVariational || doRelaxons || doIterative) {
      if ( context.getSymmetrizeMatrix() ) {
        // reinforce the condition that the scattering matrix is symmetric
        // A -> ( A^T + A ) / 2
        scatteringMatrix.symmetrize();
      }
    }
  }
  //scatteringMatrix.outputToHDF5("coupledMatrix.hdf5");


if(doIterative || doVariational) {
    Warning("Coupled BTE app only implmeneted for relaxons solver.");
  }

  if (doRelaxons) {

    if (mpi->mpiHead()) {
      std::cout << "Starting relaxons BTE solver." << std::endl;
    }

    // Calculate Du(i,j) before we diagonalize the matrix and ruin it
    // to calculate D we need the phi vectors, so we here calculate ahead of time
    // here -- they are saved internally to the class
    CoupledCoefficients coupledCoeffs(elStatisticsSweep, crystal, context);
    coupledCoeffs.calcSpecialEigenvectors(elStatisticsSweep, &phBandStructure, &elBandStructure);
    coupledCoeffs.outputDuToJSON(scatteringMatrix, context);

    // diagonalize the coupled matrix
    auto tup2 = scatteringMatrix.diagonalize();
    // EV such that Omega = V D V^-1
    // eigenvectors(state index, eigenvalue index)
    auto eigenvalues = std::get<0>(tup2);
    auto eigenvectors = std::get<1>(tup2);

    // calculate the el and ph specific heats, which are used
    // by the transport calculation.
    SpecificHeat phSpecificHeat(context, phStatisticsSweep, crystal, phBandStructure);
    phSpecificHeat.calc();

    SpecificHeat elSpecificHeat(context, elStatisticsSweep, crystal, elBandStructure);
    elSpecificHeat.calc();

    // calculate the transport properties and viscosity
    coupledCoeffs.calcFromRelaxons(scatteringMatrix, phSpecificHeat, elSpecificHeat, eigenvalues, eigenvectors);
    coupledCoeffs.print();
    // note: viscosities are output by default internally in calcFromRelaxons
    coupledCoeffs.outputToJSON("coupled_relaxons_transport_coeffs.json");

    // output relaxation times (the eigenvalues)
    // TODO also write the relaxons visulation function?
    scatteringMatrix.relaxonsToJSON("coupled_relaxons_relaxation_times.json", eigenvalues);

    if (mpi->mpiHead()) {
      std::cout << "Finished relaxons BTE solver\n\n";
      std::cout << std::string(80, '-') << "\n" << std::endl;
    }
  } // finish relaxons solver

  mpi->barrier();
}

void CoupledTransportApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getPhFC2FileName(), "PhFC2FileName");
  throwErrorIfUnset(context.getQMesh(), "qMesh");
  throwWarningIfUnset(context.getSumRuleFC2(), "sumRuleFC2");
  throwErrorIfUnset(context.getPhFC3FileName(), "PhFC3FileName");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  throwErrorIfUnset(context.getSmearingMethod(), "smearingMethod");
  if (context.getSmearingMethod() == DeltaFunction::gaussian) {
    throwErrorIfUnset(context.getSmearingWidth(), "smearingWidth");
  }
  if (!context.getElphFileName().empty()) {
    throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
    throwErrorIfUnset(context.getKMesh(), "kMesh");
    if (context.getDopings().size() == 0 &&
        context.getChemicalPotentials().size() == 0) {
      Error("Either chemical potentials or dopings must be set");
    }
  }
}
