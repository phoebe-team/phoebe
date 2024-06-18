#include "ph_el_lifetimes.h"
#include "bandstructure.h"
#include "context.h"
#include "phel_scattering_matrix.h"
#include "exceptions.h"
#include "parser.h"

void PhElLifetimesApp::run(Context &context) {

  // load harmonic hamiltonians
  auto t2 = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(t2);
  auto phononH0 = std::get<1>(t2);

  auto t1 = Parser::parseElHarmonicWannier(context, &crystal);
  auto crystalEl = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);

  // load the elph coupling
  // Note: this file contains the number of electrons
  // which is needed to understand where to place the fermi level
  auto couplingElPh = InteractionElPhWan::parse(context, crystal, phononH0);

  // Compute the window filtered phonon band structure ---------------------------
  if (mpi->mpiHead()) {
    std::cout << "\nComputing phonon band structure." << std::endl;
  }

  bool withEigenvectors = true;
  bool withVelocities = true;
  Points qPoints(crystal, context.getQMesh());
  auto t3 = ActiveBandStructure::builder(context, phononH0, qPoints, withEigenvectors, withVelocities);
  auto phBandStructure = std::get<0>(t3);
  StatisticsSweep statisticsSweep = std::get<1>(t3);

  // print some info about how window and symmetries have reduced things
  if (mpi->mpiHead()) {
    if(phBandStructure.hasWindow() != 0) {
        std::cout << "Window selection reduced phonon band structure from "
          << qPoints.getNumPoints() * phononH0.getNumBands() << " to "
          << phBandStructure.getNumStates() << " states."  << std::endl;
    }
    if(context.getUseSymmetries()) {
      std::cout << "Symmetries reduced phonon band structure from "
          << phBandStructure.getNumStates() << " to "
          << phBandStructure.irrStateIterator().size() << " states.\n" << std::endl;
    }
    std::cout << "Done computing phonon band structure.\n" << std::endl;
  }

  // build/initialize the scattering matrix and the smearing
  PhElScatteringMatrix scatteringMatrix(context, statisticsSweep,
                                        phBandStructure,
                                        &couplingElPh, &electronH0);
  scatteringMatrix.setup();
  scatteringMatrix.outputToJSON("rta_phel_relaxation_times.json");

  // solve the BTE at the relaxation time approximation level
  // we always do this, as it's the cheapest solver and is required to know
  // the diagonal for the exact method.

  if (mpi->mpiHead()) {
    std::cout << "\n" << std::string(80, '-') << "\n\n";
    std::cout << "Phonon-electron lifetimes computed.\n";
  }
}

void PhElLifetimesApp::checkRequirements(Context &context) {

  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getPhFC2FileName(), "phFC2FileName");
  throwErrorIfUnset(context.getQMesh(), "qMesh");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");

  if (std::isnan(context.getConstantRelaxationTime())) { // non constant tau
    throwErrorIfUnset(context.getElphFileName(), "elphFileName");
    throwErrorIfUnset(context.getSmearingMethod(), "smearingMethod");
    if (context.getSmearingMethod() == DeltaFunction::gaussian) {
      throwErrorIfUnset(context.getSmearingWidth(), "smearingWidth");
    }
  } else {
    if (std::isnan(context.getNumOccupiedStates()) &&
        std::isnan(context.getFermiLevel())) {
      Error("For constant tau calculations, you must provide either the number "
            "of occupied Kohn-Sham states in the valence band or the Fermi "
            "level at T=0K");
    }
  }
  if (context.getDopings().size() == 0 &&
      context.getChemicalPotentials().size() == 0) {
    Error("Either chemical potentials or dopings must be set");
  }
}
