#include "electron_wannier_transport_app.h"
#include "bandstructure.h"
#include "context.h"
#include "drift.h"
#include "el_scattering_matrix.h"
#include "electron_viscosity.h"
#include "exceptions.h"
#include "observable.h"
#include "onsager.h"
#include "parser.h"
#include "wigner_electron.h"

void ElectronWannierTransportApp::run(Context &context) {

  auto t2 = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(t2);
  auto phononH0 = std::get<1>(t2);

  auto t1 = Parser::parseElHarmonicWannier(context, &crystal);
  auto crystalEl = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);

  // load the elph coupling
  // Note: this file contains the number of electrons
  // which is needed to understand where to place the fermi level
  InteractionElPhWan couplingElPh(crystal);
  if (std::isnan(context.getConstantRelaxationTime())) {
    couplingElPh = InteractionElPhWan::parse(context, crystal, &phononH0);
  }

  // compute the band structure on the fine grid
  if (mpi->mpiHead()) {
    std::cout << "\nComputing electronic band structure." << std::endl;
  }
  Points fullPoints(crystal, context.getKMesh());
  auto t3 = ActiveBandStructure::builder(context, electronH0, fullPoints);
  auto bandStructure = std::get<0>(t3);
  auto statisticsSweep = std::get<1>(t3);

  // print some info about how window and symmetries have reduced things
  if (mpi->mpiHead()) {
    if(bandStructure.hasWindow() != 0) {
        std::cout << "Window selection reduced electronic band structure from "
                << fullPoints.getNumPoints()*electronH0.getNumBands() << " to "
                << bandStructure.getNumStates() << " states."  << std::endl;
    }
    if(context.getUseSymmetries()) {
      std::cout << "Symmetries reduced electronic band structure from "
          << bandStructure.getNumStates() << " to "
          << bandStructure.irrStateIterator().size() << " states." << std::endl;
    }
    std::cout << "Done computing electronic band structure.\n" << std::endl;
  }

  // Old code for using all the band structure
  //  bool withVelocities = true;
  //  bool withEigenvectors = true;
  //  FullBandStructure bandStructure = electronH0.populate(
  //      fullPoints, withVelocities, withEigenvectors);
  //  // set the chemical potentials to zero, load temperatures
  //  StatisticsSweep statisticsSweep(context, &bandStructure);

  // build/initialize the scattering matrix and the smearing
  ElScatteringMatrix scatteringMatrix(context, statisticsSweep, bandStructure,
                                      bandStructure, phononH0, &couplingElPh);
  scatteringMatrix.setup();
  scatteringMatrix.outputToJSON("rta_el_relaxation_times.json");

  // solve the BTE at the relaxation time approximation level
  // we always do this, as it's the cheapest solver and is required to know
  // the diagonal for the exact method.

  if (mpi->mpiHead()) {
    std::cout << "\n" << std::string(80, '-') << "\n\n";
    std::cout << "Solving BTE within the relaxation time approximation.\n";
  }

  // compute the phonon populations in the relaxation time approximation.
  // Note: this is the total phonon population n (n != f(1+f) Delta n)

  BulkEDrift driftE(statisticsSweep, bandStructure, 3);
  BulkTDrift driftT(statisticsSweep, bandStructure, 3);
  VectorBTE relaxationTimes = scatteringMatrix.getSingleModeTimes();
  VectorBTE nERTA = -driftE * relaxationTimes;
  VectorBTE nTRTA = -driftT * relaxationTimes;

  // compute the electrical conductivity
  OnsagerCoefficients transportCoefficients(statisticsSweep, crystal,
                                            bandStructure, context);
  transportCoefficients.calcFromPopulation(nERTA, nTRTA);
  transportCoefficients.print();
  transportCoefficients.outputToJSON("rta_onsager_coefficients.json");

  // compute the Wigner transport coefficients
  WignerElCoefficients wignerCoefficients(
      statisticsSweep, crystal, bandStructure, context, relaxationTimes);
  wignerCoefficients.calcFromPopulation(nERTA, nTRTA);
  wignerCoefficients.print();
  wignerCoefficients.outputToJSON("rta_wigner_coefficients.json");

  // compute the electron viscosity
  ElectronViscosity elViscosity(context, statisticsSweep, crystal,
                                bandStructure);
  elViscosity.calcRTA(relaxationTimes);
  elViscosity.print();
  elViscosity.outputToJSON("rta_electron_viscosity.json");

  // compute the specific heat
  SpecificHeat specificHeat(context, statisticsSweep, crystal, bandStructure);
  specificHeat.calc();
  specificHeat.print();
  specificHeat.outputToJSON("el_specific_heat.json");

  if (mpi->mpiHead()) {
    std::cout << "\n" << std::string(80, '-') << "\n" << std::endl;
  }

  if (!std::isnan(context.getConstantRelaxationTime())) {
    return; // if we used the constant RTA, we can't solve the BTE exactly
  }

  //---------------------------------------------------------------------------
  // start section on exact BTE solvers

  std::vector<std::string> solverBTE = context.getSolverBTE();

  bool doIterative = false;
  bool doVariational = false;
  bool doRelaxons = false;
  for (const std::string &s : solverBTE) {
    if (s.compare("iterative") == 0)
      doIterative = true;
    if (s.compare("variational") == 0)
      doVariational = true;
    if (s.compare("relaxons") == 0)
      doRelaxons = true;
  }

  // here we do validation of the input, to check for consistency
  if (doRelaxons && !context.getScatteringMatrixInMemory()) {
    Error("Relaxons require matrix kept in memory");
  }
  if (context.getScatteringMatrixInMemory() &&
      statisticsSweep.getNumCalculations() != 1) {
    Error("If scattering matrix is kept in memory, only one "
          "temperature/chemical potential is allowed in a run");
  }
  if (doRelaxons && context.getUseSymmetries()) {
    Error("Relaxon solver only works without symmetries");
    // Note: this is a problem of the theory I suppose
    // because the scattering matrix may not be anymore symmetric
    // or may require some modifications to make it work
    // that we didn't have yet thought of
  }
  if (doVariational && context.getUseSymmetries()) {
    Error("Variational solver only works without symmetries");
    // Note: this is a problem of the theory I suppose
    // because the scattering matrix may not be anymore symmetric
    // or may require some modifications to make it work
    // that we didn't yet think of
  }

  if (context.getScatteringMatrixInMemory() && !context.getUseSymmetries()) {
    if (doVariational || doRelaxons || doIterative) {
      if ( context.getSymmetrizeMatrix() ) {
        // reinforce the condition that the scattering matrix is symmetric
        // A -> ( A^T + A ) / 2
        // this helps removing negative eigenvalues which may appear due to noise
        scatteringMatrix.symmetrize();
      }
    }
  }
  //scatteringMatrix.outputToHDF5("el.scatteringMatrix.hdf5");


  if (doIterative) {
    runIterativeMethod(context, crystal, statisticsSweep, bandStructure,
                       scatteringMatrix);
  }

  if (doVariational) {
    runVariationalMethod(context, crystal, statisticsSweep, bandStructure,
                         scatteringMatrix);
  }

  if (doRelaxons) {
    if (mpi->mpiHead()) {
      std::cout << "Starting relaxons BTE solver" << std::endl;
    }
    // scatteringMatrix.a2Omega(); Important!! Must use the matrix A, non-scaled
    // this is because the scaling used for phonons here would cause instability
    // as it could contains factors like 1/0
    auto tup = scatteringMatrix.diagonalize(context.getNumRelaxonsEigenvalues());

    Eigen::VectorXd eigenvalues = std::get<0>(tup);
    ParallelMatrix<double> eigenvectors = std::get<1>(tup);
    // EV such that Omega = V D V^-1

    transportCoefficients.calcFromRelaxons(eigenvalues, eigenvectors,
                                           scatteringMatrix);
    transportCoefficients.print();
    transportCoefficients.outputToJSON("relaxons_onsager_coefficients.json");
    scatteringMatrix.relaxonsToJSON("el_relaxons_relaxation_times.json", eigenvalues);

    if (!context.getUseSymmetries()) {
      elViscosity.calcFromRelaxons(eigenvalues, eigenvectors);
      elViscosity.print();
      elViscosity.outputToJSON("relaxons_electron_viscosity.json");
    }

    if (mpi->mpiHead()) {
      std::cout << "Finished relaxons BTE solver\n\n";
      std::cout << std::string(80, '-') << "\n" << std::endl;
    }
  }
}

void ElectronWannierTransportApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");
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

  double minChemicalPotential = context.getMinChemicalPotential();
  double maxChemicalPotential = context.getMaxChemicalPotential();
  double deltaChemicalPotential = context.getDeltaChemicalPotential();
  if (context.getDopings().size() == 0 && context.getChemicalPotentials().size() == 0 &&
     (std::isnan(minChemicalPotential) || std::isnan(maxChemicalPotential) || std::isnan(deltaChemicalPotential)))  {
    Error("Either chemical potentials or dopings must be set");
  }
}

void ElectronWannierTransportApp::runVariationalMethod(
    Context &context, Crystal &crystal, StatisticsSweep &statisticsSweep,
    ActiveBandStructure &bandStructure, ElScatteringMatrix &scatteringMatrix) {

  // here we implement a conjugate gradient solution to Az = b

  if (mpi->mpiHead()) {
    std::cout << "Starting variational BTE solver\n";
    std::cout << std::endl;
  }

//  auto particle = bandStructure.getParticle();
  OnsagerCoefficients transportCoefficients(statisticsSweep, crystal,
                                            bandStructure, context);

  // initialize the transport coefficients
  // to check how these change during iterations
  Eigen::Tensor<double, 3> elCond =
      transportCoefficients.getElectricalConductivity();
  Eigen::Tensor<double, 3> thCond =
      transportCoefficients.getThermalConductivity();
  auto elCondOld = elCond.setConstant(1.);
  auto thCondOld = thCond.setConstant(1.);

  // initialize b
  // Note: we solve Az=b not with z as electron population, but as canonical
  // population, i.e. with z being f = f0 + f0 (1-f0) z
  // where f0 is the Fermi-Dirac distribution
  // so, here we compute b without the factor n(1-n)
  // note also, we cannot do divisions by n(1-n) because it could be zero
  VectorBTE preconditioning(statisticsSweep, bandStructure, 3);
  preconditioning.setConst(1.);

  // symmetrized b vectors
  BulkEDrift bE(statisticsSweep, bandStructure, 3, true);
  BulkTDrift bT(statisticsSweep, bandStructure, 3, true);
  bE.data *= -1.; // because the scattering matrix should have a -1 sign
  bT.data *= -1.; // and we are solving b = -Af

  // populations, initial guess
  VectorBTE zNewE = bE;
  VectorBTE zNewT = bT;
  VectorBTE zE = zNewE;
  VectorBTE zT = zNewT;

  std::vector<VectorBTE> inVec;
  inVec.push_back(zE);
  inVec.push_back(zT);
  std::vector<VectorBTE> w0s = scatteringMatrix.dot(inVec);
  VectorBTE w0E = w0s[0];
  VectorBTE w0T = w0s[1];

  // residual
  VectorBTE rE = bE - w0E;
  VectorBTE rT = bT - w0T;

  // search direction
  VectorBTE dE = rE;
  VectorBTE dT = rT;

  double threshold = context.getConvergenceThresholdBTE();

  for (int iter = 0; iter < context.getMaxIterationsBTE(); iter++) {
    // execute CG step, as in
    // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    // compute w = A d
    std::vector<VectorBTE> inW;
    inW.push_back(dE);
    inW.push_back(dT);
    std::vector<VectorBTE> outW = scatteringMatrix.dot(inW);
    VectorBTE wE = outW[0];
    VectorBTE wT = outW[1];

    // amount of descent along the search direction
//    Eigen::MatrixXd alphaE = (rE.dot(rE)).array() / (dE.dot(wE)).array();
//    Eigen::MatrixXd alphaT = (rT.dot(rT)).array() / (dT.dot(wT)).array();
    int numCalculations = statisticsSweep.getNumCalculations();
    Eigen::MatrixXd alphaE = Eigen::MatrixXd::Zero(numCalculations,3);
    Eigen::MatrixXd alphaT = Eigen::MatrixXd::Zero(numCalculations,3);
    {
      Eigen::MatrixXd numE = rE.dot(rE);
      Eigen::MatrixXd numT = rT.dot(rT);
      Eigen::MatrixXd denE = dE.dot(wE);
      Eigen::MatrixXd denT = dT.dot(wT);
      for (int iCalc=0; iCalc<numCalculations; iCalc++) {
        for (int i : {0,1,2}) {
          if (denE(iCalc,i) != 0.) {
            alphaE(iCalc, i) = numE(iCalc, i) / denE(iCalc, i);
          }
          if (denT(iCalc,i) != 0.) {
            alphaT(iCalc, i) = numT(iCalc, i) / denT(iCalc, i);
          }
        }
      }
    }

    // new guess of population
    zNewE = dE * alphaE + zE;
    zNewT = dT * alphaT + zT;

    // new estimate of residual
    VectorBTE tmpE = wE * alphaE;
    VectorBTE tmpT = wT * alphaT;
    VectorBTE rNewE = rE - tmpE;
    VectorBTE rNewT = rT - tmpT;

    // amount of correction for the search direction
//    Eigen::MatrixXd betaE = (rNewE.dot(rNewE)).array() / (rE.dot(rE)).array();
//    Eigen::MatrixXd betaT = (rNewT.dot(rNewT)).array() / (rT.dot(rT)).array();
    Eigen::MatrixXd betaE = Eigen::MatrixXd::Zero(numCalculations,3);
    Eigen::MatrixXd betaT = Eigen::MatrixXd::Zero(numCalculations,3);
    {
      Eigen::MatrixXd numE = rNewE.dot(rNewE);
      Eigen::MatrixXd numT = rNewT.dot(rNewT);
      Eigen::MatrixXd denE = rE.dot(rE);
      Eigen::MatrixXd denT = rT.dot(rT);
      for (int iCalc=0; iCalc<numCalculations; iCalc++) {
        for (int i : {0,1,2}) {
          if (denE(iCalc,i) != 0.) {
            betaE(iCalc, i) = numE(iCalc, i) / denE(iCalc, i);
          }
          if (denT(iCalc,i) != 0.) {
            betaT(iCalc, i) = numT(iCalc, i) / denT(iCalc, i);
          }
        }
      }
    }

    // new search direction
    VectorBTE dNewE = dE * betaE + rNewE;
    VectorBTE dNewT = dT * betaT + rNewT;

    // now we update the guess of transport coefficients
    // first though, we add the factors n(1-n) that we removed from the CG
    // we need to do this because we didn't remove this factor from sMatrix
//    VectorBTE z2E = zNewE;
//    VectorBTE z2T = zNewT;
//    auto irrStates = bandStructure.irrStateIterator();
//    size_t numIrrStates = irrStates.size();
//#pragma omp parallel for
//    for (size_t iss=0; iss<numIrrStates; iss++) {
//      int is = irrStates[iss];
//      StateIndex isIdx(is);
//      double energy = bandStructure.getEnergy(isIdx);
//      int iBte = bandStructure.stateToBte(isIdx).get();
//      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
//        auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
//        auto chemPot = calcStat.chemicalPotential;
//        auto temp = calcStat.temperature;
//        double pop = sqrt(particle.getPopPopPm1(energy, temp, chemPot));
//        for (int i : {0, 1, 2}) {
//          z2E(iCalc, i, iBte) *= pop;
//          z2T(iCalc, i, iBte) *= pop;
//        }
//      }
//    }
    std::vector<VectorBTE> inF;
    inF.push_back(zNewE);
    inF.push_back(zNewT);
    auto outF = scatteringMatrix.dot(inF);
    VectorBTE az2E = outF[0];
    VectorBTE az2T = outF[1];
    transportCoefficients.calcVariational(az2E, az2T, zNewE, zNewT, bE, bT,
                                          preconditioning);
    transportCoefficients.print(iter);
    elCond = transportCoefficients.getElectricalConductivity();
    thCond = transportCoefficients.getThermalConductivity();

    // decide whether to exit or run the next iteration
    double deltaE = findMaxRelativeDifference(elCond, elCondOld);
    double deltaT = findMaxRelativeDifference(thCond, thCondOld);
    if ((deltaE < threshold) && (deltaT < threshold)) {
      // recompute our final guess for transport coefficients
      transportCoefficients.calcFromSymmetricPopulation(zNewE, zNewT);
      break;
    } else {
      elCondOld = elCond;
      thCondOld = thCond;
      zE = zNewE;
      zT = zNewT;
      rE = rNewE;
      rT = rNewT;
      dE = dNewE;
      dT = dNewT;
    }

    if (iter == context.getMaxIterationsBTE() - 1) {
      Error("Reached max BTE iterations without convergence");
    }
  }

  // nice formatting of the transport properties at the last step
  transportCoefficients.print();
  transportCoefficients.outputToJSON("variational_onsager_coefficients.json");

  if (mpi->mpiHead()) {
    std::cout << "Finished variational BTE solver\n\n";
    std::cout << std::string(80, '-') << "\n" << std::endl;
  }
}


void ElectronWannierTransportApp::runIterativeMethod(
    Context &context, Crystal &crystal, StatisticsSweep &statisticsSweep,
    ActiveBandStructure &bandStructure, ElScatteringMatrix &scatteringMatrix) {

  // here we implement a conjugate gradient solution to Az = b

  if (mpi->mpiHead()) {
    std::cout << "Starting Omini Sparavigna BTE solver\n" << std::endl;
  }

  OnsagerCoefficients transportCoefficients(statisticsSweep, crystal,
                                            bandStructure, context);

  // initialize the transport coefficients
  // to check how these change during iterations
  Eigen::Tensor<double, 3> elCond =
      transportCoefficients.getElectricalConductivity();
  Eigen::Tensor<double, 3> thCond =
      transportCoefficients.getThermalConductivity();
  auto elCondOld = elCond.setConstant(1.);
  auto thCondOld = thCond.setConstant(1.);

  VectorBTE nENext(statisticsSweep, bandStructure, 3);
  VectorBTE nTNext(statisticsSweep, bandStructure, 3);

  VectorBTE lineWidths = scatteringMatrix.getLinewidths();

  // we get the symmetrized drift vectors
  BulkEDrift driftE(statisticsSweep, bandStructure, 3, true);
  BulkTDrift driftT(statisticsSweep, bandStructure, 3, true);
  VectorBTE relaxationTimes = scatteringMatrix.getSingleModeTimes();
  VectorBTE nERTA = -driftE * relaxationTimes;
  VectorBTE nTRTA = -driftT * relaxationTimes;

  VectorBTE nEOld = nERTA;
  VectorBTE nTOld = nTRTA;

  auto threshold = context.getConvergenceThresholdBTE();

  for (int iter = 0; iter < context.getMaxIterationsBTE(); iter++) {

    std::vector<VectorBTE> nIn;
    nIn.push_back(nEOld);
    nIn.push_back(nTOld);
    auto nOut = scatteringMatrix.offDiagonalDot(nIn);
    nENext = nOut[0] / lineWidths;
    nTNext = nOut[1] / lineWidths;
    nENext = nERTA - nENext;
    nTNext = nTRTA - nTNext;

    transportCoefficients.calcFromSymmetricPopulation(nENext, nTNext);
    transportCoefficients.print(iter);
    elCond = transportCoefficients.getElectricalConductivity();
    thCond = transportCoefficients.getThermalConductivity();

    // this exit condition must be improved
    // different temperatures might converge differently

    double dE = findMaxRelativeDifference(elCond, elCondOld);
    double dT = findMaxRelativeDifference(thCond, thCondOld);
    if ((dE < threshold) && (dT < threshold)) {
      break;
    } else {
      elCondOld = elCond;
      thCondOld = thCond;
      nEOld = nENext;
      nTOld = nTNext;
    }

    if (iter == context.getMaxIterationsBTE() - 1) {
      Error("Reached max BTE iterations without convergence");
    }
  }
  transportCoefficients.print();
  transportCoefficients.outputToJSON("omini_onsager_coefficients.json");

  if (mpi->mpiHead()) {
    std::cout << "Finished Omini-Sparavigna BTE solver\n\n";
    std::cout << std::string(80, '-') << "\n" << std::endl;
  }
}
