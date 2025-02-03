#include "coupled_observables.h"
#include "onsager_utilities.h"
#include <nlohmann/json.hpp>
#include "viscosity_io.h"
#include "io.h"

CoupledCoefficients::CoupledCoefficients(StatisticsSweep& statisticsSweep_,
                                         Crystal &crystal_, Context &context_)
    : statisticsSweep(statisticsSweep_), crystal(crystal_), context(context_) {

  // TODO : change this to use the context getSpinDegeneracyFactor
  if (context.getHasSpinOrbit()) { spinFactor = 1.;
  } else {  spinFactor = 2.; }

  // matrix had to be in memory for this calculation.
  // therefore, we can only ever have one numCalc
  numCalculations = 1;
  dimensionality = crystal.getDimensionality();

  sigma.resize(numCalculations, dimensionality, dimensionality);
  seebeck.resize(numCalculations, dimensionality, dimensionality);
  kappa.resize(numCalculations, dimensionality, dimensionality);
  mobility.resize(numCalculations, dimensionality, dimensionality);
  alpha.resize(numCalculations, dimensionality, dimensionality);
  sigma.setZero();
  seebeck.setZero();
  kappa.setZero();
  mobility.setZero();
  alpha.setZero();

  // initialize the tensors to be computed without assumption about the
  // separate elph states when drag is present
  sigmaTotal.resize(numCalculations, dimensionality, dimensionality);
  seebeckTotal.resize(numCalculations, dimensionality, dimensionality);
  kappaTotal.resize(numCalculations, dimensionality, dimensionality);
  mobilityTotal.resize(numCalculations, dimensionality, dimensionality);
  sigmaTotal.setZero(); seebeckTotal.setZero(); kappaTotal.setZero(); mobilityTotal.setZero();

  // initialize the separate components
  seebeckSelf.resize(numCalculations, dimensionality, dimensionality);
  seebeckDrag.resize(numCalculations, dimensionality, dimensionality);

  alphaEl.resize(numCalculations, dimensionality, dimensionality);
  alphaPh.resize(numCalculations, dimensionality, dimensionality);

  kappaEl.resize(numCalculations, dimensionality, dimensionality);
  kappaPh.resize(numCalculations, dimensionality, dimensionality);
  kappaDrag.resize(numCalculations, dimensionality, dimensionality);

  seebeckDrag.setZero(); seebeckSelf.setZero();
  alphaEl.setZero(); alphaPh.setZero();
  kappaEl.setZero(); kappaPh.setZero(); kappaDrag.setZero();

  // intialize viscosities
  phViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  elViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  dragViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  //totalViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  phViscosity.setZero();
  elViscosity.setZero();
  dragViscosity.setZero();
  //totalViscosity.setZero();

}

void CoupledCoefficients::calcFromRelaxons(
                        CoupledScatteringMatrix& scatteringMatrix,
                        SpecificHeat& phSpecificHeat, SpecificHeat& elSpecificHeat,
                        Eigen::VectorXd& eigenvalues, ParallelMatrix<double>& eigenvectors) {

  // Note: the calcSpecialEigenvectors has been called before this, as it's needed
  // before this function to calculate phi, and then to use phi with D

  // TODO add OMP and MPI parallelism here
  // TODO maybe block the use of symmetries

  BaseBandStructure* phBandStructure = scatteringMatrix.getPhBandStructure();
  BaseBandStructure* elBandStructure = scatteringMatrix.getElBandStructure();

  // coupled transport only allowed with matrix in memory
  if (numCalculations > 1) {
    DeveloperError("Coupled relaxons only possible with one T value.");
  }

  int numElStates = int(elBandStructure->irrStateIterator().size());
  int numPhStates = int(phBandStructure->irrStateIterator().size());
  int numRelaxons = eigenvalues.size();

  int iCalc = 0;
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double T = calcStat.temperature / kBoltzmannRy;

  // electron and phonon participation ratios, summed over later
  std::vector<double> phPR(numRelaxons);
  std::vector<double> elPR(numRelaxons);

  // print info about the special eigenvectors ------------------------------
  // and save the indices that need to be skipped
  Particle electron = elBandStructure->getParticle();
  genericRelaxonEigenvectorsCheck(eigenvectors, numRelaxons, electron, theta0, theta_e, alpha0, alpha_e);

  // calculate the V components -----------------------------------------------------------
  // Here we have "ph" and "el" components, which are summed only
  // over either ph or el states and are then used to calculate
  // ph and el specific components to the transport coefficients
  Eigen::MatrixXd elV0(numRelaxons, 3); // V_a0^j = < 0 | v^j | alpha >
  Eigen::MatrixXd elVe(numRelaxons, 3); // V_ae^j = < e | v^j | alpha >
  Eigen::MatrixXd phV0(numRelaxons, 3);
  Eigen::MatrixXd phVe(numRelaxons, 3);
  Eigen::MatrixXd Ve(numRelaxons, 3);
  Eigen::MatrixXd V0(numRelaxons, 3);
  elV0.setZero(); elVe.setZero();
  phV0.setZero(); phVe.setZero();
  V0.setZero(); Ve.setZero();

  // phi related overlaps
  Eigen::Tensor<double, 3> elVphi(numRelaxons, 3, 3); // V_a(phi)^j = < theta | v^j | phi >
  Eigen::Tensor<double, 3> phVphi(numRelaxons, 3, 3);
  Eigen::Tensor<double, 3> dragVphi(numRelaxons, 3, 3);
  Eigen::Tensor<double, 3> Vphi(numRelaxons, 3, 3);
  dragVphi.setZero(); elVphi.setZero(); phVphi.setZero(); Vphi.setZero();

  // sum over the alpha and v states that this process owns
  for (auto tup : eigenvectors.getAllLocalStates()) {

    auto is = std::get<0>(tup);
    auto gamma = std::get<1>(tup);

    // sum up the participation ratios
    // Here, we expect all the el state indices will come first
    if(is < numElStates) { elPR[gamma] += eigenvectors(is,gamma) * eigenvectors(is,gamma); }
    else { phPR[gamma] += eigenvectors(is,gamma) * eigenvectors(is,gamma); }

    if(gamma >= numRelaxons) continue; // this relaxon wasn't calculated
    // negative eigenvalues are spurious, zero ones are not summed here
    //if(eigenvalues(gamma) <= 0) continue; // count them here but not later

    StateIndex isIdx(0);
    Eigen::Vector3d v;
    // set tau, avoiding div by zero issues
    double tau = abs(1./eigenvalues(gamma));
    if(eigenvalues(gamma) < 1e-10) tau = 0;

    // if boundary length isn't set, set a giant one
    double sqrtL = 1e12;
    if(!std::isnan(context.getBoundaryLength())) sqrtL = sqrt(context.getBoundaryLength() / sqrt(2.));

    if(context.getBoundaryLength() <= 0) Error("Boundary length should not be zero or less!");

    if(is < numElStates) { // electronic state

      BteIndex iBteIdx(is);
      isIdx = elBandStructure->bteToState(iBteIdx);
      v = elBandStructure->getGroupVelocity(isIdx);

      for (auto j : {0, 1, 2}) {

        // supress mean free paths which are greater than the sample width
        double lambdaSqrt = sqrt(abs(v(j)) * tau);
        double vSqrt = std::copysign(1.0, v(j)) * sqrt(abs(v(j)));
        if(lambdaSqrt > sqrtL) {
          lambdaSqrt = sqrtL;
        }

        elV0(gamma, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * theta0(is);
        elVe(gamma, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * theta_e(is);

        // for viscosity, we have to skip the special eigenvectors
        if(gamma != alpha0 && gamma != alpha_e) {
          for(auto i : {0, 1, 2}) {
            elVphi(gamma, i, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * phi(i, is);
            //theta0_phi(i,j) += theta0(is) * lambdaSqrt * vSqrt * phi(i,is);
            //theta_e_phi(i,j) += theta_e(is) * lambdaSqrt * vSqrt * phi(i,is);
          }
        }
      }
    } else { // phonon states

      isIdx = StateIndex(is-numElStates);
      v = phBandStructure->getGroupVelocity(isIdx);
      double energy = phBandStructure->getEnergy(isIdx);
      if (energy < phEnergyCutoff) { continue; }

      for (auto j : {0, 1, 2}) {

        // supress mean free paths which are greater than the sample width
        double lambdaSqrt = sqrt(abs(v(j)) * tau);
        double vSqrt = std::copysign(1.0, v(j)) * sqrt(abs(v(j)));
        if(lambdaSqrt > sqrtL) {
          lambdaSqrt = sqrtL;
        }

        phV0(gamma, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * theta0(is);
        phVe(gamma, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * theta_e(is);

        // for viscosity, we have to skip the special eigenvectors
        if(gamma != alpha0 && gamma != alpha_e) {
          for(auto i : {0, 1, 2}) {
            phVphi(gamma, i, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * phi(i, is);
          }
        }
      }
    }
    // also collect total Vs to check that the separation of electron and phonon states is ok
    for (auto j : {0, 1, 2}) {

      // supress mean free paths which are greater than the sample width
      double lambdaSqrt = sqrt(abs(v(j)) * tau);
      double vSqrt = std::copysign(1.0, v(j)) * sqrt(abs(v(j)));
      if(lambdaSqrt > sqrtL) {
        lambdaSqrt = sqrtL;
      }

      V0(gamma, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * theta0(is);
      Ve(gamma, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * theta_e(is);

      for(auto i : {0, 1, 2}) {
        if(gamma != alpha0 && gamma != alpha_e) {
          Vphi(gamma, i, j) += eigenvectors(is,gamma) * lambdaSqrt * vSqrt * phi(i, is);
        }
      }
    }
  }
  // reduce contributions from different processes transport velocities
  mpi->allReduceSum(&elV0); mpi->allReduceSum(&phV0);
  mpi->allReduceSum(&elVe); mpi->allReduceSum(&phVe);
  mpi->allReduceSum(&V0); mpi->allReduceSum(&Ve);
  // viscosity ingredients
  mpi->allReduceSum(&Vphi); mpi->allReduceSum(&phVphi); mpi->allReduceSum(&elVphi);
  // participation ratios
  mpi->allReduceSum(&phPR); mpi->allReduceSum(&elPR);

  // Calculate the transport coefficients -------------------------------------------------

  // local copies for linear algebra ops with eigen
  Eigen::Matrix3d sigmaLocal, totalSigmaLocal, selfSigmaS, dragSigmaS, totalSigmaS;
  sigmaLocal.setZero(); totalSigmaLocal.setZero(), selfSigmaS.setZero(); dragSigmaS.setZero(); totalSigmaS.setZero(); 

  // containers to calculate the specific contributions to the transport tensors
  kappaContrib.resize(numRelaxons, 3, 3);    kappaContrib.setZero();
  sigmaContrib.resize(numRelaxons, 3, 3);    sigmaContrib.setZero();
  sigmaSContrib.resize(numRelaxons, 3, 3);   sigmaSContrib.setZero();
  iiiiContrib.resize(numRelaxons);

  // TODO could parallelize this
  for(int gamma = 0; gamma < numRelaxons; gamma++) {

    if(eigenvalues(gamma) == 0) { continue; }

    if(eigenvalues(gamma) < 0) {
      for (int i = 0; i<dimensionality; i++) {
        for (int j = 0; j<dimensionality; j++) {

        // viscosities
          if(gamma != alpha0 && gamma != alpha_e) { // important -- including theta_0 or theta_e will lead to a wrong answer!

            double xxxx = sqrt(M(0) * M(0)) * Vphi(gamma,0,0) * Vphi(gamma,0,0);// * 1./eigenvalues(gamma);
            double yyyy = sqrt(M(1) * M(1)) * Vphi(gamma,1,1) * Vphi(gamma,1,1);// * 1./eigenvalues(gamma);
            iiiiContrib[gamma] += (xxxx + yyyy)/2.;
          }

          sigmaContrib(gamma,i,j) += U * Ve(gamma,i) * Ve(gamma,j); // * 1./eigenvalues(gamma);
          sigmaSContrib(gamma,i,j) += 1. / kBoltzmannRy * sqrt(Ctot * U / T) * Ve(gamma,i) * V0(gamma,j);// * 1./eigenvalues(gamma);
          kappaContrib(gamma,i,j) += Ctot / kBoltzmannRy * V0(gamma,i) * V0(gamma,j);// * 1./eigenvalues(gamma);
        }
      }
    }

    //double tau = abs(1./eigenvalues(gamma));

    for (int i = 0; i<dimensionality; i++) {
      for (int j = 0; j<dimensionality; j++) {

        // NOTE: remove energy and charge eigenvectors
        if(gamma == alpha0 || gamma == alpha_e) continue;

        // sigma
        sigmaLocal(i,j) += U * elVe(gamma,i) * elVe(gamma,j); // * tau;
        totalSigmaLocal(i,j) += U * Ve(gamma,i) * Ve(gamma,j); // * tau;
        sigmaContrib(gamma,i,j) += U * Ve(gamma,i) * Ve(gamma,j); // * tau;

        // sigmaS
        selfSigmaS(i,j) -= 1. / kBoltzmannRy * sqrt(Ctot * U / T) * elVe(gamma,i) * elV0(gamma,j); // * tau;
        dragSigmaS(i,j) -= 1. / kBoltzmannRy * sqrt(Ctot * U / T) * elVe(gamma,i) * phV0(gamma,j); // * tau;
        totalSigmaS(i,j) -= 1. / kBoltzmannRy * sqrt(Ctot * U / T) * Ve(gamma,i) * V0(gamma,j); // * tau;
        sigmaSContrib(gamma,i,j) -= 1. / kBoltzmannRy * sqrt(Ctot * U / T) * Ve(gamma,i) * V0(gamma,j); // * tau;

        // alpha
        alphaEl(0,i,j) += sqrt(Ctot * U * T) * elV0(gamma,i) * elVe(gamma,j); // * tau;
        alphaPh(0,i,j) += sqrt(Ctot * U * T) * phV0(gamma,i) * elVe(gamma,j); // * tau;

        // thermal conductivity
        kappaEl(0,i,j) += Ctot / kBoltzmannRy * (elV0(gamma,i) * elV0(gamma,j));
        kappaPh(0,i,j) += Ctot / kBoltzmannRy * (phV0(gamma,i) * phV0(gamma,j));
        kappaDrag(0,i,j) += Ctot / kBoltzmannRy * (elV0(gamma,i) * phV0(gamma,j) + phV0(gamma,i) * elV0(gamma,j));
        kappaTotal(0,i,j) += Ctot / kBoltzmannRy * V0(gamma,i) * V0(gamma,j); // * tau;
        kappaContrib(gamma,i,j) += Ctot / kBoltzmannRy * V0(gamma,i) * V0(gamma,j); // * tau;

        // viscosities
        double xxxx = sqrt(M(0) * M(0)) * Vphi(gamma,0,0) * Vphi(gamma,0,0); // * tau;
        double yyyy = sqrt(M(1) * M(1)) * Vphi(gamma,1,1) * Vphi(gamma,1,1); // * tau;
        iiiiContrib[gamma] += (xxxx + yyyy)/2.;

        for(auto k : {0, 1, 2}) {
          for(auto l : {0, 1, 2}) {
            phViscosity(0,i,j,k,l) += sqrt(A(i) * A(k)) * phVphi(gamma,i,j) * phVphi(gamma,l,k); // * tau;
            elViscosity(0,i,j,k,l) += sqrt(G(i) * G(k)) * elVphi(gamma,i,j) * elVphi(gamma,l,k); // * tau;
            dragViscosity(0,i,j,k,l) += sqrt(A(i) * G(k)) * phVphi(gamma,i,j) * elVphi(gamma,l,k); // * tau;
                                                                //(elVphi(gamma,i,j) * phVphi(gamma,l,k)
                                                                // + phVphi(gamma,i,j) * elVphi(gamma,l,k)) * 1./eigenvalues(gamma);
            //totalViscosity(0,i,j,k,l) += sqrt(M(i) * M(k)) * Vphi(gamma,i,j) * Vphi(gamma,l,k) * 1./eigenvalues(gamma);
          }
        }
      }
    }
  }

  // seebeck = matmul(L_EE_inv, L_ET)
  Eigen::Matrix3d seebeckSelfLocal = sigmaLocal.inverse() * selfSigmaS;
  Eigen::Matrix3d seebeckDragLocal = sigmaLocal.inverse() * dragSigmaS;
  Eigen::Matrix3d totalSeebeckLocal = totalSigmaLocal.inverse() * totalSigmaS;

  // copy S and sigma into final tensors to be printed
  // convert sigma -> mobility
  double doping = abs(statisticsSweep.getCalcStatistics(iCalc).doping);
  doping *= pow(distanceBohrToCm, dimensionality); // from cm^-3 to bohr^-3
  for (int i = 0; i < dimensionality; i++) {
    for(auto j : {0, 1, 2} ) {

      seebeckSelf(0,i,j) = seebeckSelfLocal(i,j);
      seebeckDrag(0,i,j) = seebeckDragLocal(i,j);
      seebeckTotal(0,i,j) = totalSeebeckLocal(i,j);

      sigma(0,i,j) = sigmaLocal(i,j);
      sigmaTotal(0,i,j) = totalSigmaLocal(i,j);

      mobility(0,i,j) = sigma(0,i,j);
      mobilityTotal(0,i,j) = sigmaTotal(0,i,j);
      if (doping > 0.) {
        mobility(0, i, j) /= doping;
        mobilityTotal(0, i, j) /= doping;
      }
    }
  }
  // sum to get total contributions
  alpha = alphaPh + alphaEl;
  kappa = kappaPh + kappaEl + kappaDrag;
  seebeck = seebeckSelf + seebeckDrag;

  // throw warnings if different results come out from parts vs total calculation
  bool sigmaFail = false;
  bool seebeckFail = false;
  bool kappaFail = false;
  for (int i = 0; i < dimensionality; i++) {
    for(auto j = 0; j < dimensionality; j++) {
      if(sigma(0,i,j) != sigmaTotal(0,i,j))     { sigmaFail = true; }
      if(seebeck(0,i,j) != seebeckTotal(0,i,j)) { seebeckFail = true; }
      if(kappa(0,i,j) != kappaTotal(0,i,j))     { kappaFail = true; }
    }
  }
  if(seebeckFail) Warning("Developer warning: Seebeck cross + self does not equal Seebeck total.");
  if(sigmaFail) Warning("Developer warning: Sigma el does not equal sigma total.");
  if(kappaFail) Warning("Developer warning: Kappa cross + selfEl + selfPh does not equal kappa total.");

  // dump the participation ratios to file here,
  // TODO this should be a designated function
  nlohmann::json output;

  std::vector<double> chemPots = { calcStat.chemicalPotential };
  std::vector<double> dopings = { calcStat.doping };
  std::vector<double> temps = { T };
  output["temperatures"] = temps;
  output["temperatureUnit"] = "K";
  output["dopingConcentrations"] = dopings;
  output["dopingConcentrationUnit"] = "cm$^{-" + std::to_string(dimensionality) + "}$";
  output["chemicalPotentials"] = chemPots;
  output["chemicalPotentialUnit"] = "eV";
  output["phononParticipationRatio"] = phPR;
  output["electronParticipationRatio"] = elPR;
  std::ofstream o("coupled_participation_ratios.json");
  o << std::setw(3) << output << std::endl;
  o.close();

}

// standard print
void CoupledCoefficients::print() {

  // prints the total tensors to the main output file
  printHelper(statisticsSweep, dimensionality, kappa, sigma, mobility, seebeck);

}

void CoupledCoefficients::outputToJSON(const std::string &outFileName) {

  if (!mpi->mpiHead()) return;

  // output the viscosities using the helper function in viscosity_io.h
  bool append = false;
  outputViscosityToJSON("coupled_relaxons_viscosity.json", "phononViscosity", phViscosity,
        append, statisticsSweep, dimensionality);
  append = true;
  outputViscosityToJSON("coupled_relaxons_viscosity.json", "electronViscosity", elViscosity,
        append, statisticsSweep, dimensionality);
  outputViscosityToJSON("coupled_relaxons_viscosity.json", "dragViscosity", dragViscosity,
        append, statisticsSweep, dimensionality);
  //outputViscosityToJSON("coupled_relaxons_viscosity.json", "totalViscosity", totalViscosity,
  //      append, statisticsSweep, dimensionality);

  // output the transport coefficients
  int numCalculations = statisticsSweep.getNumCalculations();

  std::string unitsSigma, unitsKappa, unitsViscosity;
  double convSigma, convKappa, convViscosity;
  // TODO check the kappa units, I think it's missing a kb
  if (dimensionality == 1) {
    unitsSigma = "S m";
    unitsKappa = "W m / K";
    unitsViscosity = "Pa s / m^2";
    convSigma = elConductivityAuToSi * rydbergSi * rydbergSi;
    convKappa = thConductivityAuToSi * rydbergSi * rydbergSi;
    convViscosity = viscosityAuToSi * rydbergSi * rydbergSi;
  } else if (dimensionality == 2) {
    unitsSigma = "S";
    unitsKappa = "W / K";
    unitsViscosity = "Pa s / m";
    convSigma = elConductivityAuToSi * rydbergSi;
    convKappa = thConductivityAuToSi * rydbergSi;
    convViscosity = viscosityAuToSi * rydbergSi;
  } else {
    unitsSigma = "S / m";
    unitsKappa = "W / m / K";
    unitsViscosity = "Pa s";
    convSigma = elConductivityAuToSi;
    convKappa = thConductivityAuToSi;
    convViscosity = viscosityAuToSi;
  }

  // TODO should this use dimensionality instead??
  double convMobility = mobilityAuToSi * pow(100., 2); // from m^2/Vs to cm^2/Vs
  std::string unitsMobility = "cm^2 / V / s";

  double convSeebeck = thermopowerAuToSi * 1.0e6;
  std::string unitsSeebeck = "muV / K";

  std::vector<double> temps, dopings, chemPots;
  std::vector<std::vector<std::vector<double>>> sigmaOut, sigmaTotalOut, mobilityOut, mobilityTotalOut;
  std::vector<std::vector<std::vector<double>>> kappaOut, kappaPhOut, kappaElOut, kappaDragOut, kappaTotalOut;
  std::vector<std::vector<std::vector<double>>> seebeckOut, seebeckDragOut,seebeckSelfOut,seebeckTotalOut;
  std::vector<std::vector<double>> sigmaMomOut, seebeckMomOut, kappaMomOut; 

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp * temperatureAuToSi);
    double doping = calcStat.doping;
    dopings.push_back(doping); // output in (cm^-3)
    double chemPot = calcStat.chemicalPotential;
    chemPots.push_back(chemPot * energyRyToEv); // output in eV

    // store the electrical conductivity for output
    appendTransportTensorForOutput(sigmaTotal, dimensionality, convSigma, iCalc, sigmaTotalOut);

    // store the carrier mobility for output
    // Note: in metals, one has conductivity without doping
    // and the mobility = sigma / doping-density is ill-defined
    if (abs(doping) > 0.) {
      appendTransportTensorForOutput(mobilityTotal, dimensionality, convMobility, iCalc, mobilityTotalOut);
    }

    // store thermal conductivity for output
    appendTransportTensorForOutput(kappaPh, dimensionality, convKappa, iCalc, kappaPhOut);
    appendTransportTensorForOutput(kappaEl, dimensionality, convKappa, iCalc, kappaElOut);
    appendTransportTensorForOutput(kappaDrag, dimensionality, convKappa, iCalc, kappaDragOut);
    appendTransportTensorForOutput(kappaTotal, dimensionality, convKappa, iCalc, kappaTotalOut);

    // store seebeck coefficient for output
    appendTransportTensorForOutput(seebeckDrag, dimensionality, convSeebeck, iCalc, seebeckDragOut);
    appendTransportTensorForOutput(seebeckSelf, dimensionality, convSeebeck, iCalc, seebeckSelfOut);
    appendTransportTensorForOutput(seebeckTotal, dimensionality, convSeebeck, iCalc, seebeckTotalOut);

  }

  // convert momentum contributions 
  sigma_mom *= convSigma; 
  seebeck_mom *= convSeebeck; 
  kappa_mom *= convKappa; 
  for(auto i : {0,1,2}) {
    sigmaMomOut.push_back({sigma_mom(i,0), sigma_mom(i,1), sigma_mom(i,2)});
    seebeckMomOut.push_back({seebeck_mom(i,0), seebeck_mom(i,1), seebeck_mom(i,2)});
    kappaMomOut.push_back({kappa_mom(i,0), kappa_mom(i,1), kappa_mom(i,2)});
  }

  { // so that the output json goes out of scope and it can be reused below

    // output to json
    nlohmann::json output;
    output["temperatures"] = temps;
    output["temperatureUnit"] = "K";
    output["dopingConcentrations"] = dopings;
    output["dopingConcentrationUnit"] = "cm$^{-" + std::to_string(dimensionality) + "}$";
    output["chemicalPotentials"] = chemPots;
    output["chemicalPotentialUnit"] = "eV";

    output["totalElectricalConductivity"] = sigmaTotalOut;
    output["electricalConductivityUnit"] = unitsSigma;
    output["totalMobility"] = mobilityTotalOut;
    output["mobilityUnit"] = unitsMobility;

    output["phononThermalConductivity"] = kappaPhOut;
    output["electronicThermalConductivity"] = kappaElOut;
    output["crossElPhThermalConductivity"] = kappaDragOut;
    output["totalThermalConductivity"] = kappaTotalOut;
    output["thermalConductivityUnit"] = unitsKappa;

    output["crossElPhSeebeckCoefficient"] = seebeckDragOut;
    output["selfElSeebeckCoefficient"] = seebeckSelfOut;
    output["totalSeebeckCoefficient"] = seebeckTotalOut;
    output["seebeckCoefficientUnit"] = unitsSeebeck;

    output["momentumElectricalConductivity"] = sigmaMomOut; 
    output["momentumSeebeck"] = seebeckMomOut; 
    output["momentumThermalConductivity"] = kappaMomOut; 

    std::ofstream o(outFileName);
    o << std::setw(3) << output << std::endl;
    o.close();
  }

  // now output the separated contributions
  std::vector<std::vector<std::vector<double>>> sigmaContribOut;
  std::vector<std::vector<std::vector<double>>> kappaContribOut;
  std::vector<std::vector<std::vector<double>>> sigmaSContribOut;
  double convSigmaS = convSeebeck*convSigma;

  int numRelaxons = iiiiContrib.size();

  for (int gamma = 0; gamma < numRelaxons; gamma++) {

    // convert viscosity units
    iiiiContrib[gamma] *= convViscosity;

    // store the electrical conductivity for output
    appendTransportTensorForOutput(sigmaContrib, dimensionality, convSigma, gamma, sigmaContribOut);
    // store thermal conductivity for output
    appendTransportTensorForOutput(kappaContrib, dimensionality, convKappa, gamma, kappaContribOut);
    // store seebeck coefficient for output
    appendTransportTensorForOutput(sigmaSContrib, dimensionality, convSigmaS, gamma, sigmaSContribOut);

  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["temperatureUnit"] = "K";
  output["dopingConcentrations"] = dopings;
  output["dopingConcentrationUnit"] = "cm$^{-" + std::to_string(dimensionality) + "}$";
  output["chemicalPotentials"] = chemPots;
  output["chemicalPotentialUnit"] = "eV";

  output["electricalConductivityContributions"] = sigmaContribOut;
  output["electricalConductivityUnit"] = unitsSigma;

  output["thermalConductivityContribution"] = kappaContribOut;
  output["thermalConductivityUnit"] = unitsKappa;

  output["sigmaSContribution"] = sigmaSContribOut;
  output["sigmaSCoefficientUnit"] = unitsSeebeck + " x " + unitsSigma;

  output["iiiiViscosityContribution"] = iiiiContrib;

  std::ofstream o("coupled_transport_coeffs_contributions.json");
  o << std::setw(3) << output << std::endl;
  o.close();

}

// calculate special eigenvectors
void CoupledCoefficients::calcSpecialEigenvectors(StatisticsSweep& statisticsSweep,
                                                BaseBandStructure* phBandStructure,
                                                BaseBandStructure* elBandStructure) {

  double volume = crystal.getVolumeUnitCell(dimensionality);

  int numElStates = int(elBandStructure->irrStateIterator().size());
  int numPhStates = int(phBandStructure->irrStateIterator().size());
  int numStates = numElStates + numPhStates;
  //int numRelaxons = eigenvalues.size();

  double Nk = double(context.getKMesh().prod());
  double Nq = double(context.getQMesh().prod());

  Particle phonon = phBandStructure->getParticle();
  Particle electron = elBandStructure->getParticle();

  int iCalc = 0;
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double T = calcStat.temperature / kBoltzmannRy;
  double kBT = calcStat.temperature; // note, what's in calcStat is kBT
  double chemPot = calcStat.chemicalPotential;

  // TODO remove this after debugging and replace with
  // default specific heat function in phoebe
  // Calculate the specific heat to double check the phoebe calculated values
  Cph = 0;
  Cel = 0;

  for(int is = 0; is < numStates; is++) {

    // n(n+1) for bosons, n(1-n) for fermions
    if(is<numElStates) {

      StateIndex elIdx(is);
      double energy = elBandStructure->getEnergy(elIdx);
      Cel += electron.getPopPopPm1(energy, kBT, chemPot)
                * (energy - chemPot) * (energy - chemPot);

    } else { // second part of the vector is phonon quantities

      int iPhState = is-numElStates;
      StateIndex phIdx(iPhState);

      double energy = phBandStructure->getEnergy(phIdx);
      // Discard ph states with negative energies
      if (energy < phEnergyCutoff) { continue; }
      Cph += phonon.getPopPopPm1(energy, kBT, 0) * energy * energy;

    }
  }
  Cph *= 1./(volume * Nq * kBT * T);
  Cel *= spinFactor/(volume * Nk * kBT * T);
  Ctot = Cel + Cph;

  // normalization coeff U (summed up below)
  // U = D/(V*Nk) * (1/kT) sum_km F(1-F)
  U = 0;
  // normalization coeff G ("electron specific momentum")
  // G = D/(V*Nk) * (1/kT) sum_km (hbar*k)^2 * F(1-F)
  G = Eigen::Vector3d::Zero();
  // normalization coeff A ("phonon specific momentum")
  // A = 1/(V*Nq) * (1/kT) sum_qs (hbar*q)^2 * N(1+N)
  A = Eigen::Vector3d::Zero();

  // Precalculate theta_e, theta0, phi  ----------------------------------

  // theta^0 - energy conservation eigenvector
  //   electronic states = ds * g-1 * (hE - mu) * 1/(kbT^2 * V * Nkq * Ctot)
  //   phonon states = ds * g-1 * h*omega * 1/(kbT^2 * V * Nkq * Ctot)
  theta0 = Eigen::VectorXd::Zero(numStates);

  // theta^e -- the charge conservation eigenvector
  //   electronic states = ds * g-1 * 1/(kbT * U)
  //   phonon state = 0
  theta_e = Eigen::VectorXd::Zero(numStates);

  // phi -- the three momentum conservation eigenvectors
  //     phi = sqrt(1/(kbT*volume*Nkq*M)) * g-1 * ds * hbar * wavevector
  // first part will be el momentum vectors, second half ph
  //     phi_el = sqrt(1/(kbT*volume*Nk*G)) * b-1 * D * hbar * k
  //     phi_ph = sqrt(1/(kbT*volume*Nq*A)) * b-1 * hbar * q
  phi = Eigen::MatrixXd::Zero(3, numStates);

  // spin degen vector
  Eigen::VectorXd ds = Eigen::VectorXd::Zero(numStates);

  for(int is = 0; is < numStates; is++) {

    // n(n+1) for bosons, n(1-n) for fermions
    double sqrtPopTerm;

    if(is < numElStates) {

      StateIndex elIdx(is);
      double energy = elBandStructure->getEnergy(elIdx);
      // this is in cartesian coords
      Eigen::Vector3d k = elBandStructure->getWavevector(elIdx);
      k = elBandStructure->getPoints().bzToWs(k,Points::cartesianCoordinates);

      // note, this function expects kBT
      sqrtPopTerm = sqrt(electron.getPopPopPm1(energy, kBT, chemPot));

      ds(is) = sqrt( spinFactor / Nk );

      U += sqrtPopTerm * sqrtPopTerm;
      for(int i : {0,1,2} ) {
        G(i) += k(i) * k(i) * sqrtPopTerm * sqrtPopTerm;
        phi(i, is) = sqrtPopTerm * ds(is) * k(i);
      }

      theta0(is) = sqrtPopTerm * (energy - chemPot) * ds(is);
      theta_e(is) = sqrtPopTerm * ds(is);

    } else { // second part of the vector is phonon quantities

      int iPhState = is-numElStates;
      StateIndex phIdx(iPhState);
      double energy = phBandStructure->getEnergy(phIdx);

      // Discard ph states with negative energies
      if (energy < phEnergyCutoff) { continue; }

      // this is in cartesian coords
      Eigen::Vector3d q = phBandStructure->getWavevector(phIdx);
      q = phBandStructure->getPoints().bzToWs(q,Points::cartesianCoordinates);
      sqrtPopTerm = sqrt(phonon.getPopPopPm1(energy, kBT, 0));

      ds(is) = sqrt( 1. / Nq );

      for(int i : {0,1,2} ) {
        A(i) += q(i) * q(i) * sqrtPopTerm * sqrtPopTerm;
        phi(i, is) = sqrtPopTerm * ds(is) * q(i);
      }
      theta0(is) = sqrtPopTerm * energy * ds(is);
    }
  }

  // add the normalization prefactor to U
  U *= spinFactor / (volume * Nk * kBT);
  G *= spinFactor / (volume * Nk * kBT);
  A *= 1. / (volume * Nq * kBT);
  //M = G + A;

  // apply the normalization to theta_e
  theta_e *= 1./sqrt(kBT * U * volume);
  // apply normalization to theta0
  theta0 *= 1./sqrt(kBT * T * volume * Ctot);
  // apply normalization to phi
  for(int is = 0; is < numStates; is++) {
    for(int i : {0,1,2}) {
      if(is < numElStates) { // electrons
        phi(i,is) *= 1./sqrt(kBT * volume * G(i));
      } else { // phonons
        phi(i,is) *= 1./sqrt(kBT * volume * A(i));
      }
    }
  }

  // check the norm of phi
/*
  if(mpi->mpiHead()) {
    for(int i : {0,1,2}) {
      double phiTot = 0;
      for(int is = 0; is < numStates; is++) {
        phiTot += phi(i,is) * phi(i, is);
      }
      std::cout << "phi norm " << i << " " << phiTot << std::endl;;
    }
  }
*/
  // throw errors if normalization fails
  if( abs(theta_e.dot(theta_e) - 1.) > 1e-4 || abs(theta0.dot(theta0) - 1.) > 1e-4) {
    // std::cout << theta_e.dot(theta_e) << " " << theta0.dot(theta0) << std::endl;
    Warning("Developer error: Your energy or charge conservation eigenvectors do not"
                " normalize to 1.\nThis indicates something has gone very wrong "
                "with your relaxons solve (or your mesh is super small), please report this.");
  }
}

void CoupledCoefficients::outputDuToJSON(CoupledScatteringMatrix& coupledScatteringMatrix, Context& context,
						bool isSymmetrized) {

  // Calculate real space quantities (Du, W) --------------------------------------------------

  BaseBandStructure* phBandStructure = coupledScatteringMatrix.getPhBandStructure();
  BaseBandStructure* elBandStructure = coupledScatteringMatrix.getElBandStructure();

  int numElStates = int(elBandStructure->irrStateIterator().size());
  auto calcStat = statisticsSweep.getCalcStatistics(0); // only one calc for relaxons
  double kBT = calcStat.temperature;
  double T = calcStat.temperature / kBoltzmannRy;

  // write D to file before diagonalizing, as the scattering matrix
  // will be destroyed by scalapack
  Eigen::Matrix3d Du, DuEl, DuDragEl, DuDragPh, DuPh; 
  Du.setZero(); DuPh.setZero(); DuDragPh.setZero(); 
  DuDragEl.setZero(); DuEl.setZero();

  Eigen::Matrix3d Wjie, Wji0, elWji0, phWji0; 
  Wjie.setZero(); Wji0.setZero(); elWji0.setZero(); phWji0.setZero();

  // sum over the alpha and v states that this process owns
  for (auto tup : coupledScatteringMatrix.getAllLocalStates()) {

    auto is1 = std::get<0>(tup);
    auto is2 = std::get<1>(tup);

    // if only the uppper half is filled,
    // we count the diagonal of the scattering matrix once, and the off diagonals twice
    // as one of them will be zero
    double upperTriangleFactor = 1.;
    if(context.getUseUpperTriangle() && (is1 != is2)) {
      upperTriangleFactor = 2.;
    }

    for (auto i : {0, 1, 2}) {
      for (auto j : {0, 1, 2}) {

        double duContribution = phi(i,is1) * coupledScatteringMatrix(is1,is2) * phi(j,is2);

        // always add the contribution to the total Du value
        Du(i, j) += upperTriangleFactor * duContribution;

        // electron only quadrant case
        if ( is1 < numElStates && is2 < numElStates ) {
          DuEl(i, j) += upperTriangleFactor * duContribution;
        // phonon only quadrant case
        } else if ( is1 >= numElStates && is2 >= numElStates ) {
          DuPh(i, j) += upperTriangleFactor * duContribution;
        // drag term contribution (never has upper triangle factor, only in 1 quadrant)
        // use the upper quandrant, (el,ph) as this is always filled because it's in the upper triangle
        } else if ( is1 < numElStates && is2 >= numElStates) { // upper triangle
          DuDragPh(i, j) += duContribution;
        } else if ( is1 >= numElStates && is2 < numElStates) { // lower triangle part
          DuDragEl(i, j) += duContribution;
        }
      }
    }
  }
  mpi->allReduceSum(&Du);
  mpi->allReduceSum(&DuEl);     mpi->allReduceSum(&DuPh);
  mpi->allReduceSum(&DuDragPh); mpi->allReduceSum(&DuDragEl);

  if(context.getUseUpperTriangle()) {
    DuDragEl = DuDragPh;
  }

  Eigen::Matrix3d invDuEl = DuEl.inverse();          // Du_ee
  Eigen::Matrix3d invDuDragPh = DuDragPh.inverse();  // Du_pe
  Eigen::Matrix3d invDuPh = DuPh.inverse();          // Du_pp
  Eigen::Matrix3d invDuDragEl = DuDragEl.inverse();  // Du_ep
  Eigen::MatrixXd theta0_phi(3,6), theta_e_phi(3,6);
  theta0_phi.setZero(); theta_e_phi.setZero();

  // if boundary length isn't set, set a giant one
  double sqrtL = 1e12;
  if(!std::isnan(context.getBoundaryLength())) sqrtL = sqrt(context.getBoundaryLength() / sqrt(2.));
  if(context.getBoundaryLength() <= 0) Error("Boundary length should not be zero or less!");

  // TODO lambda function to calculate vSqrt(tau), associated with supression function 
  // replace where lambda is defined everywhere, and move to constructor or free function 
  auto v_sqrtTau = [sqrtL] (double vj,double tau) { 
    // supress mean free paths which are greater than the sample width
    double lambdaSqrt = sqrt(abs(vj) * tau);
    double vSqrt = std::copysign(1.0, vj) * sqrt(abs(vj));
    if(lambdaSqrt > sqrtL) { lambdaSqrt = sqrtL; }
    return lambdaSqrt * vSqrt; 
  };

  // Calculate and write to file Wji0, Wjie, Wj0i, Wjei --------------------------------
  for (int is : elBandStructure->parallelStateIterator()) {
    auto isIdx = StateIndex(is);
    auto v = elBandStructure->getGroupVelocity(isIdx);

    for (auto j : {0, 1, 2}) {
      for (auto i : {0, 1, 2}) {
        // calculate quantities for the real-space solve
        Wji0(j,i) += phi(i,is) * v(j) * theta0(is);
        elWji0(j,i) += phi(i,is) * v(j) * theta0(is);
        Wjie(j,i) += phi(i,is) * v(j) * theta_e(is);

        theta0_phi(i,j) += theta0(is) * v_sqrtTau(v(j), invDuEl(i,i)) * phi(i,is);  
        theta_e_phi(i,j) += theta_e(is) * v_sqrtTau(v(j), invDuEl(i,i)) * phi(i,is);
      }
    }
  }
  for (int is : phBandStructure->parallelStateIterator()) {
    auto isIdx = StateIndex(is);
    double en = phBandStructure->getEnergy(isIdx);
    // discard acoustic phonon modes
    if (en < phEnergyCutoff) { continue; }
    auto v = phBandStructure->getGroupVelocity(isIdx);

    for (auto j : {0, 1, 2}) {
      for (auto i : {0, 1, 2}) {
        // note: phi and theta here are elStates long, so we need to shift the state
        // index to account for the fact that we summed over the electronic part above
        // calculate quantities for the real-space solver
        Wji0(j,i) += phi(i,is+numElStates) * v(j) * theta0(is+numElStates);
        phWji0(j,i) += phi(i,is+numElStates) * v(j) * theta0(is+numElStates);
        Wjie(j,i) += phi(i,is+numElStates) * v(j) * theta_e(is+numElStates);

        // +3 is an offset to set the second 3 of 6 values for phi, which are el ones
        theta0_phi(i,j+3) += theta0(is+numElStates) * v_sqrtTau(v(j), invDuPh(i,i)) * phi(i,is+numElStates);
        theta_e_phi(i,j+3) += theta_e(is+numElStates) * v_sqrtTau(v(j), invDuPh(i,i)) * phi(i,is+numElStates);
      }
    }
  }
  mpi->allReduceSum(&Wji0); mpi->allReduceSum(&Wjie);
  mpi->allReduceSum(&phWji0); mpi->allReduceSum(&elWji0);
  mpi->allReduceSum(&theta0_phi); mpi->allReduceSum(&theta_e_phi);

  // TODO we should fix this all 
  if(isSymmetrized) {
    symmetrize(Du); symmetrize(DuEl); symmetrize(DuPh);
    symmetrize(DuDragPh); symmetrize(DuDragEl);
    symmetrize(Wji0); symmetrize(elWji0); symmetrize(phWji0); symmetrize(Wjie);
  }

  // calculate part of transport coefficients due to momentum eigenvectors
  Eigen::Matrix3d sigmaS_mom;  
  sigma_mom.setZero(); kappa_mom.setZero(); sigmaS_mom.setZero();
  for (int i = 0; i<dimensionality; i++) {
    for (int j = 0; j<dimensionality; j++) {
      for (int alpha = 0; alpha<6; alpha++) {
        sigma_mom(i,j) += U * theta_e_phi(i,alpha) * theta_e_phi(j,alpha);
        sigmaS_mom(i,j) -= 1. / kBoltzmannRy * sqrt(Ctot * U / T) * theta_e_phi(i,alpha) * theta0_phi(j,alpha);
        kappa_mom(i,j) += Ctot / kBoltzmannRy * theta0_phi(i,alpha) * theta0_phi(j,alpha);
      }
    }
  }
  seebeck_mom = sigma_mom.inverse() * sigmaS_mom;


  // NOTE we cannot use nested vectors from the start, as
  // vector<vector> is not necessarily contiguous and MPI
  // cannot all reduce on it
  std::vector<std::vector<double>> vecDu, vecDuEl, vecDuPh, vecDuDragPh, vecDuDragEl;
  std::vector<std::vector<double>> vecWji0, vecWji0_el, vecWji0_ph, vecWjie;

  for (auto i : {0, 1, 2}) {
    std::vector<double> t1,t2,t3,t4,t5,t6,t7,t8,t9;
    for (auto j : {0, 1, 2}) {
      t1.push_back(Du(i,j) / (energyRyToFs / twoPi));
      t2.push_back(Wji0(i,j) * velocityRyToSi);
      t3.push_back(elWji0(i,j) * velocityRyToSi);
      t4.push_back(phWji0(i,j) * velocityRyToSi);
      t5.push_back(Wjie(i,j) * velocityRyToSi);
      t6.push_back(DuEl(i,j) / (energyRyToFs / twoPi));
      t7.push_back(DuPh(i,j) / (energyRyToFs / twoPi));
      t8.push_back(DuDragPh(i,j) / (energyRyToFs / twoPi));
      t9.push_back(DuDragEl(i,j) / (energyRyToFs / twoPi));
    }
    vecDu.push_back(t1);
    vecWji0.push_back(t2);
    vecWji0_el.push_back(t3);
    vecWji0_ph.push_back(t4);
    vecWjie.push_back(t5);
    vecDuEl.push_back(t6);
    vecDuPh.push_back(t7);
    vecDuDragPh.push_back(t8);
    vecDuDragEl.push_back(t9);
  }

  // this extra kBoltzmannRy is required when we calculate specific heat ...
  // TODO need to keep track of this and figure out where it's coming from
  double specificHeatConversion = kBoltzmannSi / pow(bohrRadiusSi, dimensionality) / kBoltzmannRy;

  // convert Ai to SI, in units of picograms/(mu m^3)
  double Aconversion = electronMassSi /
                       std::pow(distanceBohrToMum, dimensionality) * // convert AU mass / V -> SI
                       2. *   // factor of two is a Ry->Ha conversion required here
                       1.e15; // convert electronMassSi in kg to pico g

                       // Michele's version of this, gives thes same answer
                       // double altConv =  1./rydbergSi * // convert kBT
                       // std::pow(hBarSi/bohrRadiusSi,2) * // convert (hbar * q)^2
                       // 1./std::pow(bohrRadiusSi, dimensionality) * // convert 1/V
                       // 1e-3; //convert from kg->pg, 1/m^3 -> 1/mum^3; // converting to pico and mu

  std::string specificHeatUnits;
  std::string AiUnits;
  if (dimensionality == 1) {
    specificHeatUnits = "J / K / m";
    AiUnits = "pg/(mum)";
  } else if (dimensionality == 2) {
    specificHeatUnits = "J / K / m^2";
    AiUnits = "pg/(mum)^2";
  } else {
    specificHeatUnits = "J / K / m^3";
    AiUnits = "pg/(mum)^3";
  }

  if(mpi->mpiHead()) {
    // output to json
    std::string outFileName = "coupled_relaxons_real_space_coeffs.json";
    if(isSymmetrized)  outFileName = "sym_coupled_relaxons_real_space_coeffs.json";
    nlohmann::json output;
    output["temperature"] = kBT * temperatureAuToSi;
    output["Wji0"] = vecWji0;
    output["phononWji0"] = vecWji0_ph;
    output["electronWji0"] = vecWji0_el;
    output["Wjie"] = vecWjie;
    output["Du"] = vecDu;
    output["electronDu"] = vecDuEl;
    output["phononDu"] = vecDuPh;
    output["dragDuPh"] = vecDuDragPh;
    output["dragDuEl"] = vecDuDragEl;
    output["temperatureUnit"] = "K";
    output["wUnit"] = "m/s";
    output["DuUnit"] = "fs^{-1}";
    output["phononSpecificHeat"] = Cph * specificHeatConversion;
    output["electronSpecificHeat"] = Cel * specificHeatConversion;
    output["U"] = U * std::pow(electronSi, 2) / ( std::pow(bohrRadiusSi, 3) * energyRyToEv);
    output["UUnit"] = "Coulomb$^2$/(m$^3$*eV)";
    output["specificHeatUnit"] = specificHeatUnits;
    std::vector<double> Atemp, Gtemp;
    for(int i = 0; i < 3; i++) {
      Atemp.push_back(A(i) * Aconversion );
      Gtemp.push_back(G(i) * Aconversion );
    }
    output["Gi"] = Gtemp;
    output["GiUnit"] = AiUnits;
    output["Ai"] = Atemp;
    output["AiUnit"] = AiUnits;
    std::ofstream o(outFileName);
    o << std::setw(3) << output << std::endl;
    o.close();
  }
}

void CoupledCoefficients::symmetrize3x3Tensors() {

  // symmetrize the transport tensors
  symmetrize(sigma);
  symmetrize(kappa);
  symmetrize(mobility);
  symmetrize(seebeck);

  symmetrize(sigmaTotal);
  symmetrize(kappaTotal);
  symmetrize(mobilityTotal);
  symmetrize(seebeckTotal);

  symmetrize(seebeckSelf);
  symmetrize(seebeckDrag);
  symmetrize(kappaDrag);
  symmetrize(kappaEl);
  symmetrize(kappaPh);

  symmetrize(alphaEl);
  symmetrize(alphaPh);
  symmetrize(alpha);

  outputToJSON("sym_coupled_relaxons_transport_coeffs.json");

}

// TODO this should be a function of observable rather than of onsager,
// however, somehow Onsager does not inherit from observable...
void CoupledCoefficients::symmetrize(Eigen::Matrix3d& transportCoeffs) {

  // get symmetry rotations of the crystal in cartesian coords
  // in case there's no symmetries, we need to trick Phoebe into
  // generating a crystal which uses them.
  bool useSyms = context.getUseSymmetries();
  context.setUseSymmetries(true);
  Crystal symCrystal(crystal);
  symCrystal.generateSymmetryInformation(context);
  auto symOps = symCrystal.getSymmetryOperations();
  context.setUseSymmetries(useSyms);

  auto invLVs = crystal.getDirectUnitCell().inverse();
  auto LVs = crystal.getDirectUnitCell();

  // to hold the symmetrized coeffs
  Eigen::Matrix3d symCoeffs;
  symCoeffs.setZero();

  for(SymmetryOperation symOp: symOps) {
    Eigen::Matrix3d rotation = symOp.rotation;
    rotation = LVs * rotation * invLVs; //convert to Cartesian
    Eigen::Matrix3d rotationTranspose = rotation.transpose();
    symCoeffs += rotationTranspose * transportCoeffs * rotation;
  }
  transportCoeffs = symCoeffs * (1. / symOps.size());
}

// TODO this should be a function of observable rather than of onsager,
// however, somehow Onsager does not inherit from observable...
void CoupledCoefficients::symmetrize(Eigen::Tensor<double, 3>& allTransportCoeffs) {

  // get symmetry rotations of the crystal in cartesian coords
  // in case there's no symmetries, we need to trick Phoebe into
  // generating a crystal which uses them.
  bool useSyms = context.getUseSymmetries();
  context.setUseSymmetries(true);
  Crystal symCrystal(crystal);
  symCrystal.generateSymmetryInformation(context);
  auto symOps = symCrystal.getSymmetryOperations();
  context.setUseSymmetries(useSyms);

  auto invLVs = crystal.getDirectUnitCell().inverse();
  auto LVs = crystal.getDirectUnitCell();

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    Eigen::Matrix3d transportCoeffs;

    // copy the 3x3 matrix of a single calculation
    for (int j : {0, 1, 2}) {
      for (int i : {0, 1, 2}) {
        transportCoeffs(i,j) = allTransportCoeffs(iCalc,i,j);
      }
    }
    // to hold the symmetrized coeffs
    Eigen::Matrix3d symCoeffs;
    symCoeffs.setZero();

    for(SymmetryOperation symOp: symOps) {
      Eigen::Matrix3d rotation = symOp.rotation;
      rotation = LVs * rotation * invLVs; //convert to Cartesian
      Eigen::Matrix3d rotationTranspose = rotation.transpose();
      symCoeffs += rotationTranspose * transportCoeffs * rotation;
    }
    transportCoeffs = symCoeffs * (1. / symOps.size());

    // place them back into the full tensor
    for (int j : {0, 1, 2}) {
      for (int i : {0, 1, 2}) {
        allTransportCoeffs(iCalc,i,j) = transportCoeffs(i,j);
      }
    }
  }
}

