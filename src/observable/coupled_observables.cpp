#include "coupled_observables.h"
#include "onsager_utilities.h"
#include <nlohmann/json.hpp>
#include "viscosity_io.h"
#include "io.h"

CoupledCoefficients::CoupledCoefficients(StatisticsSweep& statisticsSweep_,
                                         Crystal &crystal_, Context &context_)
    : statisticsSweep(statisticsSweep_), crystal(crystal_), context(context_) {

  if (context.getHasSpinOrbit()) {
    spinFactor = 1.;
  } else {
    // TODO: for spin polarized calculations, we will have to set this to 1.
    spinFactor = 2.;
  }

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
  sigmaTotal.setZero();
  seebeckTotal.setZero();
  kappaTotal.setZero();
  mobilityTotal.setZero();

  // initialize the separate components
  seebeckSelf.resize(numCalculations, dimensionality, dimensionality);
  seebeckDrag.resize(numCalculations, dimensionality, dimensionality);

  alphaEl.resize(numCalculations, dimensionality, dimensionality);
  alphaPh.resize(numCalculations, dimensionality, dimensionality);

  kappaEl.resize(numCalculations, dimensionality, dimensionality);
  kappaPh.resize(numCalculations, dimensionality, dimensionality);
  kappaDrag.resize(numCalculations, dimensionality, dimensionality);

  seebeckDrag.setZero();
  seebeckSelf.setZero();
  alphaEl.setZero();
  alphaPh.setZero();
  kappaEl.setZero();
  kappaPh.setZero();
  kappaDrag.setZero();

  // intialize viscosities
  phViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  elViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  dragViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  totalViscosity = Eigen::Tensor<double, 5>(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  phViscosity.setZero();
  elViscosity.setZero();
  dragViscosity.setZero();
  totalViscosity.setZero();

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
    Error("Developer error: coupled relaxons only possible with one T value.");
  }

  int numElStates = int(elBandStructure->irrStateIterator().size());
  int numPhStates = int(phBandStructure->irrStateIterator().size());
  int numRelaxons = eigenvalues.size();

  int iCalc = 0;
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double T = calcStat.temperature / kBoltzmannRy;

  // electron and phonon participation ratios, summed over later
  Eigen::VectorXd phPR(numRelaxons);  phPR.setZero();
  Eigen::VectorXd elPR(numRelaxons);  elPR.setZero();

  // print info about the special eigenvectors ------------------------------
  // and save the indices that need to be skipped
  relaxonEigenvectorsCheck(eigenvectors, numRelaxons, numPhStates, theta0, theta_e);

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
  Eigen::Tensor<double, 3> Vphi(numRelaxons, 3, 3);
  elVphi.setZero(); phVphi.setZero(); Vphi.setZero();

  // sum over the alpha and v states that this process owns
  for (auto tup : eigenvectors.getAllLocalStates()) {

    auto is = std::get<0>(tup);
    auto gamma = std::get<1>(tup);

    // sum up the participation ratios
    // Here, we expect all the el state indices will come first
    if(is < numElStates) { elPR(gamma) += eigenvectors(is,gamma) * eigenvectors(is,gamma); }
    else { phPR(gamma) += eigenvectors(is,gamma) * eigenvectors(is,gamma); }

    if(gamma >= numRelaxons) continue; // this relaxon wasn't calculated
    // negative eigenvalues are spurious, zero ones are not summed here
    if(eigenvalues(gamma) <= 0) continue;

    StateIndex isIdx(0);
    Eigen::Vector3d v;

    if(is < numElStates) { // electronic state

      BteIndex iBteIdx(is);
      isIdx = elBandStructure->bteToState(iBteIdx);
      v = elBandStructure->getGroupVelocity(isIdx);

      for (auto j : {0, 1, 2}) {
        elV0(gamma, j) += eigenvectors(is,gamma) * v(j) * theta0(is);
        elVe(gamma, j) += eigenvectors(is,gamma) * v(j) * theta_e(is);
        // for viscosity, we have to skip the special eigenvectors
        if(gamma != alpha0 || gamma != alpha_e) {
          for(auto i : {0, 1, 2}) {
            elVphi(gamma, i, j) += eigenvectors(is,gamma) * v(j) * phi(i, is);
          }
        }
      }

    } else { // phonon states

      isIdx = StateIndex(is-numElStates);
      v = phBandStructure->getGroupVelocity(isIdx);
      double energy = phBandStructure->getEnergy(isIdx);
      if (energy < 0.001 / ryToCmm1) { continue; }

      for (auto j : {0, 1, 2}) {
        phV0(gamma, j) += eigenvectors(is,gamma) * v(j) * theta0(is);
        phVe(gamma, j) += eigenvectors(is,gamma) * v(j) * theta_e(is);

        // for viscosity, we have to skip the special eigenvectors
        if(gamma != alpha0 || gamma != alpha_e) {
          for(auto i : {0, 1, 2}) {
            phVphi(gamma, i, j) += eigenvectors(is,gamma) * v(j) * phi(i, is);
          }
        }
      }
    }
    // also collect total Vs to check that the separation of electron and phonon states is ok
    for (auto j : {0, 1, 2}) {
      V0(gamma, j) += eigenvectors(is,gamma) * v(j) * theta0(is);
      Ve(gamma, j) += eigenvectors(is,gamma) * v(j) * theta_e(is);
      for(auto i : {0, 1, 2}) {
        if(gamma != alpha0 || gamma != alpha_e) {
          Vphi(gamma, i, j) += eigenvectors(is,gamma) * v(j) * phi(i, is);
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
  Eigen::Tensor<double, 3> kappaContrib(numRelaxons, 3, 3);    kappaContrib.setZero();
  Eigen::Tensor<double, 3> sigmaContrib(numRelaxons, 3, 3);    sigmaContrib.setZero();
  Eigen::Tensor<double, 3> sigmaSContrib(numRelaxons, 3, 3);  sigmaSContrib.setZero();
  Eigen::VectorXd iiiiContrib(numRelaxons);   iiiiContrib.setZero();

  for(int gamma = 0; gamma < numRelaxons; gamma++) {

    if(eigenvalues(gamma) <= 0) continue; // should not be counted

    // TODO these should be traded for dimensionality!
    for (auto i : {0, 1, 2}) {
      for(auto j : {0, 1, 2} ) {

        // sigma
        sigmaLocal(i,j) += U * elVe(gamma,i) * elVe(gamma,j) * 1./eigenvalues(gamma);
        totalSigmaLocal(i,j) += U * Ve(gamma,i) * Ve(gamma,j) * 1./eigenvalues(gamma);
        sigmaContrib(gamma,i,j) += U * Ve(gamma,i) * Ve(gamma,j) * 1./eigenvalues(gamma);
        // sigmaS
        selfSigmaS(i,j) += 1. / kBoltzmannRy * sqrt(Ctot * U / T) * elVe(gamma,i) * elV0(gamma,j) * 1./eigenvalues(gamma);
        dragSigmaS(i,j) += 1. / kBoltzmannRy * sqrt(Ctot * U / T) * elVe(gamma,i) * phV0(gamma,j) * 1./eigenvalues(gamma);
        totalSigmaS(i,j) += 1. / kBoltzmannRy * sqrt(Ctot * U / T) * Ve(gamma,i) * V0(gamma,j) * 1./eigenvalues(gamma);
        sigmaSContrib(gamma,i,j) += 1. / kBoltzmannRy * sqrt(Ctot * U / T) * Ve(gamma,i) * V0(gamma,j) * 1./eigenvalues(gamma);
        // alpha
        alphaEl(0,i,j) += sqrt(Ctot * U * T) * elV0(gamma,i) * elVe(gamma,j) * 1./eigenvalues(gamma);
        alphaPh(0,i,j) += sqrt(Ctot * U * T) * phV0(gamma,i) * elVe(gamma,j) * 1./eigenvalues(gamma);
        // thermal conductivity
        kappaEl(0,i,j) += Ctot / kBoltzmannRy * 1./eigenvalues(gamma) * (elV0(gamma,i) * elV0(gamma,j));
        kappaPh(0,i,j) += Ctot / kBoltzmannRy * 1./eigenvalues(gamma) * (phV0(gamma,i) * phV0(gamma,j));
        kappaDrag(0,i,j) += Ctot / kBoltzmannRy * 1./eigenvalues(gamma) * (elV0(gamma,i) * phV0(gamma,j) + phV0(gamma,i) * elV0(gamma,j));
        kappaTotal(0,i,j) += Ctot / kBoltzmannRy * V0(gamma,i) * V0(gamma,j) * 1./eigenvalues(gamma);
        kappaContrib(gamma,i,j) += Ctot / kBoltzmannRy * V0(gamma,i) * V0(gamma,j) * 1./eigenvalues(gamma);

        // viscosities
        if(gamma != alpha0 || gamma != alpha_e) { // important -- including theta_0 or theta_e will lead to a wrong answer!

          iiiiContrib(gamma) += sqrt(M(0) * M(0)) * Vphi(gamma,0,0) * Vphi(gamma,0,0) * 1./eigenvalues(gamma);

          for(auto k : {0, 1, 2}) {
            for(auto l : {0, 1, 2}) {
              phViscosity(0,i,j,k,l) += sqrt(M(i) * M(k)) * phVphi(gamma,i,j) * phVphi(gamma,l,k) * 1./eigenvalues(gamma);
              elViscosity(0,i,j,k,l) += sqrt(M(i) * M(k)) * elVphi(gamma,i,j) * elVphi(gamma,l,k) * 1./eigenvalues(gamma);
              dragViscosity(0,i,j,k,l) += sqrt(M(i) * M(k)) * (elVphi(gamma,i,j) * phVphi(gamma,l,k)
                                                                   + phVphi(gamma,i,j) * elVphi(gamma,l,k)) * 1./eigenvalues(gamma);
              totalViscosity(0,i,j,k,l) += sqrt(M(i) * M(k)) * Vphi(gamma,i,j) * Vphi(gamma,l,k) * 1./eigenvalues(gamma);
            }
          }
        }
      }
    }
  }

  // need to apply a negative to S here for convention
  Eigen::Matrix3d seebeckSelfLocal = -sigmaLocal.inverse() * selfSigmaS;
  Eigen::Matrix3d seebeckDragLocal = -sigmaLocal.inverse() * dragSigmaS;
  Eigen::Matrix3d totalSeebeckLocal = -totalSigmaLocal.inverse() * totalSigmaS;

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
  kappa = kappaPh + kappaEl;
  seebeck = seebeckSelf + seebeckDrag;

}

// standard print
void CoupledCoefficients::print() {

  // prints the total tensors to the main output file
  printHelper(statisticsSweep, dimensionality, kappa, sigma, mobility, seebeck);

  // TODO prints the total viscosity components?

}

// helper function to simplify code for printing transport to output
void CoupledCoefficients::appendTransportTensorForOutput(
                        Eigen::Tensor<double, 3>& tensor, double& unitConv, int& iCalc,
                        std::vector<std::vector<std::vector<double>>>& outFormat) {

    std::vector<std::vector<double>> rows;
    for (int i = 0; i < dimensionality; i++) {
      std::vector<double> cols;
      for (int j = 0; j < dimensionality; j++) {
        cols.push_back(tensor(iCalc, i, j) * unitConv);
      }
      rows.push_back(cols);
    }
    outFormat.push_back(rows);
}

void CoupledCoefficients::outputToJSON(const std::string &outFileName) {

  if (!mpi->mpiHead()) return;

  // output the viscosities using the helper function in viscosity_io.h
  bool append = false;
  bool isPhonon = true;
  outputViscosityToJSON("coupled_relaxons_viscosity.json", "phononViscosity", phViscosity,
        isPhonon, append, statisticsSweep, dimensionality);
  isPhonon = false;
  append = true;
  outputViscosityToJSON("coupled_relaxons_viscosity.json", "electronViscosity", elViscosity,
        isPhonon, append, statisticsSweep, dimensionality);
  outputViscosityToJSON("coupled_relaxons_viscosity.json", "dragViscosity", dragViscosity,
        isPhonon, append, statisticsSweep, dimensionality);
  outputViscosityToJSON("coupled_relaxons_viscosity.json", "totalViscosity", totalViscosity,
        isPhonon, append, statisticsSweep, dimensionality);

  // output the transport coefficients
  int numCalculations = statisticsSweep.getNumCalculations();

  std::string unitsSigma, unitsKappa;
  double convSigma, convKappa;
  // TODO check the kappa units, I think it's missing a kb
  if (dimensionality == 1) {
    unitsSigma = "S m";
    unitsKappa = "W m / K";
    convSigma = elConductivityAuToSi * rydbergSi * rydbergSi;
    convKappa = thConductivityAuToSi * rydbergSi * rydbergSi;
  } else if (dimensionality == 2) {
    unitsSigma = "S";
    unitsKappa = "W / K";
    convSigma = elConductivityAuToSi * rydbergSi;
    convKappa = thConductivityAuToSi * rydbergSi;
  } else {
    unitsSigma = "S / m";
    unitsKappa = "W / m / K";
    convSigma = elConductivityAuToSi;
    convKappa = thConductivityAuToSi;
  }

  // TODO should this use dimensionality instead??
  double convMobility = mobilityAuToSi * pow(100., 2); // from m^2/Vs to cm^2/Vs
  std::string unitsMobility = "cm^2 / V / s";

  double convSeebeck = thermopowerAuToSi * 1.0e6;
  std::string unitsSeebeck = "muV / K";

  std::vector<double> temps, dopings, chemPots;
  std::vector<std::vector<std::vector<double>>> sigmaOut;
  std::vector<std::vector<std::vector<double>>> sigmaTotalOut;
  std::vector<std::vector<std::vector<double>>> mobilityOut;
  std::vector<std::vector<std::vector<double>>> mobilityTotalOut;

  std::vector<std::vector<std::vector<double>>> kappaOut;
  std::vector<std::vector<std::vector<double>>> kappaPhOut;
  std::vector<std::vector<std::vector<double>>> kappaElOut;
  std::vector<std::vector<std::vector<double>>> kappaDragOut;
  std::vector<std::vector<std::vector<double>>> kappaTotalOut;

  std::vector<std::vector<std::vector<double>>> seebeckOut;
  std::vector<std::vector<std::vector<double>>> seebeckDragOut;
  std::vector<std::vector<std::vector<double>>> seebeckSelfOut;
  std::vector<std::vector<std::vector<double>>> seebeckTotalOut;

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
    appendTransportTensorForOutput(sigma, convSigma, iCalc, sigmaOut);
    appendTransportTensorForOutput(sigmaTotal, convSigma, iCalc, sigmaTotalOut);

    // store the carrier mobility for output
    // Note: in metals, one has conductivity without doping
    // and the mobility = sigma / doping-density is ill-defined
    if (abs(doping) > 0.) {
      appendTransportTensorForOutput(mobility, convMobility, iCalc, mobilityOut);
      appendTransportTensorForOutput(mobilityTotal, convMobility, iCalc, mobilityTotalOut);
    }

    // store thermal conductivity for output
    appendTransportTensorForOutput(kappa, convKappa, iCalc, kappaOut);
    appendTransportTensorForOutput(kappaPh, convKappa, iCalc, kappaPhOut);
    appendTransportTensorForOutput(kappaEl, convKappa, iCalc, kappaElOut);
    appendTransportTensorForOutput(kappaDrag, convKappa, iCalc, kappaDragOut);
    appendTransportTensorForOutput(kappaTotal, convKappa, iCalc, kappaTotalOut);

    // store seebeck coefficient for output
    appendTransportTensorForOutput(seebeck, convSeebeck, iCalc, seebeckOut);
    appendTransportTensorForOutput(seebeckDrag, convSeebeck, iCalc, seebeckDragOut);
    appendTransportTensorForOutput(seebeckSelf, convSeebeck, iCalc, seebeckSelfOut);
    appendTransportTensorForOutput(seebeckTotal, convSeebeck, iCalc, seebeckTotalOut);

  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["temperatureUnit"] = "K";
  output["dopingConcentrations"] = dopings;
  output["dopingConcentrationUnit"] = "cm$^{-" + std::to_string(dimensionality) + "}$";
  output["chemicalPotentials"] = chemPots;
  output["chemicalPotentialUnit"] = "eV";

  output["electricalConductivity"] = sigmaOut;
  output["totalElectricalConductivity"] = sigmaTotalOut;
  output["electricalConductivityUnit"] = unitsSigma;
  output["mobility"] = mobilityOut;
  output["totalMobility"] = mobilityTotalOut;
  output["mobilityUnit"] = unitsMobility;

  output["thermalConductivity"] = kappaOut;
  output["phononThermalConductivity"] = kappaPhOut;
  output["electronicThermalConductivity"] = kappaElOut;
  output["dragThermalConductivity"] = kappaDragOut;
  output["totalThermalConductivity"] = kappaTotalOut;
  output["thermalConductivityUnit"] = unitsKappa;

  output["seebeckCoefficient"] = seebeckOut;
  output["dragSeebeckCoefficient"] = seebeckDragOut;
  output["selfSeebeckCoefficient"] = seebeckSelfOut;
  output["totalSeebeckCoefficient"] = seebeckTotalOut;
  output["seebeckCoefficientUnit"] = unitsSeebeck;

  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();

}

void CoupledCoefficients::relaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors,
                        int& numRelaxons, int& numPhStates, Eigen::VectorXd& theta0, Eigen::VectorXd& theta_e) {

  Eigen::VectorXd prod0(numRelaxons);
  Eigen::VectorXd prod_e(numRelaxons);
  prod0.setZero(); prod_e.setZero();
  //Eigen::Vector3d vecprodphi1(numRelaxons);
  //Eigen::Vector3d vecprodphi2(numRelaxons);
  //Eigen::Vector3d vecprodphi3(numRelaxons);

  // sum over the alpha and v states that this process owns
  for (auto tup : eigenvectors.getAllLocalStates()) {

    auto is = std::get<0>(tup);
    auto gamma = std::get<1>(tup);

    prod0(gamma) += eigenvectors(is,gamma) * theta0(is);
    prod_e(gamma) += eigenvectors(is,gamma) * theta_e(is);
    //vecprodphi1[gamma] += eigenvectors(is,gamma) * phi(0,is);
    //vecprodphi2[gamma] += eigenvectors(is,gamma) * phi(1,is);
    //vecprodphi3[gamma] += eigenvectors(is,gamma) * phi(2,is);

  }
  // scalar products with vectors
  mpi->allReduceSum(&prod0); mpi->allReduceSum(&prod_e);
  //mpi->allReduceSum(&vecprodphi1); mpi->allReduceSum(&vecprodphi2); mpi->allReduceSum(&vecprodphi3);

  // find the element with the maximum product
  prod0 = prod0.cwiseAbs();
  prod_e = prod_e.cwiseAbs();
  Eigen::Index maxCol0, idxAlpha0;
  Eigen::Index maxCol_e, idxAlpha_e;
  float maxTheta0 = prod0.maxCoeff(&idxAlpha0, &maxCol0);
  float maxThetae = prod_e.maxCoeff(&idxAlpha_e, &maxCol_e);

  if(mpi->mpiHead()) {
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << "Maximum scalar product theta_0.theta_alpha = " << maxTheta0 << " at index " << idxAlpha0 << "." << std::endl;
    std::cout << "First ten products with theta_0:";
    for(int gamma = 0; gamma < 10; gamma++) { std::cout << " " << prod0(gamma); }
    std::cout << "\n\nMaximum scalar product theta_e.theta_alpha = " << maxThetae << " at index " << idxAlpha_e << "." << std::endl;
    if(numRelaxons - numPhStates >= 10) { // avoid a segfault in an edge case of few el states
      std::cout << "First ten products with theta_e starting with the numPhStates + 1 indexed eigenvector:";
      for(int gamma = 0; gamma < 10; gamma++) { std::cout << " " << prod_e(gamma+numPhStates); }
    }
    std::cout << std::endl;
  }

  // save these indices to the class objects
  // if they weren't really found, we leave these indices
  // as -1 so that no relaxons are skipped
  if(maxTheta0 >= 0.75) alpha0 = idxAlpha0;
  if(maxThetae >= 0.75) alpha_e = idxAlpha_e;

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
  double Nkq = (Nk + Nq)/2.;

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
      if (energy < 0.001 / ryToCmm1) { continue; }
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
  //     phi = sqrt(1/(kbT*volume*Nkq*M)) * g-1 * ds * hbar * wavevector;
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

      ds(is) = sqrt( spinFactor * Nkq / Nk );

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
      // TODO phononCutoff should be standardized across the code
      // Discard ph states with negative energies
      if (energy < 0.001 / ryToCmm1) { continue; }
      // this is in cartesian coords
      Eigen::Vector3d q = phBandStructure->getWavevector(phIdx);
      q = phBandStructure->getPoints().bzToWs(q,Points::cartesianCoordinates);
      sqrtPopTerm = sqrt(phonon.getPopPopPm1(energy, kBT, 0));

      ds(is) = sqrt( Nkq / Nq );

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
  M = G + A;

  // apply the normalization to theta_e
  theta_e *= 1./sqrt(kBT * U * Nkq * volume);
  // apply normalization to theta0
  theta0 *= 1./sqrt(kBT * T * volume * Nkq * Ctot);
  // apply normalization to phi
  for(int is = 0; is < numStates; is++) {
    for(int i : {0,1,2}) phi(i,is) *= 1./sqrt(kBT * volume * Nkq * M(i));
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
    Warning("Developer error: Your energy or charge conservation eigenvectors do not"
                " normalize to 1.\nThis indicates something has gone very wrong "
                "with your relaxons solve (or your mesh is super small), please report this.");
  }
}

void CoupledCoefficients::outputDuToJSON(CoupledScatteringMatrix& coupledScatteringMatrix, Context& context) {

  BaseBandStructure* phBandStructure = coupledScatteringMatrix.getPhBandStructure();
  BaseBandStructure* elBandStructure = coupledScatteringMatrix.getElBandStructure();

  int numElStates = int(elBandStructure->irrStateIterator().size());
  auto calcStat = statisticsSweep.getCalcStatistics(0); // only one calc for relaxons
  double kBT = calcStat.temperature;

  // write D to file before diagonalizing, as the scattering matrix
  // will be destroyed by scalapack
  Eigen::MatrixXd Du(3,3); Du.setZero();
  Eigen::MatrixXd Wjie(3,3); Wjie.setZero();
  Eigen::MatrixXd Wji0(3,3); Wji0.setZero();
  Eigen::MatrixXd elWji0(3,3); elWji0.setZero();
  Eigen::MatrixXd phWji0(3,3); phWji0.setZero();

  // sum over the alpha and v states that this process owns
  for (auto tup : coupledScatteringMatrix.getAllLocalStates()) {

    auto is1 = std::get<0>(tup);
    auto is2 = std::get<1>(tup);
    for (auto i : {0, 1, 2}) {
      for (auto j : {0, 1, 2}) {
        if(context.getUseUpperTriangle()) {
          if( i == j ) {
            Du(i,j) += phi(i,is1) * coupledScatteringMatrix(is1,is2) * phi(j,is2);
          } else {
            Du(i,j) += 2. * phi(i,is1) * coupledScatteringMatrix(is1,is2) * phi(j,is2);
          }
        } else {
          Du(i,j) += phi(i,is1) * coupledScatteringMatrix(is1,is2) * phi(j,is2);
        }
      }
    }
  }
  mpi->allReduceSum(&Du);

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
      }
    }
  }
  for (int is : phBandStructure->parallelStateIterator()) {
    auto isIdx = StateIndex(is);
    double en = phBandStructure->getEnergy(isIdx);
    // discard acoustic phonon modes
    if (en < 0.001 / ryToCmm1) { continue; }
    auto v = phBandStructure->getGroupVelocity(isIdx);
    for (auto j : {0, 1, 2}) {
      for (auto i : {0, 1, 2}) {
        // note: phi and theta here are elStates long, so we need to shift the state
        // index to account for the fact that we summed over the electronic part above
        // calculate qunatities for the real-space solve
        Wji0(j,i) += phi(i,is+numElStates) * v(j) * theta0(is+numElStates);
        phWji0(j,i) += phi(i,is+numElStates) * v(j) * theta0(is+numElStates);
        Wjie(j,i) += phi(i,is+numElStates) * v(j) * theta_e(is+numElStates);
      }
    }
  }
  mpi->allReduceSum(&Wji0); mpi->allReduceSum(&Wjie);
  mpi->allReduceSum(&phWji0);
  mpi->allReduceSum(&elWji0);

  // NOTE we cannot use nested vectors from the start, as
  // vector<vector> is not necessarily contiguous and MPI
  // cannot all reduce on it
  std::vector<std::vector<double>> vecDu;
  std::vector<std::vector<double>> vecWji0;
  std::vector<std::vector<double>> vecWji0_el;
  std::vector<std::vector<double>> vecWji0_ph;
  std::vector<std::vector<double>> vecWjie;
  for (auto i : {0, 1, 2}) {
    std::vector<double> temp1,temp2,temp3,temp4,temp5;
    for (auto j : {0, 1, 2}) {
      temp1.push_back(Du(i,j) / (energyRyToFs / twoPi));
      temp2.push_back(Wji0(i,j) * velocityRyToSi);
      temp3.push_back(elWji0(i,j) * velocityRyToSi);
      temp4.push_back(phWji0(i,j) * velocityRyToSi);
      temp5.push_back(Wjie(i,j) * velocityRyToSi);
    }
    vecDu.push_back(temp1);
    vecWji0.push_back(temp2);
    vecWji0_el.push_back(temp3);
    vecWji0_ph.push_back(temp4);
    vecWjie.push_back(temp5);
  }

  // this extra kBoltzmannRy is required when we calculate specific heat ...
  // TODO need to keep track of this and figure out where it's coming from
  double specificHeatConversion = kBoltzmannSi / pow(bohrRadiusSi, 3) / kBoltzmannRy;

  // convert Ai to SI, in units of picograms/(mu m^3)
  double Aconversion = electronMassSi /
                       std::pow(distanceBohrToMum,dimensionality) * // convert AU mass / V -> SI
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
    nlohmann::json output;
    output["temperature"] = kBT * temperatureAuToSi;
    output["Wji0"] = vecWji0;
    output["phononWji0"] = vecWji0_ph;
    output["electronWji0"] = vecWji0_el;
    output["Wjie"] = vecWjie;
    output["Du"] = vecDu;
    output["temperatureUnit"] = "K";
    output["wUnit"] = "m/s";
    output["DuUnit"] = "fs^{-1}";
    output["phononSpecificHeat"] = Cph * specificHeatConversion;
    output["electronSpecificHeat"] = Cel * specificHeatConversion;
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
