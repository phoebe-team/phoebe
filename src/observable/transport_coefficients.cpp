#include "transport_coefficients.h"
#include "transport_io.h"
#include "viscosity_io.h"
#include "relaxons.h"
#include <nlohmann/json.hpp>

TransportCoefficients::TransportCoefficients(Context &context, StatisticsSweep &statisticsSweep, Crystal& crystal, BaseBandStructure &bandStructure)
    : context(context), statisticsSweep(statisticsSweep), crystal(crystal), bandStructure(bandStructure), particle(bandStructure.getParticle()) {

  // TODO : change this to use the context getSpinDegeneracyFactor
  if (context.getHasSpinOrbit() || particle.isPhonon()) {
    spinFactor = 1.;
  } else {
    spinFactor = 2.;
  }

  // matrix had to be in memory for this calculation.
  // therefore, we can only ever have one numCalc
  numCalculations = statisticsSweep.getNumCalculations();
  dimensionality = crystal.getDimensionality();

  specificHeat.resize(numCalculations);

  // set up and zero all the containers for transport coefficients
  for (auto coeff : {&sigma, &seebeck, &kappa, &mobility}) {
    coeff->resize(numCalculations, dimensionality, dimensionality);
    coeff->setZero();
  }

  // intialize viscosity
  viscosity.resize(numCalculations, dimensionality, dimensionality, dimensionality, dimensionality);
  viscosity.setZero();
}

// standard print
void TransportCoefficients::print() {

  // print viscosities
  std::string viscosityName = (particle.isPhonon()) ? "Phonon" : "Electron" ;
  printViscosity(viscosityName, viscosity, statisticsSweep, dimensionality);

  // prints the total tensors to the main output file
  printHelper(statisticsSweep, dimensionality, kappa, sigma, mobility, seebeck);
}

void TransportCoefficients::outputToJSON() {

  if (!mpi->mpiHead()) return;

  // output the viscosity
  bool append = false; // it's a new file to write to
  std::string viscosityName = (particle.isPhonon()) ? "phononViscosity" : "electronViscosity" ;
  std::string outFileName = (particle.isPhonon()) ? "relaxons_ph_viscosity.json" : "relaxons_el_viscosity.json";
  outputViscosityToJSON(outFileName, viscosityName, viscosity, append, statisticsSweep, dimensionality);

  // output the conductivities
  if(particle.isPhonon()) {
    outputPhononThermalCondToJSON("relaxons_phonon_thermal_cond.json", statisticsSweep, dimensionality, kappa);
  } else {
    outputElectronicCoeffsToJSON("relaxons_onsager_coefficients.json", statisticsSweep, dimensionality, kappa, sigma, mobility, seebeck);
  }

}

// calculate special eigenvectors, output real space quantities
void TransportCoefficients::prepareRelaxons(ScatteringMatrix& scatteringMatrix) {

  if(statisticsSweep.getNumCalculations() != 1) DeveloperError("prepareRelaxons must be called with 1 calc.");

  // we need a dummy variable for theta_e, as it doesn't matter for phonons
  //Eigen::VectorXd theta_e(bandStructure.getNumStates());
  genericCalcSpecialEigenvectors(context, bandStructure, statisticsSweep,
                          spinFactor, theta0, theta_e, phi, specificHeat(0), U, A);

  // output the real space information
  genericOutputRealSpaceToJSON(context, scatteringMatrix, bandStructure, statisticsSweep,
                                theta0, theta_e, phi, specificHeat(0), A);
}

/* Calc relaxons transport coefficients for electron or phonon only case */
void TransportCoefficients::calcFromRelaxons(const Eigen::VectorXd &eigenvalues, ParallelMatrix<double> &eigenvectors) {

  // Note: the calcSpecialEigenvectors has been called before this, as it's
  // needed before this function to calculate phi, and then to use phi with D

  // TODO add OMP and MPI parallelism here

  if(!context.getEnforceDetailedBalance()) {
    Warning("Viscosity calculated without the enforcing detailed balance condition of the scattering matrix"
      "\ncan have major issues -- if the charge and energy eigenvectors are not well found (better than 75% overlap),"
      " they may make a large, spurious contribution to viscosity!");
  }
  if (numCalculations > 1) {
    DeveloperError("Relaxons viscosity cannot be calculated for more than one T or mu value.");
  }

  numRelaxons = (context.getNumRelaxonsEigenvalues() > 0) ? context.getNumRelaxonsEigenvalues() : eigenvectors.rows();
  Particle particle = bandStructure.getParticle();
  int iCalc = 0; // zero index, because we only run one for relaxons
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double T = calcStat.temperature / kBoltzmannRy;
  double kBT = kBoltzmannRy * T;
  double mu = calcStat.chemicalPotential;

  // print info about the special eigenvectors ------------------------------
  // and save the indices that need to be skipped
  if(mpi->mpiHead()) std::cout << "Checking scalar products of scattering matrix eigenvectors with special eigenvectors: -------------" << std::endl;
  alpha0 = relaxonEigenvectorOverlap(eigenvectors, theta0, "theta0");
  if(particle.isElectron()) {
    alpha_e = relaxonEigenvectorOverlap(eigenvectors, theta_e, "theta_e");
  }

  // drift eigenvector overlaps ----------
  // for now, we don't save these drift eigenvector indices
  {
    relaxonEigenvectorOverlap(eigenvectors, phi(0, Eigen::placeholders::all), "phi_x");
    relaxonEigenvectorOverlap(eigenvectors, phi(1, Eigen::placeholders::all), "phi_y");
    relaxonEigenvectorOverlap(eigenvectors, phi(2, Eigen::placeholders::all), "phi_z");
    if(mpi->mpiHead()) std::cout << std::endl; // just a new line for better print out
  }

  // calculate the V components
  // -----------------------------------------------------------
  Eigen::MatrixXd Ve = Eigen::MatrixXd::Zero(numRelaxons, 3);
  Eigen::MatrixXd V0 = Eigen::MatrixXd::Zero(numRelaxons, 3);
  Eigen::Tensor<double, 3> Vphi(numRelaxons, 3, 3);
  Vphi.setZero();

  // sum over the alpha and v states that this process owns
  for (auto [is, gamma] : eigenvectors.getAllLocalStates()) {

    if (gamma >= numRelaxons)
      continue; // this relaxon wasn't calculated

    // negative eigenvalues are spurious, zero ones are not summed here
    // if(eigenvalues(gamma) <= 0) continue; // count them here but not later

    StateIndex isIdx = StateIndex(is);
    Eigen::Vector3d v = bandStructure.getGroupVelocity(isIdx);
    // don't sum over acoustic phonons
    if (particle.isPhonon() && bandStructure.getEnergy(isIdx) < phEnergyCutoff) {
      continue;
    }

    // set tau, avoiding div by zero issues
    //double tau = abs(1. / eigenvalues(gamma));
    if (eigenvalues(gamma) < 1e-10) continue;

    // also collect total Vs to check that the separation of electron and phonon states is ok
    for (auto j : {0, 1, 2}) {

      V0(gamma, j) += eigenvectors(is,gamma) * v(j) * theta0(is);
      Ve(gamma, j) += eigenvectors(is,gamma) * v(j) * theta_e(is);

      for (auto i : {0, 1, 2}) {
        if (gamma != alpha0 && gamma != alpha_e) {
          Vphi(gamma, i, j) +=  eigenvectors(is, gamma) * v(j) * phi(i, is);
        }
      }
    }
  }
  // reduce contributions from different processes transport velocities
  mpi->allReduceSum(&V0);
  mpi->allReduceSum(&Ve);
  mpi->allReduceSum(&Vphi);

  // TODO Output velocities to file -------------------------------------------------

  // local copies for linear algebra ops with eigen
  Eigen::Matrix3d sigmaLocal = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d sigmaS = Eigen::Matrix3d::Zero();

  // containers to calculate the specific contributions to the transport tensors
  Eigen::Tensor<double, 3> kappaContrib(numRelaxons, 3, 3), sigmaContrib(numRelaxons, 3, 3), sigmaSContrib(numRelaxons, 3, 3);
  std::vector<double> iiiiContrib(numRelaxons);
  kappaContrib.setZero();
  sigmaContrib.setZero();
  sigmaSContrib.setZero();

  // TODO could parallelize this
  for (int alpha = 0; alpha < numRelaxons; alpha++) {

    if (eigenvalues(alpha) <= 0) {
      continue;
    }
    double tau = abs(1. / eigenvalues(alpha));

    // NOTE: remove energy and charge eigenvectors
    if (alpha == alpha0 || alpha == alpha_e)  continue;

    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {

        // thermal conductivity --------------------------
        kappa(0,i,j) += specificHeat(0) / kBoltzmannRy * V0(alpha,i) * V0(alpha,j) * tau;
        kappaContrib(alpha,i,j) += specificHeat(0) / kBoltzmannRy * V0(alpha,i) * V0(alpha,j) * tau;

        // viscosities ----------------------------------------------------
        double xxxx = sqrt(A(0) * A(0)) * Vphi(alpha, 0, 0) * Vphi(alpha, 0, 0) * tau;
        double yyyy =  sqrt(A(1) * A(1)) * Vphi(alpha, 1, 1) * Vphi(alpha, 1, 1) * tau;
        iiiiContrib[alpha] += (xxxx + yyyy) / 2.;

        for(auto k : {0, 1, 2}) {
          for(auto l : {0, 1, 2}) {
            viscosity(0,i,j,k,l) += sqrt(A(i) * A(k)) * Vphi(alpha,i,j) * Vphi(alpha,l,k) * tau;
          }
        }

        // do the electrical conductivity specific quantities --------------------------
        if(particle.isElectron()) {

          // sigma
          sigmaLocal(i, j) += U * Ve(alpha, i) * Ve(alpha, j) * tau;
          sigmaContrib(alpha, i, j) += U * Ve(alpha, i) * Ve(alpha, j) * tau;

          // sigmaS
          sigmaS(i,j) -= 1. / kBoltzmannRy * sqrt(specificHeat(0) * U / T) * Ve(alpha,i) * V0(alpha,j) * tau;
          sigmaSContrib(alpha,i,j) -= 1. / kBoltzmannRy * sqrt(specificHeat(0) * U / T) * Ve(alpha,i) * V0(alpha,j) * tau;

          // alpha
          //alpha(0,i,j) += sqrt(Ctot * U * T) * V0(gamma,i) * Ve(gamma,j) * tau; // check before unlocking
        }
      }
    }
  }

  // output relaxons information: contribution breakdown, relaxons eigenvectors, out of eq distribution to file
  // ----------------------------------------------------------------------------------------------------------

  // TODO : these should be simplified as there's just too much passing happing here.
  // Maybe a structure of relaxon basics... or a reference to a transport coeffs object?

  // output relaxons eigenvectors ------------------------------
  std::vector<BaseBandStructure*> bs = {&bandStructure};
  outputRelaxonsToHDF5(eigenvectors, eigenvalues, bs, theta0, theta_e, phi);

  outputRelaxonContributionsToJSON(statisticsSweep, particle, dimensionality,
                                    sigmaContrib, kappaContrib,
                                    sigmaSContrib, iiiiContrib);

  // output out of eq distributions ----------------------------
  // NOTE: specific heat units need this extract kBoltzmann factor, which should be later removed when this is fixed

  // delta pop for grad T
  outputRelaxonDeltaPopToHDF5(eigenvectors, eigenvalues, bandStructure, V0, sqrt( (specificHeat(0)) / ( kBT * T )), 0, "_gradT", false, dimensionality, kBT, mu, numRelaxons, alpha0, alpha_e);

  if(particle.isElectron()) // delta pop for delta V
    outputRelaxonDeltaPopToHDF5(eigenvectors, eigenvalues, bandStructure, Ve, sqrt( U / ( kBT )), 0, "_E", true, dimensionality, kBT, mu, numRelaxons, alpha0, alpha_e);

  // delta pop for u_xyz
  std::vector<std::string> xyz = {"_x","_y","_z"};
  for (int j = 0; j < dimensionality; j++) {
    // This is awful, but we have to convert the eigen:::tensor slice to matrix. the slicing methods cause problems,
    // and also require copies anyway, so here we are explicitly copying.
    Eigen::MatrixXd Vphi_slice(numRelaxons, 3);
    for (int alpha = 0; alpha < numRelaxons; alpha++) {
      for (int i = 0; i < dimensionality; i++)
        Vphi_slice(alpha, i) = Vphi(alpha, i, j);
    }
    outputRelaxonDeltaPopToHDF5(eigenvectors, eigenvalues, bandStructure, Vphi_slice, sqrt( A(j) / ( kBT * T )), 0, "_u"+xyz[j], true, dimensionality, kBT, mu, numRelaxons, alpha0, alpha_e);
  }

  // output relaxon velocities -----------------------------
  outputRelaxonVelocitiesToHDF5(eigenvalues, V0, Ve, Vphi, particle, numRelaxons);

  // copy S and sigma into final tensors to be printed,  convert sigma -> mobility
  if(particle.isElectron()) {

    // seebeck = matmul(L_EE_inv, L_ET)
    Eigen::Matrix3d seebeckLocal = sigmaLocal.inverse() * sigmaS;

    double doping = abs(statisticsSweep.getCalcStatistics(iCalc).doping);
    doping *= pow(distanceBohrToCm, dimensionality); // from cm^-3 to bohr^-3
    for (int i = 0; i < dimensionality; i++) {
      for (auto j : {0, 1, 2}) {

        seebeck(0, i, j) = seebeckLocal(i, j);
        sigma(0, i, j) = sigmaLocal(i, j);
        mobility(0, i, j) = sigma(0, i, j);

        if (doping > 0.) {
          mobility(0, i, j) /= doping;
        }
      }
    }
  }
}

void TransportCoefficients::outputRelaxonContributionsToJSON(StatisticsSweep& statisticsSweep, const Particle &particle,
    int dimensionality, const Eigen::Tensor<double, 3> sigmaContrib, const Eigen::Tensor<double, 3> kappaContrib,
    const Eigen::Tensor<double, 3> sigmaSContrib, std::vector<double> iiiiContrib) {

  // output the transport coefficients
  int numCalculations = statisticsSweep.getNumCalculations();
  if(numCalculations > 1) DeveloperError("Relaxons cannot be run with more than one temperature!");

  auto [unitsSigma, unitsKappa, unitsViscosity, unitsSeebeck, unitsMobility,
    convSigma, convKappa, convViscosity, convSeebeck, convMobility] = getTransportUnitsWithDimensions(dimensionality);

  std::vector<double> temps, dopings, chemPots;

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp * temperatureAuToSi);
    double doping = calcStat.doping;
    dopings.push_back(doping); // output in (cm^-3)
    double chemPot = calcStat.chemicalPotential;
    chemPots.push_back(chemPot * energyRyToEv); // output in eV

  }

  // now output the separated contributions
  std::vector<std::vector<std::vector<double>>> sigmaContribOut, kappaContribOut, sigmaSContribOut;
  double convSigmaS = convSeebeck * convSigma;

  int numRelaxons = iiiiContrib.size();

  for (int gamma = 0; gamma < numRelaxons; gamma++) {

    // convert viscosity units
    iiiiContrib[gamma] *= convViscosity;

    appendTransportTensorForOutput(kappaContrib, dimensionality, convKappa,
                                   gamma, kappaContribOut);
    if(particle.isPhonon()) continue;
    appendTransportTensorForOutput(sigmaContrib, dimensionality, convSigma,
                                   gamma, sigmaContribOut);
    appendTransportTensorForOutput(sigmaSContrib, dimensionality, convSigmaS,
                                   gamma, sigmaSContribOut);
  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["temperatureUnit"] = "K";
  output["dopingConcentrations"] = dopings;
  output["dopingConcentrationUnit"] =
      "cm$^{-" + std::to_string(dimensionality) + "}$";
  output["chemicalPotentials"] = chemPots;
  output["chemicalPotentialUnit"] = "eV";

  if(particle.isElectron()) {
    output["electricalConductivityContributions"] = sigmaContribOut;
    output["electricalConductivityUnit"] = unitsSigma;

    output["sigmaSContribution"] = sigmaSContribOut;
    output["sigmaSCoefficientUnit"] = unitsSeebeck + " x " + unitsSigma;
  }
  output["thermalConductivityContribution"] = kappaContribOut;
  output["thermalConductivityUnit"] = unitsKappa;

  output["iiiiViscosityContribution"] = iiiiContrib;

  std::ofstream o( (particle.isElectron()) ? "relaxons_el_transport_contributions.json" : "relaxons_ph_transport_contributions.json");
  o << std::setw(3) << output << std::endl;
  o.close();

}