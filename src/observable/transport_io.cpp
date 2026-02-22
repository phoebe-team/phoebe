#include "transport_io.h"
#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include <nlohmann/json.hpp>

// helper function to simplify code for printing transport to output
void appendTransportTensorForOutput(const Eigen::Tensor<double, 3>& tensor, int dimensionality,
                        double unitConv, int iCalc,
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

void printHelper(StatisticsSweep& statisticsSweep, int dimensionality,
                                const Eigen::Tensor<double, 3>& kappa,
                                const Eigen::Tensor<double, 3>& sigma,
                                const Eigen::Tensor<double, 3>& mobility,
                                const Eigen::Tensor<double, 3>& seebeck) {

  // only the head process should print
  if (!mpi->mpiHead()) return;

  Particle particle = statisticsSweep.getParticle();
  int numCalculations = statisticsSweep.getNumCalculations();

  if(numCalculations > 50) {
    std::cout << "\nBecause there are more than 50 calculations in this run,\n"
	    << "the transport tensors will not be printed to output, but can\n"
	    << "still be found in the corresponding output json file.\n"
	    << std::endl;
    return;
  }

  auto [unitsSigma, unitsKappa, unitsViscosity, unitsSeebeck, unitsMobility,
    convSigma, convKappa, convViscosity, convSeebeck, convMobility] = getTransportUnitsWithDimensions(dimensionality);

  std::cout << "\n";
  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    double doping = calcStat.doping;
    double chemPot = calcStat.chemicalPotential;

    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "iCalc = " << iCalc << ", T = " << temp * temperatureAuToSi;
    std::cout << "\n";

    std::cout.precision(4);
    std::cout << " (K)"
              << ", mu = " << chemPot * energyRyToEv << " (eV)";

    std::cout << std::scientific;
    std::cout << ", n = " << doping << " (cm^-3)" << std::endl;

    if(particle.isElectron()) {  // for phonon only, print just thermal conductivity
      
      std::cout << "Electrical Conductivity (" << unitsSigma << ")\n";
      std::cout.precision(5);
      for (int i = 0; i < dimensionality; i++) {
        std::cout << "  " << std::scientific;
        for (int j = 0; j < dimensionality; j++) {
          std::cout << " " << std::setw(13) << std::right;
          std::cout << sigma(iCalc, i, j) * convSigma;
        }
        std::cout << "\n";
      }
      std::cout << "\n";

      // Note: in metals, one has conductivity without doping
      // and the mobility = sigma / doping-density is ill-defined
      if (abs(doping) > 0.) {
        std::cout << "Carrier mobility (" << unitsMobility << ")\n";
        std::cout.precision(5);
        for (int i = 0; i < dimensionality; i++) {
          std::cout << "  " << std::scientific;
          for (int j = 0; j < dimensionality; j++) {
            std::cout << " " << std::setw(13) << std::right;
            std::cout << mobility(iCalc, i, j) * convMobility;
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }
    }

    std::cout << "Thermal Conductivity (" << unitsKappa << ")\n";
    std::cout.precision(5);
    for (int i = 0; i < dimensionality; i++) {
      std::cout << "  " << std::scientific;
      for (int j = 0; j < dimensionality; j++) {
        std::cout << " " << std::setw(13) << std::right;
        std::cout << kappa(iCalc, i, j) * convKappa;
      }
      std::cout << "\n";
    }
    std::cout << "\n";

    if(particle.isElectron()) { // for phonon only, print just thermal conductivity
      std::cout << "Seebeck Coefficient (" << unitsSeebeck << ")\n";
      std::cout.precision(5);
      for (int i = 0; i < dimensionality; i++) {
        std::cout << "  " << std::scientific;
        for (int j = 0; j < dimensionality; j++) {
          std::cout << " " << std::setw(13) << std::right;
          std::cout << seebeck(iCalc, i, j) * convSeebeck;
        }
        std::cout << "\n";
      }
      std::cout << std::endl;
    }
  }
}

void printHelper(const int iter, StatisticsSweep& statisticsSweep,
                                int dimensionality,
                                const Eigen::Tensor<double, 3>& kappa,
                                const Eigen::Tensor<double, 3>& sigma) {

  // only the head process should print
  if (!mpi->mpiHead()) return;

  Particle particle = statisticsSweep.getParticle();
  int numCalculations = statisticsSweep.getNumCalculations();

  // get the time
  time_t currentTime;
  currentTime = time(nullptr);
  // and format the time nicely
  char s[200];
  struct tm *p = localtime(&currentTime);
  strftime(s, 200, "%F, %T", p);

  std::cout << "Iteration: " << iter << " | " << s << "\n";
  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    
    if(particle.isElectron()) { // sigma only in the electronic case 
      std::cout << std::fixed;
      std::cout.precision(2);
      std::cout << "T = " << temp * temperatureAuToSi << ", sigma = ";
      std::cout.precision(5);
      for (int i = 0; i < dimensionality; i++) {
        std::cout << std::scientific;
        std::cout << sigma(iCalc, i, i) * elConductivityAuToSi << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "T = " << temp * temperatureAuToSi << ", k = ";
    std::cout.precision(5);
    for (int i = 0; i < dimensionality; i++) {
      std::cout << std::scientific;
      std::cout << kappa(iCalc, i, i) * thConductivityAuToSi << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void outputElectronicCoeffsToJSON(const std::string &outFileName,
                                StatisticsSweep& statisticsSweep,
                                int dimensionality,
                                const Eigen::Tensor<double, 3>& kappa,
                                const Eigen::Tensor<double, 3>& sigma,
                                const Eigen::Tensor<double, 3>& mobility,
                                const Eigen::Tensor<double, 3>& seebeck)  {

  if (!mpi->mpiHead()) return;
  
  // TODO change this to the "append transport coefficients" helper function 

  Kokkos::Profiling::pushRegion("onsagerToJSON");

  int numCalculations = statisticsSweep.getNumCalculations();

  auto [unitsSigma, unitsKappa, unitsViscosity, unitsSeebeck, unitsMobility,
    convSigma, convKappa, convViscosity, convSeebeck, convMobility] = getTransportUnitsWithDimensions(dimensionality);

  std::vector<double> temps, dopings, chemPots;
  std::vector<std::vector<std::vector<double>>> sigmaOut, mobilityOut, kappaOut, seebeckOut;
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
    std::vector<std::vector<double>> rows;
    for (int i = 0; i < dimensionality; i++) {
      std::vector<double> cols;
      for (int j = 0; j < dimensionality; j++) {
        cols.push_back(sigma(iCalc, i, j) * convSigma);
      }
      rows.push_back(cols);
    }
    sigmaOut.push_back(rows);

    // store the carrier mobility for output
    // Note: in metals, one has conductivity without doping
    // and the mobility = sigma / doping-density is ill-defined
    if (abs(doping) > 0.) {
      rows.clear();
      for (int i = 0; i < dimensionality; i++) {
        std::vector<double> cols;
        for (int j = 0; j < dimensionality; j++) {
          cols.push_back(mobility(iCalc, i, j) * convMobility);
        }
        rows.push_back(cols);
      }
      mobilityOut.push_back(rows);
    }

    // store thermal conductivity for output
    rows.clear();
    for (int i = 0; i < dimensionality; i++) {
      std::vector<double> cols;
      for (int j = 0; j < dimensionality; j++) {
        cols.push_back(kappa(iCalc, i, j) * convKappa);
      }
      rows.push_back(cols);
    }
    kappaOut.push_back(rows);

    // store seebeck coefficient for output
    rows.clear();
    for (int i = 0; i < dimensionality; i++) {
      std::vector<double> cols;
      for (int j = 0; j < dimensionality; j++) {
        cols.push_back(seebeck(iCalc, i, j) * convSeebeck);
      }
      rows.push_back(cols);
    }
    seebeckOut.push_back(rows);
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
  output["electricalConductivityUnit"] = unitsSigma;
  output["mobility"] = mobilityOut;
  output["mobilityUnit"] = unitsMobility;
  output["electronicThermalConductivity"] = kappaOut;
  output["electronicThermalConductivityUnit"] = unitsKappa;
  output["seebeckCoefficient"] = seebeckOut;
  output["seebeckCoefficientUnit"] = unitsSeebeck;
  output["particleType"] = "electron";
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();

  Kokkos::Profiling::popRegion();

}

void outputPhononThermalCondToJSON(const std::string &outFileName, 
                              StatisticsSweep& statisticsSweep, 
                              int dimensionality,
                              const Eigen::Tensor<double, 3>& kappa) { 
                                
  if (!mpi->mpiHead()) return;
  
  // TODO change this to the "append transport coefficients" helper function 
  
  // we'll just use the kappa one for now
  auto [unitsSigma, unitsKappa, unitsViscosity, unitsSeebeck, unitsMobility,
    convSigma, convKappa, convViscosity, convSeebeck, convMobility] = getTransportUnitsWithDimensions(dimensionality);

  std::vector<double> temps;
  std::vector<std::vector<std::vector<double>>> kappaOut;
  for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp * temperatureAuToSi);

    // store conductivity
    std::vector<std::vector<double>> rows;
    for (int i = 0; i < dimensionality; i++) {
      std::vector<double> cols;
      for (int j = 0; j < dimensionality; j++) {
        cols.push_back(kappa(iCalc, i, j) * convKappa);
      }
      rows.push_back(cols);
    }
    kappaOut.push_back(rows);
  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["thermalConductivity"] = kappaOut;
  output["temperatureUnit"] = "K";
  output["thermalConductivityUnit"] = unitsKappa;
  output["particleType"] = "phonon";
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

// replace this with a "transportCoeff" struct that contains data, string, and conversion from au->SI
std::tuple<std::string, std::string, std::string, std::string, std::string, double, double, double, double, double>
        getTransportUnitsWithDimensions(const int dimensionality) {

  std::string unitsSigma, unitsKappa, unitsViscosity, unitsSeebeck;
  double convSigma, convKappa, convViscosity, convSeebeck;
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
  // seebeck units are not dimension dept
  convSeebeck = thermopowerAuToSi * 1.0e6;
  unitsSeebeck = "muV / K";

  // FIXME this should likely be using dimensionality 
  double convMobility = mobilityAuToSi * pow(100., 2); // from m^2/Vs to cm^2/Vs
  std::string unitsMobility = "cm^2 / V / s";
  
  return std::make_tuple(unitsSigma, unitsKappa, unitsViscosity, unitsSeebeck, unitsMobility, convSigma, convKappa, convViscosity, convSeebeck, convMobility);
}