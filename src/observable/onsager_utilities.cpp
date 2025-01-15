#include "statistics_sweep.h"
#include "onsager_utilities.h"
#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include <nlohmann/json.hpp>

void onsagerToTransportCoeffs(StatisticsSweep& statisticsSweep, int& dimensionality,
                        Eigen::Tensor<double, 3>& LEE, Eigen::Tensor<double, 3>& LTE,
                        Eigen::Tensor<double, 3>& LET, Eigen::Tensor<double, 3>& LTT,
                        Eigen::Tensor<double, 3>& kappa, Eigen::Tensor<double, 3>& sigma,
                        Eigen::Tensor<double, 3>& mobility, Eigen::Tensor<double, 3>& seebeck) {

  Kokkos::Profiling::pushRegion("onsagerToTransportCoeffs");

  int numCalculations = statisticsSweep.getNumCalculations();

  sigma = LEE;
  mobility = sigma;

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    Eigen::MatrixXd thisLEE(dimensionality, dimensionality);
    Eigen::MatrixXd thisLET(dimensionality, dimensionality);
    Eigen::MatrixXd thisLTE(dimensionality, dimensionality);
    Eigen::MatrixXd thisLTT(dimensionality, dimensionality);
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        thisLEE(i, j) = LEE(iCalc, i, j);
        thisLET(i, j) = LET(iCalc, i, j);
        thisLTE(i, j) = LTE(iCalc, i, j);
        thisLTT(i, j) = LTT(iCalc, i, j);
      }
    }

    // seebeck = matmul(L_EE_inv, L_ET)
    Eigen::MatrixXd thisSeebeck = (thisLEE.inverse()) * thisLET;

    // k = L_tt - L_TE L_EE^-1 L_ET
    Eigen::MatrixXd thisKappa = thisLTE * (thisLEE.inverse());
    thisKappa = -(thisLTT - thisKappa * thisLET);
    double doping = abs(statisticsSweep.getCalcStatistics(iCalc).doping);
    doping *= pow(distanceBohrToCm, dimensionality); // from cm^-3 to bohr^-3
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        seebeck(iCalc, i, j) = thisSeebeck(i, j);
        kappa(iCalc, i, j) = thisKappa(i, j);
        if (doping > 0.) {
          mobility(iCalc, i, j) /= doping;
        }
      }
    }
  }
  Kokkos::Profiling::popRegion();
}

void printHelper(StatisticsSweep& statisticsSweep, int& dimensionality,
                                Eigen::Tensor<double, 3>& kappa,
                                Eigen::Tensor<double, 3>& sigma,
                                Eigen::Tensor<double, 3>& mobility,
                                Eigen::Tensor<double, 3>& seebeck) {

  // only the head process should print
  if (!mpi->mpiHead()) return;

  int numCalculations = statisticsSweep.getNumCalculations();

  if(numCalculations > 50) { 
    std::cout << "\nBecause there are more than 50 calculations in this run,\n"
	    << "the transport tensors will not be printed to output, but can\n"
	    << "still be found in the corresponding output json file.\n" 
	    << std::endl;
    return; 
  }

  std::string unitsSigma, unitsKappa;
  double convSigma, convKappa;
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

  double convMobility = mobilityAuToSi * 100 * 100; // from m^2/Vs to cm^2/Vs
  std::string unitsMobility = "cm^2 / V / s";

  double convSeebeck = thermopowerAuToSi * 1e6;
  std::string unitsSeebeck = "muV / K";

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

void printHelper(const int &iter, StatisticsSweep& statisticsSweep,
                                int& dimensionality,
                                Eigen::Tensor<double, 3>& kappa,
                                Eigen::Tensor<double, 3>& sigma) {

  // only the head process should print
  if (!mpi->mpiHead()) return;

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
    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "T = " << temp * temperatureAuToSi << ", sigma = ";
    std::cout.precision(5);
    for (int i = 0; i < dimensionality; i++) {
      std::cout << std::scientific;
      std::cout << sigma(iCalc, i, i) * elConductivityAuToSi << " ";
    }
    std::cout << "\n";
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

void outputCoeffsToJSON(const std::string &outFileName,
                                StatisticsSweep& statisticsSweep,
                                int& dimensionality,
                                Eigen::Tensor<double, 3>& kappa,
                                Eigen::Tensor<double, 3>& sigma,
                                Eigen::Tensor<double, 3>& mobility,
                                Eigen::Tensor<double, 3>& seebeck)  {

  if (!mpi->mpiHead()) return;

  Kokkos::Profiling::pushRegion("onsagerToJSON");

  int numCalculations = statisticsSweep.getNumCalculations();

  std::string unitsSigma, unitsKappa;
  double convSigma, convKappa;
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

  double convMobility = mobilityAuToSi * pow(100., 2); // from m^2/Vs to cm^2/Vs
  std::string unitsMobility = "cm^2 / V / s";

  double convSeebeck = thermopowerAuToSi * 1.0e6;
  std::string unitsSeebeck = "muV / K";

  std::vector<double> temps, dopings, chemPots;
  std::vector<std::vector<std::vector<double>>> sigmaOut;
  std::vector<std::vector<std::vector<double>>> mobilityOut;
  std::vector<std::vector<std::vector<double>>> kappaOut;
  std::vector<std::vector<std::vector<double>>> seebeckOut;
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

