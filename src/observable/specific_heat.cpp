#include "specific_heat.h"

#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include "constants.h"
#include "mpiHelper.h"

SpecificHeat::SpecificHeat(StatisticsSweep &statisticsSweep_, Crystal &crystal_,
        BaseBandStructure &bandStructure_) :
        Observable(statisticsSweep_, crystal_), bandStructure(bandStructure_) {
    scalar = Eigen::VectorXd::Zero(numCalcs);
}
;

// copy constructor
SpecificHeat::SpecificHeat(const SpecificHeat &that) :
        Observable(that), bandStructure(that.bandStructure) {
}

// copy assignment
SpecificHeat& SpecificHeat::operator =(const SpecificHeat &that) {
    Observable::operator=(that);
    if (this != &that) {
        bandStructure = that.bandStructure;
    }
    return *this;
}

void SpecificHeat::calc() {

  double norm = 1. / bandStructure.getNumPoints(true)
          / crystal.getVolumeUnitCell(dimensionality);
  scalar.setZero();
  auto particle = bandStructure.getParticle();
  for (long iCalc = 0; iCalc < numCalcs; iCalc++) {
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    double chemPot = calcStat.chemicalPotential;

    double sum = 0.;
    #pragma omp parallel for reduction(+ : sum)
    for (long is = 0; is < bandStructure.getNumStates(); is++) {
      auto en = bandStructure.getEnergy(is);
      auto dndt = particle.getDndt(en, temp, chemPot);

      // exclude acoustic phonons, cutoff at 0.1 cm^-1
      if (en < 0.1 / ryToCmm1 && particle.isPhonon()) {
        continue;
      }
      sum += dndt * en * norm;
    }
    scalar(iCalc) = sum;
  }
}

void SpecificHeat::print() {
  if ( ! mpi->mpiHead()) return;

  std::string units;
  if (dimensionality == 1) {
      units = "J / K / m";
  } else if (dimensionality == 2) {
      units = "J / K / m^2";
  } else {
      units = "J / K / m^3";
  }

  double conversion = kBoltzmannSi / pow(bohrRadiusSi, 3);

  std::cout << "\n";
  std::cout << "Specific heat (" << units << ")\n";

  for (long iCalc = 0; iCalc < numCalcs; iCalc++) {

    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;

    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "Temperature: " << temp * temperatureAuToSi
            << " (K), C = ";
    std::cout << std::scientific;
    std::cout.precision(5);
    std::cout << scalar(iCalc) * conversion;
    std::cout << std::endl;
  }
}

void SpecificHeat::outputToJSON(std::string outFileName) {
  if ( ! mpi->mpiHead()) return;

  std::string units;
  if (dimensionality == 1) {
      units = "J / K / m";
  } else if (dimensionality == 2) {
      units = "J / K / m^2";
  } else {
      units = "J / K / m^3";
  }

  double conversion = kBoltzmannSi / pow(bohrRadiusSi, 3);
  auto particle = bandStructure.getParticle();

  std::vector<double> temps;
  std::vector<double> specificHeat;
  for (long iCalc = 0; iCalc < numCalcs; iCalc++) {

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp*temperatureAuToSi);

    // store the specific heat
    specificHeat.push_back(scalar(iCalc) * conversion);

  }
  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["specificHeat"] = specificHeat;
  output["temperatureUnit"] = "K";
  output["specificHeatUnit"] = units;
  output["particleType"] = particle.isPhonon() ? "phonon" : "electron";
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();

}

int SpecificHeat::whichType() {
    return isScalar;
}

const double& SpecificHeat::get(const ChemPotIndex &imu, const TempIndex &it) {
    auto i = glob2Loc(imu, it);
    return scalar(i);
}

