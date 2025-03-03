#include "specific_heat.h"

#include "constants.h"
#include "mpiHelper.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include "io.h"

SpecificHeat::SpecificHeat(Context &context_, StatisticsSweep &statisticsSweep_,
                           Crystal &crystal_, BaseBandStructure &bandStructure_)
    : Observable(context_, statisticsSweep_, crystal_),
      bandStructure(bandStructure_) {

  scalar = Eigen::VectorXd::Zero(numCalculations);

  if ((context.getHasSpinOrbit() && bandStructure.getParticle().isElectron())
                        || bandStructure.getParticle().isPhonon()) {
    spinFactor = 1.;
  } else {
    // TODO: for spin polarized calculations, we will have to set this
    // to 1.
    spinFactor = 2.;
  }
}

// C = 1/(V*N) * sum(states) * dn/dT * energy
//   = 1/(V*N) * sum(states) * n(n+/-1) * energy^2 / kB T^2
void SpecificHeat::calc() {

  auto particle = bandStructure.getParticle();
  double norm = 1. / crystal.getVolumeUnitCell(dimensionality);

  if (particle.isPhonon())   { norm /= context.getQMesh().prod(); }
  if (particle.isElectron()) { norm /= context.getKMesh().prod(); }
  scalar.setZero();

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    double chemPot = calcStat.chemicalPotential;

    double sum = 0.;
    std::vector<int> iss = bandStructure.irrStateIterator();
    int niss = iss.size();
#pragma omp parallel for reduction(+ : sum) default(none) shared(bandStructure,particle,temp,norm,chemPot,iss,niss,phEnergyCutoff)
    for(int iis = 0; iis < niss; iis++){

      int is = iss[iis];
      StateIndex isIdx(is);
      auto en = bandStructure.getEnergy(isIdx);

      // exclude acoustic phonons, cutoff at 0.1 cm^-1
      if (en < phEnergyCutoff && particle.isPhonon()) {
        continue;
      }

      auto dndt = particle.getDndt(en, temp, chemPot);
      auto rots = bandStructure.getRotationsStar(isIdx);
      // here the number of symmetry rotations is used to
      // weight components to the sum
      sum += abs(dndt) * abs(en-chemPot) * norm * rots.size();
    }
    // as T is actually KBT, we need to add this extra factor here
    // as a result
    scalar(iCalc) = sum * kBoltzmannRy * spinFactor;
  }
}

void SpecificHeat::print() {

  if (!mpi->mpiHead()) return;

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
  if (bandStructure.getParticle().isPhonon()) {
    std::cout << "Phonon specific heat (" << units << ")\n";
  } else if (bandStructure.getParticle().isElectron()) {
    std::cout << "Electron specific heat (" << units << ")\n";
  }

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;

    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "Temperature: " << temp * temperatureAuToSi << " (K), C = ";
    std::cout << std::scientific;
    std::cout.precision(5);
    std::cout << scalar(iCalc) * conversion;
    std::cout << std::endl;
  }
}

void SpecificHeat::outputToJSON(const std::string &outFileName) {
  if (!mpi->mpiHead())
    return;

  std::string units;
  if (dimensionality == 1) {
    units = "J / K / m";
  } else if (dimensionality == 2) {
    units = "J / K / m^2";
  } else {
    units = "J / K / m^3";
  }

  // this has units of energy over volume
  // TODO this should be double checked for 2D material
  double conversion = rydbergSi / pow(bohrRadiusSi, 3);
  auto particle = bandStructure.getParticle();

  std::vector<double> temps;
  std::vector<double> specificHeat;
  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp * temperatureAuToSi);

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

int SpecificHeat::whichType() { return isScalar; }

const double &SpecificHeat::get(const ChemPotIndex &imu, const TempIndex &it) {
  auto i = glob2Loc(imu, it);
  return scalar(i);
}

const double &SpecificHeat::get(const int &iCalc) { return scalar(iCalc); }
