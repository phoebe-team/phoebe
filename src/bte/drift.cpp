#include "drift.h"

BulkTDrift::BulkTDrift(StatisticsSweep &statisticsSweep_,
                       BaseBandStructure &bandStructure_,
                       const int &dimensionality_,
                       const bool& symmetrize)
    : VectorBTE(statisticsSweep_, bandStructure_, dimensionality_) {

  Particle particle = bandStructure.getParticle();
  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();
#pragma omp parallel for
  for(int iis = 0; iis < niss; iis++){
    int is = iss[iis];
    StateIndex isIdx(is);
    double energy = bandStructure.getEnergy(isIdx);
    Eigen::Vector3d vel = bandStructure.getGroupVelocity(isIdx);
    int iBte = bandStructure.stateToBte(isIdx).get();
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      auto chemPot = calcStat.chemicalPotential;
      auto temp = calcStat.temperature;
      if(particle.isPhonon()) chemPot = 0.; // to be safe in the case of stat sweep in phel 
      for (int i = 0; i < 3; i++) {
        operator()(iCalc, i, iBte) =
            particle.getDndt(energy, temp, chemPot, symmetrize) * vel(i);
      }
    }
  }
  mpi->allReduceSum(&data);
}

BulkEDrift::BulkEDrift(StatisticsSweep &statisticsSweep_,
                       BaseBandStructure &bandStructure_,
                       const int &dimensionality_,
                       const bool& symmetrize)
    : VectorBTE(statisticsSweep_, bandStructure_, dimensionality_) {

  Particle particle = bandStructure.getParticle();
  std::vector<int> iss = bandStructure.parallelIrrStateIterator();
  int niss = iss.size();
#pragma omp parallel for
  for(int iis = 0; iis < niss; iis++){
    int is = iss[iis];
    StateIndex isIdx(is);
    double energy = bandStructure.getEnergy(isIdx);
    Eigen::Vector3d vel = bandStructure.getGroupVelocity(isIdx);
    int iBte = bandStructure.stateToBte(isIdx).get();
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      auto chemPot = calcStat.chemicalPotential;
      auto temp = calcStat.temperature;
      double x = particle.getDnde(energy, temp, chemPot, symmetrize);
      for (int i = 0; i < 3; i++) {
        // note: this is tuned for electrons
        // EDrift = e v dn/de
        operator()(iCalc, i, iBte) = x * vel(i);
      }
    }
  }
  mpi->allReduceSum(&data);
}