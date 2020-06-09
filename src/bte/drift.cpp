#include <math.h>
#include "drift.h"

BulkTDrift::BulkTDrift(StatisticsSweep & statisticsSweep_,
		FullBandStructure<FullPoints> & bandStructure_,
		const long & dimensionality_) :
		VectorBTE(statisticsSweep_, bandStructure_, dimensionality_) {

	Statistics statistics = bandStructure.getStatistics();
	for ( long is=0; is<numStates; is++ ) {
		double energy = bandStructure.getEnergy(is);
		Eigen::Vector3d velocity = bandStructure.getGroupVelocity(is);

		for ( long iCalc=0; iCalc<numCalcs; iCalc++ ) {
			auto [imu,it,idim] = loc2Glob(iCalc);
			double vel = velocity(idim);
			auto calcStat = statisticsSweep.getCalcStatistics(it,imu);
			auto chemicalPotential = calcStat.chemicalPotential;
			auto temperature = calcStat.temperature;

			double x = statistics.getDndt(energy, temperature,
					chemicalPotential) * vel;
			data(iCalc,is) = x;
		}
	}
}

BulkEDrift::BulkEDrift(StatisticsSweep & statisticsSweep_,
		FullBandStructure<FullPoints> & bandStructure_,
		const long & dimensionality_) :
				VectorBTE(statisticsSweep_, bandStructure_, dimensionality_) {

	Statistics statistics = bandStructure.getStatistics();
	for ( long is=0; is<numStates; is++ ) {
		double energy = bandStructure.getEnergy(is);
		Eigen::Vector3d velocity = bandStructure.getGroupVelocity(is);

		for ( long iCalc=0; iCalc<numCalcs; iCalc++ ) {
			auto [imu,it,idim] = loc2Glob(iCalc);
			double vel = velocity(idim);
			auto calcStat = statisticsSweep.getCalcStatistics(it,imu);
			auto chemicalPotential = calcStat.chemicalPotential;
			auto temperature = calcStat.temperature;

			double x = statistics.getDnde(energy, temperature,
					chemicalPotential) * vel;
			data(iCalc,is) = x;
		}
	}
}

Vector0::Vector0(StatisticsSweep & statisticsSweep_,
		FullBandStructure<FullPoints> & bandStructure_,
		SpecificHeat & specificHeat) :
		VectorBTE(statisticsSweep_, bandStructure_, 1) {

	data.setZero();

	Statistics statistics = bandStructure.getStatistics();

	for ( long iCalc=0; iCalc<numCalcs; iCalc++ ) {
		auto [imu,it,idim] = loc2Glob(iCalc);
		auto calcStat = statisticsSweep.getCalcStatistics(it,imu);
		double temp = calcStat.temperature;
		double chemPot = calcStat.chemicalPotential;

		for ( long is=0; is<numStates; is++ ) {
			double energy = bandStructure.getEnergy(is);
			double dnde = statistics.getDnde(energy,temp,chemPot);
			// note dnde = n(n+1)/T  (for bosons)
			auto c = specificHeat.get(imu,it);
			double x = - dnde / temp / c;
			data(iCalc,is) += std::sqrt(x) * energy;
			// we use std::sqrt because we overwrote sqrt() in the base class
		}
	}
}

