#include "wigner_phonon_thermal_cond.h"
#include <iomanip>
#include "constants.h"

WignerPhononThermalConductivity::WignerPhononThermalConductivity(
    Context &context_, StatisticsSweep &statisticsSweep_, Crystal &crystal_,
    BaseBandStructure &bandStructure_, VectorBTE &relaxationTimes)
    : PhononThermalConductivity(context_, statisticsSweep_, crystal_, bandStructure_),
      smaRelTimes(relaxationTimes) {

  wignerCorrection = Eigen::Tensor<double, 3>(numCalculations, dimensionality, dimensionality);
  wignerCorrection.setZero();

  auto particle = bandStructure.getParticle();
  int dimensionality = crystal.getDimensionality();

    // set up units for writing to file
  thCondUnits = "W /(m K)";
  if (dimensionality == 3) {
    thCondConversion = thConductivityAuToSi; 
  } else if (dimensionality == 2) {
    // multiply by the height of the cell / thickness of the cell to convert 3D -> 2D.  
    // Because the unit cell volume is already reduced for dimensionality, 
    // we only need to divide by thickness. 
    //double height = crystal.getDirectUnitCell()(2,2);
    thCondConversion = thConductivityAuToSi * (1. / context.getThickness()); 
  } else {
    Warning("1D conductivity should be manually adjusted for the cross-section of the material.");
    thCondConversion = thConductivityAuToSi; 
  }

  Eigen::VectorXd norm(numCalculations);
  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temperature = calcStat.temperature;
    norm(iCalc) = 1. / context.getQMesh().prod() /
                  crystal.getVolumeUnitCell(dimensionality) / temperature /
                  temperature / 2.;
  }

  for (int iq : bandStructure.parallelIrrPointsIterator()) {
    WavevectorIndex iqIdx(iq);

    auto velocities = bandStructure.getVelocities(iqIdx);
    auto energies = bandStructure.getEnergies(iqIdx);
    auto numBands = int(energies.size());

    // calculate bose factors
    Eigen::MatrixXd bose(numCalculations, numBands);
    for (int ib1 = 0; ib1 < numBands; ib1++) {
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
        auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
        double temperature = calcStat.temperature;
        bose(iCalc, ib1) = particle.getPopulation(energies(ib1), temperature);
        // exclude acoustic phonons, cutoff at 0.1 cm^-1
        // setting this to zero here causes acoustic ph
        // contribution below to evaluate to zero
        if (energies(ib1) < 0.1 / ryToCmm1) {
          bose(iCalc, ib1) = 0.0;
        }
      }
    }

    BandIndex ibIdx(0);
    int is = bandStructure.getIndex(iqIdx, ibIdx);
    StateIndex isIdx(is);
    auto rots = bandStructure.getRotationsStar(isIdx);

    // apply rotation to velocity
    for (const Eigen::Matrix3d &rot : rots) {

      Eigen::Tensor<std::complex<double>, 3> velRot = velocities.constant(0.);
      for (int ib1 = 0; ib1 < numBands; ib1++) {
        for (int ib2 = 0; ib2 < numBands; ib2++) {
          for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
              velRot(ib1, ib2, i) += rot(i, j) * velocities(ib1, ib2, j);
            }
          }
        }
      }

      // calculate wigner correction
      for (int ib1 = 0; ib1 < numBands; ib1++) {
        for (int ib2 = 0; ib2 < numBands; ib2++) {
          int is1 = bandStructure.getIndex(iqIdx, BandIndex(ib1));
          int is2 = bandStructure.getIndex(iqIdx, BandIndex(ib2));
          auto is1Idx = StateIndex(is1);
          auto is2Idx = StateIndex(is2);
          int iBte1 = bandStructure.stateToBte(is1Idx).get();
          int iBte2 = bandStructure.stateToBte(is2Idx).get();

          for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
            for (int ic1 = 0; ic1 < dimensionality; ic1++) {
              for (int ic2 = 0; ic2 < dimensionality; ic2++) {

                double num =
                    energies(ib1) * bose(iCalc, ib1) * (bose(iCalc, ib1) + 1.) +
                    energies(ib2) * bose(iCalc, ib2) * (bose(iCalc, ib2) + 1.);

                if ( (bose(iCalc, ib1) <= 0.) || (bose(iCalc, ib2) <=0.0) ) continue; //this avoids numerical problems for the acoustic phonons

                double vel =
                    (velRot(ib1, ib2, ic1) * velRot(ib2, ib1, ic2)).real();
                double den = 4. * pow(energies(ib1) - energies(ib2), 2) +
                             pow(1. / smaRelTimes(iCalc, 0, iBte1) +
                                     1. / smaRelTimes(iCalc, 0, iBte2),
                                 2);

                wignerCorrection(iCalc, ic1, ic2) +=
                    (energies(ib1) + energies(ib2)) * vel * num / den *
                    (1. / smaRelTimes(iCalc, 0, iBte1) +
                     1. / smaRelTimes(iCalc, 0, iBte2)) *
                    norm(iCalc);
              }
            }
          }
        }
      }
    }
  }
  mpi->allReduceSum(&wignerCorrection);
}

// copy constructor
WignerPhononThermalConductivity::WignerPhononThermalConductivity(
    const WignerPhononThermalConductivity &that)
    : PhononThermalConductivity(that), smaRelTimes(that.smaRelTimes),
      wignerCorrection(that.wignerCorrection) {}

// copy assignment
WignerPhononThermalConductivity &WignerPhononThermalConductivity::operator=(
    const WignerPhononThermalConductivity &that) {
  PhononThermalConductivity::operator=(that);
  if (this != &that) {
    smaRelTimes = that.smaRelTimes;
    wignerCorrection = that.wignerCorrection;
  }
  return *this;
}

void WignerPhononThermalConductivity::calcFromPopulation(VectorBTE &n) {
  PhononThermalConductivity::calcFromPopulation(n);
  // TO DO: here we use setZero because we include the diagonal elements when computing the Wigner conductivity
  // in other words, the Wigner conductivity includes both populations and coherences contributions
  tensordxd.setZero();
  tensordxd += wignerCorrection;
}

void WignerPhononThermalConductivity::calcVariational(VectorBTE &af,
                                                      VectorBTE &f,
                                                      VectorBTE &scalingCG) {
  PhononThermalConductivity::calcVariational(af, f, scalingCG);
  Error("Developer error: Wigner conductivity is currently limited to RTA approximation, implementation with iterative will be done in the future");
  tensordxd += wignerCorrection;
}

void WignerPhononThermalConductivity::calcFromRelaxons(
    Context &context, StatisticsSweep &statisticsSweep,
    ParallelMatrix<double> &eigenvectors,
    PhScatteringMatrix &scatteringMatrix, const Eigen::VectorXd &eigenvalues) {
  PhononThermalConductivity::calcFromRelaxons(context, statisticsSweep,
                                              eigenvectors,
                                              scatteringMatrix, eigenvalues);
  Error("Developer error: Wigner conductivity is currently limited to RTA approximation, implementation with iterative will be done in the future");         
  tensordxd += wignerCorrection;
}

void WignerPhononThermalConductivity::print() {

  if (!mpi->mpiHead()) return; 

  std::cout << "\n";
  std::cout << "Wigner Thermal Conductivity (" << thCondUnits << ")\n";

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;

    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "Temperature: " << temp * temperatureAuToSi << " (K)\n";
    std::cout.precision(5);
    for (int i = 0; i < dimensionality; i++) {
      std::cout << "  " << std::scientific;
      for (int j = 0; j < dimensionality; j++) {
        std::cout << " " << std::setw(13) << std::right;
        std::cout << tensordxd(iCalc, i, j) * thCondConversion;
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
}
