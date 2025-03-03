#include "wigner_electron.h"
#include "constants.h"
#include <iomanip>
#include "onsager_utilities.h"

WignerElCoefficients::WignerElCoefficients(StatisticsSweep &statisticsSweep_,
                                           Crystal &crystal_,
                                           BaseBandStructure &bandStructure_,
                                           Context &context_,
                                           VectorBTE &relaxationTimes)
    : OnsagerCoefficients(statisticsSweep_, crystal_, bandStructure_, context_),
      smaRelTimes(relaxationTimes) {

  // TODO is this the wrong size if we use symmetry?
  contributionLEE.resize(numCalculations, bandStructure_.getNumStates(), dimensionality, dimensionality);
  contributionLET.resize(numCalculations, bandStructure_.getNumStates(), dimensionality, dimensionality);
  //contributionLTT.resize(numCalculations, bandStructure_.getNumStates(), dimensionality, dimensionality);

  correctionLEE.resize(numCalculations, dimensionality, dimensionality);
  correctionLTE.resize(numCalculations, dimensionality, dimensionality);
  correctionLET.resize(numCalculations, dimensionality, dimensionality);
  correctionLTT.resize(numCalculations, dimensionality, dimensionality);

  contributionLEE.setZero();
  contributionLET.setZero();
  //contributionLTT.setZero();

  correctionLEE.setZero();
  correctionLTE.setZero();
  correctionLET.setZero();
  correctionLTT.setZero();

  auto particle = bandStructure.getParticle();

  double norm = spinFactor / context.getKMesh().prod() /
                crystal.getVolumeUnitCell(dimensionality) / 2.;

  Eigen::Tensor<std::complex<double>, 4> fE, fT;

  for (int ik : mpi->divideWorkIter(bandStructure.getNumPoints())) {

    WavevectorIndex ikIdx(ik);
    Eigen::Tensor<std::complex<double>, 3> velocities = bandStructure.getVelocities(ikIdx);
    auto energies = bandStructure.getEnergies(ikIdx);
    int numBands = energies.size();

    Eigen::Vector3d k = bandStructure.getWavevector(ikIdx);
    auto t = bandStructure.getRotationToIrreducible(k, Points::cartesianCoordinates);
    int ikIrr = std::get<0>(t);

    // we do the calculation in two steps

    //-----------------------------------------------------------------------
    // Step 1: compute the off-diagonal population term (at given wavevector)

    fE.resize(numBands, numBands, dimensionality, numCalculations);
    fT.resize(numBands, numBands, dimensionality, numCalculations);
    fE.setZero();
    fT.setZero();

    // TODO could probably OMP parallel this or the below band loops?
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      double chemPot = calcStat.chemicalPotential;
      double temp = calcStat.temperature;
      Eigen::VectorXd fermi(numBands);
      Eigen::VectorXd dfdt(numBands);
      for (int ib1 = 0; ib1 < numBands; ib1++) {
        fermi(ib1) = particle.getPopulation(energies(ib1), temp, chemPot);
        dfdt(ib1) = particle.getDndt(energies(ib1), temp, chemPot);
      }

      for (int ib1 = 0; ib1 < numBands; ib1++) {
        for (int ib2 = 0; ib2 < numBands; ib2++) {
          // discard diagonal contributions, which are included in the
          // standard BTE already
          if (ib1 == ib2) {
            continue;
          }
          int is1 = bandStructure.getIndex(WavevectorIndex(ikIrr), BandIndex(ib1));
          int is2 = bandStructure.getIndex(WavevectorIndex(ikIrr), BandIndex(ib2));
          StateIndex is1Idx(is1);
          StateIndex is2Idx(is2);
          int iBte1 = bandStructure.stateToBte(is1Idx).get();
          int iBte2 = bandStructure.stateToBte(is2Idx).get();

          if( abs(energies(ib1) - energies(ib2)) <  0.0001 / energyRyToEv) {
            velocities(ib1, ib2, 0) = 0; 
            velocities(ib1, ib2, 1) = 0; 
            velocities(ib1, ib2, 2) = 0; 
            velocities(ib2, ib1, 0) = 0; 
            velocities(ib2, ib1, 1) = 0; 
            velocities(ib2, ib1, 2) = 0; 
            //if( abs(velocities(ib1, ib2, 0) + velocities(ib1, ib2, 1) + velocities(ib1, ib2, 2)) > 1e-15) 
            //  std::cout << velocities(ib1, ib2, 0) << " " << velocities(ib1, ib2, 1) << " " << velocities(ib1, ib2, 2) << std::endl;
            //if( abs(velocities(ib1, ib2, 0) + velocities(ib1, ib2, 1) + velocities(ib1, ib2, 2)) > 1e-15) 
            //  std::cout << velocities(ib2, ib1, 0) << " " << velocities(ib2, ib1, 1) << " " << velocities(ib2, ib1, 2) << std::endl;
            //continue; 
          }

          std::complex<double> xC = {1. / smaRelTimes(iCalc, 0, iBte1) +
                                         1. / smaRelTimes(iCalc, 0, iBte2),
                                     2. * (energies(ib1) - energies(ib2))};

          for (int ic1 = 0; ic1 < dimensionality; ic1++) {
            fE(ib1, ib2, ic1, iCalc) = -2. * velocities(ib1, ib2, ic1) / xC *
                                       (fermi(ib1) - fermi(ib2)) /
                                       (energies(ib1) - energies(ib2));
            fT(ib1, ib2, ic1, iCalc) =
                2. * velocities(ib1, ib2, ic1) / xC * (dfdt(ib1) + dfdt(ib2));
          }
        }
      }
    }

    //---------------------------------------------------------------------
    // Step 2: now compute the anti-commutator for the transport coefficient

    for (int ib1 = 0; ib1 < numBands; ib1++) {
      for (int ib2 = 0; ib2 < numBands; ib2++) {

        if (ib1 == ib2)  continue; // diagonal terms are counted in the standard BTE
        int is1 = bandStructure.getIndex(WavevectorIndex(ikIrr), BandIndex(ib1));

        for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

          double chemicalPotential =
              statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;

          for (int ic1 = 0; ic1 < dimensionality; ic1++) {
            for (int ic2 = 0; ic2 < dimensionality; ic2++) {

              double xE = std::real(
                  velocities(ib1, ib2, ic1) * fE(ib2, ib1, ic2, iCalc) +
                  velocities(ib2, ib1, ic1) * fE(ib1, ib2, ic2, iCalc));
              double xT = std::real(
                  velocities(ib1, ib2, ic1) * fT(ib2, ib1, ic2, iCalc) +
                  velocities(ib2, ib1, ic1) * fT(ib1, ib2, ic2, iCalc));

              // store the contributions to Wigner transport
              // NOTE: this could be a bit memory heavy if num states is very large?
              contributionLEE(iCalc, is1, ic1, ic2) += norm * xE;
              contributionLET(iCalc, is1, ic1, ic2) -= norm * xT;
              //contributionLTT(iCalc, is1, ic1, ic2) -= norm * (energies(ib1) - chemicalPotential) * xT;

              correctionLEE(iCalc, ic1, ic2) += norm * xE;
              correctionLET(iCalc, ic1, ic2) -= norm * xT;
              correctionLTE(iCalc, ic1, ic2) -=
                  norm * (energies(ib1) - chemicalPotential) * xE;
              correctionLTT(iCalc, ic1, ic2) -=
                  norm * (energies(ib1) - chemicalPotential) * xT;
            }
          }
        }
      }
    }
  }
  // uncomment these to get the wigner contributions 
  if(numCalculations == 1) { 
    mpi->allReduceSum(&contributionLEE);
    mpi->allReduceSum(&contributionLET); // this causes an issue if ncalcs > 1
  } 

  //mpi->allReduceSum(&contributionLTT);

  mpi->allReduceSum(&correctionLEE);
  mpi->allReduceSum(&correctionLTE);
  mpi->allReduceSum(&correctionLET);
  mpi->allReduceSum(&correctionLTT);
}

void WignerElCoefficients::calcFromPopulation(VectorBTE &nE, VectorBTE &nT) {

  OnsagerCoefficients::calcFromPopulation(nE, nT);
  LEE += correctionLEE;
  LTE += correctionLTE;
  LET += correctionLET;
  LTT += correctionLTT;

  // calcTransportCoefficients is called twice, also in base calcFromPopulation.
  // Could this be improved?
  onsagerToTransportCoeffs(statisticsSweep, dimensionality,
                        LEE, LTE, LET, LTT, kappa, sigma, mobility, seebeck);
}

void WignerElCoefficients::print() {
  if (!mpi->mpiHead())
    return;
  std::cout << "Estimates with the Wigner transport equation.\n";
  OnsagerCoefficients::print();
}


void WignerElCoefficients::outputContributionsToJSON(const std::string &outFileName) {

  // TODO for now we are not writing kappa to output

  if (!mpi->mpiHead()) return;
  if(numCalculations > 1)
    Error("Cannot output Wigner contributions when numCalculations>1!");
  //if(context.getUseSymmetries()) {
  //  Error("Output Wigner contributions is not checked with symmetries on.");
 // }

  std::string unitsSigma; //, unitsKappa;
  double convSigma; //, convKappa;
  if (dimensionality == 1) {
    unitsSigma = "S m";
    //unitsKappa = "W m / K";
    convSigma = elConductivityAuToSi * rydbergSi * rydbergSi;
    //convKappa = thConductivityAuToSi * rydbergSi * rydbergSi;
  } else if (dimensionality == 2) {
    unitsSigma = "S";
    //unitsKappa = "W / K";
    convSigma = elConductivityAuToSi * rydbergSi;
    //convKappa = thConductivityAuToSi * rydbergSi;
  } else {
    unitsSigma = "S / m";
    //unitsKappa = "W / m / K";
    convSigma = elConductivityAuToSi;
    //convKappa = thConductivityAuToSi;
  }

  double convSeebeck = thermopowerAuToSi * 1.0e6;
  std::string unitsSeebeck = "muV / K";

  std::vector<double> temps, dopings, chemPots;
  std::vector<std::vector<std::vector<std::vector<double>>>> sigmaOut;
  //std::vector<std::vector<std::vector<double>>> kappaOut;
  std::vector<std::vector<std::vector<std::vector<double>>>> seebeckOut;
  //for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

  int iCalc = 0;

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp * temperatureAuToSi);
    double doping = calcStat.doping;
    dopings.push_back(doping); // output in (cm^-3)
    double chemPot = calcStat.chemicalPotential;
    chemPots.push_back(chemPot * energyRyToEv); // output in eV

    //for (int ibte = 0; ibte < contributionLEE.dimension(1); ibte++) {
    for (int ik : bandStructure.irrPointsIterator()) {
      auto ikIndex = WavevectorIndex(ik);

      std::vector<std::vector<std::vector<double>>> sigmaPoint;
      std::vector<std::vector<std::vector<double>>> seebeckPoint;

      // loop over bands here
      // get numBands at this point, in case it's an active band structure
      for (int ib = 0; ib < bandStructure.getNumBands(ikIndex); ib++) {

        auto ibIndex = BandIndex(ib);
        int is = bandStructure.getIndex(ikIndex, ibIndex);
        StateIndex isIdx(is);

        std::vector<std::vector<double>> sigmaBand;
        std::vector<std::vector<double>> seebeckBand;
        Eigen::Matrix3d invLEE;
        Eigen::Matrix3d matLET;
        for (int i = 0; i < dimensionality; i++) {
          for (int j = 0; j < dimensionality; j++) {
            invLEE(i,j) = contributionLEE(iCalc, is, i, j);
            matLET(i,j) = contributionLET(iCalc, is, i, j);
          }
        }
        Eigen::Matrix3d Scontrib = invLEE.inverse() * matLET;

        for (int i = 0; i < dimensionality; i++) {
          std::vector<double> sigmaDim;
          std::vector<double> seebeckDim;
          for (int j = 0; j < dimensionality; j++) {
            sigmaDim.push_back(contributionLEE(iCalc, is, i, j) * convSigma);
            // seebeck = - matmul(L_EE_inv, L_ET)
            seebeckDim.push_back(Scontrib(i,j) * convSeebeck);
          }
          sigmaBand.push_back(sigmaDim);
          seebeckBand.push_back(sigmaDim);
        }
        sigmaPoint.push_back(sigmaBand);
        seebeckPoint.push_back(sigmaBand);
      }
      sigmaOut.push_back(sigmaPoint);
      seebeckOut.push_back(seebeckPoint);
    }
  //}

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
  //output["mobility"] = mobilityOut;
  //output["mobilityUnit"] = unitsMobility;
  //output["electronicThermalConductivity"] = kappaOut;
  //output["electronicThermalConductivityUnit"] = unitsKappa;
  output["seebeckCoefficient"] = seebeckOut;
  output["seebeckCoefficientUnit"] = unitsSeebeck;
  output["particleType"] = "electron";
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}
