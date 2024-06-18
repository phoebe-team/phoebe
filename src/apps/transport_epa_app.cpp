#include "transport_epa_app.h"
#include "bandstructure.h"
#include "constants.h"
#include "context.h"
#include "delta_function.h"
#include "eigen.h"
#include "exceptions.h"
#include "interaction_epa.h"
#include "io.h"
#include "onsager.h"
#include "particle.h"
#include "qe_input_parser.h"
#include "statistics_sweep.h"
#include "utilities.h"
#include <nlohmann/json.hpp>
#include <vector>
#include "dos_app.h"

void TransportEpaApp::run(Context &context) {

  // parse QE-xml file
  auto t1 = QEParser::parseElHarmonicFourier(context);
  auto crystal = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);

  //--------------------------------
  // Setup energy grid
  double minEnergy = context.getFermiLevel() - context.getEpaEnergyRange();
  double maxEnergy = context.getFermiLevel() + context.getEpaEnergyRange();
  double energyStep = context.getEpaEnergyStep();

  // in principle, we should add 1 to account for ends of energy interval
  // I will not do that, because will work with the centers of energy steps
  size_t numEnergies = size_t((maxEnergy - minEnergy) / energyStep);
  // energies at the centers of energy steps
  Eigen::VectorXd energies(numEnergies);
  for (size_t i = 0; i < numEnergies; ++i) {
    // add 0.5 to be in the middle of the energy step
    energies(i) = (double(i) + 0.5) * energyStep + minEnergy;
  }

  // Read and setup k-point mesh for interpolating band structure
  Points fullPoints(crystal, context.getKMesh());
  fullPoints.setIrreduciblePoints();

  // Building electronic band structure -----------------------

  // filter to only the bands relevant to transport
  electronH0.trimBands(context, minEnergy, maxEnergy);

  // Fourier's interpolation of the electronic band structure
  bool withVelocities = true;
  bool withEigenvectors = false;
  FullBandStructure bandStructure =
      electronH0.populate(fullPoints, withVelocities, withEigenvectors);

  // set temperatures, chemical potentials and carrier concentrations
  StatisticsSweep statisticsSweep(context, &bandStructure);

  if (mpi->mpiHead()) {
    std::cout << "Starting EPA with " << numEnergies << " energies and "
              << bandStructure.getNumStates() << " states." << std::endl;
    std::cout << std::setprecision(4) << "Energy grid ranges from " << minEnergy*energyRyToEv << " to " 
              << maxEnergy*energyRyToEv 
              << " eV with " << numEnergies << " bins." << std::endl;
  }

  //--------------------------------
  // calc EPA velocities
  auto t2 = calcEnergyProjVelocity(context, bandStructure, energies);
  Eigen::Tensor<double, 3> energyProjVelocity = std::get<0>(t2);
  Eigen::VectorXd dos = std::get<1>(t2);
  {
    // output to file (convert to std::vector for the sake of reusing dos output function from dos_app)
    // braces to make this go out of scope after writing 
    std::vector<double> dosVec(dos.data(), dos.data() + dos.rows() * dos.cols());
    std::vector<double> energyVec(energies.data(), energies.data() + energies.rows() * energies.cols());
    Particle particle = bandStructure.getParticle();
    outputDOSToJSON(energyVec, dosVec, particle, "epa_dos.json"); 
  } 
  //--------------------------------
  // Calculate EPA scattering rates
  VectorEPA scatteringRates =
      getScatteringRates(context, statisticsSweep, energies, crystal, dos);

  outputToJSON("epa_relaxation_times.json", scatteringRates, statisticsSweep,
               energies, context);

  //--------------------------------
  // compute transport coefficients
  if (mpi->mpiHead()) {
    std::cout << "\nComputing transport coefficients" << std::endl;
  }

  OnsagerCoefficients transCoefficients(statisticsSweep, crystal, bandStructure,
                                        context);
  transCoefficients.calcFromEPA(scatteringRates, energyProjVelocity, energies);
  transCoefficients.print();
  transCoefficients.outputToJSON("epa_onsager_coefficients.json");
}

void TransportEpaApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getEpaFileName(), "epaFileName");
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getKMesh(), "kMesh");

  throwErrorIfUnset(context.getElectronFourierCutoff(),
                    "electronFourierCutoff");
  throwErrorIfUnset(context.getEpaEnergyStep(), "epaElectronStep");
  throwErrorIfUnset(context.getEpaEnergyRange(), "epaElectronRange");
  throwErrorIfUnset(context.getTemperatures(), "temperatures");
  if (context.getDopings().size() == 0 &&
      context.getChemicalPotentials().size() == 0) {
    Error("Either chemical potentials or doping must be set");
  }
}

std::tuple<Eigen::Tensor<double, 3>, Eigen::VectorXd>
TransportEpaApp::calcEnergyProjVelocity(Context &context,
                                        FullBandStructure &bandStructure,
                                        const Eigen::VectorXd &energies) {

  //--------------------------------
  // set up tetrahedron method
  TetrahedronDeltaFunction tetrahedrons(bandStructure);

  int numEnergies = int(energies.size());
  int numPoints = std::get<0>(bandStructure.getPoints().getMesh()).prod();

  if (mpi->mpiHead()) {
    std::cout << "\nCalculating energy projected velocity tensor" << std::endl;
  }

  auto crystal = bandStructure.getPoints().getCrystal();
  int dim = context.getDimensionality();
  double norm = pow(twoPi, dim) / crystal.getVolumeUnitCell(dim) / numPoints;
  double normDos = 1. / crystal.getVolumeUnitCell(dim) / numPoints;

  Eigen::Tensor<double, 3> energyProjVelocity(3, 3, numEnergies);
  energyProjVelocity.setZero();
  Eigen::VectorXd dos(numEnergies);
  dos.setZero();

  LoopPrint loopPrint("calculating energy projected velocity", "states",
      numEnergies);
  std::vector<size_t> iEnergies = mpi->divideWorkIter(numEnergies);
  size_t niEnergies = iEnergies.size();
#pragma omp parallel for
  for (size_t iiEnergy = 0; iiEnergy < niEnergies; iiEnergy++) {
    int iEnergy = iEnergies[iiEnergy];
#pragma omp critical
    { loopPrint.update(); }
    for (int iState : bandStructure.irrStateIterator()) {
      StateIndex isIdx(iState);
      auto rotations = bandStructure.getRotationsStar(isIdx);
      Eigen::Vector3d velIrr = bandStructure.getGroupVelocity(isIdx);
      double deltaFunction = tetrahedrons.getSmearing(energies(iEnergy), isIdx);

      // integrate DOS
      auto degeneracy = double(rotations.size());
      dos(iEnergy) += deltaFunction * degeneracy * normDos;

      for (const Eigen::Matrix3d &r : rotations) {
        Eigen::Vector3d velocity = r * velIrr;
        for (int j  = 0; j < 3; j++) {
          for (int i = 0; i < 3; i++) {
            energyProjVelocity(i, j, iEnergy) +=
              velocity(i) * velocity(j) * deltaFunction * norm;
          }
        }
      }
    }
  }
  mpi->allReduceSum(&energyProjVelocity);
  mpi->allReduceSum(&dos);
  loopPrint.close();
  return std::make_tuple(energyProjVelocity, dos);
}

VectorEPA TransportEpaApp::getScatteringRates(
    Context &context, StatisticsSweep &statisticsSweep,
    const Eigen::VectorXd &energies, Crystal &crystal,
    const Eigen::VectorXd &dos) {

  int numEnergies = int(energies.size());

  /*If constant relaxation time is specified in input, we don't need to
  calculate EPA lifetimes*/
  double constantRelaxationTime = context.getConstantRelaxationTime();
  if (constantRelaxationTime > 0.) {
    VectorEPA crtRate(statisticsSweep, numEnergies, 1);
    crtRate.setConst(twoPi / constantRelaxationTime);
    return crtRate;
  }

  double spinFactor = 2.;
  if (context.getHasSpinOrbit()) {
    spinFactor = 1;
  }

  auto particle = Particle(Particle::electron);

  int numCalculations = statisticsSweep.getNumCalculations();
  double energyStep = context.getEpaEnergyStep();

  // get vector containing averaged phonon frequencies per mode
  InteractionEpa couplingEpa = InteractionEpa::parseEpaCoupling(context);

  Eigen::VectorXd phEnergies = couplingEpa.getPhEnergies();
  int numModes = int(phEnergies.size());

  // precompute bose-einstein populations
  Eigen::MatrixXd bose(numModes, numCalculations);
  {
    auto phParticle = Particle(Particle::phonon);
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      double temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
      for (int iPhFreq = 0; iPhFreq < numModes; iPhFreq++) {
        bose(iPhFreq, iCalc) =
            phParticle.getPopulation(phEnergies(iPhFreq), temp);
      }
    }
  }

  LoopPrint loopPrint("calculation of EPA scattering rates", "energy bins per MPI process",
                      int(mpi->divideWorkIter(numEnergies).size()));

  VectorEPA epaRate(statisticsSweep, numEnergies, 1);

  double norm = twoPi / spinFactor *
                crystal.getVolumeUnitCell(crystal.getDimensionality());

  std::vector<size_t> iEnergies = mpi->divideWorkIter(numEnergies);
  size_t niEnergies = iEnergies.size();

#pragma omp parallel for
  for (size_t iiEnergy = 0; iiEnergy < niEnergies; iiEnergy++) {
    int iEnergy = iEnergies[iiEnergy];

#pragma omp critical
    { loopPrint.update(); }

    // loop over phonon frequencies
    for (int iPhFreq = 0; iPhFreq < numModes; iPhFreq++) {

      // note: phEnergies(iPhFreq)/energyStep =
      // # of bin-jumps the electron does after scattering

      // compute the dos for electron in the final state for the two
      // scatterings mechanisms, and do a linear interpolation
      int iJump = int(phEnergies(iPhFreq) / energyStep);
      double iInterp = phEnergies(iPhFreq) / energyStep - double(iJump);
      auto largeIndex = int(iEnergy + iJump + 1);
      auto smallIndex = int(iEnergy - iJump - 1);
      double dosEmission = 0.;
      double dosAbsorption = 0.;
      // Avoid some index out of bound errors
      if (smallIndex >= 0) {
        dosEmission =
            dos(smallIndex) * iInterp + dos(iEnergy - iJump) * (1. - iInterp);
      }
      if (largeIndex < dos.size()) {
        dosAbsorption =
            dos(iEnergy + iJump) * (1. - iInterp) + dos(largeIndex) * iInterp;
      }

      //------------------------------------
      // estimate strength of el-ph coupling |g|^2
      double en = energies(iEnergy);
      double enP = energies(iEnergy) + phEnergies(iPhFreq);
      double enM = energies(iEnergy) - phEnergies(iPhFreq);
      double gAbsorption = couplingEpa.getCoupling(iPhFreq, en, enP);
      double gEmission = couplingEpa.getCoupling(iPhFreq, en, enM);

      // population of phonons, electron after emission/absorption
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
        auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
        double temp = calcStat.temperature;
        double chemPot = calcStat.chemicalPotential;

        double nFermiAbsorption = particle.getPopulation(
            energies[iEnergy] + phEnergies(iPhFreq), temp, chemPot);
        double nFermiEmission = particle.getPopulation(
            energies[iEnergy] - phEnergies(iPhFreq), temp, chemPot);

        //-----------------------------
        // now the scattering rate
        // Note that g (the coupling) is already squared

        double rate = gAbsorption * (bose(iPhFreq, iCalc) + nFermiAbsorption) *
                          dosAbsorption +
                      gEmission * (bose(iPhFreq, iCalc) + 1 - nFermiEmission) *
                          dosEmission;

        epaRate.data(iCalc, iEnergy) += norm * rate;
      }
    }
  }
  mpi->allReduceSum(&epaRate.data);
  loopPrint.close();

  return epaRate;
}

/* helper function to output scattering rates at each energy to JSON */
void TransportEpaApp::outputToJSON(const std::string &outFileName,
                                   VectorEPA &scatteringRates,
                                   StatisticsSweep &statisticsSweep,
                                   Eigen::VectorXd &energiesEPA, Context &context) {

  if (!mpi->mpiHead()) {
    return;
  }

  std::string particleType = "electron";
  double energyConversion = energyRyToEv;
  std::string energyUnit = "eV";
  double energyToTime = energyRyToFs;

  int dimensionality = context.getDimensionality();

  // need to store as a vector format with dimensions
  // iCalc, ik. ib, iDim (where iState is unfolded into
  // ik, ib) for the velocities and lifetimes, no dim for energies
  std::vector<std::vector<double>> outTimes;
  std::vector<std::vector<double>> outLinewidths;
  std::vector<std::vector<double>> energies;
  std::vector<double> temps;
  std::vector<double> chemPots;
  std::vector<double> dopings;

  auto numEnergies = int(energiesEPA.size());

  for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
    auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStatistics.temperature;
    double chemPot = calcStatistics.chemicalPotential;
    double doping = calcStatistics.doping;
    temps.push_back(temp * temperatureAuToSi);
    chemPots.push_back(chemPot * energyConversion);
    dopings.push_back(doping);

    // containers to hold data in std vectors
    std::vector<double> tempT;
    std::vector<double> tempL;
    std::vector<double> tempE;
    // loop over energy values on which the calculation was done
    for (int iEnergy = 0; iEnergy < numEnergies; ++iEnergy) {
      double ene = energiesEPA(iEnergy);
      double tau = 1. / scatteringRates.data(iCalc, iEnergy);
      double linewidth = 1. / tau;
      tempE.push_back(ene * energyConversion);
      tempT.push_back(tau * energyToTime);
      tempL.push_back(linewidth * energyConversion);
    }
    outTimes.push_back(tempT);
    outLinewidths.push_back(tempL);
    energies.push_back(tempE);
  }

  // output to json
  nlohmann::json output;
  output["temperatures"] = temps;
  output["temperatureUnit"] = "K";
  output["chemicalPotentials"] = chemPots;
  output["chemicalPotentialUnit"] = "eV";
  output["dopingConcentrations"] = dopings;
  output["dopingConcentrationUnit"] = "cm$^{-" + std::to_string(dimensionality) + "}$";
  output["linewidths"] = outLinewidths;
  output["linewidthsUnit"] = energyUnit;
  output["relaxationTimes"] = outTimes;
  output["relaxationTimeUnit"] = "fs";
  output["energies"] = energies;
  output["energyUnit"] = energyUnit;
  output["particleType"] = particleType;
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}
