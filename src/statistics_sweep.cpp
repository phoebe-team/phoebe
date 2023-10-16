#include "statistics_sweep.h"

#include "bandstructure.h"
#include "constants.h"
#include "context.h"
#include "mpiHelper.h"
#include "utilities.h"
#include <algorithm>
#include <cmath>
#include <iomanip>

StatisticsSweep::StatisticsSweep(Context &context,
                                 FullBandStructure *fullBandStructure)
    : particle(fullBandStructure != nullptr ? fullBandStructure->getParticle()
                                            : Particle(Particle::phonon)),
      isDistributed(fullBandStructure != nullptr
                        ? fullBandStructure->getIsDistributed()
                        : false) {

  Kokkos::Profiling::pushRegion("StatisticsSweep constructor");

  Eigen::VectorXd temperatures = context.getTemperatures();
  nTemp = int(temperatures.size());
  if (nTemp == 0) {
    double minTemperature = context.getMinTemperature();
    double maxTemperature = context.getMaxTemperature();
    double deltaTemperature = context.getDeltaTemperature();

    if (std::isnan(minTemperature) || std::isnan(maxTemperature) || std::isnan(deltaTemperature)) {
      Error("Temperatures haven't been set in user input");
    }

    int i = 0;
    while (minTemperature <= maxTemperature) {
      temperatures.conservativeResize(i + 1);
      temperatures(i) = minTemperature;
      minTemperature += deltaTemperature;
      ++i;
    }
    nTemp = int(temperatures.size());
  }

  if (particle.isPhonon()) {
    nDop = 1;
    nChemPot = 1;
    numCalculations = nTemp * std::max(nChemPot, nDop);
    infoCalculations = Eigen::MatrixXd::Zero(numCalculations, 3);
    for (int it = 0; it < nTemp; it++) {
      double temp = temperatures(it);
      infoCalculations(it, 0) = temp;
      // note: the other two columns are set to zero above
    }

  } else {// isElectron()
    // in this case, the class is more complicated, as it is tasked with
    // the calculation of the chemical potential or the doping
    // so, we first load parameters to compute these quantities
    bool hasSpinOrbit = context.getHasSpinOrbit();
    if (hasSpinOrbit) {
      spinFactor = 1.;
    } else {// count spin degeneracy
      spinFactor = 2.;
    }

    volume = fullBandStructure->getPoints().getCrystal().getVolumeUnitCell();

    // flatten the energies (easier to work with)
    numPoints = fullBandStructure->getNumPoints();// local # of points
    numBands = fullBandStructure->getNumBands();
    energies.resize(numBands * numPoints);

    // this is written to handle a distributed band structure. It is important
    // to use the getStateIndices method, because getEnergy(stateIndex) expects
    // a global energies idx, and looping over getNumStates will only cover
    // [0,numLocalStates]. getStateIndices provides a list of global state
    // indices which belong to this process.
    std::vector<std::tuple<WavevectorIndex, BandIndex>> stateList =
        fullBandStructure->getStateIndices();
    #pragma omp parallel for
    for (int is = 0; is < fullBandStructure->getNumStates(); is++) {
      auto isk = std::get<0>(stateList[is]);
      auto isb = std::get<1>(stateList[is]);
      energies[is] = fullBandStructure->getEnergy(isk, isb);
    }

    // full # of points in grid
    numPoints = fullBandStructure->getNumPoints(true);

    // determine ground state properties
    // i.e. we define the number of filled bands and the fermi energy
    occupiedStates = context.getNumOccupiedStates();
    if (std::isnan(occupiedStates)) {
      // in this case we try to compute it from the Fermi-level
      fermiLevel = context.getFermiLevel();
      if (std::isnan(fermiLevel)) {
        Error("Must provide either the Fermi level or the number of"
              " occupied states");
      }

      // NOTE: the intel compiler has trouble compiling omp reductions
      // so, we do the reduction "manually"
      occupiedStates = 0.;
      #pragma omp parallel
      {
    	  double occupiedStatesPrivate = 0.;
      	#pragma omp for
	      for (int is = 0; is < fullBandStructure->getNumStates(); is++) {
	        if (energies[is] < fermiLevel) {
	          occupiedStatesPrivate += 1.;
	        }
	    }
	    #pragma omp critical
	    occupiedStates += occupiedStatesPrivate;
      }

      occupiedStates /= double(numPoints);
      if (isDistributed) {
        mpi->allReduceSum(&occupiedStates);
      }
    } else {
      occupiedStates /= spinFactor;

      // initial guess for chemical potential will be Ef
      // if distributed, all processes need this guess
      {
        // calculate mean of distributed energies
        // use it as an initial guess for fermi level
        double eSum = std::accumulate(energies.begin(), energies.end(), 0.);
        if (isDistributed) mpi->allReduceSum(&eSum);
        fermiLevel = eSum / double(numBands * numPoints);
      }
      // refine this guess at zero temperature and zero doping
      fermiLevel = findChemicalPotentialFromDoping(0., 0.);
    }

    // load input sweep parameters
    Eigen::VectorXd dopings = context.getDopings();
    Eigen::VectorXd chemicalPotentials = context.getChemicalPotentials();

    nChemPot = int(chemicalPotentials.size());
    nDop = int(dopings.size());

    // if chemical potentials and dopings are not supplied,
    // check for a min/max mu and dmu energy spacing in the input file
    if ((nDop == 0) && (nChemPot == 0)) {
      double minChemicalPotential = context.getMinChemicalPotential();
      double maxChemicalPotential = context.getMaxChemicalPotential();
      double deltaChemicalPotential = context.getDeltaChemicalPotential();

      if (std::isnan(minChemicalPotential) || std::isnan(maxChemicalPotential) || std::isnan(deltaChemicalPotential)) {
        Error("Didn't find chemical potentials or doping in input");
      }
      // set up energy grid of chemical potentials between min and max,
      // spaced by dmu
      int i = 0;
      while (minChemicalPotential <= maxChemicalPotential) {
        chemicalPotentials.conservativeResize(i + 1);
        chemicalPotentials(i) = minChemicalPotential;
        minChemicalPotential += deltaChemicalPotential;
        ++i;
      }
      nChemPot = int(chemicalPotentials.size());
    }

    // now have two cases:
    // if doping is passed in input, compute chemical potential
    // or vice versa

    Eigen::MatrixXd calcTable(nTemp * std::max(nChemPot, nDop), 3);
    calcTable.setZero();

    // in this case, I have dopings, and want to find chemical potentials
    if (nChemPot == 0) {
      nChemPot = nDop;
      for (int it = 0; it < nTemp; it++) {
        for (int id = 0; id < nDop; id++) {
          double temp = temperatures(it);
          double doping = dopings(id);
          double chemPot = findChemicalPotentialFromDoping(doping, temp);

          int iCalc = compress2Indices(it, id, nTemp, nDop);
          calcTable(iCalc, 0) = temp;
          calcTable(iCalc, 1) = chemPot;
          calcTable(iCalc, 2) = doping;
        }
      }

      // in this case, I have chemical potentials
    } else if (nDop == 0) {
      nDop = nChemPot;

      for (int it = 0; it < nTemp; it++) {
        for (int imu = 0; imu < nChemPot; imu++) {
          double temp = temperatures(it);
          double chemPot = chemicalPotentials(imu);
          double doping = findDopingFromChemicalPotential(chemPot, temp);

          int iCalc = compress2Indices(it, imu, nTemp, nChemPot);
          calcTable(iCalc, 0) = temp;
          calcTable(iCalc, 1) = chemPot;
          calcTable(iCalc, 2) = doping;
        }
      }
    }

    // clean memory
    energies.resize(0);
    // save potentials and dopings
    infoCalculations = calcTable;
    numCalculations = int(calcTable.rows());
  }
  printInfo();
  Kokkos::Profiling::popRegion();
}

// copy constructor
StatisticsSweep::StatisticsSweep(const StatisticsSweep &that)
    : particle(that.particle) {
  numCalculations = that.numCalculations;
  infoCalculations = that.infoCalculations;
  nTemp = that.nTemp;
  nChemPot = that.nChemPot;
  nDop = that.nDop;
}

// copy assignment
StatisticsSweep &StatisticsSweep::operator=(const StatisticsSweep &that) {
  if (this != &that) {
    particle = that.particle;
    numCalculations = that.numCalculations;
    infoCalculations = that.infoCalculations;
    nTemp = that.nTemp;
    nChemPot = that.nChemPot;
    nDop = that.nDop;
  }
  return *this;
}

double StatisticsSweep::fPop(const double &chemPot, const double &temp) {
  // fPop = 1/NK \sum_\mu FermiDirac(\mu) - N
  // Note that I don`t normalize the integral, which is the same thing I did
  // for computing the particle number

  double fPop_ = 0.;
  int numStates = int(energies.size());
  #pragma omp parallel
  {
    double fPopPrivate = 0.;
    #pragma omp for
    for (int is = 0; is < numStates; is++) {
      double x = energies[is];
      fPopPrivate += particle.getPopulation(x, temp, chemPot);
      }
    #pragma omp critical
    fPop_ += fPopPrivate;
  }

  // in the distributed case, this needs to be summed over all processes,
  // and also needs to be available to all processes for calculation of next
  // iteration's chem potential
  if (isDistributed) mpi->allReduceSum(&fPop_);
  fPop_ /= double(numPoints);
  fPop_ = numElectronsDoped - fPop_;
  return fPop_;
}

double
StatisticsSweep::findChemicalPotentialFromDoping(const double &doping,
                                                 const double &temperature) {
  // given the carrier concentration, finds the fermi energy
  // temperature is set inside glob
  // To find fermi energy, I must find the root of \sum_s f(s) - N = 0
  // the root is found with a bisection algorithm
  // Might be numerically unstable for VERY small doping concentration

  // numElectronsDoped is the total number of electrons in the unit cell
  // numElectrons is the number of electrons in the unit cell before doping
  // doping > 0 means p-doping (fermi level in the valence band)

  numElectronsDoped =
      occupiedStates - doping * volume * pow(distanceBohrToCm, 3) / spinFactor;
  // bisection method: I need to find the root of N - \int fermi dirac = 0
  // initial guess
  double chemicalPotential = fermiLevel;

  // Corner cases
  // if numElectronsDoped > numBands, it's a non-valid doping
  if (numElectronsDoped > float(numBands)) {
    Error("The requested number of occupied states is larger than the "
          "bands present in the Hamiltonian.\n"
          "numBands: " + std::to_string(numBands) + " numElectrons: " + std::to_string(numElectronsDoped)
          + "\nThis likely means you've selected a non-physical doping value, such as\n"
          "a very small doping for a metal, or you didn't Wannierize enough bands."
          "\nThis can also happen if you had bands under your disentanglement window which"
          "you did not cut out using exclude_bands in Wannier90. See a note about this in the elphWannier tutorial.");
  }
  if (numElectronsDoped < 0.) {
    Error("The number of occupied states is negative");
  }

  // if we are looking for the fermi level at T=0 and n=0, we have a corner case
  // when we have completely empty bands or completely full bands.
  if (doping == 0. && temperature == 0.) {// case of computing fermi level
    if (numElectronsDoped == 0.) {
      fermiLevel = *min_element(energies.begin(), energies.end());
      chemicalPotential = fermiLevel;
      return chemicalPotential;
    } else if (numElectronsDoped == float(numBands)) {
      fermiLevel = *max_element(energies.begin(), energies.end());
      chemicalPotential = fermiLevel;
      return chemicalPotential;
    }
  }

  double aX = 0; double bX = 0;
  // if this is a weird case where this processor has zero
  // states, the below lines will cause a seg fault
  if(energies.size() > 0) {
    // I choose the following (generous) boundaries
    aX = *min_element(energies.begin(), energies.end()) - 1.;
    bX = *max_element(energies.begin(), energies.end()) + 1.;
    // note: +-1 Ry = 13 eV should work for most dopings and temperatures,
    // even in corner cases
  }

  // if energies are distributed, each process needs to have the global
  // minimum and maximum of the energies
  if (isDistributed) {
    mpi->allReduceMin(&aX);
    mpi->allReduceMax(&bX);
  }
  double aY = fPop(aX, temperature);
  double bY = fPop(bX, temperature);

  // check if starting values are bad
  if (sgn(aY) == sgn(bY)) {
    Error("I should revisit the boundary limits for bisection method");
  }

  for (int iter = 0; iter < maxIter; iter++) {
    if (mpi->mpiHead() && iter == maxIter - 1) {
      Error("Max iteration reached in finding mu");
    }
    // x value is midpoint of prior values
    double cX = (aX + bX) / 2.;
    double cY = fPop(cX, temperature);

    // exit condition: the guess is exact or didn't change much
    if ((cY == 0.) || (abs(bX - aX) < 1.0e-8)) {
      chemicalPotential = cX;
      break;// go out of the loop
    }

    // check the sign
    if (sgn(cY) == sgn(aY)) {
      aX = cX;
    } else {
      bX = cX;
    }
  }
  return chemicalPotential;
}

double StatisticsSweep::findDopingFromChemicalPotential(
    const double &chemicalPotential, const double &temperature) {

  double fPop = 0.;
  int numStates = int(energies.size());
  #pragma omp parallel
  {
    double fPopPrivate = 0.;
    #pragma omp for
    for (int is = 0; is < numStates; is++) {
      double x = energies[is];
      fPopPrivate += particle.getPopulation(x, temperature, chemicalPotential);
    }
    #pragma omp critical
    fPop += fPopPrivate;
  }

  if (isDistributed) mpi->allReduceSum(&fPop);
  fPop /= double(numPoints);
  double doping = (occupiedStates - fPop);
  doping *= spinFactor / volume / pow(distanceBohrToCm, 3);
  return doping;
}

CalcStatistics StatisticsSweep::getCalcStatistics(const int &index) {
  CalcStatistics sc = {};
  sc.temperature = infoCalculations(index, 0);
  sc.chemicalPotential = infoCalculations(index, 1);// chemical potential
  sc.doping = infoCalculations(index, 2);           // doping
  return sc;
}

CalcStatistics
StatisticsSweep::getCalcStatistics(const TempIndex &iTemp,
                                   const ChemPotIndex &iChemPot) {
  int index = compress2Indices(iTemp.get(), iChemPot.get(), nTemp, nChemPot);
  return getCalcStatistics(index);
}

int StatisticsSweep::getNumCalculations() const { return numCalculations; }

int StatisticsSweep::getNumChemicalPotentials() const { return nChemPot; }

int StatisticsSweep::getNumTemperatures() const { return nTemp; }

void StatisticsSweep::printInfo() {
  if (!mpi->mpiHead()) return;
  std::cout << "\n";
  std::cout << "Statistical parameters for the calculation\n";

  if (particle.isElectron()) {
    std::cout << "Fermi level: " << fermiLevel * energyRyToEv << " (eV)"
              << std::endl;
    std::cout << "Index, temperature, chemical potential, doping concentration\n";
  } else {
    std::cout << "Index, temperature\n";
  }

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    double temp = infoCalculations(iCalc, 0);
    double chemPot = infoCalculations(iCalc, 1);
    double doping = infoCalculations(iCalc, 2);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "iCalc = " << iCalc << ", T = " << temp * temperatureAuToSi
              << " (K)";
    if (particle.isPhonon()) {
      std::cout << "\n";
    } else {
      std::cout << ", mu = " << chemPot * energyRyToEv << " (eV)"
                << ", n = " << std::scientific
                << doping << " (cm^-3)" << std::endl;
    }
  }
  std::cout << std::endl;
}
