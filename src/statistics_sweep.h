#ifndef SWEEP_H
#define SWEEP_H

#include <memory>

#include "bandstructure.h"
#include "context.h"
#include "particle.h"
#include "utilities.h"

/** Container for temperature, chemical potential, doping, to be used.
 * To be used for loops over temperature and chemical potentials.
 * Note, temperatures here are in Ry, so this is kB*T
 */
struct CalcStatistics {
  double temperature;
  double chemicalPotential;
  double doping;
};

/** Object for controlling the loop over temperatures and chemical potentials.
 * Given temperatures and chemical potentials (or doping), we build a list of
 * pairs of temperatures and chemical potentials (and doping). This list can be
 * accessed to repeat calculations on a range of parameters.
 */
class StatisticsSweep {

 public:
  /** Constructor of the loop over temperatures and chemical potentials.
   * For phonons, the constructor simply stores the temperatures.
   * For electrons, the class finds chemical potentials from doping (or
   * vice versa, depending on user input), and builds a table of all
   * pairs of temperatures and chemical potentials.
   * @param context: object with user input.
   */
  explicit StatisticsSweep(Context &context,
                           FullBandStructure *fullBandStructure = nullptr);

  /** Copy constructor
   */
  StatisticsSweep(const StatisticsSweep &that);

  /** Copy assignment
   */
  StatisticsSweep &operator=(const StatisticsSweep &that);

  /** returns the CalcStatistics object containing temperature,
   * chemical potential and temperature, given an index on the number of
   * "calculations" = numTemperature*numChemicalPotentials.
   * @param index: an integer, ranging in [0,numCalculations[
   * @return calcStatistics: a struct object with temperature, chemical
   * potential and doping concentration, for this particular index.
   */
  CalcStatistics getCalcStatistics(const int &index);

  /** returns the CalcStatistics object containing temperature,
   * chemical potential and temperature, given indices on temperature and
   * chemical potential.
   * @param iTemp: a strong-typed index, ranging in [0,numTemperatures[
   * @param iChemPot: a strong-typed index, ranging in
   * [0,numChemicalPotentials[
   * @return calcStatistics: a struct object with temperature, chemical
   * potential and doping concentration.
   */
  CalcStatistics getCalcStatistics(const TempIndex &iTemp,
                                          const ChemPotIndex &iChemPot);

  /** Returns the number of "calculations" i.e. the number of temperatures
   * times the number of chemical potentials we will compute.
   */
  int getNumCalculations() const;

  /** Returns for how many chemical potentials we are computing properties.
   */
  int getNumChemicalPotentials() const;

  /** Returns for how many temperatures we are computing properties.
   */
  int getNumTemperatures() const;

  /**
   * Tells us which kind of bandstructure this statistics sweep matches. 
   * The only time this matters is for the sake of chemical potentials. 
   * @return Particle: the particle which was used to calculate the stat sweep. 
  */
  inline Particle getParticle() const { return particle; }

  /** Prints to screen the information on the temperature, doping and
   * chemical potentials to be computed in the current run.
   */
  void printInfo();

  /** calculate and print the number of free carriers
  * @param bandStructure : the active band structure to calculate from
  **/
  void calcNumFreeCarriers(BaseBandStructure* bandStructure);

 protected:

  Particle particle;
  int numCalculations = 0;
  Eigen::MatrixXd infoCalculations;
  int nTemp = 0;
  int nChemPot = 0;
  int nDop = 0;
  bool isDistributed = false;
  int dimensionality = 0;

  // note: these three private methods are only meant to be used
  // during the construction of this class, as they depend on energies

  /** Computes the electronic chemical potential, given the doping
   * concentration.
   */
  double findChemicalPotentialFromDoping(const double &doping,
                                         const double &temperature);

  /** The opposite of the previous method, computes the doping concentration
   * given the electronic chemical potential.
   */
  double findDopingFromChemicalPotential(const double &chemicalPotential,
                                         const double &temperature);

  // Auxiliary function for finding the chemical potential.
  double fPop(const double &chemPot, const double &temp);

  // this block is temporary variables for electronic calculations
  const int maxIter = 100;
  int numPoints = 0;
  int numBands = 0;
  std::vector<double> energies;
  double numElectronsDoped = 0.;
  double volume = 0.;
  double spinFactor = 0.;
  double occupiedStates = 0.;
  double fermiLevel = 0.;
};

#endif
