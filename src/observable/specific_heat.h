#ifndef SPECIFIC_HEAT_H
#define SPECIFIC_HEAT_H

#include "observable.h"

/** Class for computing and storing the specific heat of a crystal.
 * for either electrons or phonons, it returns:
 * C = 1/(V*N) * sum(states) * dn/dT * energy
 *   = 1/(V*N) * sum(states) * n(n+/-1) * energy^2 / kB T^2
 */
class SpecificHeat : public Observable {
public:
  /** Constructor method
   * @param statisticsSweep: a StatisticsSweep object containing information
   * on the temperature loop
   * @param crystal: a Crystal object. Mostly used for the volume
   * @param bandStructure: bandStructure to use for computing specific heat
   */
  SpecificHeat(Context &context_, StatisticsSweep &statisticsSweep_,
               Crystal &crystal_, BaseBandStructure &bandStructure_);

  /** Copy constructor
   */
  SpecificHeat(const SpecificHeat &that);

  /** Copy assignment operator
   */
  SpecificHeat &operator=(const SpecificHeat &that);

  /** Computes the specific heat at all requested temperatures.
   */
  virtual void calc();

  /** Prints to screen the specific heat at all desired temperatures.
   */
  void print();

  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON(const std::string &outFileName);

  /** returns the specific heat computed at a specific chemical potential
   * and temperature. The indices are controlled by the statisticsSweep
   * object.
   * @param ChemPotIndex: strong typed index of chemical potential.
   * @param TempIndex: strong typed index of temperature.
   */
  const double &get(const ChemPotIndex &imu, const TempIndex &it);
  const double &get(const int &iCalc);

protected:

  int whichType() override;
  int spinFactor;
  BaseBandStructure &bandStructure;

};

#endif
