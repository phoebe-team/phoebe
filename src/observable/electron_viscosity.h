#ifndef ELECTRON_VISCOSITY_H
#define ELECTRON_VISCOSITY_H

#include "drift.h"
#include "observable.h"
#include "el_scattering.h"

/** Object to compute and store the electron viscosity.
 */
class ElectronViscosity : public Observable {
public:
  /** Object constructor. Simply stores references to the inputs
   * @param statisticsSweep: object with info on the temperature and chemical
   * potential loops
   * @param crystal: object with the crystal structure
   * @param BaseBandStructure: object with the quasiparticle energies and
   * velocities computed on a mesh of wavevectors.
   */
  ElectronViscosity(Context &context_, StatisticsSweep &statisticsSweep_,
                  Crystal &crystal_, BaseBandStructure &bandStructure_);

  /** Compute the viscosity within the relaxation time approximation
   * Stores it internally.
   * @param n: the relaxation times.
   */
  virtual void calcRTA(VectorBTE &tau);

  /** Prints the viscosity to screen for the user.
   */
  void print();

  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON(const std::string& outFileName);

protected:

  int whichType() override;
  BaseBandStructure& bandStructure;
  double spinFactor = 2;

};

#endif
