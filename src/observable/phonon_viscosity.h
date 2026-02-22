#ifndef PHONON_VISCOSITY_H
#define PHONON_VISCOSITY_H

#include "drift.h"
#include "observable.h"
#include "ph_scattering_matrix.h"
#include "viscosity_io.h"

/** Object to compute and store the phonon viscosity.
 */
class PhononViscosity : public Observable {
public:
  /** Object constructor. Simply stores references to the inputs
   * @param statisticsSweep: object with info on the temperature and chemical
   * potential loops
   * @param crystal: object with the crystal structure
   * @param BaseBandStructure: object with the quasiparticle energies and
   * velocities computed on a mesh of wavevectors.
   */
  PhononViscosity(Context &context_, StatisticsSweep &statisticsSweep_,
                  Crystal &crystal_, BaseBandStructure &bandStructure_);

  /** Compute the viscosity within the relaxation time approximation
   * Stores it internally.
   * @param n: the absolute phonon population.
   */
  virtual void calcRTA(VectorBTE &n);

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
  double spinFactor = 1;
  
};

#endif
