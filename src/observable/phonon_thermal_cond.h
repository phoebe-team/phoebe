#ifndef PHONON_CONDUCTIVITY_H
#define PHONON_CONDUCTIVITY_H

#include "observable.h"
#include "ph_scattering_matrix.h"
#include "specific_heat.h"

/** Class to compute the phonon thermal conductivity
 *
 */
class PhononThermalConductivity : public Observable {
public:
  /** Constructor method
   * @param statisticsSweep: a StatisticsSweep object containing information
   * on the temperature loop
   * @param crystal: a Crystal object
   * @param bandStructure: the bandStructure that is used to compute thermal
   * conductivity. This should be aligned with the phonon population vector
   * used in the BTE.
   */
  PhononThermalConductivity(Context &context_,
                            StatisticsSweep &statisticsSweep_,
                            Crystal &crystal_,
                            BaseBandStructure &bandStructure_);

  /** Copy constructor
   *
   */
  PhononThermalConductivity(const PhononThermalConductivity &that);

  /** Copy assignment operator
   *
   */
  PhononThermalConductivity &operator=(const PhononThermalConductivity &that);

  /** Difference operator
   * We use this method to check how much the thermal conductivity changes
   * w.r.t. the previous step in an iterative BTE solver
   */
  PhononThermalConductivity operator-(const PhononThermalConductivity &that);

  /** Compute the thermal conductivity from the phonon populations
   * @param n: the phonon population out-of-equilibrium. Note that this
   * method uses the absolute value of phonon populations n.
   */
  virtual void calcFromPopulation(VectorBTE &n);

  /** Compute the thermal conductivity from canonical phonon populations
   * where the "canonical" population f is defined as n = bose(bose+1)f
   * where "bose" is the Bose-Einstein population.
   * Iterative schemes are probably computing this canonical population.
   * @param f: canonical phonon population.
   */
  virtual void calcFromCanonicalPopulation(VectorBTE &f);

  /** Compute the thermal conductivity using a variational estimator
   * See Eq. 26 of https://link.aps.org/doi/10.1103/PhysRevB.88.045430
   * @param af: the product of the scattering matrix A with the canonical
   * population, rescaled with a Conjugate gradient preconditioning.
   * @param f: the canonical phonon population
   * @param scalingCG: the conjugate gradient preconditioning, which in detail
   * is the sqrt of the diagonal elements of the scattering matrix A.
   */
  virtual void calcVariational(VectorBTE &af, VectorBTE &f, VectorBTE &scalingCG);

  /** Compute the thermal conductivity using the relaxon method
   * i.e. Eq. 13 of Phys.Rev.X 6, 041013 (2016)
   * @param specificHeat: the specific heat
   * @param relaxonV: the vector of relaxon velocities V (eq. 10)
   * @param relaxationTimes: the reciprocal of the scattering matrix
   * eigenvalues (from eq.7), i.e. the relaxation times of the system.
   */
  virtual void calcFromRelaxons(Context &context, StatisticsSweep &statisticsSweep,
                        ParallelMatrix<double> &eigenvectors,
                        PhScatteringMatrix &scatteringMatrix,
                        const Eigen::VectorXd &eigenvalues);

  /** Prints to screen the thermal conductivity at various temperatures
   * in a a nicely formatted way.
   */
  virtual void print();

  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON(const std::string& outFileName);

  /** Prints to screen the thermal conductivity at various temperatures
   * To be used during an iterative BTE solver, this prints only the diagonal
   * elements of the thermal conductivity tensor.
   * @param iter: the iteration number to notify the user.
   */
  void print(const int &iter);

  Eigen::Tensor<double,3> getThermalConductivity();

protected:
  int whichType() override;
  BaseBandStructure &bandStructure;
  // unit information for writing to files
  std::string thCondUnits;
  double thCondConversion;

};

#endif
