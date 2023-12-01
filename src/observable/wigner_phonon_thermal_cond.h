#ifndef WIGNER_PHONON_CONDUCTIVITY_H
#define WIGNER_PHONON_CONDUCTIVITY_H

#include "phonon_thermal_cond.h"

/** Class to compute the phonon thermal conductivity
 *
 */
class WignerPhononThermalConductivity : public PhononThermalConductivity {
public:
  /** Constructor method
   * @param statisticsSweep: a StatisticsSweep object containing information
   * on the temperature loop
   * @param crystal: a Crystal object
   * @param bandStructure: the bandStructure that is used to compute thermal
   * conductivity. This should be aligned with the phonon population vector
   * used in the BTE.
   * @param lineWidths: a VectorBTE object containing the lineWidths.
   * Hence, it is expected that this class is called after the scattering matrix
   * has been computed.
   */
  WignerPhononThermalConductivity(Context &context_,
                                  StatisticsSweep &statisticsSweep_,
                                  Crystal &crystal_,
                                  BaseBandStructure &bandStructure_,
                                  VectorBTE &relaxationTimes);

  /** Compute the thermal conductivity from the phonon populations
   * @param n: the phonon population out-of-equilibrium. Note that this
   * method uses the absolute value of phonon populations n.
   */
  void calcFromPopulation(VectorBTE &n) override;

  /** Compute the thermal conductivity using a variational estimator
   * See Eq. 26 of https://link.aps.org/doi/10.1103/PhysRevB.88.045430
   * @param af: the product of the scattering matrix A with the canonical
   * population, rescaled with a Conjugate gradient preconditioning.
   * @param f: the canonical phonon population
   * @param scalingCG: the conjugate gradient preconditioning, which in detail
   * is the sqrt of the diagonal elements of the scattering matrix A.
   */
  void calcVariational(VectorBTE &af, VectorBTE &f, VectorBTE &scalingCG) override;

  /** Compute the thermal conductivity using the relaxon method
   * i.e. Eq. 13 of Phys.Rev.X 6, 041013 (2016)
   * @param specificHeat: the specific heat
   * @param relaxonV: the vector of relaxon velocities V (eq. 10)
   * @param relaxationTimes: the reciprocal of the scattering matrix
   * eigenvalues (from eq.7), i.e. the relaxation times of the system.
   */
  void calcFromRelaxons(Context &context, StatisticsSweep &statisticsSweep,
                        ParallelMatrix<double> &eigenvectors,
                        PhScatteringMatrix &scatteringMatrix,
                        const Eigen::VectorXd &eigenvalues) override;

  /** Prints to screen the thermal conductivity at various temperatures
   * in a a nicely formatted way.
   */
  void print() override;

protected:
  VectorBTE &smaRelTimes;
  Eigen::Tensor<double, 3> wignerCorrection;
};

#endif
