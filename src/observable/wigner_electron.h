#ifndef WIGNER_ELECTRON_H
#define WIGNER_ELECTRON_H

#include "onsager.h"

/** Class to compute the phonon thermal conductivity
 *
 */
class WignerElCoefficients : public OnsagerCoefficients {
 public:
  /** Constructor method
   * @param statisticsSweep: a StatisticsSweep object containing information
   * on the temperature loop
   * @param crystal: a Crystal object
   * @param bandStructure: the bandStructure that is used to compute thermal
   * conductivity. This should be aligned with the phonon population vector
   * used in the BTE.
   * @param linewidths: a VectorBTE object containing the linewidths.
   * Hence, it is expected that this class is called after the scattering matrix
   * has been computed.
   */
  WignerElCoefficients(StatisticsSweep &statisticsSweep_,
                       Crystal &crystal_,
                       BaseBandStructure &bandStructure_,
                       Context &context_,
                       VectorBTE &relaxationTimes);

  /** Compute the thermal conductivity from the phonon populations
   * @param n: the phonon population out-of-equilibrium. Note that this
   * method uses the absolute value of phonon populations n.
   */
  void calcFromPopulation(VectorBTE &nE, VectorBTE &nT) override;

  /** Output contributions to wigner transport at each state
   * into a JSON file. For now only written for electrical conductivity, LET. 
   * Currently not used. 
  */
  void outputContributionsToJSON(const std::string &outFileName);

  /** Prints to screen the thermal conductivity at various temperatures
   * in a a nicely formatted way.
   */
  void print() override;

 protected:
  VectorBTE &smaRelTimes;
  Eigen::Tensor<double, 3> correctionLEE, correctionLTE, correctionLET, correctionLTT;
  Eigen::Tensor<double, 4>  contributionLEE, contributionLET;
};

#endif

