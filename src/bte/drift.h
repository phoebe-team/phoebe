#ifndef DRIFT_H
#define DRIFT_H

#include "specific_heat.h"
#include "vector_bte.h"

/** Object describing the thermal diffusion operator of the BTE
 */
class BulkTDrift : public VectorBTE {
public:
  /** Default constructor
   *
   * @param statisticsSweep_: object with temperatures and chemical potentials
   * @param bandStructure_: object with the band structure
   * @param dimensionality_: dimensionality of space (should be 3)
   * @param symmetrize: default false. If true, the diffusion/"b" vector of the
   * BTE is rescaled by a factor sqrt(f(1-f)) or sqrt(n(n+1)). This is being
   * used for example in the electronic BTE solver, which is being built with
   * the symmetrized scattering matrix.
   */
  BulkTDrift(StatisticsSweep &statisticsSweep_,
             BaseBandStructure &bandStructure_,
             const int &dimensionality_ = 3,
             const bool& symmetrize = false);
};

/** Object describing the electric-field drift operator of the BTE
 */
class BulkEDrift : public VectorBTE {
public:
  /** Default constructor
   *
   * @param statisticsSweep_: object with temperatures and chemical potentials
   * @param bandStructure_: object with the band structure
   * @param dimensionality_: dimensionality of space (should be 3)
   * @param symmetrize: default false. If true, the diffusion/"b" vector of the
   * BTE is rescaled by a factor sqrt(f(1-f)) or sqrt(n(n+1)). This is being
   * used for example in the electronic BTE solver, which is being built with
   * the symmetrized scattering matrix.
   */
  BulkEDrift(StatisticsSweep &statisticsSweep_,
             BaseBandStructure &bandStructure_,
             const int &dimensionality_ = 3,
             const bool& symmetrize = false);
};

#endif
