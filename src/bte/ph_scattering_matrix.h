#ifndef PH_SCATTERING_MATRIX_H
#define PH_SCATTERING_MATRIX_H

#include "interaction_3ph.h"
#include "phonon_h0.h"
#include "base_ph_scattering_matrix.h"
#include "vector_bte.h"

/** class representing the phonon scattering matrix.
 * This class contains the logic to compute the phonon scattering matrix.
 * The parent class ScatteringMatrix instead contains the logic for managing
 * the operations with phonon distribution vectors.
 */
class PhScatteringMatrix : virtual public BasePhScatteringMatrix {
 public:
  /** Default constructor
   * @param context: the user-initialized variables.
   * @param statisticsSweep: the object containing the information on the
   * temperatures to be used in the calculation.
   * @param innerBandStructure: this is the band structure object used for
   * integrating the sum over q2 wavevectors.
   * @param outerBandStructure: this is the bandStructure object used for
   * integrating the sum over q1 wavevectors.
   * @param h0: the object used for constructing phonon energies.
   *
   * Note: inner and outer band structures may be different, for example, if we
   * want to compute the phonon linewidths on a path, the outer band structure
   * contains the phonon dispersion relation along high symmetry lines, whereas
   * the inner band structure contains the dispersion relation on a full grid
   * of wavevectors. For transport calculations instead, inner=outer.
   * Especially in the case of computing linewidths on a path, it might be
   * necessary to compute phonon energies on a wavevector q3=q1+q2: for this
   * reason, we need to pass also the phonon h0 object.
   */
  PhScatteringMatrix(Context &context_, StatisticsSweep &statisticsSweep_,
                     BaseBandStructure &innerBandStructure_,
                     BaseBandStructure &outerBandStructure_,
                     PhononH0 *phononH0_ = nullptr);

 protected:

  PhononH0 *phononH0;

  // implementation of the scattering matrix
  void builder(std::shared_ptr<VectorBTE> linewidth,
               std::vector<VectorBTE> &inPopulations,
               std::vector<VectorBTE> &outPopulations) override;

 // friend functions for adding scattering rates,
 // these live in ph_scattering.cpp
 // TODO write docstrings for these

  friend void addPhPhScattering(BasePhScatteringMatrix &matrix, Context &context,
                                std::vector<VectorBTE> &inPopulations,
                                std::vector<VectorBTE> &outPopulations,
                                int &switchCase,
                                std::vector<std::tuple<std::vector<int>, int>> qPairIterator,
                                Eigen::MatrixXd &innerBose, Eigen::MatrixXd &outerBose,
                                BaseBandStructure &innerBandStructure,
                                BaseBandStructure &outerBandStructure,
                                PhononH0& phononH0,
                                Interaction3Ph& coupling3Ph,
                                std::shared_ptr<VectorBTE> linewidth);

  friend void addIsotopeScattering(BasePhScatteringMatrix &matrix, Context &context,
                                std::vector<VectorBTE> &inPopulations,
                                std::vector<VectorBTE> &outPopulations, int &switchCase,
                                std::vector<std::tuple<std::vector<int>, int>> qPairIterator,
                                Eigen::MatrixXd &innerBose, Eigen::MatrixXd &outerBose,
                                BaseBandStructure &innerBandStructure,
                                BaseBandStructure &outerBandStructure,
                                std::shared_ptr<VectorBTE> linewidth);

};

#endif
