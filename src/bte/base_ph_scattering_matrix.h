#ifndef BASE_PH_SCATTERING_MATRIX_H
#define BASE_PH_SCATTERING_MATRIX_H

#include "interaction_elph.h"
#include "interaction_3ph.h"
#include "phonon_h0.h"
#include "electron_h0_wannier.h"
#include "scattering_matrix.h"
#include "vector_bte.h"

/** class representing the phonon scattering matrix.
 * This class contains the logic to compute the phonon scattering matrix.
 * The parent class ScatteringMatrix instead contains the logic for managing
 * the operations with phonon distribution vectors.
 */
class BasePhScatteringMatrix : virtual public ScatteringMatrix {
 public:

  /** Constructor that just calls the constructor of scattering matrix */
  BasePhScatteringMatrix(Context &context_,
                        StatisticsSweep &statisticsSweep_,
                        BaseBandStructure &innerBandStructure_,
                        BaseBandStructure &outerBandStructure_);

 protected:

  // friend functions for adding scattering rates,
  // these live in ph_scattering.cpp, descriptions in ph_scattering.h
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

  friend void addPhElScattering(BasePhScatteringMatrix& matrix, Context& context,
                                BaseBandStructure& phBandStructure,
                                BaseBandStructure& elBandStructure,
                                InteractionElPhWan& couplingElPhWan,
                                std::shared_ptr<VectorBTE> linewidth);

};

#endif
