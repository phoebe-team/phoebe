#ifndef BASE_EL_SCATTERING_MATRIX_H
#define BASE_EL_SCATTERING_MATRIX_H

#include "phonon_h0.h"
#include "interaction_elph.h"
#include "electron_h0_wannier.h"
#include "scattering_matrix.h"
#include "vector_bte.h"

/** class representing the electron scattering matrix.
 * This class holds the absolute essentials (electronH0) that any
 * electron scattering matrix must have
 */
class BaseElScatteringMatrix : virtual public ScatteringMatrix {
 public:

  /** Constructor that just calls the constructor of scattering matrix */
  BaseElScatteringMatrix(Context &context_,
                        StatisticsSweep &statisticsSweep_,
                        BaseBandStructure &innerBandStructure_,
                        BaseBandStructure &outerBandStructure_);

 protected:

  std::shared_ptr<VectorBTE> linewidthMR;

  // friend functions for adding scattering rates,
  // these live in el_scattering.cpp. See el_scattering header file for more info.
  friend void addElPhScattering(BaseElScatteringMatrix &matrix, Context &context,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations,
                       int &switchCase,
                       std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                       Eigen::MatrixXd &innerFermi, //Eigen::MatrixXd &outerBose,
                       BaseBandStructure &innerBandStructure,
                       BaseBandStructure &outerBandStructure,
                       PhononH0 &phononH0,
                       InteractionElPhWan& couplingElPhWan,
                       std::shared_ptr<VectorBTE> linewidth);

  friend void addChargedImpurityScattering(BaseElScatteringMatrix &matrix, Context &context,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations,
                       int &switchCase,
                       std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                       BaseBandStructure &innerBandStructure,
                       BaseBandStructure &outerBandStructure,
                       std::shared_ptr<VectorBTE> linewidth);

};

#endif
