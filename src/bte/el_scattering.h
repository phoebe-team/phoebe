#ifndef EL_SCATTERING_H
#define EL_SCATTERING_H

#include "base_el_scattering_matrix.h"

// friend functions for adding scattering rates,
// these live in el_scattering.cpp

// TODO finish adding doxygen strings to these
/**
 * @param matrix: a el scattering matrix object
 * @param context: object with user parameters for this calculation
 * @param inPopulations:
 **/

void addElPhScattering(BaseElScatteringMatrix &matrix, Context &context,
                  std::vector<VectorBTE> &inPopulations,
                  std::vector<VectorBTE> &outPopulations,
                  int &switchCase,
                  std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                  Eigen::MatrixXd &innerFermi, Eigen::MatrixXd &outerBose,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure,
                  PhononH0 &phononH0,
                  InteractionElPhWan &couplingElPhWan,
                  std::shared_ptr<VectorBTE> linewidth);

void addChargedImpurityScattering(BaseElScatteringMatrix &matrix, Context &context,
                  std::vector<VectorBTE> &inPopulations,
                  std::vector<VectorBTE> &outPopulations,
                  int &switchCase,
                  std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure,
                  std::shared_ptr<VectorBTE> linewidth);

void add_eeDMFT(BaseElScatteringMatrix &matrix, const Context &context,
                  //std::vector<VectorBTE> &inPopulations,
                  //std::vector<VectorBTE> &outPopulations,
                  const int &switchCase,
                  BaseBandStructure &outerBandStructure,
                  std::shared_ptr<VectorBTE> linewidth); 

#endif
