#ifndef PH_SCATTERING_H
#define PH_SCATTERING_H

#include "ph_scattering_matrix.h"

 // friend functions for adding scattering rates,
 // these live in ph_scattering.cpp
 // TODO write docstrings for these

void addPhPhScattering(BasePhScatteringMatrix &matrix, Context &context,
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

void addIsotopeScattering(BasePhScatteringMatrix &matrix, Context &context,
                                std::vector<VectorBTE> &inPopulations,
                                std::vector<VectorBTE> &outPopulations, int &switchCase,
                                std::vector<std::tuple<std::vector<int>, int>> qPairIterator,
                                Eigen::MatrixXd &innerBose, Eigen::MatrixXd &outerBose,
                                BaseBandStructure &innerBandStructure,
                                BaseBandStructure &outerBandStructure,
                                std::shared_ptr<VectorBTE> linewidth);

#endif
