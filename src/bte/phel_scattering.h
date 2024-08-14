#ifndef PHEL_SCATTERING_H
#define PHEL_SCATTERING_H

#include "base_ph_scattering_matrix.h"
#include "coupled_scattering_matrix.h"

// friend functions for adding scattering rates,
// these live in ph_scattering.cpp
// TODO write docstrings for these

void addPhElScattering(BasePhScatteringMatrix &matrix, Context &context,
                BaseBandStructure &phBandStructure,
                BaseBandStructure &elBandStructure,
                StatisticsSweep& statisticsSweep, 
                InteractionElPhWan &couplingElPhWan,
                std::shared_ptr<VectorBTE> linewidth);

void phononElectronAcousticSumRule(CoupledScatteringMatrix &matrix,
                Context& context,
                std::shared_ptr<CoupledVectorBTE> phElLinewidths,
                BaseBandStructure& elBandStructure,
                BaseBandStructure& phBandStructure);

#endif
