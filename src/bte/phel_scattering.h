#ifndef PHEL_SCATTERING_H
#define PHEL_SCATTERING_H

#include "base_ph_scattering_matrix.h"

 // friend functions for adding scattering rates,
 // these live in ph_scattering.cpp
 // TODO write docstrings for these

 void addPhElScattering(BasePhScatteringMatrix &matrix, Context &context,
                BaseBandStructure &phBandStructure,
                ElectronH0Wannier* electronH0,
                InteractionElPhWan *couplingElPhWan,
                std::shared_ptr<VectorBTE> linewidth);

#endif
