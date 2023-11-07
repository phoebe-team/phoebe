#include "base_el_scattering_matrix.h"

BaseElScatteringMatrix::BaseElScatteringMatrix(Context &context_,
                                   StatisticsSweep &statisticsSweep_,
                                   BaseBandStructure &innerBandStructure_,
                                   BaseBandStructure &outerBandStructure_)
    : ScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_) {

}
