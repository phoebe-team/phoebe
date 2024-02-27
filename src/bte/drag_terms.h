#ifndef DRAG_TERM_H
#define DRAG_TERM_H

#include "coupled_scattering_matrix.h"
#include "electron_h0_wannier.h"

// TODO finish adding doxygen strings to these
/**
 * @param matrix: a el scattering matrix object
 * @param context: object with user parameters for this calculation
 * @param kqPairIterator: which wavevector states to fill in
 **/

const int Del = 0;
const int Dph = 1;

// TODO change this to a coupled scattering matrix
void addDragTerm(CoupledScatteringMatrix &matrix, Context &context,
                  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator,
                  int dragTermType,
                  ElectronH0Wannier* electronH0,
                  InteractionElPhWan *couplingElPhWan,
                  BaseBandStructure &innerBandStructure, // phonon
                  BaseBandStructure &outerBandStructure); // electron

// TODO change this to a coupled scattering matrix
void addDragTerm2(CoupledScatteringMatrix &matrix, Context &context,
                  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator,
                  int dragTermType,
                  ElectronH0Wannier* electronH0,
                  InteractionElPhWan *couplingElPhWan,
                  BaseBandStructure &innerBandStructure, // phonon
                  BaseBandStructure &outerBandStructure); // electron

#endif
