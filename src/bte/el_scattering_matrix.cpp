#include "el_scattering_matrix.h"
#include "constants.h"
#include "helper_el_scattering.h"
#include "mpiHelper.h"
#include "periodic_table.h"
#include "general_scattering.h"
#include "el_scattering.h"

ElScatteringMatrix::ElScatteringMatrix(Context &context_,
                                       StatisticsSweep &statisticsSweep_,
                                       BaseBandStructure &innerBandStructure_,
                                       BaseBandStructure &outerBandStructure_,
                                       PhononH0 &h0_,
                                       InteractionElPhWan *couplingElPhWan_)
    : ScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
     BaseElScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
      couplingElPhWan(couplingElPhWan_), phononH0(h0_) {

  isMatrixOmega = true;
  highMemory = context.getScatteringMatrixInMemory();
  numElStates = numStates; // just in case
}

void ElScatteringMatrix::builder(std::shared_ptr<VectorBTE> linewidth,
                                 std::vector<VectorBTE> &inPopulations,
                                 std::vector<VectorBTE> &outPopulations) {

  Kokkos::Profiling::pushRegion("ElScatteringMatrix::builder");

  // 3 cases:
  // theMatrix and linewidth is passed: we compute and store in memory the
  // scattering
  //       matrix and the diagonal
  // inPopulation+outPopulation is passed: we compute the action of the
  //       scattering matrix on the in vector, returning outVec = sMatrix*vector
  // only linewidth is passed: we compute only the linewidths
  // Note: this determination is the same for all scattering matrices.
  // Perhaps we should make a general function for it in the parent class?
  int switchCase = 0;
  if (theMatrix.rows() != 0 && linewidth != nullptr && inPopulations.empty() && outPopulations.empty()) {
    switchCase = 0;
  } else if (theMatrix.rows() == 0 && linewidth == nullptr && !inPopulations.empty() && !outPopulations.empty()) {
    switchCase = 1;
  } else if (theMatrix.rows() == 0 && linewidth != nullptr && inPopulations.empty() && outPopulations.empty()) {
    switchCase = 2;
  } else {
    Error("Developer error: El matrix builder found a non-supported case");
  }

  if ((linewidth != nullptr) && (linewidth->dimensionality != 1)) {
    Error("Developer error: The linewidths shouldn't have dimensionality");
  }

  // set up the MRTA container
  linewidthMR = std::make_shared<VectorBTE>(statisticsSweep, outerBandStructure, 1);

  // precompute particle occupations
  //Eigen::MatrixXd outerFermi = precomputeOccupations(outerBandStructure);
  Eigen::MatrixXd innerFermi = precomputeOccupations(innerBandStructure);

  // compute wavevector pairs for the calculation
  bool rowMajor = true;
  std::vector<std::tuple<std::vector<int>, int>> kPairIterator =
                                 getIteratorWavevectorPairs(switchCase, rowMajor);

  // add scattering contributions ---------------------------------------
  // add elph scattering
  // TODO are we sure this should get two Fermi's and not have one of them be a Bose?
  addElPhScattering(*this, context, inPopulations, outPopulations, switchCase,
                                  kPairIterator, innerFermi, //outerFermi,
                                  innerBandStructure, outerBandStructure, phononH0,
                                  couplingElPhWan, linewidth);

  // add charged impurity electron scattering  -------------------
/*  addChargedImpurityScattering(*this, context, inPopulations, outPopulations,
                       switchCase, kPairIterator,
                       innerBandStructure, outerBandStructure, linewidth);
*/
  // TODO was there previously an all reduce between these two on
  //the linewidths? why is that?

  // Add boundary scattering
  if (!std::isnan(context.getBoundaryLength())) {
    if (context.getBoundaryLength() > 0.) {
      addBoundaryScattering(*this, context, inPopulations, outPopulations,
                            switchCase, outerBandStructure, linewidth);
    }
  }

  // all reduce the linewidths
  if (switchCase == 1) {
    for (unsigned int iVec = 0; iVec < inPopulations.size(); iVec++) {
      mpi->allReduceSum(&outPopulations[iVec].data);
    }
  } else {
    mpi->allReduceSum(&linewidth->data);
    mpi->allReduceSum(&linewidthMR->data);
  }

  // reinforce the condition that the scattering matrix is symmetric
  // A -> ( A^T + A ) / 2
  if ( context.getSymmetrizeMatrix() && context.getScatteringMatrixInMemory()) {
    symmetrize();
  }

  // Average over degenerate eigenstates.
  // we turn it off for now and leave the code if needed in the future
  // TODO   ^^ do we?
  if (switchCase == 2) {
    degeneracyAveragingLinewidths(linewidth);
    degeneracyAveragingLinewidths(linewidthMR);
  }

  // use the off diagonals to calculate the linewidths, 
  // to ensure the special eigenvectors can be found/preserve conservation of momentum 
  // that might be ruined by the delta functions 
  reinforceLinewidths();

 // we place the linewidths back in the diagonal of the scattering matrix
  // this because we may need an MPI_allReduce on the linewidths
  if (switchCase == 0) {// case of matrix construction
    int iCalc = 0;
    if (context.getUseSymmetries()) {
      // numStates is defined in scattering.cpp as # of irrStates
      // from the outer band structure
      for (int iBte = 0; iBte < numStates; iBte++) {
        BteIndex iBteIdx(iBte);
        // zero the diagonal of the matrix
        for (int i : {0, 1, 2}) {
          CartIndex iCart(i);
          int iMati = getSMatrixIndex(iBteIdx, iCart);
          for (int j : {0, 1, 2}) {
            CartIndex jCart(j);
            int iMatj = getSMatrixIndex(iBteIdx, jCart);
            theMatrix(iMati, iMatj) = 0.;
          }
          theMatrix(iMati, iMati) += linewidth->operator()(iCalc, 0, iBte);
        }
      }
    }
    else {
      for (int is = 0; is < numStates; is++) {
        theMatrix(is, is) = linewidth->operator()(iCalc, 0, is);
      }
    }
  }
  Kokkos::Profiling::popRegion();

  // before closing, write the MR relaxation times to file
  outputLifetimesToJSON("rta_el_momentum_relaxation_times.json", linewidthMR);
}

// function called on shared ptrs of linewidths
VectorBTE ElScatteringMatrix::getSingleModeMRTimes() {
  return getTimesFromVectorBTE(*linewidthMR);
}
