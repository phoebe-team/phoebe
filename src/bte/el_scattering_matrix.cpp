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
                                       PhononH0 &h0_)
    : ScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
     BaseElScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
       phononH0(h0_) {

  isMatrixOmega = true;
  highMemory = context.getScatteringMatrixInMemory();
  numElStates = numStates; // just in case
}

void ElScatteringMatrix::builder(std::shared_ptr<VectorBTE> linewidth,
                                 std::vector<VectorBTE> &inPopulations,
                                 std::vector<VectorBTE> &outPopulations) {

  if(mpi->mpiHead())
    std::cout << "============== Building electron scattering matrix ==============" << std::endl;

  Kokkos::Profiling::pushRegion("ElScatteringMatrix::builder");

  // set in the parent object what kind of matrix this is                            
  setMatrixCase(linewidth, inPopulations, outPopulations);
      
  if ((linewidth != nullptr) && (linewidth->dimensionality != 1)) {
    DeveloperError("The linewidths shouldn't have dimensionality");
  }

  // set up the MRTA container
  linewidthMR = std::make_shared<VectorBTE>(statisticsSweep, outerBandStructure, 1);

  // precompute particle occupations
  //Eigen::MatrixXd outerFermi = precomputeOccupations(outerBandStructure);
  Eigen::MatrixXd innerFermi = precomputeOccupations(innerBandStructure);

  // compute wavevector pairs for the calculation
  bool rowMajor = true;
  std::vector<std::tuple<std::vector<int>, int>> kPairIterator =
                                 getIteratorWavevectorPairs(rowMajor);

  // add scattering contributions ---------------------------------------
  // add elph scattering
  // TODO are we sure this should get two Fermi's and not have one of them be a Bose?

  { // let the interaction elph go out of scope after this, it takes a lot of memory

  // load the elph coupling
  // Note: this file contains the number of electrons
  // which is needed to understand where to place the fermi level
  Crystal crystal = innerBandStructure.getPoints().getCrystal();
  InteractionElPhWan couplingElPh =
      InteractionElPhWan::parse(context, crystal, phononH0);

  addElPhScattering(*this, context, inPopulations, outPopulations, 
                                  kPairIterator, innerFermi, //outerFermi,
                                  innerBandStructure, outerBandStructure, phononH0,
                                  couplingElPh, linewidth);
  }
  // add charged impurity electron scattering  -------------------
/*  addChargedImpurityScattering(*this, context, inPopulations, outPopulations,
                       kPairIterator,
                       innerBandStructure, outerBandStructure, linewidth);
*/
  // TODO was there previously an all reduce between these two on
  //the linewidths? why is that?
  // probably because boundary scattering was earlier not distributed 

  // add DMFT fermi liquid contribution  -------------------
  // currently we don't add ee time to linewidthMR. I think this is correct. 
  //add_eeDMFT(*this, context, outerBandStructure, linewidth);

  // Add boundary scattering ------------------------------------
  if (!std::isnan(context.getBoundaryLength())) {
    if (context.getBoundaryLength() > 0.) {
      addBoundaryScattering(*this, context, inPopulations, outPopulations,
                            outerBandStructure, linewidth);
    }
  }

  // all reduce the linewidths
  if (matrixCase == matrixVectorProduct) {
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
  if (matrixCase == linewidthOnly) {
    degeneracyAveragingLinewidths(linewidth);
    degeneracyAveragingLinewidths(linewidthMR);
  }

  // use the off diagonals to calculate the linewidths,
  // to ensure the special eigenvectors can be found/preserve conservation of momentum
  // that might be ruined by the delta functions
  //enforceDetailedBalance();

 // we place the linewidths back in the diagonal of the scattering matrix
  // this because we may need an MPI_allReduce on the linewidths
  if (matrixCase == fullMatrix) {// case of matrix construction
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

  // before closing, write the relaxation times to file 
  // remember to first convert to a vector BTE object without symmetrization
  if(matrixCase != matrixVectorProduct) {

    getLinewidths(*linewidthMR).outputToJSON("mrta_el_relaxation_times.json", outerBandStructure);
    getLinewidths(*linewidth).outputToJSON("rta_el_relaxation_times.json", outerBandStructure);
    
    if(outputUNTimes) { 
      getLinewidths(*internalDiagonalNormal).outputToJSON("rta_el_N_relaxation_times.json", outerBandStructure); 
      getLinewidths(*internalDiagonalUmklapp).outputToJSON("rta_el_U_relaxation_times.json", outerBandStructure); 
    }
}
}

// function called on shared ptrs of linewidths
VectorBTE ElScatteringMatrix::getSingleModeMRTimes() {
  return getSingleModeTimes(*linewidthMR);
}
