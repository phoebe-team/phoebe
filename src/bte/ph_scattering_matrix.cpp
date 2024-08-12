//#include "ph_scattering_matrix.h"
#include "constants.h"
#include "helper_3rd_state.h"
#include "io.h"
#include "mpiHelper.h"
#include <cmath>
#include "general_scattering.h"
#include "ph_scattering.h"
#include "ifc3_parser.h"

PhScatteringMatrix::PhScatteringMatrix(Context &context_,
                                       StatisticsSweep &statisticsSweep_,
                                       BaseBandStructure &innerBandStructure_,
                                       BaseBandStructure &outerBandStructure_,
                                       Interaction3Ph *coupling3Ph_,
                                       PhononH0 *phononH0_)
    : ScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
     BasePhScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
      coupling3Ph(coupling3Ph_), phononH0(phononH0_) {

  if (&innerBandStructure != &outerBandStructure && phononH0 == nullptr) {
    Error("Developer error: PhScatteringMatrix needs phononh0 for incommensurate grids");
  }

}

void PhScatteringMatrix::builder(std::shared_ptr<VectorBTE> linewidth,
                                 std::vector<VectorBTE> &inPopulations,
                                 std::vector<VectorBTE> &outPopulations) {

  // 3 cases:
  // theMatrix and linewidth is passed: we compute and store in memory the
  // scattering matrix and the diagonal
  // inPopulation+outPopulation is passed: we compute the action of the
  //       scattering matrix on the in vector, returning outVec = sMatrix*vector
  // only linewidth is passed: we compute only the linewidths

  int switchCase = 0;
  if (theMatrix.rows() != 0 && linewidth != nullptr && inPopulations.empty() &&
      outPopulations.empty()) {
    switchCase = 0;  // build matrix and linewidths
  } else if (theMatrix.rows() == 0 && linewidth == nullptr &&
             !inPopulations.empty() && !outPopulations.empty()) {
    switchCase = 1;
  } else if (theMatrix.rows() == 0 && linewidth != nullptr &&
             inPopulations.empty() && outPopulations.empty()) {
    switchCase = 2;
  } else {
    Error("Developer error: builderPh found a non-supported case");
  }

  if ((linewidth != nullptr) && (linewidth->dimensionality != 1)) {
    Error("Developer error: The linewidths shouldn't have dimensionality!");
  }

  // add in the different scattering contributions -------------------

  // precompute the Bose factors TODO check these
  Eigen::MatrixXd outerBose = precomputeOccupations(outerBandStructure);
  Eigen::MatrixXd innerBose = precomputeOccupations(innerBandStructure);

  // generate the points on which these processes will be computed
  std::vector<std::tuple<std::vector<int>, int>> qPairIterator =
                                        getIteratorWavevectorPairs(switchCase);

  // here we call the function to add ph-ph scattering
  {
    // read this in and let it go out of scope afterwards 
    Interaction3Ph coupling3Ph = 
        IFC3Parser::parse(context, innerBandStructure.getPoints().getCrystal());

    addPhPhScattering(*this, context, inPopulations, outPopulations,
                                    switchCase, qPairIterator,
                                    innerBose, outerBose,
                                    innerBandStructure, outerBandStructure,
                                    *phononH0, coupling3Ph, linewidth);
  }

  // NOTE: this is very slightly OMP thread num dependent.
  // That's because it uses the phonon eigenvectors, which can be slightly
  // different phase-wise when diagonalized with a different number of threads
  // Isotope scattering
  if (context.getWithIsotopeScattering()) {
    addIsotopeScattering(*this, context, inPopulations, outPopulations,
                                  switchCase, qPairIterator,
                                  innerBose, outerBose,
                                  innerBandStructure, outerBandStructure,
                                  linewidth);
  }


  // Add boundary scattering
  if (!std::isnan(context.getBoundaryLength())) {
    if (context.getBoundaryLength() > 0.) {
      addBoundaryScattering(*this, context, inPopulations, outPopulations,
                                  switchCase, outerBandStructure, linewidth);
    }
  }

  // TODO add phel scattering
  //if(!context.getElphFileName().empty()) {
      // this is a weird case because it requires another band structure object,
      // which we don't have access to in the regular phonon class here.
      //
      // we could make it so that the phononElectron scattering object... which maybe
      // will also compute the drag terms, does all the creating of the band structure and whatnot that
      // is currently gumming up phonon_transport_app...
  //

  // MPI reduce the distributed data now that all the scattering is accounted for
  if (switchCase == 1) {
    for (auto & outPopulation : outPopulations) {
      mpi->allReduceSum(&outPopulation.data);
    }
  } else {
    mpi->allReduceSum(&linewidth->data);
    if(outputUNTimes) {
      mpi->allReduceSum(&internalDiagonalUmklapp->data);
      mpi->allReduceSum(&internalDiagonalNormal->data);
    }
  }

  // Average over degenerate eigenstates.
  // we turn it off for now and leave the code if needed in the future << what does this mean?
  if (switchCase == 2) {
    degeneracyAveragingLinewidths(linewidth);
    if(outputUNTimes) {
      degeneracyAveragingLinewidths(internalDiagonalUmklapp);
      degeneracyAveragingLinewidths(internalDiagonalNormal);
    }
  }

  // recalculate the phonon linewidths from the off diagonals 
  //a2Omega(); // TODO remove
  //reinforceLinewidths();

  // some phonons like acoustic modes at the gamma, with omega = 0,
  // might have zero frequencies, and infinite populations. We set those
  // matrix elements to zero.
  if (switchCase == 0) {
    // case of matrix construction
    if (context.getUseSymmetries()) {
      for (auto iBte1 : excludeIndices) {
        linewidth->data.col(iBte1).setZero();
        for (auto iBte2 : excludeIndices) {
          for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
              BteIndex iBte1Idx(iBte1);
              BteIndex iBte2Idx(iBte2);
              CartIndex iCart1(i);
              CartIndex iCart2(j);
              int iMat1 = getSMatrixIndex(iBte1Idx, iCart1);
              int iMat2 = getSMatrixIndex(iBte2Idx, iCart2);
              theMatrix(iMat1, iMat2) = 0.;
            }
          }
        }
      }
    } else {
      for (auto iBte1 : excludeIndices) {
        linewidth->data.col(iBte1).setZero();
        for (auto iBte2 : excludeIndices) {
          theMatrix(iBte1, iBte2) = 0.;
        }
      }
    }
  } else if (switchCase == 1) {
    // case of matrix-vector multiplication
    for (auto iBte1 : excludeIndices) {
      for (auto & outPopulation : outPopulations) {
        outPopulation.data.col(iBte1).setZero();
      }
    }

  } else if (switchCase == 2) {
    // case of linewidth construction
    for (auto iBte1 : excludeIndices) {
      linewidth->data.col(iBte1).setZero();
      // TODO may need to toss U and N indices here?
    }
  }

  // we place the linewidths back in the diagonal of the scattering matrix
  // this because we may need an MPI_allReduce on the linewidths
  if (switchCase == 0) { // case of matrix construction
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
    } else {
      for (int is = 0; is < numStates; is++) {
        theMatrix(is, is) = linewidth->operator()(iCalc, 0, is);
      }
    }
  }
}



