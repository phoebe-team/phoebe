//#include "ph_scattering_matrix.h"
#include "constants.h"
#include "helper_3rd_state.h"
#include "io.h"
#include "mpiHelper.h"
#include <cmath>
#include "parser.h"
#include "general_scattering.h"
#include "ph_scattering.h"
#include "phel_scattering.h"
#include "ifc3_parser.h"
#include "interaction_elph.h"

PhScatteringMatrix::PhScatteringMatrix(Context &context_,
                                       StatisticsSweep &statisticsSweep_,
                                       BaseBandStructure &innerBandStructure_,
                                       BaseBandStructure &outerBandStructure_,
                                       PhononH0 *phononH0_)
    : ScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
     BasePhScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
     phononH0(phononH0_) {

  if (&innerBandStructure != &outerBandStructure && phononH0 == nullptr) {
    DeveloperError("PhScatteringMatrix needs phononH0 for incommensurate grids");
  }
}

void PhScatteringMatrix::builder(std::shared_ptr<VectorBTE> linewidth,
                                 std::vector<VectorBTE> &inPopulations,
                                 std::vector<VectorBTE> &outPopulations) {

  if(mpi->mpiHead()) 
    std::cout << "============== Building phonon scattering matrix ==============\n" << std::endl; 

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
    DeveloperError("builderPh found a non-supported case");
  }

  if ((linewidth != nullptr) && (linewidth->dimensionality != 1)) {
    DeveloperError("The linewidths shouldn't have dimensionality!");
  }

  // add in the different scattering contributions -------------------

  // precompute the Bose factors TODO check these
  Eigen::MatrixXd outerBose = precomputeOccupations(outerBandStructure);
  Eigen::MatrixXd innerBose = precomputeOccupations(innerBandStructure);

  // generate the points on which these processes will be computed
  std::vector<std::tuple<std::vector<int>, int>> qPairIterator =
                                        getIteratorWavevectorPairs(switchCase);

  Crystal crystal = innerBandStructure.getPoints().getCrystal();

  // here we call the function to add ph-ph scattering
  if(!context.getPhFC3FileName().empty()) {
    // read this in and let it go out of scope afterwards 
    Interaction3Ph coupling3Ph = IFC3Parser::parse(context, crystal);

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

  // MPI reduce the distributed data 
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

  if(!context.getElphFileName().empty()) {

    // output the phph linewidths
    outputLifetimesToJSON("rta_phph_relaxation_times.json", linewidth);

    // IMPORTANT NOTE: the ph-el scattering does not receive symmetrization factor
    // because it doesn't have these factors of n(n+1) in the scattering rates.
    // Therefore, we should symmetrize here, then add these term afterwards.
    // Only needs to be done if matrix is in memory already 
    if(highMemory) a2Omega();
    mpi->barrier(); // need to finish this before adding phel scattering

    // later add these to the linewidths
    std::shared_ptr<VectorBTE> phelLinewidths =
         std::make_shared<VectorBTE>(statisticsSweep, outerBandStructure, 1);

    // load the elph coupling
    auto couplingElPh = InteractionElPhWan::parse(context, crystal, *phononH0);

    // load electron band structure
    auto t = Parser::parseElHarmonicWannier(context, &crystal);
    auto crystalEl = std::get<0>(t);
    auto electronH0 = std::get<1>(t);
    Points fullPoints(crystal, context.getKMesh());

    auto tup = ActiveBandStructure::builder(context, electronH0, fullPoints);
    auto elBandStructure = std::get<0>(tup);
    auto elStatisticsSweep = std::get<1>(tup);

    // check that the crystal in the elph calculation is the
    // same as the one in the phph calculation
    if (crystal.getDirectUnitCell() != crystalEl.getDirectUnitCell()) {
      Warning("Phonon-electron scattering requested, "
              "but crystals used for ph-ph and \n"
              "ph-el scattering are not the same!");
    }

    // Phel gerates it's k-q pair iterator, as it's only a
    // linewidth calculation and therefore can be parallelized differently.
    // NOTE: this does not update the Smatrix diagonal, only linewidth object. Therefore,
    // requires the replacing of the linewidths object into the SMatrix diagonal at the
    // end of this function
    addPhElScattering(*this, context, 
                      innerBandStructure, elBandStructure, elStatisticsSweep,  
                      couplingElPh, phelLinewidths);

    // all reduce the calculated phel linewidths 
    mpi->allReduceSum(&phelLinewidths->data);

    // output these phel linewidths
    outputLifetimesToJSON("rta_phel_relaxation_times.json", phelLinewidths);

    // Add in the phel contribution
    // TODO better to just add the vectorBTE objects? 
    linewidth->data = linewidth->data + phelLinewidths->data;

    //std::cout << phelLinewidths->data << std::endl;

    // convert the matrix back to A to carry on as usual
    if(highMemory) omega2A(); 

  }// braces to have elph coupling go out of scope

  // Average over degenerate eigenstates.
  if (switchCase == 2) {
    degeneracyAveragingLinewidths(linewidth);
    if(outputUNTimes) {
      degeneracyAveragingLinewidths(internalDiagonalUmklapp);
      degeneracyAveragingLinewidths(internalDiagonalNormal);
    }
  }

  // recalculate the phonon linewidths from the off diagonals 
  //a2Omega(); // TODO remove
  // we should do this if phel is not involved, otherwise it wipes out phel 
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
          // note this is plus equals because it needs to be!
          // look closely at how the rates are written in Fugallo et al.
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



