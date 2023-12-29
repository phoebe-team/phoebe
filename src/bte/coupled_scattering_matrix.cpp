#include "coupled_scattering_matrix.h"
#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include <cmath>
#include "coupled_vector_bte.h"
#include "phel_scattering.h"
#include <map>

CoupledScatteringMatrix::CoupledScatteringMatrix(Context &context_,
                                      StatisticsSweep &statisticsSweep_,
                                      BaseBandStructure &innerBandStructure_, // phonon
                                      BaseBandStructure &outerBandStructure_, // electron
                                      Interaction3Ph *coupling3Ph_,
                                      InteractionElPhWan *couplingElPh_,
                                      ElectronH0Wannier *electronH0_,
                                      PhononH0 *phononH0_)
    : ScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
      BaseElScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
      BasePhScatteringMatrix(context_, statisticsSweep_, innerBandStructure_, outerBandStructure_),
        coupling3Ph(coupling3Ph_), couplingElPh(couplingElPh_), electronH0(electronH0_), phononH0(phononH0_) {

  // we need to overwrite num states as well as the internalDiagonal object,
  // as both of these are now different than the standard smatrix
  internalDiagonal = std::make_shared<CoupledVectorBTE>(statisticsSweep, innerBandStructure_, outerBandStructure_, 1);
  // set up the MRTA container
  linewidthMR = std::make_shared<VectorBTE>(statisticsSweep, outerBandStructure, 1);

  numStates = internalDiagonal->getNumStates();
  numPhStates = int(innerBandStructure.irrStateIterator().size());
  numElStates = int(outerBandStructure.irrStateIterator().size());
  isCoupled = true;
  // TODO this is only actually true after we call phononOnlyA2Omega at the bottom of the scattering
  // process addition section.
  // Otherwise, because ph scattering not symmetrized and el scattering is symmetrized,
  // we would have a very mismatched matrix
  isMatrixOmega = true;

  // inner band structure is the phonon one, so we set exclude indices
  // TODO replace this with the new global one
  if (innerBandStructure.getParticle().isPhonon()) {
    for (int iBte = 0; iBte < numPhStates; iBte++) {
      auto iBteIdx = BteIndex(iBte);
      StateIndex isIdx = innerBandStructure.bteToState(iBteIdx);
      double en = innerBandStructure.getEnergy(isIdx);
      // TODO need to standardize this
      if (en < 0.1 / ryToCmm1) { // cutoff at 0.1 cm^-1
        excludeIndices.push_back(iBte);
      }

      Eigen::Vector3d k = innerBandStructure.getWavevector(isIdx);
      if (k.squaredNorm() > 1e-8 && en < 0.) {
        Warning("Found a phonon mode q!=0 with negative energies. "
                "Consider improving the quality of your DFT phonon calculation.\n");
      }
    }
  }

  // we only output U and N in RTA, as otherwise we would need to do either:
  // 1) the full scattering matrix calc multiple times
  // 2) store 3 copies of the scattering matrix,
  // both of which are bad options
  if ( context.getOutputUNTimes() ) {

    outputUNTimes = true;
    internalDiagonalNormal = std::make_shared<CoupledVectorBTE>(statisticsSweep, innerBandStructure, outerBandStructure, 1);
    internalDiagonalUmklapp = std::make_shared<CoupledVectorBTE>(statisticsSweep, innerBandStructure, outerBandStructure, 1);

    // warn users that if they run an exact bte solver, the U and N will still only come out in RTA
    if( context.getSolverBTE().size() > 0 ) {
      Warning("You've set outputUNTimes to true in input file, but requested an exact BTE solver.\n"
        "Be aware that U and N separation is only currently implemented in the RTA case.");
    }
  }

  // enforce proper bandstructure definitions
  if(!innerBandStructure.getParticle().isPhonon() &&
                        !outerBandStructure.getParticle().isElectron()) {
    Error("Developer error: Tried to create CMatrix with bandstructures of wrong particle type!");
  }
  // scattering matrix also must be in memory
  if(!highMemory) {
    Error("Developer error: Cannot construct coupled matrix without full matrix in memory.");
  }
  // block symmetry use as relaxons solver cannot benefit from this,
  // and relaxons are the only point of this matrix
  if (context.getUseSymmetries()) {
    Error("Developer error: Currently cannot use symmetry for the calculation of the coupled scattering matrix.");
  }
}

void CoupledScatteringMatrix::builder(std::shared_ptr<VectorBTE> linewidth,
                                 std::vector<VectorBTE> &inPopulations,
                                 std::vector<VectorBTE> &outPopulations) {

  // here, inPop and outPop are empty. Linewidth should be a coupledVectorBTE object.

  // by defintion this is switch case 0 -- matrix construction only.
  // We will never have only linewidth or matrix-vector product
  int switchCase = 0;

  // internal diagonal should be allocated, as well as the matrix
  if (linewidth == nullptr || theMatrix.rows() == 0) {
    Error("Developer error: Attempted to construct coupled bte scattering matrix without linewidths or matrix.");
  }
  // linewidths have to be a certain shape here
  if ((linewidth != nullptr) && (linewidth->dimensionality != 1)) {
    Error("Developer error: The linewidths shouldn't have dimensionality!");
  }

  // add in the different scattering contributions -------------------

  // precompute the fermi and bose factors
  Eigen::MatrixXd boseOccupations = precomputeOccupations(innerBandStructure);
  Eigen::MatrixXd fermiOccupations = precomputeOccupations(outerBandStructure);

  // generate the points on which these processes will be computed
  // these points are the pairs local to each MPI process, so things should be
  // well distributed. These points have the indices of the bandstructure objects rather than
  // the scattering matrix objects, as that's what is needed by the functions.
  // The indices are shifted to the coupled scattering matrix indices in each funciton
  // just before it's written to the matrix.
  auto allPairIterators = getIteratorWavevectorPairs();
  std::vector<std::tuple<std::vector<int>, int>> kPairIterator = allPairIterators[0];
  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator = allPairIterators[1];
  std::vector<std::tuple<std::vector<int>, int>> qkPairIterator = allPairIterators[2];
  std::vector<std::tuple<std::vector<int>, int>> qPairIterator = allPairIterators[3];

  // add el-ph scattering -----------------------------------------
  addElPhScattering(*this, context, inPopulations, outPopulations,
                                  switchCase, kPairIterator,
                                  fermiOccupations,
                                  outerBandStructure, outerBandStructure,
                                  *phononH0, couplingElPh, internalDiagonal);

  // add charged impurity electron scattering  -------------------
/*  addChargedImpurityScattering(*this, context, inPopulations, outPopulations,
                       switchCase, kPairIterator,
                       innerBandStructure, outerBandStructure, linewidth);
*/
  // add ph-ph scattering ------------------------------------
  addPhPhScattering(*this, context, inPopulations, outPopulations,
                                  switchCase, qPairIterator,
                                  boseOccupations, boseOccupations,
                                  innerBandStructure, innerBandStructure,
                                  *phononH0, coupling3Ph, linewidth);

  // Isotope scattering ---------------------------------
  if (context.getWithIsotopeScattering()) {
    addIsotopeScattering(*this, context, inPopulations, outPopulations,
                            switchCase, qPairIterator,
                            boseOccupations, boseOccupations,
                            innerBandStructure, innerBandStructure, internalDiagonal);
  }

  // TODO check boundary scattering
  // Add boundary scattering ----------------------
  // Call this twice for each section of the diagonal,
  // in one case handing it the phonon bands, in the other the electron bands.
  if (!std::isnan(context.getBoundaryLength())) {
    if (context.getBoundaryLength() > 0.) {
      // phonon boundary scattering
      addBoundaryScattering(*this, context, inPopulations, outPopulations,
                            switchCase, innerBandStructure, linewidth);
      // electron boundary scattering
      addBoundaryScattering(*this, context, inPopulations, outPopulations,
                            switchCase, outerBandStructure, linewidth);
    }
  }

  // MPI reduce the distributed data now that all the scattering is accounted for
  mpi->allReduceSum(&linewidth->data);

  // unfortunately because of how things are setup, we need to call this function on the
  // phonon part of the matrix.
  // The electron part is by default symmetrized Omega matrix, but the phonon part is A
  // and must be converted.
  // The drag terms are also already symmetrized.
  // IMPORTANT NOTE: the ph-el scattering does not receive symmetrization because
  // it doesn't have these factors of n(n+1) in the scattering rates.
  // Additionally, the drag terms are already symmetrized
  // Therefore, we should symmetrize here, then add these term afterwards.
  phononOnlyA2Omega();
  mpi->barrier(); // need to finish this before adding phel scattering

  {
  // here we define a separate CoupledVectorBTE object -- the original
  // one had to be all reduced pre-symmetrization. We define this to
  // later add these post-symmetrization terms to the linewidths
  std::shared_ptr<CoupledVectorBTE> postSymLinewidths =
         std::make_shared<CoupledVectorBTE>(statisticsSweep, innerBandStructure, outerBandStructure, 1);

  // Add phel scattering ---------------------------------------
  // Because this has very different convergence than the standard transport,
  // it calculates internally a third, denser el bandstructure
  // It also internally generates it's k-q pair iterator, as it's only a
  // linewidth calculation and therefore can be parallelized differently.
  // NOTE: this does not update the Smatrix diagonal, only linewidth object. Therefore,
  // requires the replacing of the linewidths object into the SMatrix diagonal at the
  // end of this function
  addPhElScattering(*this, context, innerBandStructure, electronH0, couplingElPh, postSymLinewidths);
  mpi->barrier();

  // all reduce the calculated phel linewidths 
  mpi->allReduceSum(&postSymLinewidths->data);
  // TODO maybe output these phel linewidths? 

  // Add drag terms ----------------------------------------------
  if(context.getUseDragTerms()) { 
    // first add the el drag term 
    // TODO replace these 0 and 1s with something smarter 
    addDragTerm(*this, context, kqPairIterator, 0, electronH0,
                         couplingElPh, innerBandStructure, outerBandStructure);
    // now the ph drag term
    addDragTerm(*this, context, qkPairIterator, 1, electronH0,
                          couplingElPh, innerBandStructure, outerBandStructure);
  }

  // use drag ASR to correct the dra terms and recompute the phel linewidths 
  phononElectronAcousticSumRule(*this, context, postSymLinewidths, // phel linewidths
                                outerBandStructure,   // electron bands
                                innerBandStructure);  // phonon bands 

  // Add in the phel contribution
  // NOTE: would be nicer to use the add operatore from VectorBTE, but inheritance
  // is causing trouble -- Jenny
  linewidth->data = linewidth->data + postSymLinewidths->data;

  }// braces to have postSymLinewidths go out of scope

// TODO do we need to keep this here
  /*if(outputUNTimes) {
    mpi->allReduceSum(&internalDiagonalUmklapp->data);
    mpi->allReduceSum(&internalDiagonalNormal->data);
  }*/

  // Average over degenerate eigenstates.
  // we turn it off for now and leave the code if needed in the future << what does this mean?
  // TODO why is this only in switchcase 2? Feels like it should absolutely also be in 0,
  // before things are re-saved to the diagonal
  if (switchCase == 2) {
    degeneracyAveragingLinewidths(linewidth);
    if(outputUNTimes) {
      degeneracyAveragingLinewidths(internalDiagonalUmklapp);
      degeneracyAveragingLinewidths(internalDiagonalNormal);
    }
  }

  // TODO maybe make these functions of the parent scattering matrix?
  // we repeat the same thing here every time..

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
              // in coupled bte, we don't want to wipe out
              // el-el states in the first quadrant in this loop.
              // They actually aren't affected by the excludeIndices.
              if(iBte1Idx.get() < numElStates && iBte2Idx.get() < numElStates) {
                continue;
              }
              int iMat1 = getSMatrixIndex(iBte1Idx, iCart1);
              int iMat2 = getSMatrixIndex(iBte2Idx, iCart2);
              if(theMatrix.indicesAreLocal(iMat1,iMat2)) theMatrix(iMat1, iMat2) = 0.;
            }
          }
        }
      }
    } else {
      for (auto iBte1 : excludeIndices) {
        if(iBte1 < numElStates) {
          continue;
        }
        linewidth->data.col(iBte1).setZero();
        for (auto iBte2 : excludeIndices) {
          // in coupled bte, we don't want to wipe out
          // el-el states in the first quadrant in this loop.
          // They actually aren't affected by the excludeIndices.
          if(iBte1 < numElStates && iBte2 < numElStates) {
           continue;
          }
          if(theMatrix.indicesAreLocal(iBte1,iBte2)) theMatrix(iBte1, iBte2) = 0.;
        }
      }
    }
  }

  // TODO debug the "replaceLinewidths" function and use it instead
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
            if(theMatrix.indicesAreLocal(iMati,iMati))  theMatrix(iMati, iMatj) = 0.;
          }
          if(theMatrix.indicesAreLocal(iMati,iMati)) theMatrix(iMati, iMati) += linewidth->operator()(iCalc, 0, iBte);
        }
      }
    } else {
      for (int is = 0; is < numStates; is++) {
        if(theMatrix.indicesAreLocal(is,is)) theMatrix(is, is) = linewidth->operator()(iCalc, 0, is);
      }
    }
  }

  // apply the spin degen factors
  reweightQuadrants();

  // use the off diagonals to calculate the linewidths, 
  // to ensure the special eigenvectors can be found/preserve conservation of momentum 
  // that might be ruined by the delta functions 
  reinforceLinewidths();

  if(mpi->mpiHead()) std::cout << "\nFinished computing the coupled scattering matrix." << std::endl;

}

// set,unset the scaling of omega = A/sqrt(bose1*bose1+1)/sqrt(bose2*bose2+1)
// There's a more elegant way to do this, but the simplest way for now
// is to use the scattering rate functions the way they are (
void CoupledScatteringMatrix::phononOnlyA2Omega() {

  if(context.getUseSymmetries()) {
    Error("Developer error: Cannot use phononOnlyA2Omega with sym!");
  }

  int iCalc = 0; // as there can only be one temperature

  auto particle = innerBandStructure.getParticle();
  auto calcStatistics = statisticsSweep.getCalcStatistics(iCalc);
  double temp = calcStatistics.temperature;
  double chemPot = 0;

  auto allLocalStates = theMatrix.getAllLocalStates();
  size_t numAllLocalStates = allLocalStates.size();
#pragma omp parallel for
  for (size_t iTup=0; iTup<numAllLocalStates; iTup++) {

    auto tup = allLocalStates[iTup];

    // when there are no symmetries, ibte = imat
    // However, for band structure access,
    // remember that these states are in quadrant 4 for ph-self,
    // and need to be shifted back to bandstructure indices
    int iBte1 = std::get<0>(tup);
    int iBte2 = std::get<1>(tup);

    // we skip any state that's not a phonon one
    if(iBte1 < numElStates || iBte2 < numElStates) {
      continue;
    }
    // TODO excludeIndices... are they for ph indices or global ones?
    // skip any excluded states to avoid divergences
    if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) !=
        excludeIndices.end())
      continue;
    if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte2) !=
        excludeIndices.end())
      continue;

    int iBtePhBands1 = iBte1 - numElStates;
    int iBtePhBands2 = iBte2 - numElStates;
    BteIndex iBte1Idx(iBtePhBands1);
    BteIndex iBte2Idx(iBtePhBands2);

    StateIndex is1Idx = innerBandStructure.bteToState(iBte1Idx);
    StateIndex is2Idx = innerBandStructure.bteToState(iBte2Idx);
    double en1 = innerBandStructure.getEnergy(is1Idx);
    double en2 = innerBandStructure.getEnergy(is2Idx);

    // n(n+1) for bosons, n(1-n) for fermions
    double term1 = particle.getPopPopPm1(en1, temp, chemPot);
    double term2 = particle.getPopPopPm1(en2, temp, chemPot);

    if (iBte1 == iBte2) {
      internalDiagonal->operator()(0, 0, iBte1) /= term1;
    }
    theMatrix(iBte1, iBte2) /= sqrt(term1 * term2);
  }
}


/* return irr k index from a matrix state index
 * helper function to simplify the below function */
int CoupledScatteringMatrix::bteStateToWavevector(BteIndex& iBte, BaseBandStructure& bandStructure) {

    // map the index on the irr points of BTE to band structure index
    StateIndex is = bandStructure.bteToState(iBte);
    auto tup2 = bandStructure.getIndex(is);
    WavevectorIndex ikIndex = std::get<0>(tup2);
    int ikIrr = ikIndex.get();
    return ikIrr;
}

void addWavevectorToMap(std::unordered_map<int,std::vector<int>>& pairMap, int& ik1, int& ik2) {

  if(pairMap.find(ik1) == pairMap.end()) { // kpoint isn't in list, add it
    std::vector<int> temp = {ik2};
    pairMap[ik1] = temp;
  } else { // it's in the map and we need to check if k2 is as well
           // which can happen because there are duplicates due to bands
    if (std::find(pairMap[ik1].begin(),pairMap[ik1].end(),ik2) == pairMap[ik1].end()) { // k2 not in the list
      pairMap[ik1].push_back(ik2);
    }
    // final possibility is it's already in the list
  }
}

// helper function to add some dummy indices to each
// MPI procs iterator of indices to make sure they have the same number
// If this isn't the case, a pooled calculation will hang
void mpiPoolsIteratorCorrection(std::vector<std::tuple<std::vector<int>, int>>& pairIterator) {

  if (mpi->getSize(mpi->intraPoolComm) > 1) {
    auto myNumK1 = int(pairIterator.size());
    int numK1 = myNumK1;
    mpi->allReduceMax(&numK1, mpi->intraPoolComm);

    while (myNumK1 < numK1) {
      std::vector<int> dummyVec;
      dummyVec.push_back(-1);
      auto tt = std::make_tuple(dummyVec,-1);
      pairIterator.push_back(tt);
      myNumK1++;
    }
  }
}

std::vector<std::vector<std::tuple<std::vector<int>, int>>>
  CoupledScatteringMatrix::getIteratorWavevectorPairs([[maybe_unused]] const int& switchCase,
                                                      [[maybe_unused]] const bool& rowMajor) {

  // this function gets the k,k, k,q, or q,q pairs needed to calculate scattering rates
  // for the coupled scattering matrix. The matrix is always in memory,
  // so we use the function to get the local states and sort them for return.

  // We need to first shift the matrix states back to the states that correspond
  // to their bandstructure objects

  // get all the local matrix states
  auto matrixStateIter = theMatrix.getAllLocalStates();

  // Coupled matrix has the special case of equal bandstructures (for quadrants)
  // with scattering matrix in mem

  // maps of k and q states for self terms
  std::unordered_map<int, std::vector<int>> kPairMap;
  std::unordered_map<int, std::vector<int>> qPairMap;
  // maps of k,q pairs and vice versa for drag terms
  std::unordered_map<int, std::vector<int>> qkPairMap;
  std::unordered_map<int, std::vector<int>> kqPairMap;

  // select out the states in each quadrant
  for(auto matrixState : matrixStateIter) {

    // unpack the state info into matrix indices
    int iMat1 = std::get<0>(matrixState);
    int iMat2 = std::get<1>(matrixState);

    // convert to BTE indices (just removing cartesian direction, if sym was present)
    auto tup = getSMatrixIndex(iMat1);
    BteIndex iBte1 = std::get<0>(tup);
    tup = getSMatrixIndex(iMat2);
    BteIndex iBte2 = std::get<0>(tup);

    // if it's the el-el one, we do nothing. s1 = el, s2 = el
    if (iBte1.get() < numElStates && iBte2.get() < numElStates) {
      int ik1 = bteStateToWavevector(iBte1, outerBandStructure);
      int ik2 = bteStateToWavevector(iBte2, outerBandStructure);
      // call this to calculate only the upper triangle
      if(ik1 > ik2 && !context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) continue;
      addWavevectorToMap(kPairMap, ik1, ik2);
    }
    // quadrant el-ph drag, upper right. s1 = el, s2 = ph
    else if(iBte1.get() < numElStates && iBte2.get() >= numElStates) {
      iBte2 = BteIndex(iBte2.get() - numElStates);
      int ik1 = bteStateToWavevector(iBte1, outerBandStructure);
      int iq2 = bteStateToWavevector(iBte2, innerBandStructure);
      if(ik1 > iq2+numElStates && !context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) continue;
      addWavevectorToMap(kqPairMap, ik1, iq2);
    }
    // quadrant ph-el drag, lower left. s1 = ph, s2 = el
    else if(iBte1.get() >= numElStates && iBte2.get() < numElStates) {
      iBte1 = BteIndex(iBte1.get() - numElStates);
      int iq1 = bteStateToWavevector(iBte1, innerBandStructure);
      int ik2 = bteStateToWavevector(iBte2, outerBandStructure);
      if(iq1+numElStates > ik2 && !context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) continue;
      // need to be switched, as final return will be single k states, q batch
      addWavevectorToMap(qkPairMap, ik2, iq1);
    }
    // quadrant ph self, lower right. s1 = ph, s2 = ph
    else if(iBte1.get() >= numElStates && iBte2.get() >= numElStates) {
      iBte1 = BteIndex(iBte1.get() - numElStates);
      iBte2 = BteIndex(iBte2.get() - numElStates);
      int iq1 = bteStateToWavevector(iBte1, innerBandStructure);
      int iq2 = bteStateToWavevector(iBte2, innerBandStructure);
      if(iq1+numElStates > iq2+numElStates && !context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) {
        continue;
      }
      // important to switch this, as ph-ph scattering expects single q2, q1 batch of states
      addWavevectorToMap(qPairMap, iq2, iq1);
    }
    else {
      Error("Developer error: Somehow we found an out of bounds coupled matrix state.");
    }
  }

  // now that we've collected the maps, convert them into the format that the
  // scattering rate functions require

  // this should be ordered k2indices, k1
  std::vector<std::tuple<std::vector<int>, int>> kPairIterator;
  // this should be ordered q1indices, q2
  std::vector<std::tuple<std::vector<int>, int>> qPairIterator;
  // both drag terms expect indexes as qindices, k
  // NOTE: realize these are indeed different lists, as the states which are local are different
  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator;
  std::vector<std::tuple<std::vector<int>, int>> qkPairIterator;

  for(auto [ik1, ik2Indices] : kPairMap) {
    std::tuple<std::vector<int>,int> temp = std::make_tuple(ik2Indices,ik1);
    kPairIterator.push_back(temp);
  }
  for(auto [ik1, iq2Indices] : kqPairMap) {
    std::tuple<std::vector<int>,int> temp = std::make_tuple(iq2Indices,ik1);
    kqPairIterator.push_back(temp);
  }
  for(auto [ik1, iq2Indices] : qkPairMap) {
    std::tuple<std::vector<int>,int> temp = std::make_tuple(iq2Indices,ik1);
    qkPairIterator.push_back(temp);
  }
  for(auto [iq2, iq1Indices] : qPairMap) {
    std::tuple<std::vector<int>,int> temp = std::make_tuple(iq1Indices,iq2);
    qPairIterator.push_back(temp);
  }

  // Patch to make pooled interpolation of coupling work
  // we need to make sure each MPI process in the pool calls
  // the calculation of the coupling interpolation the same number of times
  // Hence, we correct pairIterator so that it does have the same size across the pool
  mpiPoolsIteratorCorrection(kPairIterator);
  mpiPoolsIteratorCorrection(kqPairIterator);
  mpiPoolsIteratorCorrection(qkPairIterator);

  std::vector<std::vector<std::tuple<std::vector<int>, int>>> returnContainer;
  returnContainer.push_back(kPairIterator);
  returnContainer.push_back(kqPairIterator);
  returnContainer.push_back(qkPairIterator);
  returnContainer.push_back(qPairIterator);
  return returnContainer;
}

// reweight the matrix quadrants
void CoupledScatteringMatrix::reweightQuadrants() {

  // TODO if we use linewidths also apply 2 to them

  double spinFactor = 2.; // nonspin pol = 2
  if (context.getHasSpinOrbit()) { spinFactor = 1.; }

  double Nk = double(context.getKMesh().prod());
  double Nq = double(context.getQMesh().prod());
  //double Nkq = (Nk + Nq)/2.;

  // loop over states and apply the reweighting factors of eq 28
  for(auto matrixState : theMatrix.getAllLocalStates()) {

    // unpack the state info into matrix indices
    int iMat1 = std::get<0>(matrixState);
    int iMat2 = std::get<1>(matrixState);

    // if it's the el-el one, s1 = el, s2 = el, we apply factor of 1/2
    if (iMat1 < numElStates && iMat2 < numElStates) {
      //theMatrix(iMat1, iMat2) *= 1./Nk;
      theMatrix(iMat1, iMat2) *= 1.; //1./spinFactor; // TODO put this back later, scattering matrix is missing a 2 that this is compensating for
                                                      // but I would have to add this 2 into the scattering matrix for electrons, which would
                                                      // trash all the normal solves which are already compensating for this in later parts of the code
    }
    // quadrant el-ph drag, upper right. s1 = el, s2 = ph, apply 1/sqrt(2)
    else if(iMat1 < numElStates && iMat2 >= numElStates) {
      //theMatrix(iMat1, iMat2) *= 1./Nq; //1./sqrt(spinFactor) * sqrt(Nk / Nq);
      theMatrix(iMat1, iMat2) *= 1./sqrt(spinFactor) * sqrt(Nk / Nq);
    }
    // quadrant ph-el drag, lower left. s1 = ph, s2 = el, apply 1/sqrt(2)
    else if(iMat1 >= numElStates && iMat2 < numElStates) {
      //theMatrix(iMat1, iMat2) *= 1./Nk; //1./sqrt(spinFactor) * sqrt(Nq / Nk);
      theMatrix(iMat1, iMat2) *= 1./sqrt(spinFactor) * sqrt(Nq / Nk);
    }
    // quadrant ph self, lower right. s1 = ph, s2 = ph
//    else if(iMat1 >= numElStates && iMat2 >= numElStates) {
//      //theMatrix(iMat1, iMat2) *=  1./Nq; //1.; // do nothing
//      theMatrix(iMat1, iMat2) *=  1.; // do nothing
//    }
//    else {
//      Error("Developer error: Somehow we found an out of bounds coupled matrix state in reweight.");
//    }
  }
}



BaseBandStructure* CoupledScatteringMatrix::getPhBandStructure() { return &innerBandStructure; }
BaseBandStructure* CoupledScatteringMatrix::getElBandStructure() { return &outerBandStructure; }


