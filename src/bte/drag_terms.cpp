#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include "periodic_table.h"
#include "interaction_elph.h"
#include "drag_terms.h"
#include "vector_bte.h"

const double phEnergyCutoff = 0.001 / ryToCmm1; // discard states with small omega
                                                // to avoid divergences

void addDragTerm(CoupledScatteringMatrix &matrix, Context &context,
                  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator,
                  const int dragTermType, // 0 for el (kn, qs), 1 for ph drag (qs,kn)
                  ElectronH0Wannier *electronH0,
                  InteractionElPhWan *couplingElPhWan,
                  BaseBandStructure &phononBandStructure, // phonon
                  BaseBandStructure &electronBandStructure) { // electron

  // throw errors if wrong bandstructure types are provided 
  if(!phononBandStructure.getParticle().isPhonon() || !electronBandStructure.getParticle().isElectron()) {
    Error("Developer error: either ph or el bandstructure supplied to the "
                "drag calculation is the wrong type!");
  }

  // Notes:
  // The matrix we are trying to fill in here is the coupled BTE matrix
  // * We are trying to fill here the upper right hand side of it, D_{el-ph},
  // and the lower left hand side of it, D_{ph-el}, as
  // these calculations are very structurally similar.
  // D_ph has dimensions qa, km, and D_el has dimensions km, qa --
  // both sum over intermediate states k',m'
  //
  // kqPairIterator will tell us the qa, km states for both of these

  // our three states will be:
  // k = k, initial state for el, final state for ph
  // q = q, final state for el, initial for ph
  // kp = k', intermediate state we are summing over

/* In pseudocode, the below follows as: 

  loop over k {
    # Fourier transform + rotation on elph matrix elements related to k

 	loop over a batch of q points {

 		# set up phonon quantities for q 
 		# set up electronic quantities for k'+

 		loop over k'+ points {
 			# sum up k'+ part of the D term
 			  # electronic quantities for k'-
 			  # the matrix elements g(k,k'+,q)
 		} 

 		# set up electronic quantities for k'-
 		# generate elph matrix elements for g(k,k'-,q)

 		loop over k'- points {
 			# sum up k'- part of the D term, using 
 			  # electronic quantities for k'-
 			  # the matrix elements g(k,k'-,q)
 		}
 		matrixElement = (sum_(k'+) contribution) - (sum_(k'-) contribution)
 	}
 }
 */

  StatisticsSweep &statisticsSweep = matrix.statisticsSweep;

  // becaues the summation is a BZ integral over kpoints, we use the electronic mesh
  // for smearing
  DeltaFunction *smearing = DeltaFunction::smearingFactory(context, electronBandStructure);
  if (smearing->getType() == DeltaFunction::tetrahedron) {
    DeveloperError("Tetrahedron smearing for transport untested and thus blocked");
  }

  // set up basic properties
  int numCalculations = statisticsSweep.getNumCalculations();
  bool withSymmetries = context.getUseSymmetries(); 
  if (withSymmetries) { // NOTE, not allowed with sym for now
    DeveloperError("Drag term calculation is not implemented with symmetry.");
  }

  // TODO this should be the correct norm, but worth double checking
  double norm = 0.;
  if (dragTermType == Del) { norm = 1. / context.getKMesh().prod(); }
  else { norm = 1. / context.getKMesh().prod(); }

  // TODO change this to the same in phel scattering as well
  // precompute the q-dependent part of the polar correction
  Eigen::MatrixXcd polarData = couplingElPhWan->precomputeQDependentPolar(phononBandStructure);

  // set up the loopPrint object which prints out progress 
  std::string dragName = (dragTermType == Del) ? "el-ph" : "ph-el";
  LoopPrint loopPrint("computing " + dragName + " drag terms ",
                                     "k,q pairs", int(kqPairIterator.size()));

  // loop over final and initial state pairs

  // the outer loop goes over the initial kpoint, and the internal 
  // loop is over batches of q points (or equivalently because of conservation of momentum,
  // over batches k' points). 
  //
  // We use k in the outer loop, as the InteractionElPhWan object is set up
  // to do the first rotation + FT over a k vector, followed by batches 
  // of qpoints for the second rotation + FT
  for (auto pair : kqPairIterator) {

    // update the progress bar 
    loopPrint.update();

    // unpack the k index and the corresponding list of qpoint indices
    int ik = std::get<1>(pair);
    WavevectorIndex ikIdx(ik);
    auto iQIndexes = std::get<0>(pair);

    // Dummy call to make pooled coupling calculation work. 
    // We need to make sure calcCouplingSquared is called the same # of times. 
    // This is also taken care of while generating the indices. 
    // Here we call calcCoupling. This block is useful if, for example, 
    // we have a pool of size 2, and the 1st MPI  process has 7 k-points, 
    // the 2nd MPI process has 6. This block makes
    // the 2nd process call calcCouplingSquared 7 times as well.
    if (ik == -1) {

      Eigen::Vector3d kCartesian = Eigen::Vector3d::Zero();
      int numWannier = couplingElPhWan->getCouplingDimensions()(4);
      Eigen::MatrixXcd eigenVectorK = Eigen::MatrixXcd::Zero(numWannier, 1);
      couplingElPhWan->cacheElPh(eigenVectorK, kCartesian);
      // since this is just a dummy call used to help other MPI processes
      // compute the coupling, and not to compute matrix elements, we can skip
      // to the next loop iteration
      continue;
    }

    // Set the electronic state energies, velocities, eigenvectors for initial k state
    // from the provided electronic band structure 
    Eigen::Vector3d kCartesian = electronBandStructure.getWavevector(ikIdx);
    Eigen::VectorXd stateEnergiesK = electronBandStructure.getEnergies(ikIdx);
    int nbK = int(stateEnergiesK.size());
    Eigen::MatrixXd vKs = electronBandStructure.getGroupVelocities(ikIdx);
    Eigen::MatrixXcd eigenVectorK = electronBandStructure.getEigenvectors(ikIdx);

    // perform the first fourier transform + rotation for state k
    couplingElPhWan->cacheElPh(eigenVectorK, kCartesian);

    // prepare batches of intermediate states, kp, which are determined by the points helper
    // by memory usage
    int nq = int(iQIndexes.size());

    // TODO: this was written to take nk batches, but the function should work the same
    // way. Double check this.
    // Number of batches of k' states which will be used for this k,q pair
    int numBatches = couplingElPhWan->estimateNumBatches(nq, nbK);
    // keep track of how many intermediate states we use
    size_t batchTotal = 0;

    // loop over batches of final Q states, with the batch size set to the maximum 
    // possible given available memory. This is to optimize usage and data transfer for GPUs 
    // Note that Q states also determine the intermediate Kp states
    for (int iBatch = 0; iBatch < numBatches; iBatch++) {

      // start and end q point indices for current batch
      size_t start = nq * iBatch / numBatches;
      size_t end = nq * (iBatch + 1) / numBatches;
      size_t batchSize = end - start;
      batchTotal += batchSize;

      // Kp, intermediate state quantities
      // For this implementation, there's two kinds of k'
      //    k' = k + q, and 
      //    k' = k - q
      // Here, we set up both, though we can recycle the elph matrix elements from k' = k + q
      // for k' = k - q, as g(k,k',q)^2 = g(k,k',-q)^2
      std::vector<Eigen::Vector3d> allKpPlusCartesian(batchSize);   // list of kp = k + q wavevectors
      std::vector<Eigen::Vector3d> allKpMinusCartesian(batchSize);  // list of kp = k - q wavevectors

      // final state quantities, q
      std::vector<Eigen::Vector3d> allQCartesian(batchSize);        // list of q wavevectors
      std::vector<Eigen::MatrixXcd> allEigenVectorsQ(batchSize);    // phonon eigenvectors
      std::vector<Eigen::VectorXd> allStateEnergiesQ(batchSize);    // phonon energies
      std::vector<Eigen::MatrixXd> allVQs(batchSize);               // phonon group velocities
      std::vector<Eigen::VectorXcd> allPolarData(batchSize);        // related to long range elph elements

      Kokkos::Profiling::pushRegion("drag preprocessing loop");

      // do prep work for all values of q in the current batch,
      // and store all quantities needed to calculate couplings later
#pragma omp parallel for default(none) shared(iQIndexes, allKpPlusCartesian, allKpMinusCartesian, batchSize, start, allQCartesian, allStateEnergiesQ, allVQs, allEigenVectorsQ, kCartesian, allPolarData, polarData, phononBandStructure)
      for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {

        // Set phonon state quantities from band structure ------------
        int iQ = iQIndexes[start + iQBatch];    // index the q state within this batch
        WavevectorIndex iQIdx(iQ);
        allQCartesian[iQBatch] = phononBandStructure.getWavevector(iQIdx);  // ph wavevector in cartesian
        allPolarData[iQBatch] = polarData.row(iQ);                          // long range polar data
        allStateEnergiesQ[iQBatch] = phononBandStructure.getEnergies(iQIdx);  // ph energies
        allVQs[iQBatch] = phononBandStructure.getGroupVelocities(iQIdx);      // ph vels in cartesian
        allEigenVectorsQ[iQBatch] = phononBandStructure.getEigenvectors(iQIdx);  // ph eigenvectors 

        // Set up intermediate electron state wavevectors 
        // (energies, etc, done on the fly below)
        // There are two rates which contribute to the rate at each k,q index of the matrix
        // We set up a loop above to count the "plus" and "minus" contributions one at a time
        // TODO calc coupling squared expects q = k - k', and for the drag terms we need
        // k' = k + q  -> q = k' - k (the standard one used in coupling squared)
        // k' = k - q  -> -q = k' - k
        Eigen::Vector3d kpPlusCartesian = kCartesian + allQCartesian[iQBatch];
        Eigen::Vector3d kpMinusCartesian = kCartesian - allQCartesian[iQBatch];
        allKpPlusCartesian[iQBatch] = kpPlusCartesian;
        allKpMinusCartesian[iQBatch] = kpMinusCartesian;

      }
      Kokkos::Profiling::popRegion();

      // TODO set up the p / m loop above before the points? 

      // populate kp band properties -------------------------
      // NOTE: kp must be generated on the fly, as we have not enforced
      // k = q meshes, and therefore we do not know if kP will be on the same mesh as k
      // (and, likely it's not)
      bool withEigenvectors = true;     // we need these for the transfom in calcCoupling
      bool withVelocities = false;       
      // need velocities for adaptive smearing 
      if (smearing->getType() == DeltaFunction::adaptiveGaussian
                                || smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
        withVelocities = true;
      }
      
      // generate the kp+ quantities from the Wannier Hamiltonian
      auto infoElKp = electronH0->populate(allKpPlusCartesian, withVelocities, withEigenvectors);

      std::vector<Eigen::VectorXd> allStatesEnergiesKpPlus = std::get<0>(infoElKp);
      std::vector<Eigen::MatrixXcd> allEigenVectorsKpPlus = std::get<1>(infoElKp);
      std::vector<Eigen::Tensor<std::complex<double>,3>> allVKpPlus = std::get<2>(infoElKp);

      // generate the kp-, k' = k - q quantities
      infoElKp = electronH0->populate(allKpMinusCartesian, withVelocities, withEigenvectors);

      std::vector<Eigen::VectorXd> allStatesEnergiesKpMinus = std::get<0>(infoElKp);
      std::vector<Eigen::Tensor<std::complex<double>,3>> allVKpMinus = std::get<2>(infoElKp);

      // do the remaining fourier transforms + basis rotations
      // after this call, the couplingElPhWan object contains the coupling for
      // this batch of points, and therefore is indexed by iQBatch
      //
      // We calculate the couplings for g(k,k',q) then use them for g(k,k',-q) as well
      couplingElPhWan->calcCouplingSquared(eigenVectorK, allEigenVectorsKpPlus,
                                           allEigenVectorsQ, allQCartesian, allPolarData);

      Kokkos::Profiling::pushRegion("symmetrize drag coupling");

#pragma omp parallel for
      for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {
        matrix.symmetrizeCoupling(
            couplingElPhWan->getCouplingSquared(iQBatch),
            stateEnergiesK, allStatesEnergiesKpPlus[iQBatch], allStateEnergiesQ[iQBatch]
        );
      }
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("drag terms postprocessingloop");

      // do postprocessing loop with batch of couplings to calculate the rates
      for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {

        // get the qpoint index
        int iQ = iQIndexes[start + iQBatch];

        // grab the coupling matrix elements for the batch of q points
        Eigen::Tensor<double, 3>& coupling = couplingElPhWan->getCouplingSquared(iQBatch);

        Eigen::Vector3d qCartesian = allQCartesian[iQBatch];
        auto t3 = phononBandStructure.getRotationToIrreducible(qCartesian, Points::cartesianCoordinates);
        int iQIrr = std::get<0>(t3);
        // TODO Could add this later if we someday want symmetry
        //Eigen::Matrix3d rotation = std::get<1>(t3); // unused for now as we block sym

        WavevectorIndex iQIdx(iQ);
        WavevectorIndex iQIrrIdx(iQIrr);

        // pull out the energies, etc, for this batch of points
        Eigen::VectorXd stateEnergiesQ = allStateEnergiesQ[iQBatch];

        // TODO if we can reuse the matrix elements, this might be the spot to put the new loop of +/-

        Eigen::VectorXd stateEnergiesKpPlus = allStatesEnergiesKpPlus[iQBatch];
        Eigen::VectorXd stateEnergiesKpMinus = allStatesEnergiesKpMinus[iQBatch];
        Eigen::Tensor<std::complex<double>,3> vKpPlus = allVKpPlus[iQBatch];
        Eigen::Tensor<std::complex<double>,3> vKpMinus = allVKpMinus[iQBatch];

        int nbQ = int(stateEnergiesQ.size());
        // this should be the same for both kpPlus and kpMinus, because
        // we generate these on the fly and don't filter out any bands
        int nbKp = int(stateEnergiesKpPlus.size());

        // pre compute the cosh terms 
        Eigen::MatrixXd coshDataKpPlus(nbKp, numCalculations);
        Eigen::MatrixXd coshDataKpMinus(nbKp, numCalculations);

#pragma omp parallel for collapse(2)
        for (int ibKp = 0; ibKp < nbKp; ibKp++) {
          for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

            double enKpPlus = stateEnergiesKpPlus(ibKp);
            double enKpMinus = stateEnergiesKpMinus(ibKp);
            double kT = statisticsSweep.getCalcStatistics(iCalc).temperature;
            double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
            coshDataKpPlus(ibKp, iCalc) = 0.5 / cosh(0.5 * (enKpPlus - chemPot) / kT);
            coshDataKpMinus(ibKp, iCalc) = 0.5 / cosh(0.5 * (enKpMinus - chemPot) / kT);

          }
        }

        // Loop over state bands and now calculate scattering rates
        for (int ibQ = 0; ibQ < nbQ; ibQ++) {

          double enQ = stateEnergiesQ(ibQ);
          int isQ = phononBandStructure.getIndex(iQIdx, BandIndex(ibQ));
          int isQIrr = phononBandStructure.getIndex(iQIrrIdx, BandIndex(ibQ));
          StateIndex isQIdx(isQ);
          StateIndex isQIrrIdx(isQIrr);
          BteIndex indQIdx = phononBandStructure.stateToBte(isQIrrIdx);
          int iBteQ = indQIdx.get();

          // remove small divergent phonon energies
          if (enQ < phEnergyCutoff) { continue; }

          // initial electron state
          for (int ibK = 0; ibK < nbK; ibK++) {

            double enK = stateEnergiesK(ibK);
            int isK = electronBandStructure.getIndex(ikIdx, BandIndex(ibK));
            StateIndex isKIdx(isK);
            BteIndex indKIdx = electronBandStructure.stateToBte(isKIdx);
            int iBteK = indKIdx.get();

            // intermediate electron states
            for (int ibKp = 0; ibKp < nbKp; ibKp++) {

              double enKpPlus = stateEnergiesKpPlus(ibKp);
              double enKpMinus = stateEnergiesKpMinus(ibKp);

              // set up the delta functions
              double deltaPlus = 0;  // for k' = k + q, delta(E_kp - omega - E_k)
              double deltaMinus = 0; // for k' = k - q, delta(E_k - omega - E_kp)

              if (smearing->getType() == DeltaFunction::gaussian) {

                deltaPlus = smearing->getSmearing(enKpPlus - enK - enQ);
                deltaMinus = smearing->getSmearing(enKpMinus - enK + enQ);

              } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {

                // in this case, the delta functions are all |vk - vk'|
                Eigen::Vector3d vdiffPlus = vKs.row(ibK);
                Eigen::Vector3d vdiffMinus = vKs.row(ibK);
                for (int i : {0,1,2}) {
                  vdiffPlus(i) -= vKpPlus(ibKp, ibKp, i).real();
                  vdiffMinus(i) -= vKpMinus(ibKp, ibKp, i).real();
                }
                deltaPlus = smearing->getSmearing(enKpPlus - enK - enQ, vdiffPlus);
                deltaMinus = smearing->getSmearing(enKpMinus - enK + enQ, vdiffMinus);

              } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {

                Eigen::Vector3d vQ = allVQs[iQBatch].row(ibQ);
                Eigen::Vector3d vK = vKs.row(ibK);
                Eigen::Vector3d vKpP, vKpM;
                for (int i : {0,1,2}) {
                  vKpP(i) = vKpPlus(ibKp, ibKp, i).real();
                  vKpM(i) = vKpMinus(ibKp, ibKp, i).real();
                }
                deltaPlus = smearing->getSmearing(enKpPlus - enK - enQ, vQ, vK, vKpP);
                deltaMinus = smearing->getSmearing(enKpMinus - enK + enQ, vQ, vK, vKpM);
              } else { // tetrahedron
                // we actually block this for transport in the parent scattering matrix obj
                // Therefore, I merely here again throw an error if this is used
                Error("Developer error: Tetrahedron not implemented for CBTE.");
              }

//              if(mpi->mpiHead()) {
//                    std::cout << "nK nkP nPh " << ibK << " " << ibKp << " " <<  ibQ <<  " Ek1 Ek2 omega " << enK << " " << enKp << " " << enQ << " delta 123 " << delta1 << " " << delta2 << " "<< delta3 << " coupling " << coupling(ibK, ibKp, ibQ) << std::endl;
//              }

              // if nothing contributes, go to the next triplet
              if (deltaMinus <= 0. && deltaPlus <= 0.) { continue; }

              // loop on temperature (for now this is only ever one temp at a time)
              for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

                double coshKpPlus = coshDataKpPlus(ibKp, iCalc); // symmetrization term
                double coshKpMinus = coshDataKpMinus(ibKp, iCalc); // symmetrization term

                // Note: in these statements, because |g|^2 in coupling needs
                // the factor of * sqrt(hbar/(2M0*omega)) to convert it to the SE el-ph
                // matrix elements. We apply /enQ and cancel a factor of 2

                // Note: we can't do both Delph and Dphel at once because of the block cyclic distribution
                // of the scattering matrix. There's no guarantee that A_ij and A_ji will both
                // be available to the same MPI process.
                // Therefore, this function is called with an argument to either one or the other
                double dragRate = 0;
                // Calculate D_k,q  += 2pi/N * |g|^2 * [delta(enKp - enQ - enK)] * 1/(2cosh)
                // where kp is either k+ or k-
                dragRate += pi * norm / enQ * coupling(ibK, ibKp, ibQ) * (deltaPlus) * coshKpPlus;
                dragRate -= pi * norm / enQ * coupling(ibK, ibKp, ibQ) * (deltaMinus) * coshKpMinus;

                // the other scattering classes use a "switchCase" variable to determine
                // if we're peforming a matrix*vector product, just filling in linewidths,
                // or filling the whole matrix.
                // In this class, for now we only use the whole matrix, and therefore I only
                // implement switch case 0.
                //
                // Additionally, there's no reason here to fill in the "linewidths" variable,
                // as we'll never have two of the same state, when dimensions are nph, nel states

                // shift the indices if it's necessary
                // if it's not a coupled matrix, these will be just iBte1 and iBte2
                // We have to define shifted versions, or they will be further shifted every loop.
                //
                // Additionally, these are only needed in no-sym case,
                // as coupled matrix never has sym, is always case = 0
                int iBteKShift = iBteK;
                int iBteQShift = iBteQ;
                if(matrix.isCoupled) {
                  Particle el(Particle::electron); Particle ph(Particle::phonon);
                  if(dragTermType == Del) { // indexing is K, Q
                    auto tup = matrix.shiftToCoupledIndices(iBteKShift, iBteQShift, el, ph);
                    iBteKShift = std::get<0>(tup);
                    iBteQShift = std::get<1>(tup);
                  } else { // ph-el drag, indexing is Q, K
                    auto tup = matrix.shiftToCoupledIndices(iBteQShift, iBteKShift, ph, el);
                    iBteQShift = std::get<0>(tup);
                    iBteKShift = std::get<1>(tup);
                  }
                }

                if (withSymmetries) {
                    Error("For now, the drag terms aren't implemented with symmetries.\n"
                        "This is because we only use them with a relaxons solver, "
                        "which doesn't allow symmetries.");
                } else {
                  if (dragTermType == Del) {
                    if (matrix.theMatrix.indicesAreLocal(iBteKShift, iBteQShift)) {
                      matrix.theMatrix(iBteKShift, iBteQShift) += dragRate;
                    }
                  } else {
                    if (matrix.theMatrix.indicesAreLocal(iBteQShift, iBteKShift)) {
                      matrix.theMatrix(iBteQShift, iBteKShift) += dragRate;
                    }
                  }
                } // end symmetries if
              } // end icalc loop
            } // end band Kp loop
          } // band K loop
        } // band Q loop
      Kokkos::Profiling::popRegion();
      } // loop over points in qbatch
    } // loop over batches
  } // pair iterator loop
  loopPrint.close();
}

