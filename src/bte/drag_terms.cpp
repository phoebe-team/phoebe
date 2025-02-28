#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include "periodic_table.h"
#include "interaction_elph.h"
#include "drag_terms.h"
#include "vector_bte.h"


void addDragTerm(CoupledScatteringMatrix &matrix, Context &context,
                  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator,
                  const int& dragTermType, // 0 for el (kn, qs), 1 for ph drag (qs,kn)
                  InteractionElPhWan &couplingElPhWan,
                  BaseBandStructure &phononBandStructure, // phonon
                  BaseBandStructure &electronBandStructure) { // electron

  // throw errors if wrong bandstructure types are provided
  if(!phononBandStructure.getParticle().isPhonon() || !electronBandStructure.getParticle().isElectron()) {
    DeveloperError("Either ph or el bandstructure supplied to the "
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
      loop over k'+, k'- points {
        k' = k +/- q
 		    # set up electronic quantities for k'
          # calculate cosh
          # calculate K' energies
        # calculate the matrix elements g(k,k',q)
 			  # sum the components to calculate contribution to D
 	    }
    }
  }
 */

  StatisticsSweep &statisticsSweep = matrix.statisticsSweep;

  // the summation is a BZ integral over kpoints, we use the electronic mesh for smearing
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
  double spinFactor = 2.; // nonspin pol = 2
  if (context.getHasSpinOrbit()) { spinFactor = 1.; }

  //if (dragTermType == Del) { norm = sqrt(spinFactor) / (sqrt( double(context.getKMesh().prod()) ) * sqrt(double(context.getQMesh().prod()))); }
  //else { norm = sqrt(spinFactor) / (sqrt(double(context.getQMesh().prod())) * sqrt(double(context.getKMesh().prod()))); }
  //norm = spinFactor/(double(context.getKMesh().prod()));
  double Nk = double(context.getKMesh().prod()); 
  double Nq = double(context.getQMesh().prod()); 
  norm = sqrt(spinFactor) / (sqrt( Nk * Nq));

  // TODO change this to the same in phel scattering as well
  // precompute the q-dependent part of the polar correction
  // TODO for now we precompute BOTH q+ and q-. It would be much smarter
  // to figure out how to use only q+ so that we do not have to do this twice...
  Eigen::MatrixXcd polarDataQPlus = couplingElPhWan.precomputeQDependentPolar(phononBandStructure);
  bool useMinusQ = true; // trigger -q calculation using the already computed bandstructure
  Eigen::MatrixXcd polarDataQMinus = couplingElPhWan.precomputeQDependentPolar(phononBandStructure, useMinusQ);

  // set up the loopPrint object which prints out progress
  std::string dragName = (dragTermType == Del) ? "el-ph" : "ph-el";
  LoopPrint loopPrint("computing " + dragName + " drag terms",
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
      int numWannier = couplingElPhWan.getCouplingDimensions()(4);
      Eigen::MatrixXcd eigenVectorK = Eigen::MatrixXcd::Zero(numWannier, 1);

      couplingElPhWan.cacheElPh(eigenVectorK, kCartesian);
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

    // pre compute the cosh term
    Eigen::MatrixXd coshDataK(nbK, numCalculations);

    #pragma omp parallel for collapse(2)
    for (int ibK = 0; ibK < nbK; ibK++) {
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) { // only one calculation here

        double enK = stateEnergiesK(ibK);
        double kT = statisticsSweep.getCalcStatistics(iCalc).temperature;
        double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
        coshDataK(ibK, iCalc) = 2 * cosh(0.5 * (enK - chemPot) / kT); //0.5 / cosh(0.5 * (enK - chemPot) / kT);
      }
    }

    // perform the first fourier transform + rotation for state k
    couplingElPhWan.cacheElPh(eigenVectorK, kCartesian); // U_k = eigenVectorK, FT using phase e^{ik.Re}

    // prepare batches of intermediate states, kp, which are determined by the points helper
    // by memory usage
    int nq = int(iQIndexes.size());

    // TODO: this was written to take nk batches, but the function should work the same
    // way. Double check this.
    // Number of batches of k' states which will be used for this k,q pair
    int numBatches = couplingElPhWan.estimateNumBatches(nq, nbK);
    // keep track of how many intermediate states we use
    size_t batchTotal = 0;

    // loop over batches of final Q states, with the batch size set to the maximum
    // possible given available memory. This is to optimize usage and data transfer for GPUs
    // Note that Q states also determine the intermediate Kp states
    //
    // Here, we do some additional filtering of states in the batch, in order to
    // reduce the batch size.
    //
    // We do this by first checking the q/kp energies, then removing points which
    // will not satisfy the relevant delta functions from the batch.
    //
    // Note: can we use OMP here? If generating energies for kp OTF this is probably
    // unfavorable as that code is already OMP heavy
    for (int iBatch = 0; iBatch < numBatches; iBatch++) {

      // There are two rates which contribute to the rate at each k,q index of the matrix
      // We set up a loop above to count the "plus" and "minus" contributions one at a time

      // Kp, intermediate state quantities
      // For this implementation, there's two kinds of k'
      //    k+' = k + q,
      //    k-' = k - q
      // Because both correspond to the same initial k state,
      // we can re-use the first fourier transform done by the cacheElph call above.
      // Additionally, remember q is fixed by the phonon state -- only k' varies.

      // The below loop will cover the generation of k+' and k-' ----------------------------------
      // Here, k+' = 0, k-' = 1

      // TODO can we OMP this? -- TODO be careful about the repercussions of moving this line
      for (auto isKpMinus: {0,1}) {

        // start and end q point indices for current batch
        // we will revise these by removing points that don't make sense
        size_t start = nq * iBatch / numBatches;
        size_t end = nq * (iBatch + 1) / numBatches;
        size_t batchSize = end - start;
        batchTotal += batchSize;
        size_t revisedBatchSize = 0; // increment this each time we find a useful kp point

        // first, we will determine which kp points actually matter, and build a list
        // of the important kp and q points
        std::vector<int> filteredQIndices;

        // temp container for the list of kp wavevectors
        std::vector<Eigen::Vector3d> allKpCartesian;
        std::vector<Eigen::MatrixXcd> allEigenVectorsKp;    // electron2 eigenvectors
        std::vector<Eigen::VectorXd> allStateEnergiesKp;    // electron2 energies
        std::vector<Eigen::MatrixXd> allVKps;               // electron2 group velocities

        // all container for final state quantities, q
        std::vector<Eigen::Vector3d> allQCartesian;        // list of q wavevectors
        std::vector<Eigen::MatrixXcd> allEigenVectorsQ;    // phonon eigenvectors
        std::vector<Eigen::VectorXd> allStateEnergiesQ;    // phonon energies
        std::vector<Eigen::MatrixXd> allVQs;               // phonon group velocities
        std::vector<Eigen::VectorXcd> allPolarData;        // related to long range elph elements

        Kokkos::Profiling::pushRegion("drag preprocessing loop: q states");

        // do prep work for all values of q in the current batch
        // TODO OMP is nasty when we are pushing back... can we figure out how to use this still?
        // TODO perhaps a critical block around the push back?
        #pragma omp parallel for default(none) shared(iQIndexes, batchSize,revisedBatchSize, kCartesian, start, isKpMinus, phononBandStructure,filteredQIndices, polarDataQMinus, polarDataQPlus, electronBandStructure, allKpCartesian, allPolarData, allStateEnergiesKp, allVKps, allEigenVectorsKp, allQCartesian, allStateEnergiesQ, allVQs, allEigenVectorsQ)
        for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {

          // Set phonon state quantities from band structure ----------------------------------------------------
          int iQ = iQIndexes[start + iQBatch];    // index the q state within this batch
          WavevectorIndex iQIdx(iQ);

          // temporary phonon containers
          Eigen::Vector3d thisQCartesian = phononBandStructure.getWavevector(iQIdx);

          // something went wrong here, what is it
          if(isKpMinus == 1) { // kp is minus
            // here we need -q, we have to flip the indices
            thisQCartesian = -1*thisQCartesian;       // ph wavevector in cartesian, flipped to account for k,k',-q
            // REMAP Q
            Eigen::Vector3d qCrys = phononBandStructure.getPoints().cartesianToCrystal(thisQCartesian);
            iQIdx = WavevectorIndex(phononBandStructure.getPointIndex(qCrys));
          }

          // do prep work for all values of k' in this batch --------------------------
          // (energies, etc, done on the fly below)
          // There are two rates which contribute to the rate at each k,q index of the matrix
          // We set up a loop above to count the "plus" and "minus" contributions one at a time
          Eigen::Vector3d kpCartesian = {0,0,0};
          Eigen::VectorXcd kpPolarData;
          kpPolarData.setZero();

    	    // remember: k+' = 0, k-' = 1
          if(isKpMinus == 0) { // kp is plus
            kpCartesian = kCartesian + thisQCartesian;
            kpPolarData = polarDataQPlus.row(iQBatch);
          }
          if(isKpMinus == 1) { // kp is minus
            // flip k and kp, setting k1 = k-q, k2 = k, q3 = q
            kpCartesian = kCartesian + thisQCartesian; // k' = k - q
            kpPolarData = polarDataQMinus.row(iQBatch);   // TODO check that this is the right syntax vs the old version
          }

          // check if this point is on the mesh or not
          Eigen::Vector3d kpCrys = electronBandStructure.getPoints().cartesianToCrystal(kpCartesian);
          if(electronBandStructure.getPointIndex(kpCrys, true) == -1) {// if point is not on the mesh, continue
            continue;
          }

          // if this kp point is on the el bandstructure, here we save all the ph and el' information
          #pragma omp critical
          {

            revisedBatchSize++;
            filteredQIndices.push_back(iQ);

            WavevectorIndex ikpIdx = WavevectorIndex(electronBandStructure.getPointIndex(kpCrys));
            allKpCartesian.push_back(kpCartesian);                                      // kP wavevector
            allPolarData.push_back(kpPolarData);                                        // long range polar data
            allStateEnergiesKp.push_back(electronBandStructure.getEnergies(ikpIdx));     // el Kp energies
            allVKps.push_back(electronBandStructure.getGroupVelocities(ikpIdx));         // el Kp vels in cartesian
            allEigenVectorsKp.push_back(electronBandStructure.getEigenvectors(ikpIdx));  // el Kp eigenvectors

            allQCartesian.push_back(phononBandStructure.getWavevector(iQIdx));       // ph wavevector in cartesian
            allStateEnergiesQ.push_back(phononBandStructure.getEnergies(iQIdx));     // ph energies
            allVQs.push_back(phononBandStructure.getGroupVelocities(iQIdx));         // ph vels in cartesian
            allEigenVectorsQ.push_back(phononBandStructure.getEigenvectors(iQIdx));  // ph eigenvectors
          }
        } // close the q preproc loop
        Kokkos::Profiling::popRegion();

        batchSize = revisedBatchSize;

        // do the remaining fourier transforms + basis rotations ----------------------------
	      //
        // after this call, the couplingElPhWan object contains the coupling for
        // this batch of points, and therefore is indexed by iQBatch
        couplingElPhWan.calcCouplingSquared(eigenVectorK, allEigenVectorsKp, allEigenVectorsQ,
					                                   allQCartesian, kCartesian, allPolarData);

        // symmetrize the coupling matrix elements for improved numerical stability
        Kokkos::Profiling::pushRegion("symmetrize drag coupling");

        #pragma omp parallel for
        for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {
          matrix.symmetrizeCoupling(couplingElPhWan.getCouplingSquared(iQBatch),
                    stateEnergiesK, allStateEnergiesKp[iQBatch], allStateEnergiesQ[iQBatch]
          );
        }
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion("drag terms postprocessing loop");

        // do postprocessing loop with batch of couplings to calculate the rates
        for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {

          // get the qpoint index
          int iQ = filteredQIndices[iQBatch]; //iQIndexes[start + iQBatch];

          // grab the coupling matrix elements for the batch of q points
          // returns |g(m,m',nu)|^2
          Eigen::Tensor<double, 3>& couplingSq = couplingElPhWan.getCouplingSquared(iQBatch);

          Eigen::Vector3d qCartesian = allQCartesian[iQBatch];
          WavevectorIndex iQIdx(iQ);

          // pull out the energies, etc, for this batch of points
          Eigen::VectorXd stateEnergiesQ = allStateEnergiesQ[iQBatch];
          Eigen::VectorXd stateEnergiesKp = allStateEnergiesKp[iQBatch];
          Eigen::MatrixXd vKp = allVKps[iQBatch];
          auto kpCartesian = allKpCartesian[iQBatch]; // TODO remove this it's a test statement

	        // number of bands
          int nbQ = int(stateEnergiesQ.size());
          int nbKp = int(stateEnergiesKp.size());

          //Eigen::Vector3d kCrys = electronBandStructure.getPoints().cartesianToCrystal(kCartesian);
          //Eigen::Vector3d kpCrys = electronBandStructure.getPoints().cartesianToCrystal(kpCartesian);
          Eigen::Vector3d qCrys = phononBandStructure.getPoints().cartesianToCrystal(qCartesian);

          // Calculate the scattering rate  -------------------------------------------
          // Loop over state bands
          for (int ibQ = 0; ibQ < nbQ; ibQ++) {

            double enQ = stateEnergiesQ(ibQ);
            int isQ = phononBandStructure.getIndex(iQIdx, BandIndex(ibQ));
            StateIndex isQIdx(isQ);
            BteIndex bteQIdx = phononBandStructure.stateToBte(isQIdx);
            int iBteQ = bteQIdx.get();

            // remove small divergent phonon energies
            if (enQ < phEnergyCutoff) { continue; }

            // initial electron state
            for (int ibK = 0; ibK < nbK; ibK++) {

              double enK = stateEnergiesK(ibK);
              int isK = electronBandStructure.getIndex(ikIdx, BandIndex(ibK));
              StateIndex isKIdx(isK);
              BteIndex indKIdx = electronBandStructure.stateToBte(isKIdx);
              int iBteK = indKIdx.get();

              // loop over intermediate electron state bands
              for (int ibKp = 0; ibKp < nbKp; ibKp++) {

                double enKp = stateEnergiesKp(ibKp);

                // set up the delta functions
                // for k' = k + q, delta(E_kp - omega - E_k) = delta(E_kp - E_k - omega)
                // for k' = k - q, delta(E_k - omega - E_kp) = delta(E_kp - E_k + omega)
                double delta = 0;
                double enQSign = enQ;
                if(!isKpMinus) { enQSign = enQ * -1.; }

                if (smearing->getType() == DeltaFunction::gaussian) {

                  delta = smearing->getSmearing(enKp - enK + enQSign);

                } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {

                  // in this case, the delta functions are all |vk - vk'|
                  Eigen::Vector3d vdiff = vKs.row(ibK) - vKp.row(ibKp);
                  delta = smearing->getSmearing(enKp - enK + enQSign, vdiff);

                } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {

                  Eigen::Vector3d vQ = allVQs[iQBatch].row(ibQ);
                  Eigen::Vector3d vK = vKs.row(ibK);
                  Eigen::Vector3d vKprime = vKp.row(ibKp);
                  delta = smearing->getSmearing(enKp - enK + enQSign, vQ, vK, vKprime);

                } else { // tetrahedron, which is not supported because for drag we will
                  // always want a population mesh, and tetrahedron cannot be used with non-uniform mesh
                  // we actually block this for transport in the parent scattering matrix obj
                  // Therefore, I merely here again throw an error if this is used
                  DeveloperError("Tetrahedron not implemented for CBTE.");
                }

                // if nothing contributes, go to the next triplet
                if (delta <= 0.) { continue; }

                // loop on temperature (for now this is only one temp/mu at a time)
                for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

                  // Note: in these statements, because |g|^2 in coupling needs
                  // the factor of * sqrt(hbar/(2M0*omega)) to convert it to the SE el-ph
                  // matrix elements. We apply /enQ and cancel a factor of 2

                  // Note: we can't do both Delph and Dphel at once because of the block cyclic distribution
                  // of the scattering matrix. There's no guarantee that A_ij and A_ji will both
                  // be available to the same MPI process.
                  // Therefore, this function is called with an argument to either one or the other

                  // Calculate D^+(k,q) = 2pi/N * Sum_(k'=k+q) |g(k,k+q,q)|^2 * [delta(enKp_+ - enQ - enK)] * 1/(2cosh(k+q))
		              // OR
                  // Calculate D^-(k,q) = -2pi/N * Sum_(k'=k-q) |g(k,k-q,q)|^2 * [delta(enKp_- - enQ + enK)] * 1/(2cosh(k-q))

                  double kT = statisticsSweep.getCalcStatistics(iCalc).temperature;
                  double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
                  double coshKp = 1./(2. * cosh(0.5 * (enKp - chemPot) / kT));

                  double dragRate = 0;
                  //Eigen::Vector3d qCrysM = phononBandStructure.getPoints().cartesianToCrystal(-qCartesian);
                  //WavevectorIndex iQidxM(phononBandStructure.getPointIndex(qCrysM));
                  //int isQM = phononBandStructure.getIndex(iQidxM, BandIndex(ibQ));

                  double normTemp = norm; 
                  if( (enQ < 0.007 / energyRyToEv)) { // && (qCrys.norm() < 1e-1)) {
                    normTemp = sqrt(spinFactor) / ( Nk );
                    //std::cout << " imode omega q " << ibQ << " " << enQ << " " << qCrys.transpose() << std::endl;
                  }

                  if(!isKpMinus) { // g+ part

                    dragRate = normTemp * //1./(enK) *
                          couplingSq(ibK,ibKp,ibQ) * pi / enQ *  // 1/sqrt(omega)^2, g_SE factor
                          coshKp * delta * ( (enKp + enK - enQ)/ ( 2. * enK ) );

                  } else if(isKpMinus) { // g- part

                    dragRate = -normTemp * // 1./(enK) *
                          couplingSq(ibK,ibKp,ibQ) * pi / enQ *  // 1/sqrt(omega)
                          coshKp * delta * ( (enKp + enK + enQ)/ ( 2. * enK ) );

                  }

  		            // add this contribution to the matrix -------------------------------------

                  // the other scattering classes use a "switchCase" variable to determine
                  // if we're peforming a matrix*vector product, just filling in linewidths,
                  // or filling the whole matrix.
                  // In this class, for now we only use the whole matrix, and therefore I only
		              // implement switch case = 0
                  //
                  // Additionally, there's no reason here to fill in the "linewidths" variable,
                  // as we'll never have two of the same state, when dimensions are nph, nel states

                  // Shift the BTE indices from the ph and el state indices to the relevant k,q quadrant
		              // in order to index the matrix
                  int iBteKShift = iBteK;
                  int iBteQShift = iBteQ;
                  if(matrix.isCoupled) {

                    Particle el(Particle::electron);
		                Particle ph(Particle::phonon);
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
        } // loop over points in qbatch
        Kokkos::Profiling::popRegion();
      } // close the loop over +/- kp
    } // loop over batches
  } // pair iterator loop
  loopPrint.close();
}

