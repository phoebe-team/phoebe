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

  VectorBTE minusEllw(matrix.statisticsSweep, electronBandStructure, 1);
  VectorBTE plusEllw(matrix.statisticsSweep, electronBandStructure, 1);

  VectorBTE term1(matrix.statisticsSweep, phononBandStructure, 1);
  VectorBTE term2(matrix.statisticsSweep, phononBandStructure, 1);
  VectorBTE term3(matrix.statisticsSweep, phononBandStructure, 1);

  VectorBTE phel(matrix.statisticsSweep, phononBandStructure, 1);


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
  if (dragTermType == Del) { norm = 1. / (context.getKMesh().prod() * context.getQMesh().prod()); }
  else { norm = 1. / (context.getQMesh().prod() * context.getKMesh().prod()) ; }

  // TODO change this to the same in phel scattering as well
  // precompute the q-dependent part of the polar correction
  // TODO for now we precompute BOTH q+ and q-. It would be much smarter
  // to figure out how to use only q+ so that we do not have to do this twice... 
  Eigen::MatrixXcd polarDataQPlus = couplingElPhWan->precomputeQDependentPolar(phononBandStructure);
  bool useMinusQ = true; // trigger -q calculation using the already computed bandstructure
  Eigen::MatrixXcd polarDataQMinus = couplingElPhWan->precomputeQDependentPolar(phononBandStructure, useMinusQ);

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
    couplingElPhWan->cacheElPh(eigenVectorK, kCartesian); // U_k = eigenVectorK, FT using phase e^{ik.Re}

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

      // container for the list of kp wavevectors
      std::vector<Eigen::Vector3d> allKpCartesian(batchSize);

      // final state quantities, q
      std::vector<Eigen::Vector3d> allQCartesian(batchSize);        // list of q wavevectors
      std::vector<Eigen::MatrixXcd> allEigenVectorsQ(batchSize);    // phonon eigenvectors
      std::vector<Eigen::VectorXd> allStateEnergiesQ(batchSize);    // phonon energies
      std::vector<Eigen::MatrixXd> allVQs(batchSize);               // phonon group velocities
      std::vector<Eigen::VectorXcd> allPolarData(batchSize);        // related to long range elph elements

      Kokkos::Profiling::pushRegion("drag preprocessing loop: q states");

      // do prep work for all values of q in the current batch
      #pragma omp parallel for default(none) shared(iQIndexes, batchSize, start, allQCartesian, allStateEnergiesQ, allVQs, allEigenVectorsQ, phononBandStructure)
      for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {

        // Set phonon state quantities from band structure ----------------------------------------------------
        int iQ = iQIndexes[start + iQBatch];    // index the q state within this batch
        WavevectorIndex iQIdx(iQ);
        allQCartesian[iQBatch] = phononBandStructure.getWavevector(iQIdx);       // ph wavevector in cartesian
        allStateEnergiesQ[iQBatch] = phononBandStructure.getEnergies(iQIdx);     // ph energies
        allVQs[iQBatch] = phononBandStructure.getGroupVelocities(iQIdx);         // ph vels in cartesian
        allEigenVectorsQ[iQBatch] = phononBandStructure.getEigenvectors(iQIdx);  // ph eigenvectors 

      }
      Kokkos::Profiling::popRegion();

      // Kp, intermediate state quantities
      // For this implementation, there's two kinds of k'
      //    k+' = k + q, 
      //    k-' = k - q
      // Because both correspond to the same initial k state, 
      // we can re-use the first fourier transform done by the cacheElph call above.
      // Additionally, remember q is fixed by the phonon state -- only k' varies. 
      
      // The below loop will cover the generation of k+' and k-' ----------------------------------
      // Here, k+' = 0, k-' = 1 
      // TODO can we OMP this?
      for (auto isKpMinus: {0, 1}) { 

        Kokkos::Profiling::pushRegion("drag preprocessing loop: kPrime states");

        // do prep work for all values of k' in this batch -------------------------- 
//        #pragma omp parallel for default(none) shared(allKpCartesian, batchSize, allQCartesian, kCartesian, allPolarData, polarDataQPlus, polarDataQMinus, phononBandStructure, isKpMinus)
        for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {

          // Set up intermediate electron state wavevectors 
          // (energies, etc, done on the fly below)
          // There are two rates which contribute to the rate at each k,q index of the matrix
          // We set up a loop above to count the "plus" and "minus" contributions one at a time
          Eigen::Vector3d kpCartesian = {0,0,0};
          Eigen::VectorXcd kpPolarData; 
          kpPolarData.setZero(); 
          
    	    // remember: k+' = 0, k-' = 1 
          if(isKpMinus == 0) { // kp is plus 
            kpCartesian = kCartesian + allQCartesian[iQBatch];
            kpPolarData = polarDataQPlus.row(iQBatch);
          } 
          if(isKpMinus == 1) { // kp is minus
            kpCartesian = kCartesian - allQCartesian[iQBatch];
            kpPolarData = polarDataQMinus.row(iQBatch);   // TODO check that this is the right syntax vs the old version 
          } 
          allKpCartesian[iQBatch] = kpCartesian;      // kP wavevector 
          allPolarData[iQBatch] = kpPolarData;        // long range polar data
        }  
        Kokkos::Profiling::popRegion();

        // populate kp band properties ---------------------------------------------
	
        // NOTE: kp must be generated on the fly, as we have not enforced
        // k = q meshes, and therefore we do not know if kP will be on the same mesh as k
        // (and, likely it's not)
        bool withEigenvectors = true;     // we need these for the transfom in calcCoupling
        bool withVelocities = false;       

	      // TODO add a "requiresVelocities()" to delta functions --  need velocities for adaptive smearing 
        if (smearing->getType() == DeltaFunction::adaptiveGaussian
                                || smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
          withVelocities = true;
        }
      
        // generate the kp quantities on the fly from the Wannier Hamiltonian
        // returns allStatesEnergiesKp, allEigenVectorsKp, allVKp
        auto kpElQuantities = electronH0->populate(allKpCartesian, withVelocities, withEigenvectors);

        std::vector<Eigen::VectorXd> allStatesEnergiesKp = std::get<0>(kpElQuantities);
        std::vector<Eigen::MatrixXcd> allEigenVectorsKp = std::get<1>(kpElQuantities); // U_{k+q}, or U_{k-q}
        std::vector<Eigen::Tensor<std::complex<double>,3>> allVKp = std::get<2>(kpElQuantities);

        // do the remaining fourier transforms + basis rotations ----------------------------
	      //
        // after this call, the couplingElPhWan object contains the coupling for
        // this batch of points, and therefore is indexed by iQBatch
        //
        // Depending on which kP this is, we calculate either: 
	      //    - for k'+ -> g(k,k+q,q) 
        //    - for k'- -> g(k,k-q,-q)
        // We do this because |g(k,k-q,-q)|^2 = |g(k,k-q,q)|^2, 
        // but simply changing q -> -q makes the changes to interaction elph a bit simpler. 
        // Note: we must switch q->-q as well as U_{k+q} -> U_{k-q}, 
        // see notes by Michele. 
        // remember: k+' = 0, k-' = 1
        bool useMinusQ = bool(isKpMinus); 
        couplingElPhWan->calcCouplingSquared(eigenVectorK, allEigenVectorsKp, allEigenVectorsQ, 
					                                   allQCartesian, allPolarData, useMinusQ);

        // symmetrize the coupling matrix elements for improved numerical stability 
        Kokkos::Profiling::pushRegion("symmetrize drag coupling");

        #pragma omp parallel for
        for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {
          matrix.symmetrizeCoupling(couplingElPhWan->getCouplingSquared(iQBatch),
                    stateEnergiesK, allStatesEnergiesKp[iQBatch], allStateEnergiesQ[iQBatch]
          );
        }
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion("drag terms postprocessingloop");
 
        // do postprocessing loop with batch of couplings to calculate the rates
        for (size_t iQBatch = 0; iQBatch < batchSize; iQBatch++) {

          // get the qpoint index
          int iQ = iQIndexes[start + iQBatch];

          // grab the coupling matrix elements for the batch of q points
          // returns |g(m,m',nu)|^2
          Eigen::Tensor<double, 3>& couplingSq = couplingElPhWan->getCouplingSquared(iQBatch); 

          Eigen::Vector3d qCartesian = allQCartesian[iQBatch];
          WavevectorIndex iQIdx(iQ);

          // pull out the energies, etc, for this batch of points
          Eigen::VectorXd stateEnergiesQ = allStateEnergiesQ[iQBatch];
          Eigen::VectorXd stateEnergiesKp = allStatesEnergiesKp[iQBatch];
          Eigen::Tensor<std::complex<double>,3> vKp = allVKp[iQBatch];   

	        // number of bands 
          int nbQ = int(stateEnergiesQ.size());
          int nbKp = int(stateEnergiesKp.size());

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
                double enQSign = (isKpMinus) ? enQ : -1*enQ; // if kP is k-, use +omega, else -omega

                if (smearing->getType() == DeltaFunction::gaussian) {

                  delta = smearing->getSmearing(enKp - enK + enQSign);

                } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {

                  // in this case, the delta functions are all |vk - vk'|
                  Eigen::Vector3d vdiff = vKs.row(ibK);
                  for (int i : {0,1,2}) {
                    vdiff(i) -= vKp(ibKp, ibKp, i).real();
                  }
                  delta = smearing->getSmearing(enKp - enK + enQSign, vdiff);

                } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {

                  Eigen::Vector3d vQ = allVQs[iQBatch].row(ibQ);
                  Eigen::Vector3d vK = vKs.row(ibK);
                  Eigen::Vector3d vKprime; 
                  for (int i : {0,1,2}) {
                    vKprime(i) = vKp(ibKp, ibKp, i).real();
                  }
                  delta = smearing->getSmearing(enKp - enK + enQSign, vQ, vK, vKprime);

                } else { // tetrahedron
                  // we actually block this for transport in the parent scattering matrix obj
                  // Therefore, I merely here again throw an error if this is used
                  Error("Developer error: Tetrahedron not implemented for CBTE.");
                }

//---------------------
/*
              // remove small divergent phonon energies
              if (enQ < phEnergyCutoff) { continue; }

              double delta1 = 0; double delta2 = 0; 
              if (smearing->getType() == DeltaFunction::gaussian) {
                delta1 = smearing->getSmearing(enK - enKp + enQ); // (enKp - enK - enQ)
                delta2 = smearing->getSmearing(enK - enKp - enQ); // (enKp - enK + enQ) 
              } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {
                  Eigen::Vector3d vdiff = vKs.row(ibK);
                  for (int i : {0,1,2}) {
                    vdiff(i) -= vKp(ibKp, ibKp, i).real();
                  }                
                delta1 = smearing->getSmearing(enK - enKp + enQ, vdiff);
                delta2 = smearing->getSmearing(enK - enKp - enQ, vdiff);
              } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
                  Eigen::Vector3d vQ = allVQs[iQBatch].row(ibQ);
                  Eigen::Vector3d vK = vKs.row(ibK);
                  Eigen::Vector3d vKprime; 
                  for (int i : {0,1,2}) {
                    vKprime(i) = vKp(ibKp, ibKp, i).real();
                  }
                delta1 = smearing->getSmearing(enK - enKp + enQ, vK, vKprime, vQ);
                delta2 = smearing->getSmearing(enK - enKp - enQ, vK, vKprime, vQ);
              } else {
                Error("cant do that");
              }

              //if (delta1 <= 0. && delta2 <= 0.) { continue; } // doesn't contribute

              // loop on temperature
              for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

                //double fermi1 = outerFermi(iCalc, iBte1);
                double kT = statisticsSweep.getCalcStatistics(iCalc).temperature;
                double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
                double fermi1 = electronBandStructure.getParticle().getPopulation(enK,kT,chemPot);
                double fermi2 = electronBandStructure.getParticle().getPopulation(enKp,kT,chemPot);
                double bose3 = phononBandStructure.getParticle().getPopulation(enQ,kT,0);
                double bose3Symm = 0.5 / sinh(0.5 * (enKp - chemPot) / kT); // 1/2/sinh() term

                double normKQ = 1./(context.getKMesh().prod() * context.getQMesh().prod());

                // (enKp - enK - enQ)
                term1(0,0,iBteQ) += 2./(bose3*(1+bose3)) * normKQ * (pi / enQ * couplingSq(ibK,ibKp,ibQ)) 
                                    * delta1 * (bose3 + fermi2) * ( fermi1 * (1 - fermi1) ) * enK / (enQ);

                term2(0,0,iBteQ) += 2./(bose3*(1+bose3)) * normKQ * (pi / enQ * couplingSq(ibK,ibKp,ibQ)) 
                                    * delta1 * (bose3 + 1 - fermi1) * ( fermi2 * (1 - fermi2) ) * enK / (enQ); 

                //term3(0,0,iBteQ) += 2./(bose3*(1+bose3)) * normKQ * (pi / enQ * couplingSq(ibK, ibKp, ibQ)) 
                //                    * delta1 * (bose3 + 1 - fermi1) * (fermi2*(1.-fermi2));
                
                double f1mf = electronBandStructure.getParticle().getPopPopPm1(enK, kT, chemPot);
                term3(0,0,iBteQ) += 1./context.getKMesh().prod() * (pi) * couplingSq(ibK, ibKp, ibQ) 
                                    * delta1 * f1mf / kT;

                // double rate =
                //    coupling(ib1, ib2, ib3)
                //    * (fermi(iCalc, ik1, ib1) - fermi(iCalc, ik2, ib2))
                //    * smearing_values(ib1, ib2, ib3) * norm / en3 * pi;
                //    double rate = coupling(ib1, ib2, ib3) * fermiTerm(iCalc, ik1, ib1)
                //    * smearingValues(ib1, ib2, ib3)
                //    * norm / temperatures(iCalc) * pi * k1Weight;
              }  
*/
// ---------------------

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
                  
                  double Dsign = (isKpMinus) ? -1. : 1.; // if kP is k-, D term is -, else +

                  double kT = statisticsSweep.getCalcStatistics(iCalc).temperature;
                  double chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
                  //double coshK = coshDataK(ibK, iCalc) / (enK - chemPot); 
                  //double enQSign = (isKpMinus) ? enQ : -1*enQ; 
		  // if kP is k-, use +omega, else -omega  // defined above, reminder here

                  double coshK = 1./(2. * cosh(0.5 * (enKp - chemPot) / kT)); 
                  double dragRate = 1./(enK - chemPot) * Dsign * 1./(context.getQMesh().prod()) * 
			  pi / enQ * coshK * couplingSq(ibK,ibKp,ibQ) * delta * ((enKp + enK)/2. - chemPot + enQSign/2.); 

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
        // */
              } // end band Kp loop
            } // band K loop
          } // band Q loop
        } // loop over points in qbatch
        Kokkos::Profiling::popRegion();
      } // close the loop over +/- kp
    } // loop over batches
  } // pair iterator loop
  //mpi->allReduceSum(&plusEllw.data);
  //mpi->allReduceSum(&minusEllw.data);
  mpi->allReduceSum(&term1.data);
  mpi->allReduceSum(&term2.data);
  mpi->allReduceSum(&term3.data);
  mpi->allReduceSum(&phel.data);


  for(int i = 0; i<100; i++) { 

    //if(mpi->mpiHead()) std::cout << phel(0,0,i) << std::endl;

  }
  for(int i = 0; i<100; i++) { 

   //if(mpi->mpiHead()) std::cout << "terms 1 2 3 " << term1(0,0,i) << " " << term2(0,0,i) << " " << term3(0,0,i) << " " << term1(0,0,i) - term2(0,0,i) - term3(0,0,i)  << std::endl;

  }

  loopPrint.close();
}

