#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include "periodic_table.h"
#include "interaction_elph.h"
#include "phel_scattering_matrix.h"
#include "coupled_scattering_matrix.h"

/** Method to construct the kq pair iterator for this class.
* Slightly different than the others, so we calculate it internally.
* This is especially because the intermediate state band structure
* is created in the scattering function.
*/
std::vector<std::tuple<int, std::vector<int>>> getPhElIterator(
                                        BaseBandStructure& elBandStructure,
                                        BaseBandStructure& phBandStructure) {

  std::vector<std::tuple<int, std::vector<int>>> pairIterator;

  // here I parallelize over ik1 which is the outer loop on k-points
  std::vector<int> k1Iterator = elBandStructure.parallelIrrPointsIterator();

  // I don't parallelize the inner band structure, the inner loop
  // populate vector with integers from 0 to numPoints-1
  std::vector<int> q3Iterator = phBandStructure.irrPointsIterator();

  for (size_t ik1 : k1Iterator) {
    auto t = std::make_tuple(int(ik1), q3Iterator);
    pairIterator.push_back(t);
  }

  // add some dummy indices to each MPI procs iterator of indices
  // to make sure they have the same number
  // If this isn't the case, a pooled calculation will hang
  if (mpi->getSize(mpi->intraPoolComm) > 1) {
    auto myNumK1 = int(pairIterator.size());
    int numK1 = myNumK1;
    mpi->allReduceMax(&numK1, mpi->intraPoolComm);

    while (myNumK1 < numK1) {
      std::vector<int> dummyVec;
      dummyVec.push_back(-1);
      auto tt = std::make_tuple(-1, dummyVec);
      pairIterator.push_back(tt);
      myNumK1++;
    }
  }
  return pairIterator;
}

void addPhElScattering(BasePhScatteringMatrix &matrix, Context &context,
                      BaseBandStructure& phBandStructure,
                      BaseBandStructure& elBandStructure,
                      StatisticsSweep& statisticsSweep, 
                      InteractionElPhWan& couplingElPhWan,
                      std::shared_ptr<VectorBTE> linewidth) {

  if(mpi->mpiHead()) 
    std::cout << "\n------------- Phonon-electron scattering -------------\n" << std::endl; 

  // throw error if it's not a correct band structure
  if(!phBandStructure.getParticle().isPhonon() 
                || !elBandStructure.getParticle().isElectron()) { 
    DeveloperError("Wrong bandstructure passed to phel calculation.");
  }

  // throw an error if these meshes are not commensurate. Here, we don't want to 
  // generate states on the fly. If we lock k/q = integer, then for any given
  // set of k and q, k' will still be on the electronic band structure
  auto kMesh = context.getKMesh();
  auto qMesh = context.getQMesh(); 
  if( kMesh(0)%qMesh(0) != 0 || kMesh(1)%qMesh(1) != 0 || kMesh(2)%qMesh(2) != 0 ) {
    Error("For ph-el scattering calculations, k and q meshes must be commensurate!");
  }

  Particle elParticle(Particle::electron);
  Particle particle = phBandStructure.getParticle();
  if(!statisticsSweep.getParticle().isElectron()) { 
    DeveloperError("Statistics sweep provided to phel scattering must be electronic one!"); 
  } 

  int numCalculations = statisticsSweep.getNumCalculations();
  //Crystal crystalPh = phBandStructure.getPoints().getCrystal();

  Eigen::VectorXd temperatures(numCalculations);
  for (int iCalc=0; iCalc<numCalculations; ++iCalc) {
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temperatures(iCalc) = temp;
  }

  // note: innerNumFullPoints is the number of points in the full grid
  // may be larger than innerNumPoints, when we use ActiveBandStructure
  // note: in the equations for this rate, because there's an integraton over k,
  // this rate is 1/NK (sometimes written N_eFermi).
  double spinFactor = 2.; // nonspin pol = 2
  if (context.getHasSpinOrbit()) { spinFactor = 1.; }
  double norm = spinFactor / double(context.getKMesh().prod());

  // if this is a coupled calculation, we need to remove the spin factor and Nk here, it will be reapplied later
  if(context.getAppName().find("coupled") != std::string::npos ) { 
    norm = 1. / double(context.getQMesh().prod()); 
  }

  // Here, we use a smearing corresponding to electrons, as this needs
  // to match up with the drag terms and also with the electron quadrant
  DeltaFunction *smearing = DeltaFunction::smearingFactory(context, elBandStructure);
  if (smearing->getType() == DeltaFunction::tetrahedron) {
    DeveloperError("Tetrahedron smearing for transport untested and thus blocked");
  }

  // don't proceed if we use more than one doping concentration --
  // phph scattering only has 1 mu value, therefore the linewidths won't add to it correctly
  // TODO should this be migrated to the outer function call? 
  int numMu = statisticsSweep.getNumChemicalPotentials();
  if (numMu != 1) {
      Error("Can currenly only add ph-el scattering one doping "
        "concentration at the time. Let us know if you want to have multiple mu values as a feature.");
  }

  // now that we have the band structure, we can generate the kqPairs
  // to iterate over in parallel
  std::vector<std::tuple<int,std::vector<int>>> kqPairIterator
                = getPhElIterator(elBandStructure, phBandStructure);

  // precompute Fermi-Dirac factors
  int numKPoints = elBandStructure.getNumPoints();
  int nb1Max = 0;
  for (int ik=0; ik<numKPoints; ++ik) {
    WavevectorIndex ikIdx(ik);
    nb1Max = std::max(nb1Max, int(elBandStructure.getEnergies(ikIdx).size()));
  }

  // precompute Fermi-Dirac populations
  // TODO can we fit this into the same format as the other ones
  // to call the helper function instead?
  Eigen::Tensor<double,3> fermiTerm(numCalculations, numKPoints, nb1Max);
  fermiTerm.setZero();

  std::vector<size_t> kIterator = mpi->divideWorkIter(numKPoints);
  #pragma omp parallel for
  for (size_t iik = 0; iik < kIterator.size(); iik++) {

    int ik = kIterator[iik]; // avoid omp parallel on iterator loops 
	  
    WavevectorIndex ikIdx(ik);
    Eigen::VectorXd energies = elBandStructure.getEnergies(ikIdx);
    int nb1 = energies.size();
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
      double temp = calcStat.temperature;
      double chemPot = calcStat.chemicalPotential;
      for (int ib=0; ib<nb1; ++ib) {
        fermiTerm(iCalc, ik, ib) = elParticle.getPopPopPm1(energies(ib), temp, chemPot);
      }
    }
  }
  mpi->allReduceSum(&fermiTerm);

  // precompute the q-dependent part of the polar correction ---------
  // see the precomputeQPolarCorrection function in interaction, could be
  // used here instead
  int numQPoints = phBandStructure.getNumPoints();
  auto qPoints = phBandStructure.getPoints();
  // we just set this to the largest possible number of phonons
  int nb3Max = 3 * phBandStructure.getPoints().getCrystal().getNumAtoms();
  Eigen::MatrixXcd polarData(numQPoints, nb3Max);
  polarData.setZero();

  std::vector<size_t> qIterator = mpi->divideWorkIter(numQPoints); 

  #pragma omp parallel for
  for (int iiq = 0; iiq < int(qIterator.size()); iiq++) { 

    int iq = qIterator[iiq]; // avoid issues with omp on iterator loops
    WavevectorIndex iqIdx(iq);
    auto q3C = phBandStructure.getWavevector(iqIdx);
    auto ev3 = phBandStructure.getEigenvectors(iqIdx);
    Eigen::VectorXcd thisPolar = couplingElPhWan.polarCorrectionPart1(q3C, ev3);
    for (int i=0; i<thisPolar.size(); ++i) {
      polarData(iq, i) = thisPolar(i);
    }
  }
  mpi->allReduceSum(&polarData);

  //---------------

  // k1, k2 are the electronic states, q3 is the phonon
  // this helper returns pairs of the form vector<idxK1, std::vector(allQ3idxs)>
  LoopPrint loopPrint("computing ph-el linewidths","k,q pairs", kqPairIterator.size());

  // iterate over vector<idxK1, std::vector(allK2idxs)>
  // In this loop, k1 is fixed at the top, and we compute
  // it's electronic properties in the outer loop.
  // q3 is the list of iq3Indices, and k2 is determined using k1 and q3
  for (auto t1 : kqPairIterator) {

    loopPrint.update();

    int ik1 = std::get<0>(t1);

    WavevectorIndex ik1Idx(ik1);
    auto iq3Indexes = std::get<1>(t1);

    // dummy call to make pooled coupling calculation work. We need to make sure
    // calcCouplingSquared is called the same # of times. This is also taken
    // care of while generating the indices. Here we call calcCoupling.
    // This block is useful if e.g. we have a pool of size 2, the 1st MPI
    // process has 7 k-points, the 2nd MPI process has 6. This block makes
    // the 2nd process call calcCouplingSquared 7 times as well.
    if (ik1 == -1) {
      Eigen::Vector3d k1C = Eigen::Vector3d::Zero();
      int numWannier = couplingElPhWan.getCouplingDimensions()(4);
      Eigen::MatrixXcd eigenVector1 = Eigen::MatrixXcd::Zero(numWannier, 1);
      couplingElPhWan.cacheElPh(eigenVector1, k1C);
      // since this is just a dummy call used to help other MPI processes
      // compute the coupling, and not to compute matrix elements, we can skip
      // to the next loop iteration
      continue;
    }

    // store k1 energies, velocities, eigenvectors from elBandstructure
    Eigen::Vector3d k1Cartesian = elBandStructure.getWavevector(ik1Idx);
    Eigen::VectorXd state1Energies = elBandStructure.getEnergies(ik1Idx);
    auto nb1 = int(state1Energies.size());
    Eigen::MatrixXd v1s = elBandStructure.getGroupVelocities(ik1Idx);
    Eigen::MatrixXcd eigenVector1 = elBandStructure.getEigenvectors(ik1Idx);
    // unlike in el-ph scattering, here we only loop over irr points.
    // This means we need to multiply by the weights of the irr k1s
    // in our integration over the BZ. This returns the list of kpoints
    // that map to this irr kpoint
    double k1Weight = elBandStructure.getPoints().
                                getReducibleStarFromIrreducible(ik1).size();

    // precompute first fourier transform + rotation by k1
    couplingElPhWan.cacheElPh(eigenVector1, k1Cartesian);

    // prepare batches of q3s based on memory usage (so that this could be done on gpus)?
    size_t nq3 = size_t(iq3Indexes.size());
    size_t numBatches = couplingElPhWan.estimateNumBatches(nq3, nb1);

    // loop over batches of q3s
    // later we will loop over the q3s inside each batch
    // this is done to optimize the usage and data transfer of a GPU
    for (size_t iBatch = 0; iBatch < numBatches; iBatch++) {

      // start and end point for current batch of q3s
      size_t start = nq3 * iBatch / numBatches;
      size_t end = nq3 * (iBatch + 1) / numBatches;
      size_t batchSize = end - start;
      size_t revisedBatchSize = 0; // increment this each time we find a useful kp point

      // first, we will determine which kp points actually matter, and build a list 
      // of the important kp and q points
      std::vector<int> filteredQIndices; 
      std::vector<int> filteredK2Indices; 

      std::vector<Eigen::Vector3d> allQ3Cartesian;
      std::vector<Eigen::MatrixXcd> allEigenVectors3; 
      std::vector<Eigen::VectorXcd> allPolarData; 

      std::vector<Eigen::Vector3d> allK2Cartesian; 
      std::vector<Eigen::MatrixXcd> allEigenVectors2;

      // do prep work for all values of q3 in current batch,
      // store stuff needed for couplings later
      //
      // loop over each iq3 in the batch of q3s
      // we only take points which are on the active bandstructure -- 
      // this speeds things up dramatically and should not eliminate meaningful states, 
      // especially if convergence is tested wrt window size 

      // here the critical block is ok, as 95% of the points are discarded and we only 
      // push back a few 
      #pragma omp parallel for  
      for (size_t iq3Batch = 0; iq3Batch < batchSize; iq3Batch++) {

        size_t iq3 = iq3Indexes[start + iq3Batch];
        WavevectorIndex iq3Idx(iq3);
        Eigen::Vector3d q3Cartesian = phBandStructure.getWavevector(iq3Idx);
        Eigen::Vector3d k2Cartesian = q3Cartesian + k1Cartesian; 

        // check if this point is on the mesh or not 
        // TODO this is slow, can we make it faster? 
        Eigen::Vector3d k2Crys = elBandStructure.getPoints().cartesianToCrystal(k2Cartesian);
        int ik2 = elBandStructure.getPointIndex(k2Crys, true);
        if(ik2 == -1) {  // if point is not on the mesh, continue
          continue; 
        }

        // if this kp point is on the el bandstructure, here we save all the ph and el' information
        #pragma omp critical 
        {
          revisedBatchSize++;
          filteredQIndices.push_back(iq3);
          filteredK2Indices.push_back(ik2);

          WavevectorIndex ik2Idx = WavevectorIndex(ik2);
          allK2Cartesian.push_back(k2Cartesian);                                 // kP wavevector  // TODO might be able to remove this 
          allPolarData.push_back(polarData.row(iq3));                            // long range polar data
          allEigenVectors2.push_back(elBandStructure.getEigenvectors(ik2Idx));  // el Kp eigenvectors

          allQ3Cartesian.push_back(phBandStructure.getWavevector(iq3Idx));            // ph wavevector in cartesian 
          allEigenVectors3.push_back(phBandStructure.getEigenvectors(iq3Idx));        // ph eigenvectors
        }
      }

      batchSize = revisedBatchSize;

      // Generate couplings for fixed k1, all k2s and all Q3Cs
      couplingElPhWan.calcCouplingSquared(eigenVector1, allEigenVectors2,
                                          allEigenVectors3, allQ3Cartesian, 
                                          k1Cartesian, allPolarData);

      // do postprocessing loop with batch of couplings to calculate the scattering rates
      for (size_t iq3Batch = 0; iq3Batch < batchSize; iq3Batch++) {

        int iq3 = filteredQIndices[iq3Batch]; 
        int ik2 = filteredK2Indices[iq3Batch]; 
        WavevectorIndex iq3Idx(iq3);
        WavevectorIndex ik2Idx(ik2);

        auto k2C = allK2Cartesian[iq3Batch]; // TODO remove this it's just for testing
        auto q3C = allQ3Cartesian[iq3Batch]; // TODO remove this it's just for testing

        // for gpu would replace with compute OTF
        Eigen::VectorXd state3Energies = phBandStructure.getEnergies(iq3Idx); // iq3Idx
        Eigen::VectorXd state2Energies = elBandStructure.getEnergies(ik2Idx); // iq3Idx
        auto nb2 = int(state2Energies.size());

        // NOTE: these loops are already set up to be applicable to gpus
        // the precomputaton of the smearing values and the open mp loops could
        // be converted to GPU relevant version
        int nb3 = state3Energies.size();
        Eigen::Tensor<double,3> smearingValues(nb1, nb2, nb3);

        #pragma omp parallel for collapse(3) default(none) shared(nb1,nb2,nb3,state1Energies, state2Energies, state3Energies, phEnergyCutoff, smearingValues, elBandStructure, phBandStructure,ik1Idx,ik2Idx,iq3Idx,smearing)
        for (int ib2 = 0; ib2 < nb2; ib2++) {
          for (int ib1 = 0; ib1 < nb1; ib1++) {
            for (int ib3 = 0; ib3 < nb3; ib3++) {

              double en1 = state1Energies(ib1);
              double en2 = state2Energies(ib2);
              double en3 = state3Energies(ib3);

              // remove small divergent phonon energies
              if (en3 < phEnergyCutoff) {
                smearingValues(ib1, ib2, ib3) = 0.;
                continue;
              }
              double delta = 0;

              Eigen::Vector3d v1, v2, v3; 
              if(smearing->getType() == DeltaFunction::symAdaptiveGaussian || 
                          smearing->getType() == DeltaFunction::adaptiveGaussian) {

                StateIndex is1(elBandStructure.getIndex(ik1Idx,BandIndex(ib1)));
                StateIndex is2(elBandStructure.getIndex(ik2Idx,BandIndex(ib2)));
                StateIndex is3(phBandStructure.getIndex(iq3Idx,BandIndex(ib3)));

                v1 = elBandStructure.getGroupVelocity(is1);
                v2 = elBandStructure.getGroupVelocity(is2);
                v3 = phBandStructure.getGroupVelocity(is3);
              }

              // NOTE: for gpus, if statements need to be moved outside, as
              // this creates load balancing issues for gpu threads and cpu must
              // dispatch this decision info
              if (smearing->getType() == DeltaFunction::gaussian) {
                delta = smearing->getSmearing(en1 - en2 + en3);

              } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
                delta = smearing->getSmearing(en1 - en2 + en3, v1, v2, v3);

              // adaptive gaussian for delta(omega - (Ek' - Ek))
              // has width of W = a | v2 - v1 |
              } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {
                delta = smearing->getSmearing(en1 - en2 + en3, v1-v2);
              } else {
                Error("Tetrahedron smearing is currently not tested with phel scattering.");
              }
              smearingValues(ib1, ib2, ib3) = std::max(delta, 0.);
            }
          }
        }

        Eigen::Tensor<double, 3> coupling = couplingElPhWan.getCouplingSquared(iq3Batch);
        // symmetrize the elph coupling tensor
        matrix.symmetrizeCoupling(coupling, state1Energies, state2Energies, state3Energies);

        #pragma omp parallel for collapse(2)
        for (int ib3 = 0; ib3 < nb3; ib3++) {
          for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

            int is3 = phBandStructure.getIndex(iq3Idx, BandIndex(ib3));
            // the BTE index is an irr point which indexes VectorBTE objects
            // like the linewidths + scattering matrix -- as these are
            // only allocated for irr points when sym is on
            StateIndex isIdx3(is3);
            BteIndex ibteIdx3 = phBandStructure.stateToBte(isIdx3);
            int ibte3 = ibteIdx3.get();
            double en3 = state3Energies(ib3);

            // remove small divergent phonon energies
            if (en3 < phEnergyCutoff) {
              continue;
            }

            auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
            double temp = calcStat.temperature;
            double chemPot = calcStat.chemicalPotential;
        
            for (int ib1 = 0; ib1 < nb1; ib1++) {
              for (int ib2 = 0; ib2 < nb2; ib2++) {

              double en2 = state2Energies(ib2);
              double en1 = state1Energies(ib1);

              //double fermi1 = elBandStructure.getParticle().getPopulation(en1,temp,chemPot);
              //double fermi2 = elBandStructure.getParticle().getPopulation(en2,temp,chemPot);

              // loop on temperature
                // https://arxiv.org/pdf/1409.1268.pdf
                // double rate =
                //    coupling(ib1, ib2, ib3) 
                //    * (fermi1 - fermi2)
                //    * smearingValues(ib1, ib2, ib3) * norm / en3 * pi;

                // NOTE: although the expression above is formally correct,
                // fk-fk2 could be negative due to numerical noise.
                // so instead, we do:
                // fk-fk2 ~= dfk/dek dwq
                // However, we don't include the dwq here, as this is gSE^2, which
                // includes a factor of (1/wq)
                //double rate =
                //    coupling(ib1, ib2, ib3) * fermiTerm(iCalc, ik1, ib1)
                //    * smearingValues(ib1, ib2, ib3)
                //    * norm / temperatures(iCalc) * pi * k1Weight;

                if (smearingValues(ib1, ib2, ib3) <= 0.) { continue; }
                
                double rate = 
                    coupling(ib1, ib2, ib3) * //fermiTerm(iCalc, ik1, ib1)
                    smearingValues(ib1, ib2, ib3)
                    * norm * pi / en3 * k1Weight 
                    * sinh(0.5 * en3 / temperatures(iCalc)) / 
                    (2. * cosh( 0.5*(en2 - chemPot)/temperatures(iCalc))
                    * cosh(0.5 * (en1 - chemPot)/temperatures(iCalc))); 

                // if it's not a coupled matrix, this will be ibte3
                // We have to define shifted ibte3, or it will be further
                // shifted every loop.
                //
                // Additionally, these are only needed in no-sym case,
                // as coupled matrix never has sym, is always case = 0
                int iBte3Shift = ibte3;
                if(matrix.isCoupled) {
                  // translate into the phonon-self quadrant if it's a coupled bte
                  std::tuple<int,int> tup = matrix.shiftToCoupledIndices(ibte3, ibte3, particle, particle);
                  iBte3Shift = std::get<0>(tup);
                }

                // case of linewidth construction (the only case, for ph-el)
                linewidth->operator()(iCalc, 0, iBte3Shift) += rate;

                //NOTE: for eliashberg function, we could here add another vectorBTE object
                // as done with the linewidths here, slightly modified

              }
            }
          }
        }
      }
    }
  }
  mpi->barrier();
  // better to close loopPrint after the MPI barrier: all MPI are synced here
  loopPrint.close();
}

void phononElectronAcousticSumRule(CoupledScatteringMatrix &matrix,
	       			                    Context& context, 	
				                          std::shared_ptr<CoupledVectorBTE> phElLinewidths, 
				                          BaseBandStructure& elBandStructure,
                                  BaseBandStructure& phBandStructure) {

  if(context.getUseSymmetries()) {
    DeveloperError("Phonon-electron acoustic sum rule isn't implemented with symmetries.");
  } 

  // el statistics sweep should be stored in the coupled matrix
  StatisticsSweep& elStatisticsSweep = matrix.statisticsSweep;

  size_t Nq = phBandStructure.getNumPoints();
  size_t Nk = elBandStructure.getNumPoints();
  double Nkq = (Nk + Nq)/2.;
  size_t numPhStates = phBandStructure.getNumStates();
  size_t numElStates = elBandStructure.getNumStates();

  double spinFactor = 2.; // non spin pol = 2
  if (context.getHasSpinOrbit()) { spinFactor = 1.; }

  double volume = phBandStructure.getPoints().getCrystal().getVolumeUnitCell();
  double norm = 1./(Nkq * volume) * sqrt(spinFactor * Nkq / double(Nk)); 

  // this is for coupled matrices, only one temperature and mu value
  int iCalc = 0; 
  auto calcStat = elStatisticsSweep.getCalcStatistics(iCalc);
  double kBT = calcStat.temperature;
  double mu = calcStat.chemicalPotential;

  // Rqs = sum_km D_phel_qs,km sqrt(f(1-f))
  std::vector<double> Rqs(numPhStates);
  // used in the next step, sum_km step_fn(|Delph| - 1e-16)
  std::vector<double> weightDenominator(numPhStates); 

  // calculate Rqs and look at how far it is from zero
  for(auto matrixState : matrix.getAllLocalStates()) {

    // unpack the state info into matrix indices. 
    size_t iMat1 = std::get<0>(matrixState);
    size_t iMat2 = std::get<1>(matrixState);

    // check if this is a drag-related index
    // if iMat1 is electron, iMat2 is phonon 
    // ( we only sum over one quadrant for now) 
    if(( iMat1 < numElStates && iMat2 >= numElStates)) { //|| 
       		//(iMat1 > numElStates && iMat2 < numElStates) ) {

       size_t iMat2Ph = iMat2 - numElStates; // shift back to phonon index

       // here because symmetries are not used, we can directly use
       // the matrix indices without converting iBte/iMat -> iState
       StateIndex is1(iMat1);
       double energy = elBandStructure.getEnergy(is1);
       double f1mf = elBandStructure.getParticle().getPopPopPm1(energy, kBT, mu);
       Rqs[iMat2Ph] += matrix(iMat1,iMat2) * sqrt(f1mf); 
       // here we're summing over sum_km step_fn(|Delph| - 1e-16), where
       // if the argument is positive, the value is 1, and if it's <= 0 it's 0
       weightDenominator[iMat2Ph] += abs(matrix(iMat1,iMat2)) - 1e-16 > 0. ? 1. : 0;
    } 
  }
  mpi->allReduceSum(&Rqs);
  mpi->allReduceSum(&weightDenominator);

  //if(mpi->mpiHead()) std::cout << "Sum of the Rqs components: " << std::endl; //<< std::accumulate(Rqs.begin(),Rqs.end(),0) << std::endl;
  //for (auto i: Rqs) {
  //  std::cout << i << '\n';
  //} 
  //if(mpi->mpiHead()) { std::cout << "Weight denominator: " << std::endl; 
  //for (auto i: weightDenominator)
  //  std::cout << i << '\n';
  //}

  // now correct the drag terms by calculating
  // w_qs,km = step_fn (Delph - 1e-16) / sum_km step_fn(|Delph| - 1e-16)
  // and then 
  // Dphel_qs,km^corrected = Dphel_qs,km - w_qs,km * Rqs / sqrt(f(1-f)
  for(auto matrixState : matrix.getAllLocalStates()) {
    
    // unpack the state info into matrix indices 
    size_t iMat1 = std::get<0>(matrixState);
    size_t iMat2 = std::get<1>(matrixState);

    // if iMat1 is electron, iMat2 is phonon (upper quadrant only for now)
    if(( iMat1 < numElStates && iMat2 >= numElStates)) { 
          
      size_t iMat2Ph = iMat2 - numElStates; // shift back to phonon index

      // here because symmetries are not used, we can directly use
      // the matrix indices without converting iBte/iMat -> iState
      StateIndex is1(iMat1);
      double energy = elBandStructure.getEnergy(is1);
      double f1mf = elBandStructure.getParticle().getPopPopPm1(energy, kBT, mu);    

      if(weightDenominator[iMat2Ph] == 0) continue; // avoid nan
      if(sqrt(f1mf) < 1e-16) continue; // avoid nan

      double weightNumerator = abs(matrix(iMat1,iMat2)) - 1e-16 > 0. ? 1. : 0;   
      double weight  = weightNumerator / weightDenominator[iMat2Ph];

      // replace the matrix elements with the corrected ones
      matrix(iMat1, iMat2) = matrix(iMat1, iMat2) - weight * Rqs[iMat2Ph] / sqrt(f1mf);  

    }
  }
  /*
  // sanity check the correction 
  std::vector<double> RqsSanity(numPhStates);
  for(auto matrixState : matrix.getAllLocalStates()) {

    // unpack the state info into matrix indices. 
    size_t iMat1 = std::get<0>(matrixState);
    size_t iMat2 = std::get<1>(matrixState);

    // check if this is a drag-related index
    // if iMat1 is electron, iMat2 is phonon 
    if(( iMat1 < numElStates && iMat2 >= numElStates)) { 

      size_t iMat2Ph = iMat2 - numElStates; // shift back to phonon index

      // here because symmetries are not used, we can directly use
      // the matrix indices without converting iBte/iMat -> iState
      StateIndex is1(iMat1);
      double energy = elBandStructure.getEnergy(is1);
      double f1mf = elBandStructure.getParticle().getPopPopPm1(energy, kBT, mu);
      RqsSanity[iMat2Ph] += matrix(iMat1,iMat2) * sqrt(f1mf); 
    } 
  }
  mpi->allReduceSum(&RqsSanity);

  if(mpi->mpiHead()) std::cout << "Rqs sanity check: " << std::endl; //<< std::accumulate(Rqs.begin(),Rqs.end(),0) << std::endl;
  for (auto i: RqsSanity) {
    std::cout << i << ' ';
  }
  */ 
  /*
  // compute the corrected phel linewidths
  if(size_t(phElLinewidths->data.cols()) != numPhStates + numElStates) { 
    DeveloperError("Attempting to use ph-el ASR on a "
		    "linewidths vector of the wrong size!");
  } 
  // we will replace vectorBTE's data internal matrix with this
  Eigen::MatrixXd newPhElLinewidths(phElLinewidths->data.rows(),
	  			                          phElLinewidths->data.cols());
  newPhElLinewidths.setZero();  
  
  // sum over all the off diagonal states from the el,ph drag quandrant
  // to reconstruct the linewidths 
  for(auto matrixState : matrix.getAllLocalStates()) {

    // unpack the state info into matrix indices 
    size_t iMat1 = std::get<0>(matrixState);
    size_t iMat2 = std::get<1>(matrixState);

    // if iMat2 is phonon, iMat1 is electron
    if(( iMat2 >= numElStates && iMat1 < numElStates)) { 

      size_t iMat2Ph = iMat2 - numElStates; // shift back to phonon index, 
      					  // for use in bandstructure indexed objects

      // here because symmetries are not used, we can directly use
      // the matrix indices without converting iBte/iMat -> iState
      StateIndex is2(iMat2Ph); // phonon
      double phEnergy = phBandStructure.getEnergy(is2);
      double nnp1 = phBandStructure.getParticle().getPopPopPm1(phEnergy, kBT, 0);
      StateIndex is1(iMat1); // phonon 
      double elEnergy = elBandStructure.getEnergy(is1);
      double f1mf = elBandStructure.getParticle().getPopPopPm1(elEnergy, kBT, mu);

      //if (phEnergy < phononCutoff) { continue; } // skip the acoustic modes
      //if (sqrt(nnp1) * phEnergy < 1e-16) { continue; } // skip values which would cause divergence

      newPhElLinewidths(iCalc, iMat2) -= matrix(iMat1, iMat2) * sqrt(f1mf) * (elEnergy - mu) // * norm 
      					/ ( sqrt(nnp1) * phEnergy ); 
    }
  }
  mpi->allReduceSum(&newPhElLinewidths); 

  // print statement to print new and old linewidths side by side
if(mpi->mpiHead()) {
  std::cout << "compare " << std::endl;
  for (int i = numElStates; i<200+numElStates; i++) {

    StateIndex istate(i-numElStates);
    auto qPlusRemap = phBandStructure.getWavevector(istate);

    // jesus christ we have to improve this lookup
    auto qMinus = -1*phBandStructure.getWavevector(istate);
    Eigen::Vector3d qMinusCrys = phBandStructure.getPoints().cartesianToCrystal(qMinus);
    WavevectorIndex qIdxMinusRemap = WavevectorIndex(phBandStructure.getPointIndex(qMinusCrys));
    auto qMinusRemapCart = phBandStructure.getWavevector(qIdxMinusRemap);
    Eigen::Vector3d qMinusRemapCrys = phBandStructure.getPoints().cartesianToCrystal(qMinusRemapCart);
    Eigen::Vector3d qPlusRemapCrys = phBandStructure.getPoints().cartesianToCrystal(qPlusRemap);

    auto qPlusWS = phBandStructure.getPoints().bzToWs(qPlusRemapCrys,Points::crystalCoordinates);
    auto qMinusWS = phBandStructure.getPoints().bzToWs(qMinusRemapCrys,Points::crystalCoordinates);

    if( newPhElLinewidths(iCalc,i) <= 0 ) { 
      newPhElLinewidths(iCalc,i) = phElLinewidths->data(iCalc,i);
    }

    double x = newPhElLinewidths(iCalc,i); 
    double y = phElLinewidths->data(iCalc,i);
    if (abs(x) < 1e-15) x = 0;
    if (abs(y) < 1e-15) y = 0;

    if(x == 0 && y == 0) continue;
    //if( !( abs(abs(x/y) - 0.5) < 1e-3)) continue; 
    //std::cout <<  x << " " << y << "  " << x/y << " | "  << qPlusRemapCrys.transpose() << " " << qMinusRemapCrys.transpose() << " | " << qPlusWS.transpose() << " " << qMinusWS.transpose() << "\n" ;

  } 
}
  // replace the linewidth data
  //phElLinewidths->data = newPhElLinewidths; 
*/
}

