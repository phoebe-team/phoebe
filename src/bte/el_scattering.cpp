#include "constants.h"
#include "helper_el_scattering.h"
#include "io.h"
#include "mpiHelper.h"
#include "periodic_table.h"
#include "interaction_elph.h"
#include "el_scattering_matrix.h"
#include "vector_bte.h"
#include "delta_function.h"

// TODO maybe remove some of these that are not necessary

// 3 cases:
// theMatrix and linewidth is passed: we compute and store in memory the
// scattering
//       matrix and the diagonal
// inPopulation+outPopulation is passed: we compute the action of the
//       scattering matrix on the in vector, returning outVec = sMatrix*vector
// only linewidth is passed: we compute only the linewidths

void addElPhScattering(BaseElScatteringMatrix &matrix, Context &context,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations,
                       int &switchCase,
                       std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                       Eigen::MatrixXd &innerFermi, //Eigen::MatrixXd &outerBose,
                       BaseBandStructure &innerBandStructure,
                       BaseBandStructure &outerBandStructure,
                       PhononH0 &phononH0,
                       InteractionElPhWan &couplingElPhWan,
                       std::shared_ptr<VectorBTE> linewidth) {

  StatisticsSweep &statisticsSweep = matrix.statisticsSweep;

  DeltaFunction *smearing = DeltaFunction::smearingFactory(context, innerBandStructure);
  if (smearing->getType() == DeltaFunction::tetrahedron) {
    Error("Tetrahedron method not supported by electron scattering. It will not work properly with a\n"
          "filtering window, and we will almost always need to use this for electrons.");
    // that's because it doesn't work with the window the way it's implemented,
    // and we will almost always have a window for electrons
  }

  // generate the intermediate points to be summed over
  HelperElScattering pointHelper(innerBandStructure, outerBandStructure,
                    statisticsSweep, smearing->getType(), phononH0, couplingElPhWan);

  bool withSymmetries = context.getUseSymmetries();
  int numCalculations = statisticsSweep.getNumCalculations();

  // spin factor not applied to scattering rate, but later in Onsager coefficients
  //double spinFactor = 2.; // nonspin pol = 2
  //if (context.getHasSpinOrbit()) { spinFactor = 1.; }
  double norm = 1. / context.getKMesh().prod(); // we sum over q here, but the two meshes are the same

  LoopPrint loopPrint("computing el-ph contribution to the scattering matrix",
                                        "k-points", int(kPairIterator.size()));

  for (auto t1 : kPairIterator) {

    loopPrint.update();
    int ik1 = std::get<1>(t1);
    auto ik2Indexes = std::get<0>(t1);
    WavevectorIndex ik1Idx(ik1);

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

    Eigen::Vector3d k1C = outerBandStructure.getWavevector(ik1Idx);
    Eigen::VectorXd state1Energies = outerBandStructure.getEnergies(ik1Idx);
    auto nb1 = int(state1Energies.size());
    Eigen::MatrixXd v1s = outerBandStructure.getGroupVelocities(ik1Idx);
    Eigen::MatrixXcd eigenVector1 = outerBandStructure.getEigenvectors(ik1Idx);

    couplingElPhWan.cacheElPh(eigenVector1, k1C);

    pointHelper.prepare(k1C, ik2Indexes);

    // prepare batches based on memory usage
    auto nk2 = int(ik2Indexes.size());
    int numBatches = couplingElPhWan.estimateNumBatches(nk2, nb1);

    // loop over batches of q1s
    // later we will loop over the q1s inside each batch
    // this is done to optimize the usage and data transfer of a GPU
    for (int iBatch = 0; iBatch < numBatches; iBatch++) {

      // start and end point for current batch
      int start = nk2 * iBatch / numBatches;
      int end = nk2 * (iBatch + 1) / numBatches;
      int batch_size = end - start;

      std::vector<Eigen::Vector3d> allQ3C(batch_size);
      std::vector<Eigen::VectorXd> allStates3Energies(batch_size);
      std::vector<int> allNb3(batch_size);
      std::vector<Eigen::MatrixXcd> allEigenVectors3(batch_size);
      std::vector<Eigen::MatrixXd> allV3s(batch_size);
      std::vector<Eigen::MatrixXd> allBose3Data(batch_size);
      std::vector<Eigen::VectorXcd> allPolarData(batch_size);

      std::vector<Eigen::Vector3d> allK2C(batch_size);
      std::vector<Eigen::MatrixXcd> allEigenVectors2(batch_size);
      std::vector<Eigen::VectorXd> allState2Energies(batch_size);
      std::vector<Eigen::MatrixXd> allV2s(batch_size);

      Kokkos::Profiling::pushRegion("preprocessing loop");
      // do prep work for all values of q1 in current batch,
      // store stuff needed for couplings later
#pragma omp parallel for default(none) shared(allNb3, allEigenVectors3, allV3s, allBose3Data, ik2Indexes, pointHelper, allQ3C, allStates3Energies, batch_size, start, allK2C, allState2Energies, allV2s, allEigenVectors2, k1C, allPolarData,innerBandStructure)
      for (int ik2Batch = 0; ik2Batch < batch_size; ik2Batch++) {
        int ik2 = ik2Indexes[start + ik2Batch];
        WavevectorIndex ik2Idx(ik2);
        allK2C[ik2Batch] = innerBandStructure.getWavevector(ik2Idx);
        allState2Energies[ik2Batch] = innerBandStructure.getEnergies(ik2Idx);
        allV2s[ik2Batch] = innerBandStructure.getGroupVelocities(ik2Idx);
        allEigenVectors2[ik2Batch] = innerBandStructure.getEigenvectors(ik2Idx);
        auto t2 = pointHelper.get(k1C, ik2);
        allQ3C[ik2Batch] = std::get<0>(t2);
        allStates3Energies[ik2Batch] = std::get<1>(t2);
        allNb3[ik2Batch] = std::get<2>(t2);
        allEigenVectors3[ik2Batch] = std::get<3>(t2);
        allV3s[ik2Batch] = std::get<4>(t2);
        allBose3Data[ik2Batch] = std::get<5>(t2);
        allPolarData[ik2Batch] = std::get<6>(t2);
      }
      Kokkos::Profiling::popRegion();

      couplingElPhWan.calcCouplingSquared(eigenVector1, allEigenVectors2,
                                           allEigenVectors3, allQ3C, k1C, allPolarData);

      Kokkos::Profiling::pushRegion("symmetrize coupling");
#pragma omp parallel for
      for (int ik2Batch = 0; ik2Batch < batch_size; ik2Batch++) {
        matrix.symmetrizeCoupling(
            couplingElPhWan.getCouplingSquared(ik2Batch),
            state1Energies, allState2Energies[ik2Batch], allStates3Energies[ik2Batch]
        );
      }
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("postprocessing loop");
      // do postprocessing loop with batch of couplings
      for (int ik2Batch = 0; ik2Batch < batch_size; ik2Batch++) {

        int ik2 = ik2Indexes[start + ik2Batch];

        Eigen::Tensor<double, 3>& coupling = couplingElPhWan.getCouplingSquared(ik2Batch);

        Eigen::Vector3d k2C = allK2C[ik2Batch];
        auto t3 = innerBandStructure.getRotationToIrreducible(k2C, Points::cartesianCoordinates);
        int ik2Irr = std::get<0>(t3);
        Eigen::Matrix3d rotation = std::get<1>(t3);

        WavevectorIndex ik2Idx(ik2);
        WavevectorIndex ik2IrrIdx(ik2Irr);

        Eigen::VectorXd state2Energies = allState2Energies[ik2Batch];
        Eigen::MatrixXd v2s = allV2s[ik2Batch];

        Eigen::MatrixXd bose3Data = allBose3Data[ik2Batch];
        Eigen::MatrixXd v3s = allV3s[ik2Batch];
        Eigen::VectorXd state3Energies = allStates3Energies[ik2Batch];
        //Eigen::Vector3d QC = allQ3C[ik2Batch]; // just used for printing in tests

        auto nb2 = int(state2Energies.size());
        auto nb3 = int(state3Energies.size());

        Eigen::MatrixXd sinh3Data(nb3, numCalculations);
#pragma omp parallel for collapse(2)
        for (int ib3 = 0; ib3 < nb3; ib3++) {
          for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
            double en3 = state3Energies(ib3);
            double kT = statisticsSweep.getCalcStatistics(iCalc).temperature;
            sinh3Data(ib3, iCalc) = 0.5 / sinh(0.5 * en3 / kT);
          }
        }

        for (int ib2 = 0; ib2 < nb2; ib2++) {

          double en2 = state2Energies(ib2);
          int is2 = innerBandStructure.getIndex(ik2Idx, BandIndex(ib2));
          int is2Irr = innerBandStructure.getIndex(ik2IrrIdx, BandIndex(ib2));
          StateIndex is2Idx(is2);
          StateIndex is2IrrIdx(is2Irr);
          BteIndex ind2Idx = innerBandStructure.stateToBte(is2IrrIdx);
          int iBte2 = ind2Idx.get();

          for (int ib1 = 0; ib1 < nb1; ib1++) {

            double en1 = state1Energies(ib1);
            int is1 = outerBandStructure.getIndex(ik1Idx, BandIndex(ib1));
            StateIndex is1Idx(is1);
            BteIndex ind1Idx = outerBandStructure.stateToBte(is1Idx);
            int iBte1 = ind1Idx.get();

            for (int ib3 = 0; ib3 < nb3; ib3++) {

              double en3 = state3Energies(ib3);

              // remove small divergent phonon energies
              if (en3 < phEnergyCutoff) { continue; }

              double delta1, delta2;
              if (smearing->getType() == DeltaFunction::gaussian) {
                delta1 = smearing->getSmearing(en1 - en2 + en3);
                delta2 = smearing->getSmearing(en1 - en2 - en3);
              } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {
                Eigen::Vector3d vdiff = v1s.row(ib1) - v2s.row(ib2);
                delta1 = smearing->getSmearing(en1 - en2 + en3, vdiff);
                delta2 = smearing->getSmearing(en1 - en2 - en3, vdiff);
              } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
                Eigen::Vector3d v1 = v1s.row(ib1);
                Eigen::Vector3d v2 = v2s.row(ib2);
                Eigen::Vector3d v3 = v3s.row(ib3);
                delta1 = smearing->getSmearing(en1 - en2 + en3, v1, v2, v3);
                delta2 = smearing->getSmearing(en1 - en2 - en3, v1, v2, v3);
              } else {
                delta1 = smearing->getSmearing(en3 + en1, is2Idx);
                delta2 = smearing->getSmearing(en3 - en1, is2Idx);
              }

              if (delta1 <= 0. && delta2 <= 0.) { continue; } // doesn't contribute

              // compute the extra 1-cosTheta term needed for MRTA
              double cosTheta = 1. - (v1s.row(ib1).dot(v2s.row(ib2))
                    / (v1s.row(ib1).norm() * v2s.row(ib2).norm()));

              // loop on temperature
              for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

                //double fermi1 = outerFermi(iCalc, iBte1);
                double fermi2 = innerFermi(iCalc, iBte2);
                double bose3 = bose3Data(iCalc, ib3);
                double bose3Symm = sinh3Data(ib3, iCalc); // 1/2/sinh() term
                //double mu = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;

                // Calculate transition probability W+
                double rate = 
                    coupling(ib1, ib2, ib3) * ((fermi2 + bose3) * delta1
                       + (1. - fermi2 + bose3) * delta2) * norm / en3 * pi;

                double rateMR = rate * cosTheta;

                double rateOffDiagonal = -
                      coupling(ib1, ib2, ib3) * bose3Symm * (delta1 + delta2)
                      * norm / en3 * pi;

                if (switchCase == 0) {

                  if (withSymmetries) {
                    for (int i : {0, 1, 2}) {
                      CartIndex iIndex(i);
                      int iMat1 = matrix.getSMatrixIndex(ind1Idx, iIndex);
                      for (int j : {0, 1, 2}) {
                        CartIndex jIndex(j);
                        int iMat2 = matrix.getSMatrixIndex(ind2Idx, jIndex);
                        if (matrix.theMatrix.indicesAreLocal(iMat1, iMat2)) {
                          if (i == 0 && j == 0) {
                            linewidth->operator()(iCalc, 0, iBte1) += rate;

                            matrix.linewidthMR->operator()(iCalc, 0, iBte1) += rateMR;
                          }
                          if (is1 != is2Irr) {
                            matrix.theMatrix(iMat1, iMat2) +=
                                rotation.inverse()(i, j) * rateOffDiagonal;
                          }
                        }
                      }
                    }
                  } else {
                    // Note: we double check that the indices are local,
                    // but because we selected pairs which were local to supply to the function call,
                    // this isn't really necessary
                    if (matrix.theMatrix.indicesAreLocal(iBte1, iBte2)) {
                      linewidth->operator()(iCalc, 0, iBte1) += rate;

                      matrix.linewidthMR->operator()(iCalc, 0, iBte1) += rateMR;
		                  matrix.theMatrix(iBte1, iBte2) += rateOffDiagonal;

                      // if we're not symmetrizing the matrix, and we have
                      // dropped down to only using the upper triangle of the matrix, we must fill
                      // in linewidths twice, using detailed balance, in order to get the right rates
                      if(!context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) {
                        linewidth->operator()(iCalc, 0, iBte2) += rate;
                        matrix.linewidthMR->operator()(iCalc, 0, iBte2) += rateMR;
		                  }
                    }
                  }
                } else if (switchCase == 1) {
                  // case of matrix-vector multiplication
                  // we build the scattering matrix A = S*n(n+1)

                  for (unsigned int iVec = 0; iVec < inPopulations.size();
                       iVec++) {
                    Eigen::Vector3d inPopRot;
                    inPopRot.setZero();
                    for (int i : {0, 1, 2}) {
                      for (int j : {0, 1, 2}) {
                        inPopRot(i) += rotation.inverse()(i, j) * inPopulations[iVec](iCalc, j, iBte2);
                      }
                    }
                    for (int i : {0, 1, 2}) {
                      if (is1 != is2Irr) {
                        outPopulations[iVec](iCalc, i, iBte1) +=
                            rateOffDiagonal * inPopRot(i);
                      }
                      outPopulations[iVec](iCalc, i, iBte1) +=
                          rate * inPopulations[iVec](iCalc, i, iBte1);
                    }
                  }
                } else {
                  // case of linewidth construction
                  linewidth->operator()(iCalc, 0, iBte1) += rate;
                  matrix.linewidthMR->operator()(iCalc, 0, iBte1) += rateMR;
                }
              }
            }
          }
        }
      } // loop over ik2s in the batch
      Kokkos::Profiling::popRegion();
    } // loop over batches
  } // kpair iterator loop (defines k1)
  loopPrint.close();
}


void addChargedImpurityScattering(BaseElScatteringMatrix &matrix, Context &context,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations,
                       int &switchCase,
                       std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                       BaseBandStructure &innerBandStructure,
                       BaseBandStructure &outerBandStructure,
                       std::shared_ptr<VectorBTE> linewidth) {

  // we implement here an expression from Fiorentini and Bonini
  // https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.085204
  // Helpfully symmetrized for use in Phoebe by Michele

  // The implemented expression is:
  // S_km,(k+q)m' = 2pi*Z^2*ni*e^4       delta(E_k'm' - Ekm)       ( f_k'm'(1-f_k'm') )
  //                ----------------    --------------------- sqrt ( ---------------- )
  //               hbar (eps_r*eps0)^2   (beta ^2 + |q|^2 )^2      (  f_km  (1-f_km)  )


  // define some constants, which will become inputs from context later
  // used later to calculate the screening length
  int dimensionality = context.getDimensionality();
  double ni = -1e19 * std::pow(distanceBohrToCm, dimensionality);  // impurity density
  double Z = -1.;                      // charge of the impurity
  double epsReps0 = 9.7;               // relative permitivity * eps0, units are (e^2/(eV * mu meters))
  epsReps0 *= (energyRyToEv * distanceBohrToMum);
  double invEpsR_Eps0 = 1. / epsReps0;
  double Nk = context.getKMesh().prod(); // the BTE discretization is over kmesh
  double prefactor = ( twoPi * std::pow(Z,2) * ni ) * std::pow(invEpsR_Eps0,2) * 1./Nk;

  // the object which contains temperatures, dopings, chemical potentials
  StatisticsSweep &statisticsSweep = matrix.statisticsSweep;
  bool withSymmetries = context.getUseSymmetries();
  int numCalculations = statisticsSweep.getNumCalculations();
  double norm = 1. / context.getKMesh().prod(); // the intermediate mesh summed over

  // set up the dirac delta function we will use
  DeltaFunction *smearing = DeltaFunction::smearingFactory(context, innerBandStructure);
  // TODO switch this to for window?
  if (smearing->getType() == DeltaFunction::tetrahedron) {
    DeveloperError("Tetrahedron smearing should be blocked when a filtering window is used, which is almost all el calculations.");
  }

  if(mpi->mpiHead()) {
    std::cout << "Adding impurity scattering to the scattering matrix." << std::endl;
  }

  Kokkos::Profiling::pushRegion("impurity scattering");

  // loop over pairs as in the electron phonon calculation,
  // where the pair is a first k1 and a batch of k2s
  for (auto t1 : kPairIterator) {

    //loopPrint.update();
    int ik1 = std::get<1>(t1);
    WavevectorIndex ik1Idx(ik1);
    Eigen::Vector3d k1C = outerBandStructure.getWavevector(ik1Idx);
    Eigen::VectorXd state1Energies = outerBandStructure.getEnergies(ik1Idx);
    Eigen::MatrixXd v1s = outerBandStructure.getGroupVelocities(ik1Idx);
    auto nb1 = int(state1Energies.size());

    auto ik2Indexes = std::get<0>(t1);

    // loop over k'
    for ( int ik2 : ik2Indexes ) {

      WavevectorIndex ik2Idx(ik2);
      Eigen::Vector3d k2C = innerBandStructure.getWavevector(ik2Idx);
      Eigen::VectorXd state2Energies = innerBandStructure.getEnergies(ik2Idx);
      Eigen::MatrixXd v2s = innerBandStructure.getGroupVelocities(ik2Idx);
      int nb2 = int(state2Energies.size());

      Eigen::Vector3d qC = k2C - k1C;   // TODO make sure this is an ok way to get Q
                                        // maybe it should be folded into the BZ or something

      auto t3 = innerBandStructure.getRotationToIrreducible(k2C, Points::cartesianCoordinates);
      int ik2Irr = std::get<0>(t3);
      Eigen::Matrix3d rotation = std::get<1>(t3);
      WavevectorIndex ik2IrrIdx(ik2Irr);

      for (int ib1 = 0; ib1 < nb1; ib1++) {

        // TODO to use symmetries, do we need to get irr index here
        double en1 = state1Energies(ib1);
        int is1 = outerBandStructure.getIndex(ik1Idx, BandIndex(ib1));
        StateIndex is1Idx(is1);
        BteIndex ind1Idx = outerBandStructure.stateToBte(is1Idx);
        int iBte1 = ind1Idx.get();

        for (int ib2 = 0; ib2 < nb2; ib2++) {

          double en2 = state2Energies(ib2);
          int is2 = innerBandStructure.getIndex(ik2Idx, BandIndex(ib2));
          int is2Irr = innerBandStructure.getIndex(ik2IrrIdx, BandIndex(ib2));
          StateIndex is2Idx(is2);
          StateIndex is2IrrIdx(is2Irr);
          BteIndex ind2Idx = innerBandStructure.stateToBte(is2IrrIdx);
          int iBte2 = ind2Idx.get();

          double delta = 0;
          if (smearing->getType() == DeltaFunction::gaussian) {
            delta = smearing->getSmearing(en2 - en1);
          // TODO set up adaptive smearing properly here
          } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {
            delta += smearing->getSmearing(en1 - en2, v2s.row(ib2));
            delta += smearing->getSmearing(en1 - en2, v1s.row(ib1));
            delta *= 0.5;
          } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
            Eigen::Vector3d v1 = v1s.row(ib1);
            Eigen::Vector3d v2 = v2s.row(ib2);
            Eigen::Vector3d v3; // feed zeros for the third one, as this transition has only two
            delta = smearing->getSmearing(en1 - en2, v1, v2, v3);
          } else {
            // TODO is this ok NOTE I don't think this is ever called, but
            // I'm following what was done in isotope smearing's delta
            delta = smearing->getSmearing(en1, is2Idx);
          }

          if (delta <= 0.) { continue; } // doesn't contribute

          // loop on temperature, doping etc
          for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

            double kBT = statisticsSweep.getCalcStatistics(iCalc).temperature;
            double beta2 = ni / (epsReps0 * kBT);
            double denominator = 1./(beta2 + qC.norm()); // TODO check that norm is good here

            // Calculate transition probability
            double rate = prefactor * std::pow(denominator,2) * delta;

            // compute the extra 1-cosTheta term needed for MRTA
            double cosTheta = 1. - (v1s.row(ib1).dot(v2s.row(ib2))
                                        / (v1s.row(ib1).norm() * v2s.row(ib2).norm()));
            double rateMR = rate * cosTheta;

            if (switchCase == 0) { // case of matrix construction
              if (context.getUseSymmetries()) {
                BteIndex iBte1Idx(iBte1);
                BteIndex iBte2Idx(iBte2);
                for (int i : {0, 1, 2}) {
                  CartIndex iIndex(i);
                  int iMat1 = matrix.getSMatrixIndex(iBte1Idx, iIndex);
                  for (int j : {0, 1, 2}) {
                    CartIndex jIndex(j);
                    int iMat2 = matrix.getSMatrixIndex(iBte2Idx, jIndex);
                    if (matrix.theMatrix.indicesAreLocal(iMat1, iMat2)) {
                      if (i == 0 && j == 0) {
                        linewidth->operator()(iCalc, 0, iBte1) += rate;
                        matrix.linewidthMR->operator()(iCalc, 0, iBte1) += rateMR;
		                  }
                      if (is1 != is2Irr) {
                        matrix.theMatrix(iMat1, iMat2) += rotation.inverse()(i, j) * rate;
                      }
                    }
                  }
                }
              } else {
                if (matrix.theMatrix.indicesAreLocal(iBte1, iBte2)) {
                  linewidth->operator()(iCalc, 0, iBte1) += rate;
                  matrix.linewidthMR->operator()(iCalc, 0, iBte1) += rateMR;
		              matrix.theMatrix(iBte1, iBte2) += rate;

                  // if we're not symmetrizing the matrix, and we have
                  // dropped down to only using the upper triangle of the matrix, we must fill
                  // in linewidths twice, using detailed balance, in order to get the right ratest
                  if(!context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) {
                    linewidth->operator()(iCalc, 0, iBte2) += rate;
		                matrix.linewidthMR->operator()(iCalc, 0, iBte2) += rateMR;

		              }
                }
              }
            } else if (switchCase == 1) { // case of matrix-vector multiplication
              for (unsigned int iInput = 0; iInput < inPopulations.size(); iInput++) {

                // here we rotate the populations from the irreducible point
                Eigen::Vector3d inPopRot;
                inPopRot.setZero();
                for (int i : {0, 1, 2}) {
                  for (int j : {0, 1, 2}) {
                    inPopRot(i) += rotation.inverse()(i, j) *
                                   inPopulations[iInput](iCalc, j, iBte2);
                  }
                }
                for (int i : {0, 1, 2}) {
                  if (is1 != is2Irr) {
                    outPopulations[iInput](iCalc, i, iBte1) += rate * inPopRot(i);
                  }
                  outPopulations[iInput](iCalc, i, iBte1) +=
                      rate * inPopulations[iInput](iCalc, i, iBte1);
                }
              }

            } else { // case of linewidth construction
              linewidth->operator()(iCalc, 0, iBte1) += rate;
	            matrix.linewidthMR->operator()(iCalc, 0, iBte1) += rateMR;
            }
          }
        }
      }
    }
  }
  Kokkos::Profiling::popRegion();
}

// TODO do we need to do this differently in the iterative/variational case?
void add_eeDMFT(BaseElScatteringMatrix &matrix, const Context &context,
                //std::vector<VectorBTE> &inPopulations,
                //std::vector<VectorBTE> &outPopulations,
                const int &switchCase,
                BaseBandStructure &outerBandStructure,
                std::shared_ptr<VectorBTE> linewidth) {

  // from a DMFT calculation, we have a linewidth of the form
  //    Gamma_ee = 2 C (omega^2 + pi^2 T^2)
  // where C is a user defined coefficient determined from DMFT

  // we need to iterate over the linewidths held on this process, and add this scattering
  // rate to the matrix. Because we do this with the parallel state iterator,
  // we will have to call an allReduceSum after this is called.

  StatisticsSweep &statisticsSweep = matrix.statisticsSweep;
  bool withSymmetries = context.getUseSymmetries();
  int numCalculations = statisticsSweep.getNumCalculations();

  // C coefficient in Ry^-1
  double Cx2 = 2. * context.getEeFermiLiquidCoefficient();
  //double kBInv = 1./kBoltzmannRy; // avoid repeated division

  // TODO OMP here
  for( auto is : outerBandStructure.parallelIrrStateIterator() ) {

    // get state energies at this kpt
    StateIndex isIdx(is);
    double Ek = outerBandStructure.getEnergy(isIdx);

    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

      //Eigen::MatrixXd &innerFermi,
      //double fermi = outerFermi(iCalc, iBte);  // TODO Does the diagonal need fermi term?

      // get calculation temperature
      double mu = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
      double EkSquared = (Ek-mu) * (Ek-mu);
      double piTSquared = piSquared * std::pow(statisticsSweep.getCalcStatistics(iCalc).temperature, 2);
      double rate = Cx2 * ( EkSquared + piTSquared );

      int iBte = outerBandStructure.stateToBte(isIdx).get();

      // simply add this to the linewidth
      linewidth->operator()(iCalc, 0, iBte) += rate;

    }
  }
}
