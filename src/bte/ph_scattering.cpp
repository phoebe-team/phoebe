#include "constants.h"
#include "helper_3rd_state.h"
#include "ph_scattering_matrix.h"
#include "io.h"
#include "mpiHelper.h"
#include <cmath>

// 3 cases:
// theMatrix and linewidth is passed: we compute and store in memory the
// scattering matrix and the diagonal
// inPopulation+outPopulation is passed: we compute the action of the
//       scattering matrix on the in vector, returning outVec = sMatrix*vector
// only linewidth is passed: we compute only the linewidths

// TODO check to see how we can simplify this function

// function to add phph scattering to a scattering matrix
void addPhPhScattering(BasePhScatteringMatrix &matrix, Context &context,
                                 std::vector<VectorBTE> &inPopulations,
                                 std::vector<VectorBTE> &outPopulations,
                                 int &switchCase,
                                 std::vector<std::tuple<std::vector<int>, int>> qPairIterator,
                                 Eigen::MatrixXd &innerBose, Eigen::MatrixXd &outerBose,
                                 BaseBandStructure &innerBandStructure,
                                 BaseBandStructure &outerBandStructure,
                                 PhononH0& phononH0,
                                 Interaction3Ph& coupling3Ph,
                                 std::shared_ptr<VectorBTE> linewidth) {

  // notes: + process is (1+2) -> 3
  //        - processes are (1+3)->2 and (3+2)->1

  // copy a few small things that don't take
  // much memory but will keep the code easier to read
  auto excludeIndices = matrix.excludeIndices;
  bool outputUNTimes = matrix.outputUNTimes;
  Particle particle = innerBandStructure.getParticle();
  const double pathLinewidthCutoff = 0.000001 / ryToCmm1; 

  // setup smearing using phonon band structure
  DeltaFunction *smearing = DeltaFunction::smearingFactory(context, innerBandStructure);

  if ( // innerBandStructure != outerBandStructure &&
     smearing->getType() == DeltaFunction::tetrahedron) {
    Error("Developer error: Tetrahedron smearing for transport untested and thus blocked");
    // May work for linewidths. Although this should be double-checked
  }

  // determine if this is lifetimes on a path or regular mesh
  bool outerEqualInnerMesh = false; // case of lifetimes on a path
  if (&innerBandStructure == &outerBandStructure) {
    // case of transport calculation
    outerEqualInnerMesh = true;
  }

  // generate basic properties from the function arguments
  int numCalculations = matrix.statisticsSweep.getNumCalculations();

  // note: innerNumFullPoints is the number of points in the full grid
  // may be larger than innerNumPoints, when we use ActiveBandStructure
  double norm = 1. / context.getQMesh().prod();

  Helper3rdState pointHelper(innerBandStructure, outerBandStructure, outerBose,
                             matrix.statisticsSweep, smearing->getType(), phononH0);
  LoopPrint loopPrint("computing ph-ph contribution to scattering matrix", "q-point pairs",
                      int(qPairIterator.size()));

  /** Very important: the code must be executed with a loop over q2 outside
   * and a loop over q1 inside. This is because the 3-ph coupling must compute
   * the Fourier transform on q1 and q2. The 3-ph class splits the Fourier
   * transform in two parts, recycling the FT over q2 for more values of q1.
   * Thus, we have some speed when executing in this order.
   * PointHelper too assumes that order of loop execution.
   */
  // outer loop over q2
  for (auto tup : qPairIterator) {

    std::vector<int> iq1Indexes = std::get<0>(tup);
    int iq2 = std::get<1>(tup);
    WavevectorIndex iq2Index(iq2);

    Point q2Point = innerBandStructure.getPoint(iq2);
    Eigen::VectorXd energies2 = innerBandStructure.getEnergies(iq2Index);
    auto nb2 = int(energies2.size());
    Eigen::MatrixXd v2s = innerBandStructure.getGroupVelocities(iq2Index);
    Eigen::Vector3d q2 = innerBandStructure.getWavevector(iq2Index);
    Eigen::MatrixXcd ev2 = innerBandStructure.getEigenvectors(iq2Index);

    auto nq1 = int(iq1Indexes.size());

    auto t = innerBandStructure.getRotationToIrreducible(
        q2Point.getCoordinates(Points::cartesianCoordinates), Points::cartesianCoordinates);
    int iq2Irr = std::get<0>(t);
    WavevectorIndex iq2IrrIndex(iq2Irr);
    Eigen::Matrix3d rotation = std::get<1>(t);
    // rotation such that qIrr = R * qRed

    loopPrint.update();
    pointHelper.prepare(iq1Indexes, iq2);

    // prepare batches based on memory usage
    int numBatches = coupling3Ph.estimateNumBatches(nq1, nb2);

    // precalculate D3cached for current value of q2
    coupling3Ph.cacheD3(q2);

    // loop over batches of q1s
    // later we will loop over the q1s inside each batch
    // this is done to optimize the usage and data transfer of a GPU
    for (int iBatch = 0; iBatch < numBatches; iBatch++) {

      // start and end point for current batch
      int start = nq1 * iBatch / numBatches;
      int end = nq1 * (iBatch + 1) / numBatches;
      int batch_size = end - start;

      std::vector<Eigen::Vector3d> q1_v(batch_size);
      std::vector<Eigen::MatrixXcd> ev1_v(batch_size);
      std::vector<Eigen::VectorXd> energies1_v(batch_size);
      std::vector<Eigen::MatrixXd> v1s_v(batch_size);
      std::vector<Eigen::MatrixXcd> ev3Plus_v(batch_size);
      std::vector<Eigen::MatrixXcd> ev3Minus_v(batch_size);
      std::vector<int> nb1_v(batch_size);
      std::vector<int> nb3Plus_v(batch_size);
      std::vector<int> nb3Minus_v(batch_size);

      std::vector<Eigen::VectorXd> energies3Plus_v(batch_size);
      std::vector<Eigen::MatrixXd> v3sPlus_v(batch_size);
      std::vector<Eigen::MatrixXd> bose3PlusData_v(batch_size);
      std::vector<Eigen::VectorXd> energies3Minus_v(batch_size);
      std::vector<Eigen::MatrixXd> v3sMinus_v(batch_size);
      std::vector<Eigen::MatrixXd> bose3MinusData_v(batch_size);

      // do prep work for all values of q1 in current batch,
      // store stuff needed for couplings later
#pragma omp parallel for default(none) shared(v3sMinus_v, v3sPlus_v, bose3MinusData_v, bose3PlusData_v, energies3Minus_v, energies3Plus_v, ev1_v, ev3Minus_v, ev3Plus_v, q1_v, nb1_v, nb3Minus_v, nb3Plus_v, batch_size, iq1Indexes, start, pointHelper, q2Point, v1s_v, energies1_v, outerBandStructure)
      for (int iq1Batch = 0; iq1Batch < batch_size; iq1Batch++) {

        int iq1 = iq1Indexes[start + iq1Batch];
        WavevectorIndex iq1Index(iq1);

        // note: for computing linewidths on a path, we must distinguish
        // that q1 and q2 are on different meshes, and that q3+/- may not
        // fall into known meshes and therefore needs to be computed

        Point q1Point = outerBandStructure.getPoint(iq1);
        Eigen::VectorXd energies1 = outerBandStructure.getEnergies(iq1Index);
        auto nb1 = int(energies1.size());
        Eigen::MatrixXd v1s = outerBandStructure.getGroupVelocities(iq1Index);

        auto tup1 = pointHelper.get(q1Point, q2Point, Helper3rdState::casePlus);
        auto tup2 = pointHelper.get(q1Point, q2Point, Helper3rdState::caseMinus);

        auto energies3Plus = std::get<1>(tup1);
        auto ev3Plus = std::get<3>(tup1);
        auto v3sPlus = std::get<4>(tup1);
        auto bose3PlusData = std::get<5>(tup1);

        auto energies3Minus = std::get<1>(tup2);
        auto ev3Minus = std::get<3>(tup2);
        auto v3sMinus = std::get<4>(tup2);
        auto bose3MinusData = std::get<5>(tup2);

        q1_v[iq1Batch] = outerBandStructure.getWavevector(iq1Index);
        nb1_v[iq1Batch] = nb1;
        energies1_v[iq1Batch] = energies1;
        v1s_v[iq1Batch] = v1s;
        nb3Plus_v[iq1Batch] = int(energies3Plus.size());
        nb3Minus_v[iq1Batch] = int(energies3Minus.size());
        ev1_v[iq1Batch] = outerBandStructure.getEigenvectors(iq1Index);
        ev3Plus_v[iq1Batch] = ev3Plus;
        ev3Minus_v[iq1Batch] = ev3Minus;

        energies3Plus_v[iq1Batch] = energies3Plus;
        v3sPlus_v[iq1Batch] = v3sPlus;
        bose3PlusData_v[iq1Batch] = bose3PlusData;
        energies3Minus_v[iq1Batch] = energies3Minus;
        v3sMinus_v[iq1Batch] = v3sMinus;
        bose3MinusData_v[iq1Batch] = bose3MinusData;
      }

      // calculate batch of couplings
      auto tuple1 = coupling3Ph.getCouplingsSquared(
          q1_v, q2, ev1_v, ev2, ev3Plus_v, ev3Minus_v,
          nb1_v, nb2, nb3Plus_v, nb3Minus_v);
      auto couplingPlus_v = std::get<0>(tuple1);
      auto couplingMinus_v = std::get<1>(tuple1);

#pragma omp parallel for
      for (int iq1Batch = 0; iq1Batch < batch_size; iq1Batch++) {
        matrix.symmetrizeCoupling(
            couplingPlus_v[iq1Batch], energies1_v[iq1Batch], energies2, energies3Plus_v[iq1Batch]);
        matrix.symmetrizeCoupling(
            couplingMinus_v[iq1Batch], energies1_v[iq1Batch], energies2, energies3Minus_v[iq1Batch]);
      }

      // do postprocessing loop with batch of couplings
      for (int iq1Batch = 0; iq1Batch < batch_size; iq1Batch++) {

        int iq1 = iq1Indexes[start + iq1Batch];
        WavevectorIndex iq1Index(iq1);
        auto couplingPlus = couplingPlus_v[iq1Batch];
        auto couplingMinus = couplingMinus_v[iq1Batch];
        Eigen::VectorXd energies1 = energies1_v[iq1Batch];
        auto nb1 = int(energies1.size());
        Eigen::MatrixXd v1s = v1s_v[iq1Batch];

        auto energies3Plus = energies3Plus_v[iq1Batch];
        auto nb3Plus = energies3Plus.size();
        auto v3sPlus = v3sPlus_v[iq1Batch];
        auto bose3PlusData = bose3PlusData_v[iq1Batch];

        auto energies3Minus = energies3Minus_v[iq1Batch];
        auto nb3Minus = energies3Minus.size();
        auto v3sMinus = v3sMinus_v[iq1Batch];
        auto bose3MinusData = bose3MinusData_v[iq1Batch];

        for (int ib1 = 0; ib1 < nb1; ib1++) {

          double en1 = energies1(ib1);
          int is1 = outerBandStructure.getIndex(iq1Index, BandIndex(ib1));
          StateIndex is1Idx(is1);
          BteIndex iBte1Idx = outerBandStructure.stateToBte(is1Idx);
          int iBte1 = iBte1Idx.get();

          // discard states near gamma that might cause divergence
          if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) !=
              excludeIndices.end()) {
            continue;
         }

          for (int ib2 = 0; ib2 < nb2; ib2++) {

            double en2 = energies2(ib2);
            int is2 = innerBandStructure.getIndex(iq2Index, BandIndex(ib2));
            int is2Irr = innerBandStructure.getIndex(iq2IrrIndex, BandIndex(ib2));
            StateIndex is2Idx(is2);
            StateIndex is2IrrIdx(is2Irr);
            BteIndex iBte2Idx = innerBandStructure.stateToBte(is2IrrIdx);
            int iBte2 = iBte2Idx.get();

            if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte2) != excludeIndices.end()) {
              continue;
            }

            for (int ib3 = 0; ib3 < nb3Plus; ib3++) {

              double en3Plus = energies3Plus(ib3);
              double enProd = en1 * en2 * en3Plus;

              // block description:
              // we have two cases;
              // 1) if we are doing transport calculations,
              //    we apply the cutoff cutting away low energy phonons
              //    with the cutoff imposed on each of the three energies
              // 2) however, when computing lifetimes e.g. on a path, we impose
              //    the criterion on the product of the three energies.
              //    This is done to avoid large spikes in lifetimes along a path
              //    which can appear when the q' mesh has a very small offset
              //    that integrates very differently from q' mesh with large
              //    offset. Note anyway that the difference between the two
              //    criteria should disappear when huge q-meshes are used
              if (outerEqualInnerMesh) {
                if (en1 < phEnergyCutoff || en2 < phEnergyCutoff || en3Plus < phEnergyCutoff) {
                  continue;
                }
              } else {
                if (enProd < pathLinewidthCutoff) { continue; }
              }

              double deltaPlus;
              if (smearing->getType() == DeltaFunction::gaussian) {
                deltaPlus = smearing->getSmearing(en1 + en2 - en3Plus);
              } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {
                Eigen::Vector3d v = v2s.row(ib2) - v3sPlus.row(ib3);
                deltaPlus = smearing->getSmearing(en1 + en2 - en3Plus, v);
              } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
                Eigen::Vector3d v1 = v1s.row(ib1);
                Eigen::Vector3d v2 = v2s.row(ib2);
                Eigen::Vector3d v3 = v3sPlus.row(ib3);
                deltaPlus = smearing->getSmearing(en1 + en2 - en3Plus, v1, v2, v3);
              } else {
                deltaPlus = smearing->getSmearing(en3Plus - en1, is2Idx);
              }

              if (deltaPlus <= 0.) { continue; }

              // loop on temperature
              for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

                double bose1 = outerBose(iCalc, iBte1);
                double bose2 = innerBose(iCalc, iBte2);
                double bose3Plus = bose3PlusData(iCalc, ib3);

                // Calculate transition probability W+
                double ratePlus = pi * 0.25 * bose1 * bose2 * (bose3Plus + 1.) *
                    couplingPlus(ib1, ib2, ib3) * deltaPlus * norm / enProd;

                // if it's not a coupled matrix, these will be just iBte1 and iBte2
                // We have to define shifted versions, or they will be further
                // shifted every loop.
                //
                // Additionally, these are only needed in no-sym case,
                // as coupled matrix never has sym, is always case = 0
                int iBte1Shift = iBte1;
                int iBte2Shift = iBte2;
                BteIndex iBte1ShiftIdx(iBte1);
                BteIndex iBte2ShiftIdx(iBte2);
                if(matrix.isCoupled) {
                  // translate these into the phonon-phonon quadrant if it's a coupled bte
                  std::tuple<int,int> tup =
                        matrix.shiftToCoupledIndices(iBte1, iBte2, particle, particle);
                  iBte1Shift = std::get<0>(tup);
                  iBte2Shift = std::get<1>(tup);
                  iBte1ShiftIdx = BteIndex(iBte1Shift);
                  iBte2ShiftIdx = BteIndex(iBte2Shift);
                }

                if (switchCase == 0) { // case of matrix construction
                  if (context.getUseSymmetries()) {
                    for (int i : {0, 1, 2}) {
                      for (int j : {0, 1, 2}) {
                        CartIndex iIndex(i);
                        CartIndex jIndex(j);
                        int iMat1 = matrix.getSMatrixIndex(iBte1Idx, iIndex);
                        int iMat2 = matrix.getSMatrixIndex(iBte2Idx, jIndex);
                        if (matrix.theMatrix.indicesAreLocal(iMat1, iMat2)) {
                          if (i == 0 && j == 0) {
                            linewidth->operator()(iCalc, 0, iBte1) += 0.5 * ratePlus;
                          }
                          if (is1 != is2Irr) {
                            matrix.theMatrix(iMat1, iMat2) += rotation.inverse()(i, j) * ratePlus;
                          }
                        }
                      }
                    }
                  } else {
                    if (matrix.theMatrix.indicesAreLocal(iBte1Shift, iBte2Shift)) {
                      linewidth->operator()(iCalc, 0, iBte1Shift) += 0.5 * ratePlus;
                      matrix.theMatrix(iBte1Shift, iBte2Shift) += ratePlus;

                      // if we're not symmetrizing the matrix, and we have
                      // dropped down to only using the upper triangle of the matrix, we must fill
                      // in linewidths twice, using detailed balance, in order to get the right ratest
                      if(!context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) {
                        linewidth->operator()(iCalc, 0, iBte2Shift) += 0.5 * ratePlus;
                      }
                    }
                  }

                } else if (switchCase == 1) { // case of matrix-vector multiplication
                  // we build the scattering matrix A = S*n(n+1)
                  // here we rotate the populations from the irreducible point
                  for (unsigned int iInput = 0; iInput < inPopulations.size();
                       iInput++) {
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
                        outPopulations[iInput](iCalc, i, iBte1) += ratePlus * inPopRot(i);
                      }
                      outPopulations[iInput](iCalc, i, iBte1) +=
                          0.5 * ratePlus * inPopulations[iInput](iCalc, i, iBte1);
                    }
                  }

                } else { // case of linewidth construction
                  linewidth->operator()(iCalc, 0, iBte1) += 0.5 * ratePlus;
                  if(outputUNTimes) {
                    Point q1 = outerBandStructure.getPoint(iq1);
                    Point q2 = innerBandStructure.getPoint(iq2);
                    // check if this process is umklapp // TODO put this in hasUmklapp function
                    Eigen::Vector3d q1Cart = q1.getCoordinates(Points::cartesianCoordinates);
                    Eigen::Vector3d q2Cart = q2.getCoordinates(Points::cartesianCoordinates);
                    Eigen::Vector3d q1WS = outerBandStructure.getPoints().bzToWs(q1Cart, Points::cartesianCoordinates);
                    Eigen::Vector3d q2WS = outerBandStructure.getPoints().bzToWs(q2Cart, Points::cartesianCoordinates);
                    Eigen::Vector3d q3Cart = q1WS + q2WS;
                    Eigen::Vector3d q3fold = outerBandStructure.getPoints().bzToWs(q3Cart, Points::cartesianCoordinates);
                    bool isUmklapp = false;
                    if(abs((q3Cart-q3fold).norm()) > 1e-6) { isUmklapp = true; }
                    if(isUmklapp) {
                      matrix.internalDiagonalUmklapp->operator()(iCalc, 0, iBte1) += 0.5 * ratePlus;
                    } else {
                      matrix.internalDiagonalNormal->operator()(iCalc, 0, iBte1) += 0.5 * ratePlus;
                    }
                  }
                }
              }
            }

            for (int ib3 = 0; ib3 < nb3Minus; ib3++) {

              double en3Minus = energies3Minus(ib3);
              double enProd = en1 * en2 * en3Minus;

              // block description:
              // we have two cases;
              // 1) if we are doing transport calculations,
              //    we apply the cutoff cutting away low energy phonons
              //    with the cutoff imposed on each of the three energies.
              // 2) however, when computing lifetimes e.g. on a path, we impose
              //    the criterion on the product of the three energies.
              //    This is done to avoid large spikes in lifetimes along a path
              //    which can appear when the q' mesh has a very small offset
              //    that integrates very differently from q' mesh with large
              //    offset. Note anyway that the difference between the two
              //    criteria should disappear when huge q-meshes are used
              if (outerEqualInnerMesh) {
                if (en1 < phEnergyCutoff || en2 < phEnergyCutoff || en3Minus < phEnergyCutoff) {
                  continue;
                }
              } else {
                if (enProd < phEnergyCutoff) { continue; }
              }

              double deltaMinus1, deltaMinus2;
              if (smearing->getType() == DeltaFunction::gaussian) {
                deltaMinus1 = smearing->getSmearing(en1 + en3Minus - en2);
                deltaMinus2 = smearing->getSmearing(en2 + en3Minus - en1);
              } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {
                Eigen::Vector3d v = v2s.row(ib2) - v3sMinus.row(ib3);
                deltaMinus1 = smearing->getSmearing(en1 + en3Minus - en2, v);
                deltaMinus2 = smearing->getSmearing(en2 + en3Minus - en1, v);
              } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
                Eigen::Vector3d v1 = v1s.row(ib1);
                Eigen::Vector3d v2 = v2s.row(ib2);
                Eigen::Vector3d v3 = v3sMinus.row(ib3);
                deltaMinus1 = smearing->getSmearing(en1 + en3Minus - en2, v1, v2, v3);
                deltaMinus2 = smearing->getSmearing(en2 + en3Minus - en1, v1, v2, v3);
              } else {
                // Note: here I require inner == outer band structure
                deltaMinus1 = smearing->getSmearing(en1 + en3Minus, is2Idx);
                deltaMinus2 = smearing->getSmearing(en1 - en3Minus, is2Idx);
              }

              if (deltaMinus1 <= 0. && deltaMinus2 <= 0.)  continue;
              if (deltaMinus1 < 0.) deltaMinus1 = 0.;
              if (deltaMinus2 < 0.) deltaMinus2 = 0.;

                // TODO somehow bose3minus is ZERO! but only when we do coupled...
              for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

                double bose1 = outerBose(iCalc, iBte1);
                double bose2 = innerBose(iCalc, iBte2);
                double bose3Minus = bose3MinusData(iCalc, ib3);

                // Calculate transition probability W-
                double rateMinus1 =
                    pi * 0.25 * bose3Minus * bose1 * (bose2 + 1.) *
                    couplingMinus(ib1, ib2, ib3) * deltaMinus1 * norm / enProd;
                double rateMinus2 =
                    pi * 0.25 * bose2 * bose3Minus * (bose1 + 1.) *
                    couplingMinus(ib1, ib2, ib3) * deltaMinus2 * norm / enProd;

                // if it's coupled we translate the indices to the ph-ph quadrant
                int iBte1Shift = iBte1;   BteIndex iBte1ShiftIdx(iBte1);
                int iBte2Shift = iBte2;   BteIndex iBte2ShiftIdx(iBte2);
                if(matrix.isCoupled) {
                  std::tuple<int,int> tup =
                        matrix.shiftToCoupledIndices(iBte1, iBte2, particle, particle);
                  iBte1Shift = std::get<0>(tup);
                  iBte2Shift = std::get<1>(tup);
                  iBte1ShiftIdx = BteIndex(iBte1Shift);
                  iBte2ShiftIdx = BteIndex(iBte2Shift);
                }

                if (switchCase == 0) { // case of matrix construction
                  if (context.getUseSymmetries()) {
                    for (int i : {0, 1, 2}) {
                      for (int j : {0, 1, 2}) {
                        CartIndex iIndex(i);
                        CartIndex jIndex(j);
                        int iMat1 = matrix.getSMatrixIndex(iBte1Idx, iIndex);
                        int iMat2 = matrix.getSMatrixIndex(iBte2Idx, jIndex);
                        if (matrix.theMatrix.indicesAreLocal(iMat1, iMat2)) {
                          if (i == 0 && j == 0) {
                            linewidth->operator()(iCalc, 0, iBte1) +=
                                0.5 * (rateMinus1 + rateMinus2);
                          }
                          if (is1 != is2Irr) {
                            matrix.theMatrix(iMat1, iMat2) -=
                                rotation.inverse()(i, j) * (rateMinus1 + rateMinus2);
                          }
                        }
                      }
                    }
                  } else {
                    // these are unshifted if it's not coupled
                    if (matrix.theMatrix.indicesAreLocal(iBte1Shift, iBte2Shift)) {
                      linewidth->operator()(iCalc, 0, iBte1Shift) += 0.5 * (rateMinus1 + rateMinus2);
                      matrix.theMatrix(iBte1Shift, iBte2Shift) -= rateMinus1 + rateMinus2;

                      // if we're not symmetrizing the matrix, and we have
                      // dropped down to only using the upper triangle of the matrix, we must fill
                      // in linewidths twice, using detailed balance, in order to get the right ratest
                      if(!context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) {
                        linewidth->operator()(iCalc, 0, iBte2Shift) += 0.5 * (rateMinus1 + rateMinus2);
                      }
                    }

                  }
                } else if (switchCase == 1) { // matrix-vector multiplication
                  for (unsigned int iInput = 0; iInput < inPopulations.size(); iInput++) {
                    Eigen::Vector3d inPopRot;
                    inPopRot.setZero();
                    for (int i : {0, 1, 2}) {
                      for (int j : {0, 1, 2}) {
                        inPopRot(i) += rotation.inverse()(i, j) *
                                       inPopulations[iInput](iCalc, j, iBte2);
                      }
                    }

                    for (int i : {0, 1, 2}) {
                      // off-diagonal term
                      if (is1 != is2Irr) { // avoid double counting terms
                        outPopulations[iInput](iCalc, i, iBte1) -=
                            (rateMinus1 + rateMinus2) * inPopRot(i);
                      }
                      // diagonal term
                      outPopulations[iInput](iCalc, i, iBte1) +=
                          0.5 * (rateMinus1 + rateMinus2) *
                          inPopulations[iInput](iCalc, i, iBte1);
                    }
                  }
                } else {
                  linewidth->operator()(iCalc, 0, iBte1) += 0.5 * (rateMinus1 + rateMinus2);
                  if(outputUNTimes) {
                    Point q1 = outerBandStructure.getPoint(iq1);
                    Point q2 = innerBandStructure.getPoint(iq2);
                    Eigen::Vector3d q1Cart = q1.getCoordinates(Points::cartesianCoordinates);
                    Eigen::Vector3d q2Cart = q2.getCoordinates(Points::cartesianCoordinates);
                    Eigen::Vector3d q1WS = outerBandStructure.getPoints().bzToWs(q1Cart, Points::cartesianCoordinates);
                    Eigen::Vector3d q2WS = outerBandStructure.getPoints().bzToWs(q2Cart, Points::cartesianCoordinates);
                    Eigen::Vector3d q3Cart = q1WS - q2WS;
                    Eigen::Vector3d q3fold = outerBandStructure.getPoints().bzToWs(q3Cart, Points::cartesianCoordinates);
                    bool isUmklapp = false;
                    if(abs((q3Cart-q3fold).norm()) > 1e-6) {
                      isUmklapp = true;
                    }
                    if(isUmklapp) {
                      matrix.internalDiagonalUmklapp->operator()(iCalc, 0, iBte1) += 0.5*(rateMinus1);
                    } else {
                      matrix.internalDiagonalNormal->operator()(iCalc, 0, iBte1) += 0.5*(rateMinus1);
                    }

                    // check the second point
                    Eigen::Vector3d q3Cart2 = q2WS - q1WS;
                    Eigen::Vector3d q3fold2 = outerBandStructure.getPoints().bzToWs(q3Cart2, Points::cartesianCoordinates);
                    isUmklapp = false;
                    if(abs((q3Cart2-q3fold2).norm()) > 1e-6) {
                      isUmklapp = true;
                    }
                    if(isUmklapp) {
                      matrix.internalDiagonalUmklapp->operator()(iCalc, 0, iBte1) += 0.5*(rateMinus2);
                    } else {
                      matrix.internalDiagonalNormal->operator()(iCalc, 0, iBte1) += 0.5*(rateMinus2);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  loopPrint.close();
}

// ISOTOPE SCATTERING =====================================================

void addIsotopeScattering(BasePhScatteringMatrix &matrix, Context &context,
                                std::vector<VectorBTE> &inPopulations,
                                std::vector<VectorBTE> &outPopulations, int &switchCase,
                                std::vector<std::tuple<std::vector<int>, int>> qPairIterator,
                                Eigen::MatrixXd &innerBose, Eigen::MatrixXd &outerBose,
                                BaseBandStructure &innerBandStructure,
                                BaseBandStructure &outerBandStructure,
                                std::shared_ptr<VectorBTE> linewidth) {

  // TODO add developer safety checks on these functions to make ssure it's bose , ph bands , etc

  if(mpi->mpiHead()) {
    std::cout << "\nAdding isotope scattering to the scattering matrix." << std::endl;
  }

  // copy a few small things that don't take
  // much memory but will keep the code easier to read
  auto excludeIndices = matrix.excludeIndices;

  // setup smearing using phonon band structure
  DeltaFunction *smearing = DeltaFunction::smearingFactory(context, innerBandStructure);
  if (smearing->getType() == DeltaFunction::tetrahedron) {
    Error("Developer error: Tetrahedron smearing for transport untested and thus blocked");
  }

  // generate basic properties from the function arguments
  int numAtoms = innerBandStructure.getPoints().getCrystal().getNumAtoms();
  int numCalculations = matrix.statisticsSweep.getNumCalculations();
  Particle particle = innerBandStructure.getParticle();

  // note: innerNumFullPoints is the number of points in the full grid
  // may be larger than innerNumPoints, when we use ActiveBandStructure
  double norm = 1. / context.getQMesh().prod();
  bool outputUNTimes = matrix.outputUNTimes;

  // create vector with the interaction strength
  Eigen::VectorXd massVariance = Eigen::VectorXd::Zero(numAtoms);

  // load the mass variance at natural abundances for isotope scattering.
  {
    auto crystal = outerBandStructure.getPoints().getCrystal();
    int numAtoms = crystal.getNumAtoms();
    massVariance = crystal.getAtomicIsotopeCouplings();
    Eigen::VectorXd masses = crystal.getAtomicMasses();

    if (masses.size() != massVariance.size() || masses.size() != numAtoms) {
      Error("Developer error: Problem setting up mass variance: incosistent sizes");
    }
    for (int i=0; i<masses.size(); i++) {
      massVariance(i) *= masses(i) * masses(i);
    }
  }

  // loop over points pairs
  for (auto tup : qPairIterator) {

    auto iq1Indexes = std::get<0>(tup);
    int iq2 = std::get<1>(tup);

    // collect information about s2
    WavevectorIndex iq2Index(iq2);
    Eigen::VectorXd state2Energies = innerBandStructure.getEnergies(iq2Index);
    auto nb2 = int(state2Energies.size());
    Eigen::Tensor<std::complex<double>, 3> ev2 = innerBandStructure.getPhEigenvectors(iq2Index);
    Eigen::MatrixXd v2s = innerBandStructure.getGroupVelocities(iq2Index);

    auto q2 = innerBandStructure.getPoint(iq2).getCoordinates(Points::cartesianCoordinates);
    auto t = innerBandStructure.getRotationToIrreducible(q2, Points::cartesianCoordinates);
    // rotation such that qIrr = R * qRed
    int iq2Irr = std::get<0>(t);
    Eigen::Matrix3d rotation = std::get<1>(t);

    // this index is MPI parallelized over
    for (auto iq1 : iq1Indexes) {

      WavevectorIndex iq1Index(iq1);

      // note: for computing linewidths on a path, we must distinguish
      // that q1 and q2 are on different meshes, and that q3+/- may not
      // fall into known meshes and therefore needs to be computed

      // gather s1 information
      Eigen::VectorXd state1Energies = outerBandStructure.getEnergies(iq1Index);
      auto nb1 = int(state1Energies.size());
      Eigen::Tensor<std::complex<double>, 3> ev1 = outerBandStructure.getPhEigenvectors(iq1Index);
      Eigen::MatrixXd v1s = outerBandStructure.getGroupVelocities(iq1Index);

      for (int ib1 = 0; ib1 < nb1; ib1++) {

        double en1 = state1Energies(ib1);

        int is1 = outerBandStructure.getIndex(iq1Index, BandIndex(ib1));
        StateIndex is1Idx(is1);
        int iBte1 = outerBandStructure.stateToBte(is1Idx).get();

        // stop the calculation for indices which are
        // acoustic modes at the gamma point
        if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) !=
            excludeIndices.end()) {
          continue;
        }
        if (en1 < phEnergyCutoff) {  continue; }

        for (int ib2 = 0; ib2 < nb2; ib2++) {

          double en2 = state2Energies(ib2);
          int is2Irr = innerBandStructure.getIndex(WavevectorIndex(iq2Irr),
                                                     BandIndex(ib2));
          int is2 = innerBandStructure.getIndex(WavevectorIndex(iq2), BandIndex(ib2));
          StateIndex is2IrrIdx(is2Irr);
          StateIndex is2Idx(is2);
          int iBte2 = innerBandStructure.stateToBte(is2IrrIdx).get();

          // remove gamma point acoustic phonon frequencies
          if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte2) != excludeIndices.end()) {
            continue;
          }
          if (en2 < phEnergyCutoff) { continue; }

          double deltaIso = 0;
          if (smearing->getType() == DeltaFunction::gaussian) {
            deltaIso = smearing->getSmearing(en1 - en2);
          // TODO double check the smearing on this
          } else if (smearing->getType() == DeltaFunction::adaptiveGaussian) {
            deltaIso += smearing->getSmearing(en1 - en2, v2s.row(ib2));
            deltaIso += smearing->getSmearing(en1 - en2, v1s.row(ib1));
            deltaIso *= 0.5;
          } else if (smearing->getType() == DeltaFunction::symAdaptiveGaussian) {
            Eigen::Vector3d v1 = v1s.row(ib1);
            Eigen::Vector3d v2 = v2s.row(ib2);
            Eigen::Vector3d v3; // feed zeros for the third one, as this transition has only two
            deltaIso = smearing->getSmearing(en1 - en2, v1, v2, v3);
          } else {
            deltaIso = smearing->getSmearing(en1, is2Idx);
          }

          double termIso = 0.;
          for (int iat = 0; iat < numAtoms; iat++) {
            std::complex<double> zzIso = complexZero;
            for (int kDim : {0, 1, 2}) { // cartesian indices
              zzIso += std::conj(ev1(kDim, iat, ib1)) * ev2(kDim, iat, ib2);
            }
            termIso += std::norm(zzIso) * massVariance(iat);
          }
          termIso *= pi * 0.5 * norm * en1 * en2 * deltaIso;

          for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

            double bose1 = outerBose(iCalc, iBte1);
            double bose2 = innerBose(iCalc, iBte2);

            double rateIso = termIso * (bose1 * bose2 + 0.5 * (bose1 + bose2));

            // shift the indices if it's necessary
            int iBte1Shift = iBte1;   int iBte2Shift = iBte2;
            if(matrix.isCoupled) {
              std::tuple<int,int> tup =
                    matrix.shiftToCoupledIndices(iBte1, iBte2, particle, particle);
              iBte1Shift = std::get<0>(tup);
              iBte2Shift = std::get<1>(tup);
            }

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
                        linewidth->operator()(iCalc, 0, iBte1) += rateIso;
                      }
                      if (is1 != is2Irr) {
                        matrix.theMatrix(iMat1, iMat2) -=
                            rotation.inverse()(i, j) * rateIso;
                      }
                    }
                  }
                }
              } else {
                if (matrix.theMatrix.indicesAreLocal(iBte1Shift, iBte2Shift)) {

   	          linewidth->operator()(iCalc, 0, iBte1Shift) += rateIso; 
                  // if we're not symmetrizing the matrix, and we have
                  // dropped down to only using the upper triangle of the matrix, we must fill
                  // in linewidths twice, using detailed balance, in order to get the right ratest
  		  if(!context.getSymmetrizeMatrix() && context.getUseUpperTriangle()) {
                    linewidth->operator()(iCalc, 0, iBte2Shift) += rateIso;
		  }
                  matrix.theMatrix(iBte1Shift, iBte2Shift) -= rateIso;
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
		  // off diagonals
      		  if (is1 != is2Irr) {
                    outPopulations[iInput](iCalc, i, iBte1) -= rateIso * inPopRot(i);
                  } // diagonals 
                  outPopulations[iInput](iCalc, i, iBte1) +=
                      rateIso * inPopulations[iInput](iCalc, i, iBte1);
                }
              }

            } else { // case of linewidth construction

              linewidth->operator()(iCalc, 0, iBte1) += rateIso;

              if(outputUNTimes) {
                Point q1 = outerBandStructure.getPoint(iq1);
                Point q2 = innerBandStructure.getPoint(iq2);
                // check if this process is umklapp
                // TODO put this in hasUmklapp function
                Eigen::Vector3d q1Cart = q1.getCoordinates(Points::cartesianCoordinates);
                Eigen::Vector3d q2Cart = q2.getCoordinates(Points::cartesianCoordinates);
                Eigen::Vector3d q1WS = outerBandStructure.getPoints().bzToWs(q1Cart, Points::cartesianCoordinates);
                Eigen::Vector3d q2WS = outerBandStructure.getPoints().bzToWs(q2Cart, Points::cartesianCoordinates);
                Eigen::Vector3d q3Cart = q1WS + q2WS;
                Eigen::Vector3d q3fold = outerBandStructure.getPoints().bzToWs(q3Cart, Points::cartesianCoordinates);
                bool isUmklapp = false;
                if(abs((q3Cart-q3fold).norm()) > 1e-6) { isUmklapp = true; }
                if(isUmklapp) {
                  matrix.internalDiagonalUmklapp->operator()(iCalc, 0, iBte1) += rateIso;
                } else {
                  matrix.internalDiagonalNormal->operator()(iCalc, 0, iBte1) += rateIso;
                }
              }
            }
          }
        }
      }
    }
  }
  if(mpi->mpiHead()) {
    std::cout << "Finished adding isotope scattering to the scattering matrix.\n" << std::endl;
  }
}

