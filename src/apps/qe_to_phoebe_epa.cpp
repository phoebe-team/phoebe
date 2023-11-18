#include "bandstructure.h"
#include "eigen.h"
#include "elph_qe_to_phoebe_app.h"
#include "io.h"
#include "qe_input_parser.h"
#include <iomanip>
#include <sstream>
#include <string>

void ElPhQeToPhoebeApp::epaPostProcessing(
    Context &context, Eigen::MatrixXd &elEnergies, Points &kPoints,
    Points &qPoints, const int &numElectrons, const int &numSpin,
    const int &numModes, const int &numIrrQPoints, const int &numQEBands,
    const Eigen::MatrixXd &kGridFull) {

  if (mpi->mpiHead()) {
    std::cout << "Starting EPA post-processing\n" << std::endl;
  }

  auto t1 = QEParser::parseElHarmonicFourier(context);
  auto electronH0 = std::get<1>(t1);

  // prepare energy bins
  double minEnergy = context.getEpaMinEnergy();
  double maxEnergy = context.getEpaMaxEnergy();
  if (maxEnergy < minEnergy) {
    Error("Problems in setting the EPA energy ranges");
  }

  int numEpaEnergies = context.getEpaNumBins();
  double deltaEnergy = context.getEpaDeltaEnergy();
  if (std::isnan(deltaEnergy)) {
    context.getEpaNumBins();
    deltaEnergy = (maxEnergy - minEnergy) / numEpaEnergies;
  } else {
    numEpaEnergies = int((maxEnergy - minEnergy) / deltaEnergy) + 1;
  }

  Eigen::VectorXd epaEnergies(numEpaEnergies);
  for (int i = 0; i < numEpaEnergies; i++) {
    epaEnergies[i] = i * deltaEnergy + minEnergy;
  }

  if (mpi->mpiHead()) {
    std::cout << "\nBuilding EPA with " << numEpaEnergies << " energy bins."
              << std::endl;
  }

  int numKPoints = kPoints.getNumPoints();
  double smearing2 = 2. * pow(context.getEpaSmearingEnergy(), 2);
  Eigen::Tensor<double, 3> gaussian(numEpaEnergies, numQEBands, numKPoints);
#pragma omp parallel for collapse(3) default(none)                             \
    shared(numQEBands, numKPoints, numEpaEnergies, elEnergies, epaEnergies,    \
           smearing2, gaussian)
  for (int ib1 = 0; ib1 < numQEBands; ib1++) {
    for (int ik = 0; ik < numKPoints; ik++) {
      for (int i = 0; i < numEpaEnergies; i++) {
        // note: the old EPA was using the bin center
        double arg = pow(elEnergies(ib1, ik) - epaEnergies(i), 2) / smearing2;
        gaussian(i, ib1, ik) = exp(-arg);
      }
    }
  }

  // integrated in the loop:
  Eigen::VectorXd phAvgEnergies(numModes);
  phAvgEnergies.setZero();
  Eigen::Tensor<double, 3> g2Epa(numModes, numEpaEnergies, numEpaEnergies);
  g2Epa.setZero();

  Eigen::VectorXi ikMap(numKPoints);
#pragma omp parallel for default(none)                                         \
    shared(numKPoints, kGridFull, kPoints, ikMap, Eigen::Dynamic)
  for (int ikOld = 0; ikOld < numKPoints; ikOld++) {
    Eigen::Vector3d kOld = kGridFull.col(ikOld);
    int ikNew = kPoints.getIndex(kOld);
    ikMap(ikOld) = ikNew;
  }

  double phononCutoff = 5. / ryToCmm1; // used to discard small phonon energies

  Eigen::MatrixXd normalization(numEpaEnergies, numEpaEnergies);
  normalization.setZero();

  // loop over all irreducible q-points written to file
  LoopPrint loopPrint("Computing coupling EPA", "irreducible q-points",
                      int(mpi->divideWorkIter(numIrrQPoints).size()));
  for (int iqIrr : mpi->divideWorkIter(numIrrQPoints)) {
    loopPrint.update(false);

    // read coupling from file
    auto t2 =
        readChunkGFromQE(iqIrr, context, kPoints, numModes, numQEBands, ikMap);
    auto gStar = std::get<0>(t2); // (nBands, nBands, nModes, numK, numQStar)
    // crystal coordinates of the points in the current qPoint star
    Eigen::MatrixXd qStar = std::get<3>(t2);

    int numQStar = int(qStar.cols());

    // phonon frequency averaging
    Eigen::MatrixXd phEnergies = std::get<2>(t2); // (numModes, numQPoints)

    Eigen::Vector3i qMesh = std::get<0>(qPoints.getMesh());
    for (int nu = 0; nu < numModes; nu++) { // phEnergies(numModes, numQPoints);
      phAvgEnergies(nu) += phEnergies.row(nu).sum() / double(qMesh.prod());
    }

    assert(numQEBands == gStar.dimension(0));
    assert(numModes == gStar.dimension(2));

    for (int ik = 0; ik < numKPoints; ik++) {
      Eigen::Vector3d kCrystal =
          kPoints.getPointCoordinates(ik, Points::crystalCoordinates);

      for (int iqStar = 0; iqStar < numQStar; iqStar++) {
        Eigen::Vector3d qCrystal = qStar.col(iqStar);

        // Coordinates and index of k+q point
        Eigen::Vector3d kqCrystal = kCrystal + qCrystal;
        int ikq = int(kPoints.getIndex(kqCrystal));

#pragma omp parallel for collapse(2)
        for (int j = 0; j < numEpaEnergies; j++) {
          for (int i = 0; i < numEpaEnergies; i++) {
            for (int ib2 = 0; ib2 < numQEBands; ib2++) {
              for (int ib1 = 0; ib1 < numQEBands; ib1++) {

                double gaussianX =
                    gaussian(i, ib2, ik) * gaussian(j, ib1, ikq);
                normalization(i,j) += gaussianX;

                for (int nu = 0; nu < numModes; nu++) {
                  if (phEnergies(nu, iqStar) <= phononCutoff) {
                    continue;
                  }
                  g2Epa(nu, i, j) +=
                      std::norm(gStar(ib1, ib2, nu, ik, iqStar)) * gaussianX *
                      0.5 / phEnergies(nu, iqStar);
                  // /2omega, because there is a difference between the
                  // coupling <k+q| dV_q |k> from quantum espresso
                  // and the coupling g to be used for transport calculations
                }
              }
            }
          }
        }
      }
    }
  }
  mpi->allReduceSum(&g2Epa);
  mpi->allReduceSum(&phAvgEnergies);
  mpi->allReduceSum(&normalization);
  loopPrint.close();

  for (int j = 0; j < numEpaEnergies; j++) {
    for (int i = 0; i < numEpaEnergies; i++) {
      if (normalization(i,j) > 1.0e-12) {
        for (int nu = 0; nu < numModes; nu++) {
          g2Epa(nu, i, j) /= normalization(i, j);
        }
      }
    }
  }

  if (mpi->mpiHead()) {
    std::cout << "\nStart writing el-ph coupling to file." << std::endl;
    std::string phoebePrefixQE = context.getQuantumEspressoPrefix();
    std::string outFileName = phoebePrefixQE + ".phoebe.epa.dat";
    std::ofstream outfile(outFileName);
    if (not outfile.is_open()) {
      Error("Output file couldn't be opened");
    }
    outfile << numElectrons << " " << numSpin << "\n";
    outfile << phAvgEnergies.size() << "\n";
    outfile << phAvgEnergies.transpose() << "\n";
    outfile << numEpaEnergies << "\n";
    outfile << epaEnergies.transpose() << "\n";
    for (auto i = 0; i < numModes; ++i) {
      for (auto j = 0; j < numEpaEnergies; ++j) {   // k index
        for (auto k = 0; k < numEpaEnergies; ++k) { // k+q index
          outfile << g2Epa(i, j, k) << "\n";
        }
      }
    }
    std::cout << "Done writing el-ph coupling to file.\n" << std::endl;
  }
}
