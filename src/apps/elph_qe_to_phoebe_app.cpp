#include "elph_qe_to_phoebe_app.h"
#include "bandstructure.h"
#include "eigen.h"
#include "io.h"
#include "qe_input_parser.h"
#include <iomanip>
#include <sstream>
#include <string>

void ElPhQeToPhoebeApp::run(Context &context) {
  (void)context;
  // actually, we only need the crystal
  auto t1 = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(t1);
  auto phononH0 = std::get<1>(t1);
  std::string phoebePrefixQE = context.getQuantumEspressoPrefix();
  auto t0 = readQEPhoebeHeader(crystal, phoebePrefixQE);
  Eigen::Vector3i qMesh = std::get<0>(t0);
  Eigen::Vector3i kMesh = std::get<1>(t0);
  Eigen::MatrixXd kGridFull = std::get<2>(t0);
  Eigen::MatrixXd qGridFull = std::get<3>(t0);
  Eigen::MatrixXd energies = std::get<4>(t0);
  int numIrrQPoints = std::get<5>(t0);
  int numQEBands = std::get<6>(t0);
  int numElectrons = std::get<7>(t0);
  int numSpin = std::get<8>(t0);

  Points kPoints(crystal, kMesh);
  Points qPoints(crystal, qMesh);

  int numModes = 3 * crystal.getNumAtoms();

  if (context.getElPhInterpolation() == "wannier") {

    postProcessingWannier(context, crystal, phononH0, kPoints, qPoints,
                          numQEBands, numModes, numIrrQPoints, numElectrons,
                          numSpin, energies, kGridFull, kMesh, qMesh);

  } else { // EPA

    epaPostProcessing(context, energies, kPoints, qPoints, numElectrons,
                      numSpin, numModes, numIrrQPoints, numQEBands,
                      kGridFull);
  }
}

void ElPhQeToPhoebeApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getPhFC2FileName(), "PhFC2FileName");
  throwErrorIfUnset(context.getQuantumEspressoPrefix(),
                    "QuantumEspressoPrefix");

  choices = {"wannier", "epa"};
  std::string x = context.getElPhInterpolation();
  throwErrorIfUnset(x, "elPhInterpolation");
  if (std::find(choices.begin(), choices.end(), x) == choices.end()) {
    Error("The elPhInterpolation value has not been recognized.");
  }

  if (x == "wannier") {
    throwErrorIfUnset(context.getWannier90Prefix(), "Wannier90Prefix");
  } else {
    throwErrorIfUnset(context.getEpaSmearingEnergy(), "epaSmearingEnergy");
    throwErrorIfUnset(context.getElectronFourierCutoff(), "electronFourierCutoff");
    throwErrorIfUnset(context.getEpaMinEnergy(), "epaMinEnergy");
    throwErrorIfUnset(context.getEpaMaxEnergy(), "epaMaxEnergy");
    if (std::isnan(context.getEpaDeltaEnergy())) {
      throwErrorIfUnset(context.getEpaNumBins(), "epaNumBins");
    } else {
      throwErrorIfUnset(context.getEpaDeltaEnergy(), "epaDeltaEnergy");
    }
  }
}

// read g, which is written to file on all k, q points
std::tuple<Eigen::Tensor<std::complex<double>, 5>,
           Eigen::Tensor<std::complex<double>, 3>, Eigen::MatrixXd,
           Eigen::MatrixXd>
ElPhQeToPhoebeApp::readChunkGFromQE(const int& iqIrr, Context &context,
                                    Points &kPoints,
                                    const int &numModes,
                                    const int &numQEBands,
                                    const Eigen::VectorXi &ikMap) {

  int numKPoints = kPoints.getNumPoints();

  // first, find the number of reducible q points in this file
  int nqStar;
  std::string phoebePrefixQE = context.getQuantumEspressoPrefix();
  std::stringstream ss;
  ss << std::setw(4) << std::setfill('0') << iqIrr + 1;
  std::string numString = ss.str();
  std::string fileName = phoebePrefixQE + ".phoebe." + numString + ".dat";

  std::ifstream infileQ(fileName);
  infileQ >> nqStar;

  // allocate read quantities
  Eigen::Tensor<std::complex<double>, 5> gStar(numQEBands, numQEBands, numModes,
                                               numKPoints, nqStar);
  Eigen::Tensor<std::complex<double>, 3> phononEigenvectorsStar(numModes, numModes,
                                                                nqStar);
  Eigen::MatrixXd phononEnergiesStar(numModes, nqStar);
  Eigen::MatrixXd qStar(3, nqStar);

  gStar.setZero();
  phononEigenvectorsStar.setZero();
  phononEnergiesStar.setZero();
  qStar.setZero();

  // now, we start reading everything else

  for (int iq = 0; iq < nqStar; iq++) {
    Eigen::Vector3d thisQ; // in crystal coordinates
    infileQ >> thisQ(0) >> thisQ(1) >> thisQ(2);
    for (int i : {0,1,2}) {
      qStar(i,iq) = thisQ(i);
    }
  }

  for (int iq = 0; iq < nqStar; iq++) {
    Eigen::Vector3d thisQ; // in same as above, in cartesian coordinates
    infileQ >> thisQ(0) >> thisQ(1) >> thisQ(2);
  }

  Eigen::VectorXd phononEnergies(numModes);
  for (int nu = 0; nu < numModes; nu++) {
    infileQ >> phononEnergies(nu);
  }
  for (int iqStar = 0; iqStar<nqStar; iqStar++) {
    phononEnergiesStar.col(iqStar) = phononEnergies;
  }

  for (int iq = 0; iq < nqStar; iq++) {
    for (int j = 0; j < numModes; j++) {
      for (int i = 0; i < numModes; i++) {
        // Note, in Fortran I was writing:
        // do jj = 1,nModes
        //   do k = 1,nat
        //     do i_cart = 1,3
        // This has to be aligned with what done by PhononH0
        double re, im;
        infileQ >> re >> im;
        phononEigenvectorsStar(i, j, iq) = {re, im}; // j is the eig index
      }
    }
  }

  // read empty line
  std::string line;
  std::getline(infileQ, line);

  // read the g-coupling
  for (int iq = 0; iq < nqStar; iq++) {
    for (int nu = 0; nu < numModes; nu++) {
      for (int ik = 0; ik < numKPoints; ik++) {
        for (int ib2 = 0; ib2 < numQEBands; ib2++) {
          for (int ib1 = 0; ib1 < numQEBands; ib1++) {
            double re, im;
            infileQ >> re >> im;
            gStar(ib1, ib2, nu, ikMap(ik), iq) = {re, im};
            // note: ikMap maps the index of k-point in the order of QE,
            // to the index of k-point in the order of Phoebe
          }
        }
      }
    }
  }

  return std::make_tuple(gStar, phononEigenvectorsStar, phononEnergiesStar, qStar);
}

// read g, which is written to file on all k, q points
std::tuple<Eigen::Tensor<std::complex<double>, 5>,
           Eigen::Tensor<std::complex<double>, 3>, Eigen::MatrixXd>
ElPhQeToPhoebeApp::readGFromQEFile(Context &context, const int &numModes,
                                   const int &numBands, const int &numWannier,
                                   Points &kPoints, Points &qPoints,
                                   const Eigen::MatrixXd &kGridFull,
                                   const int &numIrrQPoints,
                                   const int &numQEBands,
                                   const Eigen::MatrixXd &energies) {

  if (mpi->mpiHead()) {
    std::cout << "\nStart reading el-ph coupling from file." << std::endl;
  }

  std::string interpolation = context.getElPhInterpolation();
  int bandsOffset;
  if (interpolation == "wannier") {
    std::string wannierPrefix = context.getWannier90Prefix();
    bandsOffset = computeOffset(energies, wannierPrefix);
  } else {
    bandsOffset = 0;
  }

  int numKPoints = kPoints.getNumPoints();
  int numQPoints = qPoints.getNumPoints();

  if ( mpi->mpiHead() ) {
    double x = pow(numWannier,2) * numModes * numKPoints * numQPoints;
    std::complex<double> xx;
    x *= sizeof(xx) / pow(1024.,3);
    // the last 2 is because we will later work with 2 copies of g
    std::cout << "The app will now allocate " << x
              << " (GB) of memory per MPI process." << std::endl;
  }

  Eigen::Tensor<std::complex<double>, 5> gFull(numBands, numBands, numModes,
                                                numKPoints, numQPoints);
  Eigen::Tensor<std::complex<double>, 3> phEigenvectors(numModes, numModes,
                                                        numQPoints);
  Eigen::MatrixXd phEnergies(numModes, numQPoints);

  gFull.setZero();
  phEigenvectors.setZero();
  phEnergies.setZero();

  if (mpi->mpiHead()) {

    Eigen::VectorXi ikMap(numKPoints);
#pragma omp parallel for default(none) shared(numKPoints, kGridFull, kPoints, ikMap, Eigen::Dynamic)
    for (int ikOld = 0; ikOld < numKPoints; ikOld++) {
      Eigen::Vector3d kOld = kGridFull.col(ikOld);
      int ikNew = kPoints.getIndex(kOld);
      ikMap(ikOld) = ikNew;
    }

    std::string phoebePrefixQE = context.getQuantumEspressoPrefix();

#pragma omp parallel for default(none) shared(context, kPoints, numIrrQPoints, phoebePrefixQE, gFull, phEnergies, phEigenvectors, numModes, numKPoints, ikMap, bandsOffset, numWannier, qPoints, numQEBands)
    for (int iqIrr = 0; iqIrr < numIrrQPoints; iqIrr++) {

      auto t = readChunkGFromQE(iqIrr, context, kPoints, numModes, numQEBands,
                                ikMap);
      auto gStar = std::get<0>(t);
      auto phononEigenvectorsStar = std::get<1>(t);
      auto phononEnergiesStar = std::get<2>(t);
      auto qStar = std::get<3>(t);

      // Note: the tensor read from file contains
      // gFull(ib1, ib2, nu, ik, iq)
      // = < k+q,ib1 |  dV_{q,nu}  |  k,ib2  >
      // where k and q run over the full mesh.
      // ikMap takes care of the fact that k-points in QE have a different
      // order then phoebe.

      // reorder the q/k indices
      for (int iqStar = 0; iqStar < qStar.cols(); iqStar++) {
        Eigen::Vector3d qVec = qStar.col(iqStar);
        int iqFull = qPoints.getIndex(qVec);

        for (int nu = 0; nu < numModes; nu++) {
          for (int ik = 0; ik < numKPoints; ik++) {
            for (int ib2 = 0; ib2 < numWannier; ib2++) {
              for (int ib1 = 0; ib1 < numWannier; ib1++) {
                gFull(ib1, ib2, nu, ik, iqFull) =
                    gStar(bandsOffset + ib1, bandsOffset + ib2, nu, ik, iqStar);
              }
            }
          }
        }

        for (int j = 0; j < numModes; j++) {
          for (int i = 0; i < numModes; i++) {
            phEigenvectors(i, j, iqFull) = phononEigenvectorsStar(i, j, iqStar);
          }
        }

        for (int i = 0; i < numModes; i++) {
          phEnergies(i, iqFull) = phononEnergiesStar(i);
        }
      }
    }

    std::cout << "Done reading el-ph coupling from file.\n" << std::endl;
  }

  mpi->bcast(&gFull);
  mpi->bcast(&phEigenvectors);
  mpi->bcast(&phEnergies);

  return std::make_tuple(gFull, phEigenvectors, phEnergies);
}

std::tuple<Eigen::Vector3i, Eigen::Vector3i, Eigen::MatrixXd, Eigen::MatrixXd,
           Eigen::MatrixXd, int, int, int, int>
ElPhQeToPhoebeApp::readQEPhoebeHeader(Crystal &crystal,
                                      const std::string &phoebePrefixQE) {
  int numQEBands;             // number of Kohn-Sham states
  double numElectrons;        // number of electrons (spin degeneracy included)
  int numSpin;                // should always be one, without support for spin
  Eigen::Vector3i kMesh, qMesh;
  int bogusI;
  double bogusD;
  int numAtoms;
  int numKPoints;
  Eigen::MatrixXd qGridFull;
  Eigen::MatrixXd kGridFull;
  Eigen::MatrixXd energies;
  (void)crystal;
  int numQPoints, numIrrQPoints;


  if (mpi->mpiHead()) {

    std::string fileName = phoebePrefixQE + ".phoebe.0000.dat";
    std::ifstream infile(fileName);
    std::string line;
    if (not infile.is_open()) {
      Error(fileName + " file not found.");
    }
    std::getline(infile, line);// first line is a title

    infile >> numQEBands >> numElectrons >> numSpin;
    infile >> qMesh(0) >> qMesh(1) >> qMesh(2) >> kMesh(0) >> kMesh(1) >> kMesh(2);
    infile >> bogusD >> numAtoms;// lattice parameter and numAtoms

    // unit cell
    for (int i = 0; i < 9; i++) {
      infile >> bogusD;
    }
    // reciprocal unit cell
    for (int i = 0; i < 9; i++) {
      infile >> bogusD;
    }
    // iType
    for (int i = 0; i < numAtoms; i++) {
      infile >> bogusI;
    }
    // positions
    for (int i = 0; i < 3 * numAtoms; i++) {
      infile >> bogusD;
    }

    infile >> numQPoints >> numIrrQPoints;
    qGridFull.resize(3, numQPoints);
    for (int iq = 0; iq < numQPoints; iq++) {
      infile >> qGridFull(0, iq) >> qGridFull(1, iq) >> qGridFull(2, iq);
    }

    infile >> numKPoints;
    kGridFull.resize(3, numKPoints);
    for (int ik = 0; ik < numKPoints; ik++) {
      infile >> kGridFull(0, ik) >> kGridFull(1, ik) >> kGridFull(2, ik);
    }

    energies.resize(numQEBands, numKPoints);
    for (int ik = 0; ik < numKPoints; ik++) {
      for (int ib = 0; ib < numQEBands; ib++) {
        infile >> energies(ib, ik);
      }
    }
    assert(numAtoms == crystal.getNumAtoms());
  }

  mpi->bcast(&numQEBands);
  mpi->bcast(&numElectrons);
  mpi->bcast(&numSpin);
  mpi->bcast(&kMesh);
  mpi->bcast(&qMesh);
  mpi->bcast(&numAtoms);
  mpi->bcast(&numKPoints);
  mpi->bcast(&numQPoints);
  mpi->bcast(&numIrrQPoints);
  if (!mpi->mpiHead()) {
    qGridFull.resize(3, numQPoints);
    kGridFull.resize(3, numKPoints);
    energies.resize(numQEBands, numKPoints);
  }
  mpi->bcast(&qGridFull);
  mpi->bcast(&kGridFull);
  mpi->bcast(&energies);

  return std::make_tuple(qMesh,         kMesh,      kGridFull,    qGridFull, energies,
          numIrrQPoints, numQEBands, numElectrons, numSpin);
}
