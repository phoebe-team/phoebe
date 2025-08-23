#include "bandstructure.h"
#include "ph_scattering.h"
#include "points.h"
#include "qe_input_parser.h"
#include "gtest/gtest.h"
#include <fstream>

TEST(InteractionIsotope, Wphisoiq4) {
  // Here we compare the phonon-isotope scattering rates
  // against those from ShengBTE. The test material is cubic
  // silicon and the phonon q-mesh is 8x8x8.

  // Parse espresso ifc2 file
  Context context;
  context.setPhFC2FileName("../test/data/444_silicon.fc");
  context.setSumRuleFC2("simple");
  auto tup = QEParser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  // Total number of wave vectors
  int nq = 8 * 8 * 8;
  // For full BZ mesh in crystal coordinates
  Eigen::Vector3i qMesh;
  qMesh << 8, 8, 8;
  Points points(crystal, qMesh);

  // Number of atoms
  int numAtoms = crystal.getNumAtoms();
  // Number of phonon branches
  int numBands = 3 * numAtoms;
  // Mass of basis atoms
  auto atomicMasses = crystal.getAtomicMasses();
  // Mass variance for isotopic scattering
  // TODO should get this through a getMassVariance method
  Eigen::VectorXd massVariance(numAtoms);
  massVariance << 0.00020164270215294942,
      0.00020164270215294942; // from src/constants/periodic_table.h

  // Phonon energies initialized to zero
  Eigen::VectorXd energies(numBands);
  energies.setConstant(0.0);

  // Test q-point
  Eigen::VectorXd WIsotopeQ(numBands);
  WIsotopeQ.setConstant(0.0);

  // Reference value from ShengBTE with 0.01 Ry fixed Gaussian broadening
  Eigen::VectorXd WRef(numBands);
  WRef << 0.3327465318e-03, 0.3327465318e-03, 0.4151744806e-02,
      0.4978611835e-02, 0.6913784969e-02, 0.6913784969e-02;
  // Convert from Thz to Ry
  WRef.array() /= 32889.83;

  context.setSmearingWidth(0.01); // in rydberg
  // create a throwaway bandstructure, just used 
  // in the constructor of the DeltaFunction object
  FullBandStructure fbs = phononH0.populate(points, false, false);
  GaussianDeltaFunction smearing(fbs, context);

  // Eigenvector and angular frequencies at iq
  int iq = 4; // = (0.5, 0, 0) in crystal coordinates
  auto q1 = points.getPoint(iq);
  auto tup1 = phononH0.diagonalize(q1);
  auto ens1 = std::get<0>(tup1);
  auto ev1 = std::get<1>(tup1);

  Eigen::Tensor<std::complex<double>, 3> evt1(3, numAtoms, numBands);
  Eigen::Tensor<std::complex<double>, 3> evt2(3, numAtoms, numBands);

  for (int iq2 = 0; iq2 < nq; iq2++) { // integrate over
    auto q2 = points.getPoint(iq2);
    // Eigenvector and angular frequencies at jq
    auto tup2 = phononH0.diagonalize(q2);
    auto ens2 = std::get<0>(tup2);
    auto ev2 = std::get<1>(tup2);

    for (int i = 0; i < numBands; i++) {
      for (int j = 0; j < numBands; j++) {
        auto tup3 = decompress2Indices(i, numAtoms, 3);
        auto iat = std::get<0>(tup3);
        auto iDim = std::get<1>(tup3);
        evt1(iDim, iat, j) = ev1(i, j);
        evt2(iDim, iat, j) = ev2(i, j);
      }
    }

    // phonon branches of the test q-point
    for (int ib1 = 0; ib1 < numBands; ib1++) {
      double fac = pow(ens1(ib1), 2) / nq;
      for (int ib2 = 0; ib2 < numBands; ib2++) { // integrate over
        // using fixed gaussian for now
        double deltaWeight = smearing.getSmearing(ens1(ib1) - ens2(ib2));

        if (deltaWeight == 0.)
          continue;

        for (int p = 0; p < numAtoms; p++) { // sum over atomic basis
          double aux = 0.0;
          // inner product over Cartesian space
          for (int kDim : {0, 1, 2}) {
            // Recall that the eigenvectors were mass-normalized
            aux += pow(abs(std::conj(evt1(kDim, p, ib1)) * evt2(kDim, p, ib2)),
                       2) *
                   pow(atomicMasses(p), 2);
          }
          WIsotopeQ(ib1) += aux * deltaWeight * massVariance(p) * fac;

        } // p
      }   // jb
    }     // ib
  }       // jq

  for (int ib = 0; ib < numBands; ib++) {
    double relativeError = (WRef(ib) - WIsotopeQ(ib)) / WRef(ib);
    ASSERT_NEAR(relativeError, 0., 0.1);
    // up to 10% error, which may come from several details on how the
    // dynamical matrix is diagonalized
  }
}
