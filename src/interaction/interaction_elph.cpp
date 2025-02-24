#include "interaction_elph.h"
#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>
#include <sys/types.h>

#ifdef HDF5_AVAIL
#include <Kokkos_ScatterView.hpp>
#endif

// constructor
InteractionElPhWan::InteractionElPhWan(
    const Crystal &crystal_,
    const Eigen::Tensor<std::complex<double>, 5> &couplingWannier_,
    const Eigen::MatrixXd &wsR1Vectors_,
    const Eigen::VectorXd &wsR1VectorsDegeneracies_,
    const Eigen::MatrixXd &wsR2Vectors_,
    const Eigen::VectorXd &wsR2VectorsDegeneracies_, const int &phaseConvention,
    const PhononH0 &phononH0_)
    : crystal(crystal_), phononH0(phononH0_), phaseConvention(phaseConvention) {

  numElBands = int(couplingWannier_.dimension(0));
  numPhBands = int(couplingWannier_.dimension(2));
  numWsR2Vectors = int(couplingWannier_.dimension(3));
  numWsR1Vectors = int(couplingWannier_.dimension(4));

  usePolarCorrection = false;
  Eigen::Matrix3d epsilon = phononH0.getDielectricMatrix();
  if (epsilon.squaredNorm() > 1.0e-10) { // i.e. if epsilon wasn't computed
    if (crystal.getNumSpecies() > 1) {   // otherwise polar correction = 0
      usePolarCorrection = true;
    }
  }

  // TODO REMOVE TEMPORARY VARS
  // wsR1Vectors = wsR1Vectors_;
  // wsR1VectorsDegeneracies = wsR1VectorsDegeneracies_;
  // wsR2Vectors = wsR2Vectors_;
  // wsR2VectorsDegeneracies = wsR2VectorsDegeneracies_;
  // cachedK1.setConstant(-1000);
  // couplingWannier = couplingWannier_;

  // in the first call to this function, we must copy the el-ph tensor
  // from the CPU to the accelerator
  {
    Kokkos::realloc(couplingWannier_k, numWsR1Vectors, numWsR2Vectors,
                    numPhBands, numElBands, numElBands);
    Kokkos::realloc(wsR1VectorsDegeneracies_k, numWsR1Vectors);
    Kokkos::realloc(wsR2VectorsDegeneracies_k, numWsR2Vectors);
    Kokkos::realloc(wsR1Vectors_k, numWsR1Vectors, 3);
    Kokkos::realloc(wsR2Vectors_k, numWsR2Vectors, 3);

    // note that Eigen has left layout while kokkos has right layout
    HostComplexView5D couplingWannier_h(
        (Kokkos::complex<double> *)couplingWannier_.data(), numWsR1Vectors,
        numWsR2Vectors, numPhBands, numElBands, numElBands);
    HostDoubleView1D wsR1VectorsDegeneracies_h(
        (double *)wsR1VectorsDegeneracies_.data(), numWsR1Vectors);
    HostDoubleView1D wsR2VectorsDegeneracies_h(
        (double *)wsR2VectorsDegeneracies_.data(), numWsR2Vectors);

    HostDoubleView2D wsR1Vectors_h((double *)wsR1Vectors_.data(),
                                   numWsR1Vectors, 3);
    HostDoubleView2D wsR2Vectors_h((double *)wsR2Vectors_.data(),
                                   numWsR2Vectors, 3);

    Kokkos::deep_copy(couplingWannier_k, couplingWannier_h);
    Kokkos::deep_copy(wsR2Vectors_k, wsR2Vectors_h);
    Kokkos::deep_copy(wsR2VectorsDegeneracies_k, wsR2VectorsDegeneracies_h);
    Kokkos::deep_copy(wsR1Vectors_k, wsR1Vectors_h);
    Kokkos::deep_copy(wsR1VectorsDegeneracies_k, wsR1VectorsDegeneracies_h);

    double memoryUsed = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(memoryUsed);
  }
}

void InteractionElPhWan::resetK1() { cachedK1.setConstant(-1000); }

InteractionElPhWan::~InteractionElPhWan() {
  if (couplingWannier_k.use_count() == 1) {
    double memory = getDeviceMemoryUsage();
    kokkosDeviceMemory->removeDeviceMemoryUsage(memory);
  }
}

Eigen::Tensor<double, 3> &
InteractionElPhWan::getCouplingSquared(const int &ik2) {
  // cacheCoupling[ik2].setConstant(1.); // for testing
  // cacheCoupling[ik2] = 4. * cacheCoupling[ik2];
  return cacheCoupling[ik2];
}

// set up V_L for the regular wannier interpolation of the matrix elements
Eigen::Tensor<std::complex<double>, 3> InteractionElPhWan::getPolarCorrection(
    const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
    const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3) {
  // doi:10.1103/physrevlett.115.176401, Eq. 4, is implemented here

  Eigen::VectorXcd x = polarCorrectionPart1(q3, ev3);
  return polarCorrectionPart2(ev1, ev2, x);
}

// compute the q dependent part of the polar elph correction for all the phonon
// wavevectors in a given bandstucture.
//
// * NOTE: this could be changed to calculate it for a specific q list if needed
// * TODO additionally, could store this polar correction internally and use it
// directly when calcCouplingSquared is called. However, for now we let the
// external scattering rate calculation class handle it and pass it back to the
// calcCoupling function. This is because currently in el-ph coupling, the polar
// data is calculated by the external pointsHelper class, rather than as a
// precomputation over q at the start of the scattering rate calculation/
Eigen::MatrixXcd InteractionElPhWan::precomputeQDependentPolar(
    BaseBandStructure &phBandStructure, const bool useMinusQ) {

  if (!phBandStructure.getParticle().isPhonon()) {
    Error("Developer error: cannot use electron bands to "
          "precompute q-dept part of polar correction.");
  }

  // precompute the q-dependent part of the polar correction ---------
  int numQPoints = phBandStructure.getNumPoints();
  auto qPoints = phBandStructure.getPoints();
  // we just set this to the largest possible number of phonons
  int nbQMax = 3 * phBandStructure.getPoints().getCrystal().getNumAtoms();
  Eigen::MatrixXcd polarData(numQPoints, nbQMax);
  polarData.setZero();

  // Fine to divide over qpoints as all processes have pairs k1, allQpoints
  std::vector<size_t> qIterator = mpi->divideWorkIter(numQPoints);
#pragma omp parallel for
  for (size_t iiq = 0; iiq < size_t(qIterator.size()); iiq++) {
    size_t iq = qIterator[iiq];
    WavevectorIndex iqIdx(iq);
    auto qC = phBandStructure.getWavevector(iqIdx);
    auto evQ = phBandStructure.getEigenvectors(iqIdx);
    if (useMinusQ) {
      qC = qC * -1.;
      evQ = evQ.conjugate(); // u_{-q} = u_{q}^*, see 2017 giustino review eq 18
    }
    // prepare part of the correction for this q point
    Eigen::VectorXcd thisPolar = polarCorrectionPart1(qC, evQ);
    for (int i = 0; i < thisPolar.size(); ++i) {
      polarData(iq, i) = thisPolar(i);
    }
  }
  mpi->allReduceSum(&polarData);
  return polarData;
}

Eigen::VectorXcd
InteractionElPhWan::polarCorrectionPart1(const Eigen::Vector3d &q3,
                                         const Eigen::MatrixXcd &ev3) {

  // gather variables
  double volume = crystal.getVolumeUnitCell();
  Eigen::Matrix3d reciprocalUnitCell = crystal.getReciprocalUnitCell();
  Eigen::Matrix3d epsilon = phononH0.getDielectricMatrix();
  Eigen::Tensor<double, 3> bornCharges = phononH0.getBornCharges();
  int dimensionality = crystal.getDimensionality();
  // must be in Bohr
  Eigen::MatrixXd atomicPositions = crystal.getAtomicPositions();
  Eigen::Vector3i qCoarseMesh = phononH0.getCoarseGrid();

  return polarCorrectionPart1Static(q3, ev3, volume, reciprocalUnitCell,
                                    epsilon, bornCharges, atomicPositions,
                                    qCoarseMesh, dimensionality);
}

// compute g_L for block->wannierTransform
// TODO would it not be simpler to pass the crystal object here?
Eigen::Tensor<std::complex<double>, 3>
InteractionElPhWan::getPolarCorrectionStatic(
    const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
    const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3,
    const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
    const Eigen::Matrix3d &epsilon, const Eigen::Tensor<double, 3> &bornCharges,
    const Eigen::MatrixXd &atomicPositions, const Eigen::Vector3i &qCoarseMesh,
    const int dimensionality) {

  Eigen::VectorXcd x = polarCorrectionPart1Static(
      q3, ev3, volume, reciprocalUnitCell, epsilon, bornCharges,
      atomicPositions, qCoarseMesh, dimensionality);
  return polarCorrectionPart2(ev1, ev2, x);
}

// compute the V_L component of g_L for block->wannierTransform
Eigen::VectorXcd InteractionElPhWan::polarCorrectionPart1Static(
    const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3,
    const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
    const Eigen::Matrix3d &epsilon, const Eigen::Tensor<double, 3> &bornCharges,
    const Eigen::MatrixXd &atomicPositions, const Eigen::Vector3i &qCoarseMesh,
    const int dimensionality) {

  // for 3D:
  // doi:10.1103/physRevLett.115.176401, Eq. 4, is implemented here

  // for 2D case: https://arxiv.org/pdf/2207.10187.pdf Eq 2.

  Kokkos::Profiling::pushRegion("Interaction Elph: polarCorrectionPart1Static");

  auto numAtoms = int(atomicPositions.rows());
  auto numPhBands = int(ev3.rows());
  Eigen::VectorXcd x(numPhBands);
  x.setZero();

  if (dimensionality == 3) {

    // auxiliary terms
    double gMax = 14.;
    double chargeSquare = 2.; // = e^2/4/Pi/eps_0 in atomic units
    std::complex<double> factor = chargeSquare * fourPi / volume * complexI;

    // build a list of (q+G) vectors
    std::vector<Eigen::Vector3d> gVectors; // here we insert all (q+G)
    for (int m1 = -qCoarseMesh(0); m1 <= qCoarseMesh(0); m1++) {
      for (int m2 = -qCoarseMesh(1); m2 <= qCoarseMesh(1); m2++) {
        for (int m3 = -qCoarseMesh(2); m3 <= qCoarseMesh(2); m3++) {
          Eigen::Vector3d gVector;
          gVector << m1, m2, m3;
          gVector = reciprocalUnitCell * gVector;
          gVector += q3;
          gVectors.push_back(gVector);
        }
      }
    }

    for (Eigen::Vector3d gVector : gVectors) {

      // (q+G) * eps_inf * (q+G)
      double qEq = gVector.transpose() * epsilon * gVector;

      if (qEq > 0. && qEq / 4. < gMax) {

        // TODO what is this? Ask Norma
        std::complex<double> factor2 = factor * exp(-qEq / 4.) / qEq;

        for (int iAt = 0; iAt < numAtoms; iAt++) {

          // e^-(G+q).r ? TODO what is this phase...
          double arg = -gVector.dot(atomicPositions.row(iAt));
          std::complex<double> phase = {cos(arg), sin(arg)};
          // factor 3 = (4pi*i/V) * (e^2/(4pi*eps0) * e^(-qEq / 4.) / qEq  *
          // e^-i(G+q).r
          std::complex<double> factor3 = factor2 * phase;

          for (int iPol : {0, 1, 2}) {

            double gqDotZ = gVector(0) * bornCharges(iAt, 0, iPol) +
                            gVector(1) * bornCharges(iAt, 1, iPol) +
                            gVector(2) * bornCharges(iAt, 2, iPol);
            int k = PhononH0::getIndexEigenvector(iAt, iPol, numAtoms);

            for (int ib3 = 0; ib3 < numPhBands; ib3++) {
              //  x = (4pi*i/V) * (e^2/(4pi*eps0) * e^(-qEq / 4.) / qEq  *
              //  e^-i(G+q).r  * (q+G)*Z * ph_eig
              x(ib3) += factor3 * gqDotZ * ev3(k, ib3);
            }
          }
        }
      }
    }
  } else if (dimensionality == 2) {

    if (mpi->mpiHead())
      Warning("El-ph polar correction for 2D materials implemented but not "
              "tested!");

    // TODO in phoebe is volume when dim = 2 used actually S? I think so, let's
    // check.
    double S = volume; // ! ESPECIALLY check that this is the case in the call
                       // from B->W !
    double qNorm = q3.norm(); // TODO do I need to sqrt this
    double L = 1.; // TODO check how to optimize this, it's an "arbitrary" param
    double fq =
        1. - tanh(qNorm * L /
                  2.); // TODO check if there is a numerically smarter tanh
    double vq = twoPi * fq / qNorm;
    // "The macroscopic in-plane polarizability can be obtained from the
    // in-plane dielectric constant eps_ab of an artificially periodic stack of
    // monolayers with spacing c through alphaPar(q) = (c/4pi) sum_ab q*a(eps_ab
    // − delta_ab)*q*b TODO Norma?
    double alphaPar = 1.; // TODO how to determine this? -- the macroscopic
                          // in-plane polarizability
    double epsPar_q = 1. + vq * alphaPar;
    Eigen::Tensor<double, 3> Zpar =
        bornCharges; // TODO later add in quadrupoles

    // TODO -- how to handle this form factor? Eq. 3 -- to start, maybe leave
    // this as 1?

    double prefactor = 1. / S * vq;

    for (int iAt = 0; iAt < numAtoms; iAt++) {

      // e^-i(q.r)
      double arg = -q3.dot(atomicPositions.row(iAt));
      std::complex<double> phase = {cos(arg), sin(arg)};

      for (int iPol : {0, 1, 2}) {

        std::complex<double> iqDotZ = complexI * (q3(0) * Zpar(iAt, 0, iPol) +
                                                  q3(1) * Zpar(iAt, 1, iPol) +
                                                  q3(2) * Zpar(iAt, 2, iPol));
        // get combined eigenvector index for atom + polarization
        int kappa = PhononH0::getIndexEigenvector(iAt, iPol, numAtoms);

        for (int ib3 = 0; ib3 < numPhBands; ib3++) {
          x(ib3) += prefactor * iqDotZ * epsPar_q * ev3(kappa, ib3) * phase;
        }
      }
    }
  } else { //(dimensionality == 1)
    if (mpi->mpiHead())
      Warning("El-ph polar correction for 1D materials not implemented. Let us "
              "know if you need this!");
  }

  Kokkos::Profiling::popRegion(); // interaction elph polar correction part1
                                  // static

  return x;
}

// regardless of the use case, this calculates <psi|e^{i(G+q).r}|psi> part of
// polar correction
Eigen::Tensor<std::complex<double>, 3>
InteractionElPhWan::polarCorrectionPart2(const Eigen::MatrixXcd &ev1,
                                         const Eigen::MatrixXcd &ev2,
                                         const Eigen::VectorXcd &x) {

  // overlap = <U^+_{b2 k+q}|U_{b1 k}>
  //         = <psi_{b2 k+q}|e^{i(q+G)r}|psi_{b1 k}>
  Eigen::MatrixXcd overlap = ev2.adjoint() * ev1; // matrix size (nb2,nb1)
  overlap = overlap.transpose();                  // matrix size (nb1,nb2)

  int numPhBands = x.rows();
  Eigen::Tensor<std::complex<double>, 3> v(overlap.rows(), overlap.cols(),
                                           numPhBands);
  v.setZero();
  for (int ib3 = 0; ib3 < numPhBands; ib3++) {
    for (int i = 0; i < overlap.rows(); i++) {
      for (int j = 0; j < overlap.cols(); j++) {
        v(i, j, ib3) += x(ib3) * overlap(i, j);
      }
    }
  }
  return v;
}

void InteractionElPhWan::calcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3,
    const std::vector<Eigen::Vector3d> &q3Cs, const Eigen::Vector3d &k1C,
    const std::vector<Eigen::VectorXcd> &polarData) {

  Kokkos::Profiling::pushRegion("calcCouplingSquared");
  int numWannier = numElBands;
  int nb1 = int(eigvec1.cols());
  int numLoops = int(eigvecs2.size()); // the number of k2 and q points

#ifdef MPI_AVAIL
  int pool_rank = mpi->getRank(mpi->intraPoolComm);
  int pool_size = mpi->getSize(mpi->intraPoolComm);
  if (pool_size > 1 && mpi_requests[0] != MPI_REQUEST_NULL) {

    // Kokkos::Profiling::pushRegion("wait for reductions");
    //  wait for MPI_Ireduces from cacheCoupling
    // MPI_Waitall(pool_size, mpi_requests.data(), MPI_STATUSES_IGNORE);
    // Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("copy to GPU");
    this->elPhCached = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), elPhCached_hs[pool_rank]);
    Kokkos::Profiling::popRegion();
  }
#endif

  auto elPhCached = this->elPhCached;
  int numPhBands = this->numPhBands;
  int numWsR2Vectors = this->numWsR2Vectors;
  DoubleView2D wsR2Vectors_k = this->wsR2Vectors_k;
  DoubleView1D wsR2VectorsDegeneracies_k = this->wsR2VectorsDegeneracies_k;

  // get nb2 for each ik and find the max
  // since loops and views must be rectangular, not ragged
  IntView1D nb2s_k("nb2s", numLoops);
  int nb2max = 0;
  auto nb2s_h = Kokkos::create_mirror_view(nb2s_k);
  for (int ik = 0; ik < numLoops; ik++) {
    nb2s_h(ik) = int(eigvecs2[ik].cols());
    if (nb2s_h(ik) > nb2max) {
      nb2max = nb2s_h(ik);
    }
  }
  Kokkos::deep_copy(nb2s_k, nb2s_h);

  // Polar corrections are computed on the CPU and then transferred to GPU
  IntView1D usePolarCorrections("usePolarCorrections", numLoops);
  ComplexView4D polarCorrections(
      Kokkos::ViewAllocateWithoutInitializing("polarCorrections"), numLoops,
      numPhBands, nb1, nb2max);
  auto usePolarCorrections_h = Kokkos::create_mirror_view(usePolarCorrections);
  auto polarCorrections_h = Kokkos::create_mirror_view(polarCorrections);

  // precompute all needed polar corrections
#pragma omp parallel for
  for (int ik = 0; ik < numLoops; ik++) {

    Eigen::Vector3d q3C = q3Cs[ik];
    Eigen::MatrixXcd eigvec2 = eigvecs2[ik];
    usePolarCorrections_h(ik) = usePolarCorrection && abs(q3C.norm()) > 1.0e-8;
    if (usePolarCorrections_h(ik)) {
      Eigen::Tensor<std::complex<double>, 3> singleCorrection =
          polarCorrectionPart2(eigvec1, eigvec2, polarData[ik]);

      for (int nu = 0; nu < numPhBands; nu++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
            polarCorrections_h(ik, nu, ib1, ib2) =
                singleCorrection(ib1, ib2, nu);
          }
        }
      }
    } else {
      Kokkos::complex<double> kZero(0., 0.);
      for (int nu = 0; nu < numPhBands; nu++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
            polarCorrections_h(ik, nu, ib1, ib2) = kZero;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(polarCorrections, polarCorrections_h);
  Kokkos::deep_copy(usePolarCorrections, usePolarCorrections_h);

  // copy eigenvectors etc. to device
  DoubleView2D q3Cs_k("q3", numLoops, 3);
  ComplexView3D eigvecs2Dagger_k("ev2Dagger", numLoops, numWannier, nb2max),
      eigvecs3_k("ev3", numLoops, numPhBands, numPhBands);

  {
    auto eigvecs2Dagger_h = Kokkos::create_mirror_view(eigvecs2Dagger_k);
    auto eigvecs3_h = Kokkos::create_mirror_view(eigvecs3_k);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_k);

#pragma omp parallel for default(none)                                         \
    shared(eigvecs3_h, eigvecs2Dagger_h, nb2s_h, q3Cs_h, q3Cs_k, q3Cs, k1C,    \
               numLoops, numWannier, numPhBands, eigvecs2Dagger_k, eigvecs3_k, \
               eigvecs2, eigvecs3, phaseConvention, std::cout)
    for (size_t ik = 0; ik < size_t(numLoops); ik++) {

      for (int i = 0; i < numWannier; i++) {
        for (int j = 0; j < nb2s_h(ik); j++) {
          eigvecs2Dagger_h(ik, i, j) = std::conj(eigvecs2[ik](i, j));
        }
      }

      // copy in the phonon eigenvectors
      for (int i = 0; i < eigvecs3[ik].cols() ; i++) {
        for (int j = 0; j < eigvecs3[ik].rows(); j++) {
          // if JDFTx is used so that phaseConvention = 1, q should be negated
          if (phaseConvention == JdftxPhaseConvention) { // i,j flipped here due to row/col major,
                                      // this is intentionally a * not a dagger
                                      // e(-q) = e(q)^*
            eigvecs3_h(ik, i, j) = std::conj(eigvecs3[ik](j, i)); 
          } else {
            eigvecs3_h(ik, i, j) = eigvecs3[ik](j, i);
          }
        }
      }
      for (int i = 0; i < 3; i++) {

        // in the JDFTx case we have to use k' in the place of q in phase2
        // we also have to remember q = k-k'
        // Here we are playing an unclear trick, and replacing q -> q + k1,
        // which in phaseConvention=0, k2 = q + k1, and using that for phase 2
        if (phaseConvention == JdftxPhaseConvention) {
          q3Cs_h(ik, i) =
              (q3Cs[ik](i) + k1C(i)); // k' wavevector stored here in this case
        } else {
          q3Cs_h(ik, i) = q3Cs[ik](i);
        }
      }
    }
    Kokkos::deep_copy(eigvecs2Dagger_k, eigvecs2Dagger_h);
    Kokkos::deep_copy(eigvecs3_k, eigvecs3_h);
    Kokkos::deep_copy(q3Cs_k, q3Cs_h); // kokkos, h = host is the cpu
  }

  // now we finish the Wannier transform. We have to do the Fourier transform
  // on the lattice degrees of freedom, and then do two rotations (at k2 and q)
  // -------------------------------------------------------------------------
  // set up the phases related to phonons
  ComplexView2D phases("phases", numLoops, numWsR2Vectors);
  Kokkos::complex<double> complexI(0.0, 1.0);
  Kokkos::parallel_for(
      "Interaction elph: calculate phase 2",
      Range2D({0, 0}, {numLoops, numWsR2Vectors}),
      KOKKOS_LAMBDA(int iq, int irP) {
        double arg = 0.0;
        for (int j = 0; j < 3; j++) {
          arg += q3Cs_k(iq, j) * wsR2Vectors_k(irP, j);
        }
        phases(iq, irP) = exp(complexI * arg) / wsR2VectorsDegeneracies_k(irP);
      });
  Kokkos::fence();

  // apply phases
  ComplexView4D g3(Kokkos::ViewAllocateWithoutInitializing("g3"), numLoops,
                   numPhBands, nb1, numWannier);
  Kokkos::parallel_for(
      "Interaction elph: apply phase 2",
      Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, numWannier}),
      KOKKOS_LAMBDA(int iq, int nu, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int irP = 0; irP < numWsR2Vectors; irP++) {
          tmp += phases(iq, irP) * elPhCached(irP, nu, ib1, iw2);
        }
        g3(iq, nu, ib1, iw2) = tmp;
      });
  Kokkos::realloc(phases, 0, 0);

  // rotate using phonon eigenvectors
  ComplexView4D g4(Kokkos::ViewAllocateWithoutInitializing("g4"), numLoops,
                   numPhBands, nb1, numWannier);
  Kokkos::parallel_for(
      "Interaction elph: rotate using phonon eigenvectors",
      Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, numWannier}),
      KOKKOS_LAMBDA(int iq, int nu2, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int nu = 0; nu < numPhBands; nu++) {
          tmp += g3(iq, nu, ib1, iw2) * eigvecs3_k(iq, nu2, nu);
          // if((nu == 3 && nu2 == 6) && if(ib1 == 4 && iw2 ==3)
          // if(ib1 == 4 && iw2 ==3 && nu2 == 8) std::cout << "ib1 nu nu2 iw2 Uq
          // " << ib1 << " " << nu << " " << nu2 << " " << iw2 << " " <<
          // eigvecs3_k(iq, nu2, nu) << " " << g3(iq, nu, ib1, iw2) <<
          // std::endl;
        }
        g4(iq, nu2, ib1, iw2) = tmp;
      });
  Kokkos::realloc(g3, 0, 0, 0, 0);

  // rotate using U^dagger eigenvectors
  ComplexView4D gFinal(Kokkos::ViewAllocateWithoutInitializing("gFinal"),
                       numLoops, numPhBands, nb1, nb2max);
  Kokkos::parallel_for(
      "Interaction elph: rotate using U^dagger eigenvectors",
      Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, nb2max}),
      KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int iw2 = 0; iw2 < numWannier; iw2++) {
          tmp += eigvecs2Dagger_k(ik, iw2, ib2) * g4(ik, nu, ib1, iw2);
          // std::cout << "ib1 nu iw2 ib2 Uk " << ib1 << " " << nu << " " << iw2
          // << " " << ib2 << " " << eigvecs2Dagger_k(ik, ib2, iw2) << " " <<
          // g4(ik, nu, ib1, iw2) << std::endl;
        }
        gFinal(ik, nu, ib1, ib2) = tmp;
      });
  Kokkos::realloc(g4, 0, 0, 0, 0);

  // we now add the precomputed polar corrections, before taking the norm of g
  if (usePolarCorrection) {
    Kokkos::parallel_for(
        "re-add polar correction to g",
        Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, nb2max}),
        KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
          gFinal(ik, nu, ib1, ib2) += polarCorrections(ik, nu, ib1, ib2);
        });
  }
  Kokkos::realloc(polarCorrections, 0, 0, 0, 0);

  // finally, compute |g|^2 from g
  DoubleView4D coupling_k(Kokkos::ViewAllocateWithoutInitializing("gSq"),
                          numLoops, numPhBands, nb2max, nb1);
  Kokkos::parallel_for(
      "Interaction elph: modulus coupling",
      Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb2max, nb1}),
      KOKKOS_LAMBDA(int ik, int nu, int ib2, int ib1) {
        // notice the flip of 1 and 2 indices is intentional
        // coupling is |<k+q,ib2 | dV_nu | k,ib1>|^2
        auto tmp = gFinal(ik, nu, ib1, ib2);
        // std::cout << "nu ib1 ib2 " << nu << " " << ib1 << " " << ib2 << " "
        // << gFinal(ik, nu, ib1, ib2) << std::endl;
        coupling_k(ik, nu, ib2, ib1) =
            tmp.real() * tmp.real() + tmp.imag() * tmp.imag();
      });
  Kokkos::realloc(gFinal, 0, 0, 0, 0);

  // now, copy results back to the CPU
  cacheCoupling.resize(0);
  cacheCoupling.resize(numLoops);
  auto coupling_h = Kokkos::create_mirror_view(coupling_k);
  Kokkos::deep_copy(coupling_h, coupling_k);

#pragma omp parallel for default(none)                                         \
    shared(numLoops, cacheCoupling, coupling_h, nb1, nb2s_h, numPhBands)
  for (int ik = 0; ik < numLoops; ik++) {
    Eigen::Tensor<double, 3> coupling(nb1, nb2s_h(ik), numPhBands);
    for (int nu = 0; nu < numPhBands; nu++) {
      for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          coupling(ib1, ib2, nu) = coupling_h(ik, nu, ib2, ib1);
        }
      }
    }
    // and we save the coupling |g|^2 for later
    cacheCoupling[ik] = coupling;
  }
  Kokkos::Profiling::popRegion(); // calcCouplingSquared
}

Eigen::VectorXi InteractionElPhWan::getCouplingDimensions() {
  Eigen::VectorXi xx(5);
  for (int i : {0, 1, 2, 3, 4}) {
    xx(i) = couplingWannier_k.extent(i);
  }
  return xx;
}

int InteractionElPhWan::estimateNumBatches(const int &nk2, const int &nb1) {
  int maxNb2 = numElBands;
  int maxNb3 = numPhBands;

  double availableMemory = kokkosDeviceMemory->getAvailableMemory();

  // memory used by different tensors, that is linear in nk2
  // Note: 16 (2*8) is the size of double (complex<double>) in bytes
  double evs = 16 * (maxNb2 * numElBands + maxNb3 * numPhBands);
  double phase = 16 * numWsR2Vectors;
  double g3 = 2 * 16 * numPhBands * nb1 * numElBands;
  double g4 = 2 * 16 * numPhBands * nb1 * numElBands;
  double gFinal = 2 * 16 * numPhBands * nb1 * maxNb2;
  double coupling = 16 * nb1 * maxNb2 * numPhBands;
  double polar = 16 * numPhBands * nb1 * maxNb2;
  double maxUsage =
      nk2 * (evs + polar +
             std::max({phase + g3, g3 + g4, g4 + gFinal, gFinal + coupling}));

  // the number of batches needed
  int numBatches = std::ceil(maxUsage / availableMemory);

  double totalMemory = kokkosDeviceMemory->getTotalMemory();

  if (availableMemory < maxUsage / nk2) {
    // not enough memory to do even a single q1
    std::cerr << "total memory = " << totalMemory / 1e9
              << "(Gb), available memory = " << availableMemory / 1e9
              << "(Gb), max memory usage = " << maxUsage / 1e9
              << "(Gb), numBatches = " << numBatches << "\n";
    Error("Insufficient memory!");
  }
  return numBatches;
}

void InteractionElPhWan::cacheElPh(const Eigen::MatrixXcd &eigvec1,
                                   const Eigen::Vector3d &k1C) {

  Kokkos::Profiling::pushRegion("cacheElPh");

  auto nb1 = int(eigvec1.cols());
  Kokkos::complex<double> complexI(0.0, 1.0);

  // note: when Kokkos is compiled with GPU support, we must create elPhCached
  // and other variables as local, so that Kokkos correctly allocates these
  // quantities on the GPU. At the end of this function, elPhCached must be
  // 'copied' back into this->elPhCached. Note that no copy actually is done,
  // since Kokkos::View works similarly to a shared_ptr.
  auto elPhCached = this->elPhCached;
  int numPhBands = this->numPhBands;
  int numElBands = this->numElBands;
  int numWsR1Vectors = this->numWsR1Vectors;
  int numWsR2Vectors = this->numWsR2Vectors;

  double memory = getDeviceMemoryUsage();
  kokkosDeviceMemory->removeDeviceMemoryUsage(memory);

  int pool_rank = mpi->getRank(mpi->intraPoolComm);
  int pool_size = mpi->getSize(mpi->intraPoolComm);

#ifdef MPI_AVAIL
  mpi_requests.resize(pool_size);
  elPhCached_hs.resize(pool_size);
#endif

  ComplexView4D g1(Kokkos::ViewAllocateWithoutInitializing("g1"),
                   numWsR2Vectors, numPhBands, numElBands, numElBands);

  // note: this loop is a parallelization over the group (Pool) of MPI
  // processes, which together contain all the el-ph coupling tensor
  // First, loop over the MPI processes in the pool
  for (int iPool = 0; iPool < pool_size; iPool++) {
    Kokkos::Profiling::pushRegion("cacheElPh setup");

    // the current MPI process must first broadcast the k-point and the
    // eigenvector that will be computed now.
    // So, first broadcast the number of bands of the iPool-th process
    int poolNb1 = 0;
    if (iPool == pool_rank) {
      poolNb1 = nb1;
    }
    mpi->bcast(&poolNb1, mpi->intraPoolComm, iPool);

    // broadcast also the wavevector and the eigenvector at k for process iPool
    Eigen::Vector3d poolK1C = Eigen::Vector3d::Zero();
    Eigen::MatrixXcd poolEigvec1 = Eigen::MatrixXcd::Zero(poolNb1, numElBands);
    if (iPool == pool_rank) {
      poolK1C = k1C;
      if (phaseConvention == JdftxPhaseConvention) // jdftx convention needs this phase to be -k
        poolK1C = -k1C;
      poolEigvec1 = eigvec1;
    }
    // broadcast to other processes on this pool
    mpi->bcast(&poolK1C, mpi->intraPoolComm, iPool);
    mpi->bcast(&poolEigvec1, mpi->intraPoolComm, iPool);

    // now, copy the eigenvector and wavevector to the accelerator
    ComplexView2D eigvec1_k("ev1", poolNb1, numElBands);
    DoubleView1D poolK1C_k("k", 3);
    {
      HostComplexView2D eigvec1_h((Kokkos::complex<double> *)poolEigvec1.data(),
                                  poolNb1, numElBands);
      HostDoubleView1D poolK1C_h(poolK1C.data(), 3);
      Kokkos::deep_copy(eigvec1_k, eigvec1_h);
      Kokkos::deep_copy(poolK1C_k, poolK1C_h);
    }

    // now compute the Fourier transform on electronic coordinates.
    ComplexView5D couplingWannier_k = this->couplingWannier_k;
    DoubleView2D wsR1Vectors_k = this->wsR1Vectors_k;
    DoubleView1D wsR1VectorsDegeneracies_k = this->wsR1VectorsDegeneracies_k;
    // Kokkos::Profiling::popRegion(); // cache setup

    // first we precompute the phases
    ComplexView1D phases_k("phases", numWsR1Vectors);
    Kokkos::parallel_for(
        "phases_k", numWsR1Vectors, KOKKOS_LAMBDA(int irE) {
          double arg = 0.0;
          for (int j = 0; j < 3; j++) {
            arg += poolK1C_k(j) * wsR1Vectors_k(irE, j);
          }
          phases_k(irE) = exp(complexI * arg) / wsR1VectorsDegeneracies_k(irE);
        });
    Kokkos::fence();
    Kokkos::Profiling::popRegion(); // cache setup

    // now we complete the Fourier transform
    // We have to write two codes: one for when the GPU runs on CUDA,
    // the other for when we compile the code without GPU support

    // TODO should we switch the below gemv code to work with CUDA also?
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::parallel_for(
        "g1",
        Range4D({0, 0, 0, 0},
                {numWsR2Vectors, numPhBands, numElBands, numElBands}),
        KOKKOS_LAMBDA(int irP, int nu, int iw1, int iw2) {
          Kokkos::complex<double> tmp(0.0);
          for (int irE = 0; irE < numWsR1Vectors; irE++) {
            // important note: the first index iw2 runs over the k+q transform
            // while iw1 runs over k
            tmp += couplingWannier_k(irE, irP, nu, iw1, iw2) * phases_k(irE);
          }
          g1(irP, nu, iw1, iw2) = tmp;
        });
    Kokkos::fence();
#else

    // Here we create a view to the elph matrix elements which represents it
    // in 2D, so that we can use a matrix-vector product with the phases to
    // accelerate an otherwise very expensive loop
    //
    // tutorial description of this gemv function:
    // https://youtu.be/_qD4X66MQF8?t=2434
    // read me about gemv
    // https://github.com/kokkos/kokkos-kernels/wiki/BLAS-2%3A%3Agemv

    // product of phase factor with g
    Kokkos::View<Kokkos::complex<double> *> g1_1D(
        g1.data(), numWsR2Vectors * numPhBands * numElBands * numElBands);
    Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight> coupling_2D(
        couplingWannier_k.data(), numWsR1Vectors,
        numWsR2Vectors * numPhBands * numElBands * numElBands);
    KokkosBlas::gemv("T", Kokkos::complex<double>(1.0), coupling_2D, phases_k,
                     Kokkos::complex<double>(0.0), g1_1D);

/*
  // Previous method -- slower than the gemv call
    Kokkos::deep_copy(g1, Kokkos::complex<double>(0.0, 0.0));
    Kokkos::Experimental::ScatterView<Kokkos::complex<double> ****>
  g1scatter(g1); Kokkos::parallel_for( "g1", Range5D({0, 0, 0, 0, 0},
                {numWsR1Vectors, numWsR2Vectors, numPhBands, numElBands,
  numElBands}), KOKKOS_LAMBDA(int irE, int irP, int nu, int iw1, int iw2) { auto
  g1 = g1scatter.access(); g1(irP, nu, iw1, iw2) += couplingWannier_k(irE, irP,
  nu, iw1, iw2) * phases_k(irE);
        });
    Kokkos::Experimental::contribute(g1, g1scatter);
*/
#endif

    // now we need to add the rotation on the electronic coordinates
    // and finish the transformation on electronic coordinates
    // we distinguish two cases. If each MPI process has the whole el-ph
    // tensor, we don't need communication and directly store results in
    // elPhCached. Otherwise, we need to do an MPI reduction

    if (pool_size == 1) {
      Kokkos::realloc(elPhCached, numWsR2Vectors, numPhBands, poolNb1,
                      numElBands);

      Kokkos::parallel_for(
          "elPhCached",
          Range4D({0, 0, 0, 0},
                  {numWsR2Vectors, numPhBands, poolNb1, numElBands}),
          KOKKOS_LAMBDA(int irP, int nu, int ib1, int iw2) {
            Kokkos::complex<double> tmp(0.0);
            for (int iw1 = 0; iw1 < numElBands; iw1++) {
              tmp += g1(irP, nu, iw1, iw2) * eigvec1_k(ib1, iw1);
              // if(irP == 2 && nu == 3) std::cout << "Re' nu iw1 iw2 ib1 g1 Uk
              // " << irP << " " << nu << " " <<iw1<< " " << iw2 << " " <<ib1 <<
              // " " << eigvec1_k(ib1, iw1) << " " << g1(irP, nu, iw1, iw2) <<
              // std::endl;
            }
            elPhCached(irP, nu, ib1, iw2) = tmp;
          });
      Kokkos::fence();

    } else {

      ComplexView4D poolElPhCached_k(
          Kokkos::ViewAllocateWithoutInitializing("poolElPhCached"),
          numWsR2Vectors, numPhBands, poolNb1, numElBands);

      Kokkos::parallel_for(
          "elPhCached",
          Range4D({0, 0, 0, 0},
                  {numWsR2Vectors, numPhBands, poolNb1, numElBands}),
          KOKKOS_LAMBDA(int irP, int nu, int ib1, int iw2) {
            Kokkos::complex<double> tmp(0.0);
            for (int iw1 = 0; iw1 < numElBands; iw1++) {
              tmp += g1(irP, nu, iw1, iw2) * eigvec1_k(ib1, iw1);
            }
            poolElPhCached_k(irP, nu, ib1, iw2) = tmp;
          });

      // note: we do the reduction after the rotation, so that the tensor
      // may be a little smaller when windows are applied (nb1<numWannier)

      // do a mpi->allReduce across the pool
      // mpi->allReduceSum(&poolElPhCached_h, mpi->intraPoolComm);

      Kokkos::Profiling::pushRegion("copy elPhCached to CPU");
      // copy from accelerator to CPU
      auto poolElPhCached_h = Kokkos::create_mirror_view(poolElPhCached_k);
      Kokkos::deep_copy(poolElPhCached_h, poolElPhCached_k);
      Kokkos::Profiling::popRegion();

      elPhCached_hs[iPool] = poolElPhCached_h;

#ifdef MPI_AVAIL
      // start reduction for current iteration
      Kokkos::Profiling::pushRegion("call MPI_reduce");
      // previously, we had tried non-blocking collectives here.
      // However, this resulted in some segfaults, so we fell back to standard
      // reduce.
      if (pool_rank == iPool) {
        // MPI_Ireduce(MPI_IN_PLACE, poolElPhCached_h.data(),
        // poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool,
        // mpi->getComm(mpi->intraPoolComm), &mpi_requests[iPool]);
        MPI_Reduce(MPI_IN_PLACE, poolElPhCached_h.data(),
                   poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool,
                   mpi->getComm(mpi->intraPoolComm)); //, &mpi_requests[iPool]);
      } else {
        MPI_Reduce(poolElPhCached_h.data(), poolElPhCached_h.data(),
                   poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool,
                   mpi->getComm(mpi->intraPoolComm)); //, &mpi_requests[iPool]);
        // MPI_Ireduce(poolElPhCached_h.data(), poolElPhCached_h.data(),
        // poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool,
        // mpi->getComm(mpi->intraPoolComm), &mpi_requests[iPool]);
      }
      Kokkos::Profiling::popRegion();
#endif
    }
  }
  this->elPhCached = elPhCached;
  double newMemory = getDeviceMemoryUsage();
  kokkosDeviceMemory->addDeviceMemoryUsage(newMemory);
  Kokkos::Profiling::popRegion(); // cache elph
}

double InteractionElPhWan::getDeviceMemoryUsage() {
  double x = 16 * (this->elPhCached.size() + couplingWannier_k.size()) +
             8 * (wsR2VectorsDegeneracies_k.size() + wsR2Vectors_k.size() +
                  wsR1Vectors_k.size() + wsR1VectorsDegeneracies_k.size());
  return x;
}

void InteractionElPhWan::oldCalcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3, const Eigen::Vector3d &k1C,
    const std::vector<Eigen::Vector3d> &k2Cs,
    const std::vector<Eigen::Vector3d> &q3Cs) {

  (void)k2Cs;
  int numWannier = numElBands;
  int nb1 = eigvec1.cols();

  int numLoops = eigvecs2.size();
  cacheCoupling.resize(0);
  cacheCoupling.resize(numLoops);

  if (k1C != cachedK1 || elPhCached.size() == 0) {
    cachedK1 = k1C;

    Eigen::Tensor<std::complex<double>, 4> g1(numWannier, numWannier,
                                              numPhBands, numWsR2Vectors);
    g1.setZero();

    std::vector<std::complex<double>> phases(numWsR1Vectors);
    for (int irE = 0; irE < numWsR1Vectors; irE++) {
      double arg = k1C.dot(wsR1Vectors.col(irE));
      phases[irE] = exp(complexI * arg) / double(wsR1VectorsDegeneracies(irE));
    }
    for (int irE = 0; irE < numWsR1Vectors; irE++) {
      for (int irP = 0; irP < numWsR2Vectors; irP++) {
        for (int nu = 0; nu < numPhBands; nu++) {
          for (int iw1 = 0; iw1 < numWannier; iw1++) {
            for (int iw2 = 0; iw2 < numWannier; iw2++) {
              // important note: the first index iw2 runs over the k+q transform
              // while iw1 runs over k
              g1(iw2, iw1, nu, irP) +=
                  couplingWannier(iw2, iw1, nu, irP, irE) * phases[irE];
            }
          }
        }
      }
    }
    elPhCached_old.resize(numWannier, nb1, numPhBands, numWsR2Vectors);
    elPhCached_old.setZero();

    for (int irP = 0; irP < numWsR2Vectors; irP++) {
      for (int nu = 0; nu < numPhBands; nu++) {
        for (int iw1 = 0; iw1 < numWannier; iw1++) {
          for (int ib1 = 0; ib1 < nb1; ib1++) {
            for (int iw2 = 0; iw2 < numWannier; iw2++) {
              elPhCached_old(iw2, ib1, nu, irP) +=
                  g1(iw2, iw1, nu, irP) * eigvec1(iw1, ib1);
            }
          }
        }
      }
    }
  }

  for (int ik = 0; ik < numLoops; ik++) {
    Eigen::Vector3d q3C = q3Cs[ik];

    Eigen::MatrixXcd eigvec2 = eigvecs2[ik];
    int nb2 = eigvec2.cols();
    Eigen::MatrixXcd eigvec3 = eigvecs3[ik];

    Eigen::Tensor<std::complex<double>, 3> g3(numWannier, nb1, numPhBands);
    g3.setZero();
    std::vector<std::complex<double>> phases(numWsR2Vectors);
    for (int irP = 0; irP < numWsR2Vectors; irP++) {
      double arg = q3C.dot(wsR2Vectors.col(irP));
      phases[irP] = exp(complexI * arg) / double(wsR2VectorsDegeneracies(irP));
    }
    for (int irP = 0; irP < numWsR2Vectors; irP++) {
      for (int nu = 0; nu < numPhBands; nu++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int iw2 = 0; iw2 < numWannier; iw2++) {
            g3(iw2, ib1, nu) += phases[irP] * elPhCached_old(iw2, ib1, nu, irP);
          }
        }
      }
    }

    Eigen::Tensor<std::complex<double>, 3> g4(numWannier, nb1, numPhBands);
    g4.setZero();
    for (int nu = 0; nu < numPhBands; nu++) {
      for (int nu2 = 0; nu2 < numPhBands; nu2++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int iw2 = 0; iw2 < numWannier; iw2++) {
            g4(iw2, ib1, nu2) += g3(iw2, ib1, nu) * eigvec3(nu, nu2);
          }
        }
      }
    }

    auto eigvec2Dagger = eigvec2.adjoint();
    Eigen::Tensor<std::complex<double>, 3> gFinal(nb2, nb1, numPhBands);
    gFinal.setZero();
    for (int nu = 0; nu < numPhBands; nu++) {
      for (int ib1 = 0; ib1 < nb1; ib1++) {
        for (int iw2 = 0; iw2 < numWannier; iw2++) {
          for (int ib2 = 0; ib2 < nb2; ib2++) {
            gFinal(ib2, ib1, nu) += eigvec2Dagger(ib2, iw2) * g4(iw2, ib1, nu);
          }
        }
      }
    }

    if (usePolarCorrection && q3C.norm() > 1.0e-8) {
      gFinal += getPolarCorrection(q3C, eigvec1, eigvec2, eigvec3);
    }

    Eigen::Tensor<double, 3> coupling(nb1, nb2, numPhBands);
    for (int nu = 0; nu < numPhBands; nu++) {
      for (int ib2 = 0; ib2 < nb2; ib2++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          // notice the flip of 1 and 2 indices is intentional
          // coupling is |<k+q,ib2 | dV_nu | k,ib1>|^2
          coupling(ib1, ib2, nu) = std::norm(gFinal(ib2, ib1, nu));
        }
      }
    }
    cacheCoupling[ik] = coupling;
  }
}
