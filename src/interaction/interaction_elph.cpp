#include "interaction_elph.h"
#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>

#ifdef HDF5_AVAIL
#include <Kokkos_ScatterView.hpp>
#endif

// default constructor
InteractionElPhWan::InteractionElPhWan(
    Crystal &crystal_,
    const Eigen::Tensor<std::complex<double>, 5> &couplingWannier_,
    const Eigen::MatrixXd &elBravaisVectors_,
    const Eigen::VectorXd &elBravaisVectorsDegeneracies_,
    const Eigen::MatrixXd &phBravaisVectors_,
    const Eigen::VectorXd &phBravaisVectorsDegeneracies_, PhononH0 *phononH0_)
    : crystal(crystal_), phononH0(phononH0_) {

  numElBands = int(couplingWannier_.dimension(0));
  numPhBands = int(couplingWannier_.dimension(2));
  numPhBravaisVectors = int(couplingWannier_.dimension(3));
  numElBravaisVectors = int(couplingWannier_.dimension(4));

  usePolarCorrection = false;
  if (phononH0 != nullptr) {
    Eigen::Matrix3d epsilon = phononH0->getDielectricMatrix();
    if (epsilon.squaredNorm() > 1.0e-10) {// i.e. if epsilon wasn't computed
      if (crystal.getNumSpecies() > 1) {  // otherwise polar correction = 0
        usePolarCorrection = true;
      }
    }
  }

  // in the first call to this function, we must copy the el-ph tensor
  // from the CPU to the accelerator
  {
    Kokkos::realloc(couplingWannier_k, numElBravaisVectors, numPhBravaisVectors,
                    numPhBands, numElBands, numElBands);
    Kokkos::realloc(elBravaisVectorsDegeneracies_k, numElBravaisVectors);
    Kokkos::realloc(phBravaisVectorsDegeneracies_k, numPhBravaisVectors);
    Kokkos::realloc(elBravaisVectors_k, numElBravaisVectors, 3);
    Kokkos::realloc(phBravaisVectors_k, numPhBravaisVectors, 3);

    // note that Eigen has left layout while kokkos has right layout
    HostComplexView5D couplingWannier_h((Kokkos::complex<double> *) couplingWannier_.data(),
                                        numElBravaisVectors, numPhBravaisVectors,
                                        numPhBands, numElBands, numElBands);
    HostDoubleView1D elBravaisVectorsDegeneracies_h((double *) elBravaisVectorsDegeneracies_.data(), numElBravaisVectors);
    HostDoubleView1D phBravaisVectorsDegeneracies_h((double *) phBravaisVectorsDegeneracies_.data(), numPhBravaisVectors);

    HostDoubleView2D elBravaisVectors_h((double *) elBravaisVectors_.data(), numElBravaisVectors, 3);
    HostDoubleView2D phBravaisVectors_h((double *) phBravaisVectors_.data(), numPhBravaisVectors, 3);

    Kokkos::deep_copy(couplingWannier_k, couplingWannier_h);
    Kokkos::deep_copy(phBravaisVectors_k, phBravaisVectors_h);
    Kokkos::deep_copy(phBravaisVectorsDegeneracies_k, phBravaisVectorsDegeneracies_h);
    Kokkos::deep_copy(elBravaisVectors_k, elBravaisVectors_h);
    Kokkos::deep_copy(elBravaisVectorsDegeneracies_k, elBravaisVectorsDegeneracies_h);
    double memoryUsed = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(memoryUsed);
  }
}

InteractionElPhWan::InteractionElPhWan(Crystal &crystal_) : crystal(crystal_) {}

// copy constructor
InteractionElPhWan::InteractionElPhWan(const InteractionElPhWan &that)
    : crystal(that.crystal), phononH0(that.phononH0),
      numPhBands(that.numPhBands), numElBands(that.numElBands),
      numElBravaisVectors(that.numElBravaisVectors),
      numPhBravaisVectors(that.numPhBravaisVectors),
      cacheCoupling(that.cacheCoupling),
      usePolarCorrection(that.usePolarCorrection),
      elPhCached(that.elPhCached), couplingWannier_k(that.couplingWannier_k),
      phBravaisVectors_k(that.phBravaisVectors_k),
      phBravaisVectorsDegeneracies_k(that.phBravaisVectorsDegeneracies_k),
      elBravaisVectors_k(that.elBravaisVectors_k),
      elBravaisVectorsDegeneracies_k(that.elBravaisVectorsDegeneracies_k) {}

// assignment operator
InteractionElPhWan &
InteractionElPhWan::operator=(const InteractionElPhWan &that) {
  if (this != &that) {
    crystal = that.crystal;
    phononH0 = that.phononH0;
    numPhBands = that.numPhBands;
    numElBands = that.numElBands;
    numElBravaisVectors = that.numElBravaisVectors;
    numPhBravaisVectors = that.numPhBravaisVectors;
    cacheCoupling = that.cacheCoupling;
    usePolarCorrection = that.usePolarCorrection;
    elPhCached = that.elPhCached;
    couplingWannier_k = that.couplingWannier_k;
    phBravaisVectors_k = that.phBravaisVectors_k;
    phBravaisVectorsDegeneracies_k = that.phBravaisVectorsDegeneracies_k;
    elBravaisVectors_k = that.elBravaisVectors_k;
    elBravaisVectorsDegeneracies_k = that.elBravaisVectorsDegeneracies_k;
  }
  return *this;
}

InteractionElPhWan::~InteractionElPhWan() {
  //printf("rank %d calling interaction destructor\n", mpi->getRank());
  if(couplingWannier_k.use_count()==1){
    double memory = getDeviceMemoryUsage();
    kokkosDeviceMemory->removeDeviceMemoryUsage(memory);
  }
}

Eigen::Tensor<double, 3>&
InteractionElPhWan::getCouplingSquared(const int &ik2) {
  return cacheCoupling[ik2];
}

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
// * TODO additionally, could store this polar correction internally and use it directly
// when calcCouplingSquared is called. However, for now we let the external
// scattering rate calculation class handle it and pass it back to the calcCoupling
// function. This is because currently in el-ph coupling, the polar data is
// calculated by the external pointsHelper class, rather than as a precomputation
// over q at the start of the scattering rate calculation/
Eigen::MatrixXcd InteractionElPhWan::precomputeQDependentPolar(
                                                BaseBandStructure &phBandStructure) {

  if(!phBandStructure.getParticle().isPhonon()) {
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
  for(int iiq = 0; iiq < int(qIterator.size()); iiq++) { 
    int iq = qIterator[iiq];
    WavevectorIndex iqIdx(iq);
    auto qC = phBandStructure.getWavevector(iqIdx);
    auto evQ = phBandStructure.getEigenvectors(iqIdx);
    Eigen::VectorXcd thisPolar = polarCorrectionPart1(qC, evQ);
    for (int i=0; i<thisPolar.size(); ++i) {
      polarData(iq, i) = thisPolar(i);
    }
  }
  mpi->allReduceSum(&polarData);
  return polarData;
}

Eigen::Tensor<std::complex<double>, 3>
InteractionElPhWan::getPolarCorrectionStatic(
    const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
    const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3,
    const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
    const Eigen::Matrix3d &epsilon,
    const Eigen::Tensor<double, 3> &bornCharges,
    const Eigen::MatrixXd &atomicPositions,
    const Eigen::Vector3i &qCoarseMesh) {
  Eigen::VectorXcd x = polarCorrectionPart1Static(q3, ev3, volume, reciprocalUnitCell,
                                                  epsilon, bornCharges, atomicPositions, qCoarseMesh);
  return polarCorrectionPart2(ev1, ev2, x);
}

Eigen::VectorXcd
InteractionElPhWan::polarCorrectionPart1(const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3) {
  // gather variables
  double volume = crystal.getVolumeUnitCell();
  Eigen::Matrix3d reciprocalUnitCell = crystal.getReciprocalUnitCell();
  Eigen::Matrix3d epsilon = phononH0->getDielectricMatrix();
  Eigen::Tensor<double, 3> bornCharges = phononH0->getBornCharges();
  // must be in Bohr
  Eigen::MatrixXd atomicPositions = crystal.getAtomicPositions();
  Eigen::Vector3i qCoarseMesh = phononH0->getCoarseGrid();

  return polarCorrectionPart1Static(q3, ev3, volume, reciprocalUnitCell,
                                    epsilon, bornCharges, atomicPositions, qCoarseMesh);
}

Eigen::VectorXcd InteractionElPhWan::polarCorrectionPart1Static(
    const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3,
    const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
    const Eigen::Matrix3d &epsilon, const Eigen::Tensor<double, 3> &bornCharges,
    const Eigen::MatrixXd &atomicPositions, const Eigen::Vector3i &qCoarseMesh) {
  // doi:10.1103/physRevLett.115.176401, Eq. 4, is implemented here

  Kokkos::Profiling::pushRegion("polarCorrectionPart1Static");

  auto numAtoms = int(atomicPositions.rows());

  // auxiliary terms
  double gMax = 14.;
  double chargeSquare = 2.;// = e^2/4/Pi/eps_0 in atomic units
  std::complex<double> factor = chargeSquare * fourPi / volume * complexI;

  // build a list of (q+G) vectors
  std::vector<Eigen::Vector3d> gVectors;// here we insert all (q+G)
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

  auto numPhBands = int(ev3.rows());
  Eigen::VectorXcd x(numPhBands);
  x.setZero();
  for (Eigen::Vector3d gVector : gVectors) {
    double qEq = gVector.transpose() * epsilon * gVector;
    if (qEq > 0. && qEq / 4. < gMax) {
      std::complex<double> factor2 = factor * exp(-qEq / 4.) / qEq;
      for (int iAt = 0; iAt < numAtoms; iAt++) {
        double arg = -gVector.dot(atomicPositions.row(iAt));
        std::complex<double> phase = {cos(arg), sin(arg)};
        std::complex<double> factor3 = factor2 * phase;
        for (int iPol : {0, 1, 2}) {
          double gqDotZ = gVector(0) * bornCharges(iAt, 0, iPol) + gVector(1) * bornCharges(iAt, 1, iPol) + gVector(2) * bornCharges(iAt, 2, iPol);
          int k = PhononH0::getIndexEigenvector(iAt, iPol, numAtoms);
          for (int ib3 = 0; ib3 < numPhBands; ib3++) {
            x(ib3) += factor3 * gqDotZ * ev3(k, ib3);
          }
        }
      }
    }
  }
  Kokkos::Profiling::popRegion();
  return x;
}

Eigen::Tensor<std::complex<double>, 3>
InteractionElPhWan::polarCorrectionPart2(const Eigen::MatrixXcd &ev1, const Eigen::MatrixXcd &ev2, const Eigen::VectorXcd &x) {
  // overlap = <U^+_{b2 k+q}|U_{b1 k}>
  //         = <psi_{b2 k+q}|e^{i(q+G)r}|psi_{b1 k}>
  Eigen::MatrixXcd overlap = ev2.adjoint() * ev1;// matrix size (nb2,nb1)
  overlap = overlap.transpose();                 // matrix size (nb1,nb2)

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
    const std::vector<Eigen::Vector3d> &q3Cs,
    const std::vector<Eigen::VectorXcd> &polarData) {

  Kokkos::Profiling::pushRegion("calcCouplingSquared");
  int numWannier = numElBands;
  auto nb1 = int(eigvec1.cols());
  auto numLoops = int(eigvecs2.size());

#ifdef MPI_AVAIL
  int pool_rank = mpi->getRank(mpi->intraPoolComm);
  int pool_size = mpi->getSize(mpi->intraPoolComm);
  if(pool_size > 1 && mpi_requests[0] != MPI_REQUEST_NULL){
      Kokkos::Profiling::pushRegion("wait for reductions");
      // wait for MPI_Ireduces from cacheCoupling
      //MPI_Waitall(pool_size, mpi_requests.data(), MPI_STATUSES_IGNORE);
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("copy to GPU");
      this->elPhCached = Kokkos::create_mirror_view_and_copy(
          Kokkos::DefaultExecutionSpace(), elPhCached_hs[pool_rank]
      );
      Kokkos::Profiling::popRegion();
  }
#endif

  auto elPhCached = this->elPhCached;
  int numPhBands = this->numPhBands;
  int numPhBravaisVectors = this->numPhBravaisVectors;
  DoubleView2D phBravaisVectors_k = this->phBravaisVectors_k;
  DoubleView1D phBravaisVectorsDegeneracies_k = this->phBravaisVectorsDegeneracies_k;

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
  ComplexView4D polarCorrections(Kokkos::ViewAllocateWithoutInitializing("polarCorrections"),
                                 numLoops, numPhBands, nb1, nb2max);
  auto usePolarCorrections_h = Kokkos::create_mirror_view(usePolarCorrections);
  auto polarCorrections_h = Kokkos::create_mirror_view(polarCorrections);

  // precompute all needed polar corrections
#pragma omp parallel for
  for (int ik = 0; ik < numLoops; ik++) {
    Eigen::Vector3d q3C = q3Cs[ik];
    Eigen::MatrixXcd eigvec2 = eigvecs2[ik];
    Eigen::MatrixXcd eigvec3 = eigvecs3[ik];
    usePolarCorrections_h(ik) = usePolarCorrection && q3C.norm() > 1.0e-8;
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

#pragma omp parallel for default(none) shared(eigvecs3_h, eigvecs2Dagger_h, nb2s_h, q3Cs_h, q3Cs_k, q3Cs, numLoops, numWannier, numPhBands, eigvecs2Dagger_k, eigvecs3_k, eigvecs2, eigvecs3)
    for (int ik = 0; ik < numLoops; ik++) {
      for (int i = 0; i < numWannier; i++) {
        for (int j = 0; j < nb2s_h(ik); j++) {
          eigvecs2Dagger_h(ik, i, j) = std::conj(eigvecs2[ik](i, j));
        }
      }
      for (int i = 0; i < numPhBands; i++) {
        for (int j = 0; j < numPhBands; j++) {
          eigvecs3_h(ik, i, j) = eigvecs3[ik](j, i);
        }
      }
      for (int i = 0; i < numPhBands; i++) {
        for (int j = 0; j < numPhBands; j++) {
          eigvecs3_h(ik, i, j) = eigvecs3[ik](j, i);
        }
      }

      for (int i = 0; i < 3; i++) {
        q3Cs_h(ik, i) = q3Cs[ik](i);
      }
    }
    Kokkos::deep_copy(eigvecs2Dagger_k, eigvecs2Dagger_h);
    Kokkos::deep_copy(eigvecs3_k, eigvecs3_h);
    Kokkos::deep_copy(q3Cs_k, q3Cs_h);
  }

  // now we finish the Wannier transform. We have to do the Fourier transform
  // on the lattice degrees of freedom, and then do two rotations (at k2 and q)
  // -------------------------------------------------------------------------
  // set up the phases
  ComplexView2D phases("phases", numLoops, numPhBravaisVectors);
  Kokkos::complex<double> complexI(0.0, 1.0);
  Kokkos::parallel_for(
      "phases", Range2D({0, 0}, {numLoops, numPhBravaisVectors}),
      KOKKOS_LAMBDA(int ik, int irP) {
        double arg = 0.0;
        for (int j = 0; j < 3; j++) {
          arg += q3Cs_k(ik, j) * phBravaisVectors_k(irP, j);
        }
        phases(ik, irP) =
            exp(complexI * arg) / phBravaisVectorsDegeneracies_k(irP);
     });
   Kokkos::fence();

  // apply phases
  ComplexView4D g3(Kokkos::ViewAllocateWithoutInitializing("g3"), numLoops, numPhBands, nb1, numWannier);
  Kokkos::parallel_for(
      "g3", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, numWannier}),
      KOKKOS_LAMBDA(int ik, int nu, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int irP = 0; irP < numPhBravaisVectors; irP++) {
          tmp += phases(ik, irP) * elPhCached(irP, nu, ib1, iw2);
        }
        g3(ik, nu, ib1, iw2) = tmp;
      });
  Kokkos::realloc(phases, 0, 0);

  // rotate using phonon eigenvectors
  ComplexView4D g4(Kokkos::ViewAllocateWithoutInitializing("g4"), numLoops, numPhBands, nb1, numWannier);
  Kokkos::parallel_for(
      "g4", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, numWannier}),
      KOKKOS_LAMBDA(int ik, int nu2, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int nu = 0; nu < numPhBands; nu++) {
          tmp += g3(ik, nu, ib1, iw2) * eigvecs3_k(ik, nu2, nu);
        }
        g4(ik, nu2, ib1, iw2) = tmp;
      });
  Kokkos::realloc(g3, 0, 0, 0, 0);

  // rotate using U^dagger eigenvectors
  ComplexView4D gFinal(Kokkos::ViewAllocateWithoutInitializing("gFinal"), numLoops, numPhBands, nb1, nb2max);
  Kokkos::parallel_for(
      "gFinal", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, nb2max}),
      KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int iw2 = 0; iw2 < numWannier; iw2++) {
          tmp += eigvecs2Dagger_k(ik, iw2, ib2) * g4(ik, nu, ib1, iw2);
        }
        gFinal(ik, nu, ib1, ib2) = tmp;
      });
  Kokkos::realloc(g4, 0, 0, 0, 0);

  // we now add the precomputed polar corrections, before taking the norm of g
  if (usePolarCorrection) {
    Kokkos::parallel_for(
        "correction",
        Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, nb2max}),
        KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
          gFinal(ik, nu, ib1, ib2) += polarCorrections(ik, nu, ib1, ib2);
        });
  }
  Kokkos::realloc(polarCorrections, 0, 0, 0, 0);

  // finally, compute |g|^2 from g
  DoubleView4D coupling_k(Kokkos::ViewAllocateWithoutInitializing("coupling"), numLoops, numPhBands, nb2max, nb1);
  Kokkos::parallel_for(
      "coupling", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb2max, nb1}),
      KOKKOS_LAMBDA(int ik, int nu, int ib2, int ib1) {
        // notice the flip of 1 and 2 indices is intentional
        // coupling is |<k+q,ib2 | dV_nu | k,ib1>|^2
        auto tmp = gFinal(ik, nu, ib1, ib2);
        coupling_k(ik, nu, ib2, ib1) =
            tmp.real() * tmp.real() + tmp.imag() * tmp.imag();
      });
  Kokkos::realloc(gFinal, 0, 0, 0, 0);

  // now, copy results back to the CPU
  cacheCoupling.resize(0);
  cacheCoupling.resize(numLoops);
  auto coupling_h = Kokkos::create_mirror_view(coupling_k);
  Kokkos::deep_copy(coupling_h, coupling_k);
#pragma omp parallel for default(none) shared(numLoops, cacheCoupling, coupling_h, nb1, nb2s_h, numPhBands)
  for (int ik = 0; ik < numLoops; ik++) {
    Eigen::Tensor<double, 3> coupling(nb1, nb2s_h(ik), numPhBands);
    for (int nu = 0; nu < numPhBands; nu++) {
      for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          coupling(ib1, ib2, nu) = coupling_h(ik, nu, ib2, ib1);
        }
      }
    }
    // and we save the coupling |g|^2 it for later
    cacheCoupling[ik] = coupling;
  }
  Kokkos::Profiling::popRegion();
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
  double phase = 16 * numPhBravaisVectors;
  double g3 = 2 * 16 * numPhBands * nb1 * numElBands;
  double g4 = 2 * 16 * numPhBands * nb1 * numElBands;
  double gFinal = 2 * 16 * numPhBands * nb1 * maxNb2;
  double coupling = 16 * nb1 * maxNb2 * numPhBands;
  double polar = 16 * numPhBands * nb1 * maxNb2;
  double maxUsage =
      nk2 * (evs + polar + std::max({phase + g3, g3 + g4, g4 + gFinal, gFinal + coupling}));

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

void InteractionElPhWan::cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C) {

  Kokkos::Profiling::pushRegion("cacheElPh");
  //  int numWannier = numElBands;
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
  int numElBravaisVectors = this->numElBravaisVectors;
  int numPhBravaisVectors = this->numPhBravaisVectors;

  double memory = getDeviceMemoryUsage();
  kokkosDeviceMemory->removeDeviceMemoryUsage(memory);

  int pool_rank = mpi->getRank(mpi->intraPoolComm);
  int pool_size = mpi->getSize(mpi->intraPoolComm);

#ifdef MPI_AVAIL
  mpi_requests.resize(pool_size);
  elPhCached_hs.resize(pool_size);
#endif
  ComplexView4D g1(Kokkos::ViewAllocateWithoutInitializing("g1"),
      numPhBravaisVectors, numPhBands, numElBands, numElBands);

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
      poolEigvec1 = eigvec1;
    }
    // broadcast to other processes on this pool
    mpi->bcast(&poolK1C, mpi->intraPoolComm, iPool);
    mpi->bcast(&poolEigvec1, mpi->intraPoolComm, iPool);

    // now, copy the eigenvector and wavevector to the accelerator
    ComplexView2D eigvec1_k("ev1", poolNb1, numElBands);
    DoubleView1D poolK1C_k("k", 3);
    {
      HostComplexView2D eigvec1_h((Kokkos::complex<double> *) poolEigvec1.data(), poolNb1, numElBands);
      HostDoubleView1D poolK1C_h(poolK1C.data(), 3);
      Kokkos::deep_copy(eigvec1_k, eigvec1_h);
      Kokkos::deep_copy(poolK1C_k, poolK1C_h);
    }

    // now compute the Fourier transform on electronic coordinates.
    ComplexView5D couplingWannier_k = this->couplingWannier_k;
    DoubleView2D elBravaisVectors_k = this->elBravaisVectors_k;
    DoubleView1D elBravaisVectorsDegeneracies_k = this->elBravaisVectorsDegeneracies_k;
    Kokkos::Profiling::popRegion();

    // first we precompute the phases
    ComplexView1D phases_k("phases", numElBravaisVectors);
    Kokkos::parallel_for("phases_k", numElBravaisVectors,
        KOKKOS_LAMBDA(int irE) {
          double arg = 0.0;
          for (int j = 0; j < 3; j++) {
            arg += poolK1C_k(j) * elBravaisVectors_k(irE, j);
          }
          phases_k(irE) = exp(complexI * arg) / elBravaisVectorsDegeneracies_k(irE);
        });
    Kokkos::fence();
    Kokkos::Profiling::popRegion();

    // now we complete the Fourier transform
    // We have to write two codes: one for when the GPU runs on CUDA,
    // the other for when we compile the code without GPU support
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::parallel_for(
        "g1",
        Range4D({0, 0, 0, 0},
                {numPhBravaisVectors, numPhBands, numElBands, numElBands}),
        KOKKOS_LAMBDA(int irP, int nu, int iw1, int iw2) {
          Kokkos::complex<double> tmp(0.0);
          for (int irE = 0; irE < numElBravaisVectors; irE++) {
            // important note: the first index iw2 runs over the k+q transform
            // while iw1 runs over k
            tmp += couplingWannier_k(irE, irP, nu, iw1, iw2) * phases_k(irE);
          }
          g1(irP, nu, iw1, iw2) = tmp;
        });
   Kokkos::fence();
#else

  // Here we create a view to the elph matrix elements which represents it
  // in 2D, so that we can use a matrix-vector product with the phases to accelerate
  // an otherwise very expensive loop
  //
  // tutorial description of this gemv function:
  // https://youtu.be/_qD4X66MQF8?t=2434
  // read me about gemv https://github.com/kokkos/kokkos-kernels/wiki/BLAS-2%3A%3Agemv

  // product of phase factor with g
  Kokkos::View<Kokkos::complex<double>*> g1_1D(g1.data(), numPhBravaisVectors*numPhBands*numElBands*numElBands);
  Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutRight> coupling_2D(couplingWannier_k.data(), numElBravaisVectors, numPhBravaisVectors*numPhBands*numElBands*numElBands);
  KokkosBlas::gemv("T", Kokkos::complex<double>(1.0), coupling_2D, phases_k, Kokkos::complex<double>(0.0), g1_1D);

/*
  // Previous method -- slower than the gemv call
    Kokkos::deep_copy(g1, Kokkos::complex<double>(0.0, 0.0));
    Kokkos::Experimental::ScatterView<Kokkos::complex<double> ****> g1scatter(g1);
    Kokkos::parallel_for(
        "g1",
        Range5D({0, 0, 0, 0, 0},
                {numElBravaisVectors, numPhBravaisVectors, numPhBands, numElBands, numElBands}),
        KOKKOS_LAMBDA(int irE, int irP, int nu, int iw1, int iw2) {
          auto g1 = g1scatter.access();
          g1(irP, nu, iw1, iw2) += couplingWannier_k(irE, irP, nu, iw1, iw2) * phases_k(irE);
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
      Kokkos::realloc(elPhCached, numPhBravaisVectors, numPhBands, poolNb1, numElBands);

      Kokkos::parallel_for(
          "elPhCached",
          Range4D({0, 0, 0, 0},
                  {numPhBravaisVectors, numPhBands, poolNb1, numElBands}),
          KOKKOS_LAMBDA(int irP, int nu, int ib1, int iw2) {
            Kokkos::complex<double> tmp(0.0);
            for (int iw1 = 0; iw1 < numElBands; iw1++) {
              tmp += g1(irP, nu, iw1, iw2) * eigvec1_k(ib1, iw1);
            }
            elPhCached(irP, nu, ib1, iw2) = tmp;
          });
      Kokkos::fence();

    } else {

      ComplexView4D poolElPhCached_k(Kokkos::ViewAllocateWithoutInitializing("poolElPhCached"),
                                   numPhBravaisVectors, numPhBands, poolNb1, numElBands);

      Kokkos::parallel_for(
          "elPhCached",
          Range4D({0, 0, 0, 0},
                  {numPhBravaisVectors, numPhBands, poolNb1, numElBands}),
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
      //mpi->allReduceSum(&poolElPhCached_h, mpi->intraPoolComm);

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
      // However, this resulted in some segfaults, so we fell back to standard reduce.
      if (pool_rank == iPool) {
        //MPI_Ireduce(MPI_IN_PLACE, poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm), &mpi_requests[iPool]);
        MPI_Reduce(MPI_IN_PLACE, poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm)); //, &mpi_requests[iPool]);
      }
      else{
        MPI_Reduce(poolElPhCached_h.data(), poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm)); //, &mpi_requests[iPool]);
        //MPI_Ireduce(poolElPhCached_h.data(), poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm), &mpi_requests[iPool]);
      }
      Kokkos::Profiling::popRegion();
#endif

    }
  }
  this->elPhCached = elPhCached;
  double newMemory = getDeviceMemoryUsage();
  kokkosDeviceMemory->addDeviceMemoryUsage(newMemory);
  Kokkos::Profiling::popRegion();
}

double InteractionElPhWan::getDeviceMemoryUsage() {
  double x = 16 * (this->elPhCached.size() + couplingWannier_k.size())
      + 8 * (phBravaisVectorsDegeneracies_k.size() + phBravaisVectors_k.size() + elBravaisVectors_k.size() + elBravaisVectorsDegeneracies_k.size());
  return x;
}
