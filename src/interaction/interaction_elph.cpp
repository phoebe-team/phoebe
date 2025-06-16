#include "interaction_elph.h"
#include "interaction_elph_base.h"

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
    : InteractionElPhBase(crystal_, phononH0_,
                        static_cast<int>(couplingWannier_.dimension(2)),      // numPhBands
                        static_cast<int>(couplingWannier_.dimension(0)),      // numElBands
                        static_cast<int>(elBravaisVectors_.rows()),           // numElBravaisVectors
                        static_cast<int>(phBravaisVectors_.rows())            // numPhBravaisVectors
                        ) {

  // numElBands, numPhBands, numPhBravaisVectors, numElBravaisVectors are now set in InteractionElPhBase constructor.
  // We access them via this->numElBands, this->numPhBands, this->numPhBravaisVectors, this->numElBravaisVectors.

  // in the first call to this function, we must copy the el-ph tensor
  // from the CPU to the accelerator
  {
    // couplingWannier_k is specific to InteractionElPhWan
    // Dimensions for couplingWannier_k in Kokkos: (NElCells, NPhCells, NPhBands, NElBands_state2, NElBands_state1)
    // Eigen tensor dims: (NElBands_state1, NElBands_state2, NPhBands, NPhCells, NElCells)
    int d0_eig_elbands1 = couplingWannier_.dimension(0);
    int d1_eig_elbands2 = couplingWannier_.dimension(1);
    int d2_eig_phbands = couplingWannier_.dimension(2);
    int d3_eig_phcells = couplingWannier_.dimension(3);
    int d4_eig_elcells = couplingWannier_.dimension(4);

    Kokkos::realloc(couplingWannier_k, d4_eig_elcells, d3_eig_phcells,
                    d2_eig_phbands, d1_eig_elbands2, d0_eig_elbands1);

    // these views are members of InteractionElPhBase
    Kokkos::realloc(this->elBravaisVectorsDegeneracies_k, this->numElBravaisVectors);
    Kokkos::realloc(this->phBravaisVectorsDegeneracies_k, this->numPhBravaisVectors);
    Kokkos::realloc(this->elBravaisVectors_k, this->numElBravaisVectors, 3);
    Kokkos::realloc(this->phBravaisVectors_k, this->numPhBravaisVectors, 3);

    // note that Eigen has left layout while kokkos has right layout
    HostComplexView5D couplingWannier_h((Kokkos::complex<double> *) couplingWannier_.data(),
                                        d4_eig_elcells, d3_eig_phcells, d2_eig_phbands, d1_eig_elbands2, d0_eig_elbands1);
    HostDoubleView1D elBravaisVectorsDegeneracies_h((double *) elBravaisVectorsDegeneracies_.data(), this->numElBravaisVectors);
    HostDoubleView1D phBravaisVectorsDegeneracies_h((double *) phBravaisVectorsDegeneracies_.data(), this->numPhBravaisVectors);

    HostDoubleView2D elBravaisVectors_h((double *) elBravaisVectors_.data(), this->numElBravaisVectors, 3);
    HostDoubleView2D phBravaisVectors_h((double *) phBravaisVectors_.data(), this->numPhBravaisVectors, 3);

    Kokkos::deep_copy(couplingWannier_k, couplingWannier_h);
    Kokkos::deep_copy(this->phBravaisVectors_k, phBravaisVectors_h);
    Kokkos::deep_copy(this->phBravaisVectorsDegeneracies_k, phBravaisVectorsDegeneracies_h);
    Kokkos::deep_copy(this->elBravaisVectors_k, elBravaisVectors_h);
    Kokkos::deep_copy(this->elBravaisVectorsDegeneracies_k, elBravaisVectorsDegeneracies_h);

    if (this->kokkosDeviceMemory != nullptr) {
        double baseMemory = this->InteractionElPhBase::getDeviceMemoryUsage();
        this->kokkosDeviceMemory->addDeviceMemoryUsage(baseMemory);
        this->kokkosDeviceMemory->addDeviceMemoryUsage(16.0 * couplingWannier_k.size());
    }
  }
}

// copy constructor
InteractionElPhWan::InteractionElPhWan(const InteractionElPhWan &that)
    : InteractionElPhBase(that)
    {
    // Copy derived-class specific members
    // Kokkos views are reference counted, so this is a shallow copy of the view,
    // but a deep copy of the data if 'that' was the sole owner and this is the first copy.
    // Or rather, it shares the underlying allocation.
    this->couplingWannier_k = that.couplingWannier_k;

    // Standard container copies
    this->cacheCoupling = that.cacheCoupling;
    this->elPhCached_hs = that.elPhCached_hs;
    this->mpi_requests = that.mpi_requests;

    // Note: usePolarCorrection is a base member, handled by InteractionElPhBase(that)
    // Other base members like crystal, phononH0, numPhBands, elPhCached_k, Bravais vectors etc.
    // are handled by InteractionElPhBase(that) copy constructor.
}


// assignment operator
InteractionElPhWan &
InteractionElPhWan::operator=(const InteractionElPhWan &that) {
  if (this != &that) {
    InteractionElPhBase::operator=(that);

    this->couplingWannier_k = that.couplingWannier_k;
    this->cacheCoupling = that.cacheCoupling;
    this->elPhCached_hs = that.elPhCached_hs;
    this->mpi_requests = that.mpi_requests;
    // Base members (crystal, phononH0, numPhBands, etc., usePolarCorrection, elPhCached_k)
    // are handled by base class assignment.
  }
  return *this;
}

InteractionElPhWan::~InteractionElPhWan() {
  //printf("rank %d calling interaction destructor\n", mpi->getRank());
  if(couplingWannier_k.use_count()==1){
    double couplingWannierMemory = 16.0 * couplingWannier_k.size();
    if (this->kokkosDeviceMemory != nullptr) {
        this->kokkosDeviceMemory->removeDeviceMemoryUsage(couplingWannierMemory);
    }
  }
}

void InteractionElPhWan::calcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3,
    const std::vector<Eigen::Vector3d> &q3Cs,
    const std::vector<Eigen::VectorXcd> &polarData) override {
  Kokkos::Profiling::pushRegion("calcCouplingSquared");
  int numWannier = this->numElBands;
  auto nb1 = int(eigvec1.cols());
  auto numLoops = int(eigvecs2.size());

#ifdef MPI_AVAIL
  if (this->mpi == nullptr) { Error("MPI not initialized in calcCouplingSquared"); }
  int pool_rank = this->mpi->getRank(this->mpi->intraPoolComm);
  int pool_size = this->mpi->getSize(this->mpi->intraPoolComm);
  if(pool_size > 1 && !this->mpi_requests.empty() && this->mpi_requests[0] != MPI_REQUEST_NULL){
      Kokkos::Profiling::pushRegion("wait for reductions");
      // MPI_Waitall(this->mpi_requests.size(), this->mpi_requests.data(), MPI_STATUSES_IGNORE);
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("copy to GPU");
      if (pool_rank < this->elPhCached_hs.size()) {
        this->elPhCached_k = Kokkos::create_mirror_view_and_copy( /
            Kokkos::DefaultExecutionSpace(), this->elPhCached_hs[pool_rank]
      }
      Kokkos::Profiling::popRegion();
  }
#endif

  auto local_elPhCached_k = this->elPhCached_k;
  int local_numPhBands = this->numPhBands;
  int local_numPhBravaisVectors = this->numPhBravaisVectors;
  DeviceDoubleView2D local_phBravaisVectors_k = this->phBravaisVectors_k;
  DeviceDoubleView1D local_phBravaisVectorsDegeneracies_k = this->phBravaisVectorsDegeneracies_k;

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

  IntView1D usePolarCorrections_device_view("usePolarCorrections_device_view", numLoops);
  ComplexView4D polarCorrections_device_view(Kokkos::ViewAllocateWithoutInitializing("polarCorrections_device_view"),
                                 numLoops, local_numPhBands, nb1, nb2max);
  auto usePolarCorrections_h = Kokkos::create_mirror_view(usePolarCorrections_device_view);
  auto polarCorrections_h = Kokkos::create_mirror_view(polarCorrections_device_view);

#pragma omp parallel for
  for (int ik = 0; ik < numLoops; ik++) {
    Eigen::Vector3d q3C = q3Cs[ik];
    Eigen::MatrixXcd eigvec2_loop = eigvecs2[ik];
    Eigen::MatrixXcd eigvec3_loop = eigvecs3[ik];
    usePolarCorrections_h(ik) = this->usePolarCorrection && q3C.norm() > 1.0e-8;
      Eigen::Tensor<std::complex<double>, 3> singleCorrection =
          this->polarCorrectionPart2(eigvec1, eigvec2_loop, polarData[ik]);
      for (int nu = 0; nu < local_numPhBands; nu++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
            polarCorrections_h(ik, nu, ib1, ib2) =
                singleCorrection(ib1, ib2, nu);
          }
        }
      }
    } else {
      Kokkos::complex<double> kZero(0., 0.);
      for (int nu = 0; nu < local_numPhBands; nu++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
            polarCorrections_h(ik, nu, ib1, ib2) = kZero;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(polarCorrections_device_view, polarCorrections_h);
  Kokkos::deep_copy(usePolarCorrections_device_view, usePolarCorrections_h);

  DoubleView2D q3Cs_device_k("q3Cs_device_k", numLoops, 3);
  ComplexView3D eigvecs2Dagger_device_k("eigvecs2Dagger_device_k", numLoops, numWannier, nb2max);
  ComplexView3D eigvecs3_transformed_device_k("eigvecs3_transformed_device_k", numLoops, local_numPhBands, local_numPhBands);
  {
    auto eigvecs2Dagger_h = Kokkos::create_mirror_view(eigvecs2Dagger_device_k);
    auto eigvecs3_transformed_h = Kokkos::create_mirror_view(eigvecs3_transformed_device_k);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_device_k);

#pragma omp parallel for default(none) shared(eigvecs3_transformed_h, eigvecs2Dagger_h, nb2s_h, q3Cs_h, q3Cs_device_k, q3Cs, numLoops, numWannier, local_numPhBands, eigvecs2Dagger_device_k, eigvecs3_transformed_device_k, eigvecs2, eigvecs3)
    for (int ik = 0; ik < numLoops; ik++) {
      for (int i = 0; i < numWannier; i++) {
        for (int j = 0; j < nb2s_h(ik); j++) {
          eigvecs2Dagger_h(ik, i, j) = std::conj(eigvecs2[ik](i, j));
        }
      }
      for (int i = 0; i < local_numPhBands; i++) {
        for (int j = 0; j < local_numPhBands; j++) {
          eigvecs3_transformed_h(ik, i, j) = eigvecs3[ik](j, i);
        }
      }
      // Duplicate loop removed

      for (int i = 0; i < 3; i++) {
        q3Cs_h(ik, i) = q3Cs[ik](i);
      }
    }
    Kokkos::deep_copy(eigvecs2Dagger_device_k, eigvecs2Dagger_h);
    Kokkos::deep_copy(eigvecs3_transformed_device_k, eigvecs3_transformed_h);
    Kokkos::deep_copy(q3Cs_device_k, q3Cs_h);
  }

  ComplexView2D phases("phases", numLoops, local_numPhBravaisVectors);
  Kokkos::complex<double> complexI(0.0, 1.0);
  Kokkos::parallel_for(
      "phases_calc_g3", Range2D({0, 0}, {numLoops, local_numPhBravaisVectors}),
      KOKKOS_LAMBDA(int ik, int irP) {
        double arg = 0.0;
        for (int j = 0; j < 3; j++) {
          arg += q3Cs_device_k(ik, j) * local_phBravaisVectors_k(irP, j);
        }
        phases(ik, irP) =
            exp(complexI * arg) / local_phBravaisVectorsDegeneracies_k(irP);
     });
   Kokkos::fence();

  ComplexView4D g3(Kokkos::ViewAllocateWithoutInitializing("g3"), numLoops, local_numPhBands, nb1, numWannier);
  Kokkos::parallel_for(
      "g3_calc", Range4D({0, 0, 0, 0}, {numLoops, local_numPhBands, nb1, numWannier}),
      KOKKOS_LAMBDA(int ik, int nu, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int irP = 0; irP < local_numPhBravaisVectors; irP++) {
          tmp += phases(ik, irP) * local_elPhCached_k(irP, nu, ib1, iw2);
        }
        g3(ik, nu, ib1, iw2) = tmp;
      });
  Kokkos::realloc(phases, 0, 0);

  ComplexView4D g4(Kokkos::ViewAllocateWithoutInitializing("g4"), numLoops, local_numPhBands, nb1, numWannier);
  Kokkos::parallel_for(
      "g4_calc", Range4D({0, 0, 0, 0}, {numLoops, local_numPhBands, nb1, numWannier}),
      KOKKOS_LAMBDA(int ik, int nu2, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0., 0.);
        for (int nu = 0; nu < local_numPhBands; nu++) {
          tmp += g3(ik, nu, ib1, iw2) * eigvecs3_transformed_device_k(ik, nu2, nu);
        }
        g4(ik, nu2, ib1, iw2) = tmp;
      });
  Kokkos::realloc(g3, 0, 0, 0, 0);

  ComplexView4D gFinal(Kokkos::ViewAllocateWithoutInitializing("gFinal"), numLoops, local_numPhBands, nb1, nb2max);
  Kokkos::parallel_for(
      "gFinal_calc", Range4D({0, 0, 0, 0}, {numLoops, local_numPhBands, nb1, nb2max}),
      KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
        Kokkos::complex<double> tmp(0., 0.);
        if (ib2 < nb2s_k(ik)) {
            for (int iw2 = 0; iw2 < numWannier; iw2++) {
              tmp += eigvecs2Dagger_device_k(ik, iw2, ib2) * g4(ik, nu, ib1, iw2);
            }
        }
        gFinal(ik, nu, ib1, ib2) = tmp;
      });
  Kokkos::realloc(g4, 0, 0, 0, 0);

  if (this->usePolarCorrection) {
    Kokkos::parallel_for(
        "correction_add",
        Range4D({0, 0, 0, 0}, {numLoops, local_numPhBands, nb1, nb2max}),
        KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
          if (usePolarCorrections_device_view(ik) && ib2 < nb2s_k(ik)) {
             gFinal(ik, nu, ib1, ib2) += polarCorrections_device_view(ik, nu, ib1, ib2);
          }
        });
  }
  Kokkos::realloc(polarCorrections_device_view, 0, 0, 0, 0);
  Kokkos::realloc(usePolarCorrections_device_view, 0);


  DoubleView4D coupling_k(Kokkos::ViewAllocateWithoutInitializing("coupling"), numLoops, local_numPhBands, nb2max, nb1);
  Kokkos::parallel_for(
      "coupling_final_calc", Range4D({0, 0, 0, 0}, {numLoops, local_numPhBands, nb2max, nb1}),
      KOKKOS_LAMBDA(int ik, int nu, int ib2, int ib1) {
        Kokkos::complex<double> tmp_val(0.0,0.0);
        if (ib2 < nb2s_k(ik)){
            tmp_val = gFinal(ik, nu, ib1, ib2);
        }
        coupling_k(ik, nu, ib2, ib1) =
            tmp_val.real() * tmp_val.real() + tmp_val.imag() * tmp_val.imag();
      });
  Kokkos::realloc(gFinal, 0, 0, 0, 0);

  this->cacheCoupling.assign(numLoops, Eigen::Tensor<double,3>());
  auto coupling_h = Kokkos::create_mirror_view(coupling_k);
  Kokkos::deep_copy(coupling_h, coupling_k);

#pragma omp parallel for default(none) shared(numLoops, cacheCoupling, coupling_h, nb1, nb2s_h, local_numPhBands)
  for (int ik = 0; ik < numLoops; ik++) {
    Eigen::Tensor<double, 3> coupling_eigen(nb1, nb2s_h(ik), local_numPhBands);
    for (int nu = 0; nu < local_numPhBands; nu++) {
      for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          coupling_eigen(ib1, ib2, nu) = coupling_h(ik, nu, ib2, ib1);
        }
      }
    }
    this->cacheCoupling[ik] = coupling_eigen;
  }
  Kokkos::Profiling::popRegion();
}

Eigen::VectorXi InteractionElPhWan::getCouplingDimensions() override {
  Eigen::VectorXi xx(5);
  // couplingWannier_k (derived member) dimensions: (NElCells, NPhCells, NPhBands, NElBands_state2, NElBands_state1)
  // Original mapping to Eigen Tensor (d0..d4)
  xx(0) = couplingWannier_k.extent(4); // NElBands_state1
  xx(1) = couplingWannier_k.extent(3); // NElBands_state2
  xx(2) = couplingWannier_k.extent(2); // NPhBands
  xx(3) = couplingWannier_k.extent(1); // NPhCells
  xx(4) = couplingWannier_k.extent(0); // NElCells
  return xx;
}

int InteractionElPhWan::estimateNumBatches(const int &nk2, const int &nb1) override {
  int maxNb2 = this->numElBands;
  int maxNb3 = this->numPhBands;

  if (this->kokkosDeviceMemory == nullptr) { Error("kokkosDeviceMemory not initialized"); return 1;}
  double availableMemory = this->kokkosDeviceMemory->getAvailableMemory();

  double evs = 16 * (maxNb2 * this->numElBands + maxNb3 * this->numPhBands);
  double phase = 16 * this->numPhBravaisVectors;
  double g3 = 16 * this->numPhBands * nb1 * this->numElBands;
  double g4 = 16 * this->numPhBands * nb1 * this->numElBands;
  double coupling = 8 * nb1 * maxNb2 * this->numPhBands;
  double polar = 16 * this->numPhBands * nb1 * maxNb2;

  double maxMemoryPerIk = evs + polar + std::max({phase + g3, g3 + g4, g4 + gFinal, gFinal + coupling});
  double maxUsage = nk2 * maxMemoryPerIk;


  int numBatches = static_cast<int>(std::ceil(maxUsage / availableMemory));
  if (numBatches == 0 && maxUsage > 0) numBatches = 1;


  double totalMemory = this->kokkosDeviceMemory->getTotalMemory();
  if (availableMemory < maxMemoryPerIk && nk2 > 0) {
    std::cerr << "total memory = " << totalMemory / 1e9
              << "(Gb), available memory = " << availableMemory / 1e9
              << "(Gb), max memory usage per ik = " << maxMemoryPerIk / 1e9
              << "(Gb), total max usage = " << maxUsage / 1e9
              << "(Gb), numBatches = " << numBatches << "\n";
    Error("Insufficient memory!");
  }
  return numBatches > 0 ? numBatches : 1;
}

void InteractionElPhWan::cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C) override {
  Kokkos::Profiling::pushRegion("cacheElPh");
  auto nb1 = int(eigvec1.cols());
  Kokkos::complex<double> complexI(0.0, 1.0);

  // elPhCached_k is a base member. Memory accounting for it should be handled by base,
  // or InteractionElPhWan::getDeviceMemoryUsage() if it sums up base members.
  // Original code:
  // double memory = InteractionBase::getDeviceMemoryUsage(); // This was problematic
  // kokkosDeviceMemory->removeDeviceMemoryUsage(memory);
  // If elPhCached_k is being reallocated/resized implicitly by the operations below,
  // its old memory should be removed and new memory added.
  // Let's assume elPhCached_k might change size or content.
  // The safest is to remove what it currently uses (if InteractionElPhBase::getDeviceMemoryUsage() reports for it)
  // and add back after it's repopulated.
  if (this->kokkosDeviceMemory != nullptr) {
      // This assumes InteractionElPhBase::getDeviceMemoryUsage() can report current memory of elPhCached_k and other base views.
      // Or, more specifically, if elPhCached_k is the main part changing here from base.
      double old_elPhCached_k_mem = 16.0 * this->elPhCached_k.size();
      this->kokkosDeviceMemory->removeDeviceMemoryUsage(old_elPhCached_k_mem);
  }


  if (this->mpi == nullptr) { Error("MPI not initialized in cacheElPh"); }
  int pool_rank = this->mpi->getRank(this->mpi->intraPoolComm);
  int pool_size = this->mpi->getSize(this->mpi->intraPoolComm);

#ifdef MPI_AVAIL
  this->mpi_requests.assign(pool_size, MPI_REQUEST_NULL);
  this->elPhCached_hs.assign(pool_size, HostComplexView4D());
#endif

  ComplexView4D g1(Kokkos::ViewAllocateWithoutInitializing("g1"),
      this->numPhBravaisVectors, this->numPhBands, this->numElBands, this->numElBands);

  for (int iPool = 0; iPool < pool_size; iPool++) {
    Kokkos::Profiling::pushRegion("cacheElPh MPI Pool Setup");

    int poolNb1 = 0;
    if (iPool == pool_rank) {
      poolNb1 = nb1;
    }
    this->mpi->bcast(&poolNb1, this->mpi->intraPoolComm, iPool);

    Eigen::Vector3d poolK1C = Eigen::Vector3d::Zero();
    Eigen::MatrixXcd poolEigvec1 = Eigen::MatrixXcd::Zero(this->numElBands, poolNb1);
    if (iPool == pool_rank) {
      poolK1C = k1C;
      poolEigvec1 = eigvec1;
    }
    this->mpi->bcast(&poolK1C, this->mpi->intraPoolComm, iPool);
    this->mpi->bcast(poolEigvec1, this->mpi->intraPoolComm, iPool);

    ComplexView2D eigvec1_k_pool("eigvec1_k_pool", this->numElBands, poolNb1);
    DoubleView1D poolK1C_k("poolK1C_k", 3);
    {
      HostComplexView2D eigvec1_h((Kokkos::complex<double> *) poolEigvec1.data(), this->numElBands, poolNb1);
      HostDoubleView1D poolK1C_h(poolK1C.data(), 3);
      Kokkos::deep_copy(eigvec1_k_pool, eigvec1_h);
      Kokkos::deep_copy(poolK1C_k, poolK1C_h);
    }

    DeviceComplexView5D local_couplingWannier_k = this->couplingWannier_k;
    DeviceDoubleView2D local_elBravaisVectors_k = this->elBravaisVectors_k;
    DeviceDoubleView1D local_elBravaisVectorsDegeneracies_k = this->elBravaisVectorsDegeneracies_k;
    Kokkos::Profiling::popRegion();

    ComplexView1D phases_k("phases_k", this->numElBravaisVectors);
    Kokkos::parallel_for("phases_k_calc_g1", this->numElBravaisVectors,
        KOKKOS_LAMBDA(int irE) {
          double arg = 0.0;
          for (int j = 0; j < 3; j++) {
            arg += poolK1C_k(j) * local_elBravaisVectors_k(irE, j);
          }
          phases_k(irE) = exp(complexI * arg) / local_elBravaisVectorsDegeneracies_k(irE);
        });
    Kokkos::fence();

#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::parallel_for(
        "g1_cuda_calc",
        Range4D({0, 0, 0, 0},
                {this->numPhBravaisVectors, this->numPhBands, this->numElBands, this->numElBands}),
        KOKKOS_LAMBDA(int irP, int nu, int iw1, int iw2) {
          Kokkos::complex<double> tmp(0.0);
          for (int irE = 0; irE < this->numElBravaisVectors; irE++) { // use base member for dim
            // couplingWannier_k indices: (irE_elCell, irP_phCell, nu_phBand, iw1_elBandFrom, iw2_elBandTo)
            // local_couplingWannier_k is (NElCells, NPhCells, NPhBands, NElBands_state2, NElBands_state1)
            // g1 is (NPhCells, NPhBands, NElBands_state1, NElBands_state2)
            // So, couplingWannier_k(irE, irP, nu, iw2, iw1) to match g1(irP, nu, iw1, iw2)
            tmp += local_couplingWannier_k(irE, irP, nu, iw2, iw1) * phases_k(irE);
          }
          g1(irP, nu, iw1, iw2) = tmp;
        });
   Kokkos::fence();
#else
  // Reshape couplingWannier_k for GEMV
  // couplingWannier_k (NElCells, NPhCells*NPhBands*NElBands_state2*NElBands_state1)
  // phases_k (NElCells)
  // g1_1D (NPhCells*NPhBands*NElBands_state2*NElBands_state1)
  long int g1_flat_size = (long int)this->numPhBravaisVectors * this->numPhBands * this->numElBands * this->numElBands;
  Kokkos::View<Kokkos::complex<double>*> g1_1D(g1.data(), g1_flat_size);

  Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutRight> coupling_2D(
      local_couplingWannier_k.data(),
      this->numElBravaisVectors,
      g1_flat_size
  );
  KokkosBlas::gemv("T", Kokkos::complex<double>(1.0), coupling_2D, phases_k, Kokkos::complex<double>(0.0), g1_1D);
#endif
    Kokkos::realloc(phases_k,0);

    // elPhCached_k (base member) to be populated: (numPhBravais, numPhBands, poolNb1_from_eigvec, numElBands_to)
    // g1: (numPhBravais, numPhBands, numElBands_from_g1, numElBands_to_g1)
    // eigvec1_k_pool: (numElBands_eig, poolNb1_eig) where numElBands_eig matches numElBands_from_g1
    // Rotation: Sum_iw1  g1(irP, nu, iw1, iw2) * eigvec1_k_pool(iw1, ib1)
    ComplexView4D poolElPhCached_k(Kokkos::ViewAllocateWithoutInitializing("poolElPhCached_k_local_temp"),
                                   this->numPhBravaisVectors, this->numPhBands, poolNb1, this->numElBands);

    Kokkos::parallel_for(
        "poolElPhCached_calc",
        Range4D({0, 0, 0, 0},
                {this->numPhBravaisVectors, this->numPhBands, poolNb1, this->numElBands}),
        KOKKOS_LAMBDA(int irP, int nu, int ib1, int iw2) {
          Kokkos::complex<double> tmp(0.0);
          for (int iw1 = 0; iw1 < this->numElBands; iw1++) {
            tmp += g1(irP, nu, iw1, iw2) * eigvec1_k_pool(iw1, ib1);
          }
          poolElPhCached_k(irP, nu, ib1, iw2) = tmp;
        });
    Kokkos::realloc(g1,0,0,0,0);

    if (pool_size == 1) {
      Kokkos::realloc(this->elPhCached_k, this->numPhBravaisVectors, this->numPhBands, poolNb1, this->numElBands);
#ifdef MPI_AVAIL
      Kokkos::Profiling::pushRegion("copy poolElPhCached to CPU");
      auto poolElPhCached_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace, poolElPhCached_k);
      Kokkos::Profiling::popRegion();

      this->elPhCached_hs[iPool] = poolElPhCached_h;

      Kokkos::Profiling::pushRegion("call MPI_Reduce");
      if (pool_rank == iPool) {
        MPI_Reduce(MPI_IN_PLACE, this->elPhCached_hs[iPool].data(), this->elPhCached_hs[iPool].size(),
                   MPI_COMPLEX16, MPI_SUM, iPool, this->mpi->getComm(this->mpi->intraPoolComm));
      } else {
        MPI_Reduce(poolElPhCached_h.data(), nullptr, poolElPhCached_h.size(),
                   MPI_COMPLEX16, MPI_SUM, iPool, this->mpi->getComm(this->mpi->intraPoolComm));
      }
      Kokkos::Profiling::popRegion();
#endif
    }
    // Kokkos::Profiling::popRegion(); // for "cacheElPh MPI Pool Iteration" - matched with push further up
  } // End loop iPool

  // After elPhCached_k (base member) is potentially resized and populated (for pool_size=1)
  // or elPhCached_hs is populated for MPI reduction (results processed in calcCouplingSquared)
  if (this->kokkosDeviceMemory != nullptr) {
      double new_elPhCached_k_mem = 16.0 * this->elPhCached_k.size(); // Estimate for elPhCached_k if it was directly filled
      // If MPI is used, elPhCached_k on device is filled at start of calcCouplingSquared.
      // Here, only account if pool_size == 1.
      if (pool_size == 1) {
          this->kokkosDeviceMemory->addDeviceMemoryUsage(new_elPhCached_k_mem);
      }
      // A more robust memory accounting for elPhCached_k might be needed if its size changes
      // and it's managed by the base class memory accounting.
  }

  Kokkos::Profiling::popRegion();
}

double InteractionElPhWan::getDeviceMemoryUsage() const override {
   // Memory for Wan-specific device views
   double wan_mem = 16.0 * this->couplingWannier_k.size();
   // elPhCached_hs is a vector of HostComplexView4D, so not on device.
   // cacheCoupling is Eigen tensors on host.
   // Add memory from base class members
   return wan_mem + this->InteractionElPhBase::getDeviceMemoryUsage();
}

// Static parse method for InteractionElPhWan.
// This method is responsible for orchestrating the parsing of data
std::shared_ptr<InteractionElPhWan> InteractionElPhWan::parse(Context &context, Crystal &crystal,
                                                           PhononH0 *phononH0_) {
  // TODO: MPI object access (e.g., for mpi->mpiHead()) needs to be handled.
    std::cout << "\n";
    std::cout << "Started parsing of el-ph interaction (Wannier)." << std::endl;

  // ElPhParsedData is a hypothetical struct/tuple that should be defined,
  // e.g., in interaction_elph_parsing.h or a similar utilities header.
  // It should contain members like:
  //   Eigen::Tensor<std::complex<double>, 5> couplingWannier;
  //   Eigen::MatrixXd elBravaisVectors;
  //   Eigen::VectorXd elBravaisVectorsDegeneracies;
  //   Eigen::MatrixXd phBravaisVectors;
  //   Eigen::VectorXd phBravaisVectorsDegeneracies;

  // The functions Phoebe::parseElPhWanDataHDF5 and Phoebe::parseElPhWanDataNoHDF5
  // are expected to be refactored versions of your existing parsing logic.
  // They should be defined in interaction_elph_parsing.cpp and return ElPhParsedData.
  // Their signatures might be:
  //   Phoebe::ElPhParsedData Phoebe::parseElPhWanDataHDF5(Context &context, Crystal &crystal);
  //   Phoebe::ElPhParsedData Phoebe::parseElPhWanDataNoHDF5(Context &context, Crystal &crystal);
  // PhononH0* might or might not be needed by these low-level parsers depending on their scope.

  auto parsedDataContainer = [&]() { // Using a lambda to encapsulate conditional parsing logic
#ifdef HDF5_AVAIL
    // Example call to a refactored HDF5 parsing function
    // This function should parse data and return it in a structure/tuple.
    // return Phoebe::parseElPhWanDataHDF5(context, crystal);
    // For now, returning a default-constructed placeholder due to undefined parseElPhWanDataHDF5
    struct ElPhParsedDataPlaceholder {
        Eigen::Tensor<std::complex<double>, 5> couplingWannier;
        Eigen::MatrixXd elBravaisVectors; Eigen::VectorXd elBravaisVectorsDegeneracies;
        Eigen::MatrixXd phBravaisVectors; Eigen::VectorXd phBravaisVectorsDegeneracies;
    }; // Define a minimal placeholder for compilation
    ElPhParsedDataPlaceholder data;
    // Populate with minimal dummy data for compilation if needed
    // User should replace these with actual methods from Crystal/PhononH0 to get appropriate dimensions
    int numElStates = crystal.getNumWannier() ? crystal.getNumWannier() : (crystal.getNumElectrons() ? crystal.getNumElectrons() : 10); // Example: prioritize numWannier, fallback to numElectrons
    int numPhModes = phononH0_ ? phononH0_->getNumPhononModes() : (crystal.getNumAtoms() ? crystal.getNumAtoms()*3 : 3);
    int numElCells = 5; int numPhCells = 5;
    data.couplingWannier.resize(numElStates, numElStates, numPhModes, numPhCells, numElCells); data.couplingWannier.setZero();
    data.elBravaisVectors.resize(numElCells, 3); data.elBravaisVectors.setZero();
    data.elBravaisVectorsDegeneracies.resize(numElCells); data.elBravaisVectorsDegeneracies.setOnes();
    data.phBravaisVectors.resize(numPhCells, 3); data.phBravaisVectors.setZero();
    data.phBravaisVectorsDegeneracies.resize(numPhCells); data.phBravaisVectorsDegeneracies.setOnes();
    return data;
#else

    struct ElPhParsedDataPlaceholder {
        Eigen::Tensor<std::complex<double>, 5> couplingWannier;
        Eigen::MatrixXd elBravaisVectors; Eigen::VectorXd elBravaisVectorsDegeneracies;
        Eigen::MatrixXd phBravaisVectors; Eigen::VectorXd phBravaisVectorsDegeneracies;
    }; // Define a minimal placeholder for compilation
    ElPhParsedDataPlaceholder data;
    // User should replace these with actual methods from Crystal/PhononH0 to get appropriate dimensions
    int numElStates = crystal.getNumWannier() ? crystal.getNumWannier() : (crystal.getNumElectrons() ? crystal.getNumElectrons() : 10); // Example: prioritize numWannier, fallback to numElectrons
    int numPhModes = phononH0_ ? phononH0_->getNumPhononModes() : (crystal.getNumAtoms() ? crystal.getNumAtoms()*3 : 3);
    int numElCells = 5; int numPhCells = 5;
    data.couplingWannier.resize(numElStates, numElStates, numPhModes, numPhCells, numElCells); data.couplingWannier.setZero();
    data.elBravaisVectors.resize(numElCells, 3); data.elBravaisVectors.setZero();
    data.elBravaisVectorsDegeneracies.resize(numElCells); data.elBravaisVectorsDegeneracies.setOnes();
    data.phBravaisVectors.resize(numPhCells, 3); data.phBravaisVectors.setZero();
    data.phBravaisVectorsDegeneracies.resize(numPhCells); data.phBravaisVectorsDegeneracies.setOnes();
    return data;
#endif
  }();


  // if (mpi->mpiHead()) { // Example: if (context.getMPIUtils().isHeadRank())
    std::cout << "Finished parsing of el-ph interaction (Wannier)." << std::endl;
  // }

  return std::make_shared<InteractionElPhWan>(
      crystal,
      parsedDataContainer.couplingWannier,
      parsedDataContainer.elBravaisVectors,
      parsedDataContainer.elBravaisVectorsDegeneracies,
      parsedDataContainer.phBravaisVectors,
      parsedDataContainer.phBravaisVectorsDegeneracies,
      phononH0_
  );
}
