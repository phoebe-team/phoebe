#include "interaction_elph_svd.h"
#include "common_kokkos.h"
#include "context.h"
#include "crystal.h"
#include "exceptions.h"
#include "harmonic/phonon_h0.h"
#include "mpi/mpiController.h"
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <iostream>
#include <regex>
#include <tuple>
#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>


// Constructor
InteractionElPhSVD::InteractionElPhSVD(Crystal &crystal, Context &context,
                                       PhononH0 &phononH0)
    : InteractionElPhBase(crystal, &phononH0) {

  std::string fileName = context.getElphFileName();
  if (mpi->mpiHead()) {
    HighFive::File file(fileName, HighFive::File::ReadOnly);
    file.getDataSet("/numElBands").read(numElBands);
    file.getDataSet("/numPhModes").read(numPhBands);
    file.getDataSet("/elBravaisVectors").read(elBravaisVectors);
    file.getDataSet("/phBravaisVectors").read(phBravaisVectors);
    file.getDataSet("/elDegeneracies").read(elBravaisVectorsDegeneracies);
    file.getDataSet("/phDegeneracies").read(phBravaisVectorsDegeneracies);
  }
  // Broadcast all header data to other processes
  mpi->bcast(&numElBands);
  mpi->bcast(&numPhBands);
  mpi->bcast(elBravaisVectors.data(), elBravaisVectors.size());
  mpi->bcast(phBravaisVectors.data(), phBravaisVectors.size());
  mpi->bcast(elBravaisVectorsDegeneracies.data(), elBravaisVectorsDegeneracies.size());
  mpi->bcast(phBravaisVectorsDegeneracies.data(), phBravaisVectorsDegeneracies.size());

  numElBravaisVectors = elBravaisVectorsDegeneracies.size();
  numPhBravaisVectors = phBravaisVectorsDegeneracies.size();

  // Initialize SVD-specific variables
  numWsR1Vectors = numElBravaisVectors;
  numWsR2Vectors = numPhBravaisVectors;
  numWannierOrbitals = numElBands;
  // numGamma = 1;  // Will be updated during parsing

  // Parsing function
  parseSVDKokkos(context);

  // if (mpi->mpiHead()) {
  //    printSVDInfo();
  //  }
}

// ---------------------------------------------------------------------------
// Core HDF5 Parsing Logic
// ---------------------------------------------------------------------------

// Helper to parse the (i,j,eta) indices from a group name
inline std::tuple<int, int, int>
extractIndicesFromGroupName(const std::string &groupName) {
  static const std::regex r("slice_(\\d+)_(\\d+)_(\\d+)");
  std::smatch match;
  if (std::regex_match(groupName, match, r)) {
    return {std::stoi(match[1]), std::stoi(match[2]), std::stoi(match[3])};
  }
  throw std::invalid_argument("SVD group name has incorrect format: " +
                              groupName);
}

static inline std::tuple<int,int,int> parseSliceName(const std::string &name) {
  static const std::regex r("slice_(\\d+)_(\\d+)_(\\d+)");
  std::smatch m;
  if (!std::regex_match(name, m, r))
    throw std::invalid_argument("Bad SVD group name: " + name);
  return {std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3])};
}

// REFACTOR : new name :) 
template <typename EC, typename ER>
static void readMaybeComplex(const HighFive::DataSet &ds, EC &out) {
  try { ds.read(out); }                        // complex
  catch (...) { ER tmp; ds.read(tmp); out = tmp.template cast<std::complex<double>>(); }
}

std::vector<InteractionElPhSVD::SVDGroupData>
InteractionElPhSVD::processAllSVDGroups(const HighFive::Group &svdGroup,
                                        int &num_i, int &num_j, int &num_eta,
                                        int &RE, int &RP, int &maxGamma) {
  using CD = Kokkos::complex<double>;
  const auto names = svdGroup.listObjectNames();
  if (names.empty()) throw std::runtime_error("zSVD group is empty.");

  // pass 1: discover geometry & maxGamma
  size_t mi=0, mj=0, me=0; bool shapeInit=false; maxGamma=0;
  for (const auto &name : names) {
    auto [i,j,eta] = parseSliceName(name);
    mi=std::max(mi,size_t(i)); mj=std::max(mj,size_t(j)); me=std::max(me,size_t(eta));

    auto g = svdGroup.getGroup(name);
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Uc;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Vc;
    readMaybeComplex<
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Matrix<double,              Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >(g.getDataSet("U"), Uc);
    readMaybeComplex<
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<double,              Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    >(g.getDataSet("V"), Vc);

    int gamma=0;
    try { Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> Sc; g.getDataSet("S").read(Sc); gamma=int(Sc.size()); }
    catch (...) {
      try { Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Sm; g.getDataSet("S").read(Sm); gamma=int(Sm.size()); }
      catch (...) { Eigen::Matrix<double, Eigen::Dynamic, 1> Sr; g.getDataSet("S").read(Sr); gamma=int(Sr.size()); }
    }

    if (!shapeInit) { RE=int(Uc.rows()); RP=int(Vc.rows()); shapeInit=true; }
    else if (RE!=Uc.rows() || RP!=Vc.rows()) throw std::runtime_error("Inconsistent RE/RP across slices.");
    maxGamma = std::max(maxGamma, gamma);
  }
  num_i = int(mi)+1; num_j = int(mj)+1; num_eta = int(me)+1;

  // pass 2: read & pack per-slice into flat vectors
  std::vector<SVDGroupData> out; out.reserve(names.size());
  for (const auto &name : names) {
    auto [i,j,eta] = parseSliceName(name);
    auto g = svdGroup.getGroup(name);

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Uc;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> Sc;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Vc;

    readMaybeComplex<
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Matrix<double,              Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >(g.getDataSet("U"), Uc);

    try { Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> Sc_try; g.getDataSet("S").read(Sc_try); Sc = Sc_try; }
    catch (...) {
      try {
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Sm;
        g.getDataSet("S").read(Sm);
        Sc.resize(Sm.size());
        for (int k=0;k<Sm.size();++k) Sc(k)=Sm.data()[k];
      } catch (...) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> Sr; g.getDataSet("S").read(Sr);
        Sc = Sr.cast<std::complex<double>>();
      }
    }

    readMaybeComplex<
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<double,              Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    >(g.getDataSet("V"), Vc);

    const int gamma = int(Sc.size());

    SVDGroupData d; d.i=i; d.j=j; d.eta=eta; d.RE=RE; d.RP=RP; d.gamma=gamma;
    d.U.resize(size_t(RE)*size_t(gamma));
    d.V.resize(size_t(RP)*size_t(gamma));
    d.S.resize(size_t(gamma));

    for (int re=0; re<RE; ++re)
      for (int gg=0; gg<gamma; ++gg)
        d.U[size_t(re)*size_t(gamma)+size_t(gg)] = Uc(re,gg);

    for (int rp=0; rp<RP; ++rp)
      for (int gg=0; gg<gamma; ++gg)
        d.V[size_t(rp)*size_t(gamma)+size_t(gg)] = Vc(rp,gg); // store V(rp,g)

    for (int gg=0; gg<gamma; ++gg) d.S[size_t(gg)] = Sc(gg);

    out.push_back(std::move(d));
  }
  return out;
}

// Core routine to read data
void InteractionElPhSVD::parseSVDKokkos(Context &context) {
  const std::string fileName = context.getElphFileName();
  if (mpi->mpiHead()) {
    std::cout << "Loading SVD slices from " << fileName << " ...\n";
  }

  // Head gathers from HDF5 and packs to flat buffers, then broadcasts.
  // ---- buffers on host (std::complex for MPI) ----
  int RE=0, RP=0;
  std::vector<std::complex<double>> SY_buf;   // size = L*RE*maxGamma (layout (l,re,g))
  std::vector<std::complex<double>> Vt_buf;   // size = L*maxGamma*RP (layout (l,g,rp))
  std::vector<int> gammaLens;                 // size = L

  try {
    if (mpi->mpiHead()) {
      HighFive::File file(fileName, HighFive::File::ReadOnly);
      HighFive::Group svdGroup = file.getGroup("zSVD");

      int ni, nj, ne, mG;
      auto slices = processAllSVDGroups(svdGroup, ni, nj, ne, RE, RP, mG);
      numI=ni; numJ=nj; numEta=ne; maxGamma=mG;

      const int L = numI * numJ * numEta;
      SY_buf.assign(size_t(L)*size_t(RE)*size_t(maxGamma), std::complex<double>(0,0));
      Vt_buf.assign(size_t(L)*size_t(maxGamma)*size_t(RP), std::complex<double>(0,0));
      gammaLens.assign(L, 0);

      auto mapL = [J=numJ, E=numEta](int i,int j,int eta){ return i*(J*E) + j*E + eta; };

      for (const auto &d : slices) {
        const int l = mapL(d.i, d.j, d.eta);
        gammaLens[l] = d.gamma;

        // SY(l,re,g) = U(re,g)*S(g)
        for (int re=0; re<RE; ++re)
          for (int g=0; g<d.gamma; ++g) {
            const auto val = d.U[size_t(re)*size_t(d.gamma)+size_t(g)] * d.S[size_t(g)];
            const size_t off = (size_t(l)*RE + size_t(re))*size_t(maxGamma) + size_t(g);
            SY_buf[off] = val;
          }

        // Vt(l,g,rp) = V(rp,g)
        for (int g=0; g<d.gamma; ++g)
          for (int rp=0; rp<RP; ++rp) {
            const auto val = d.V[size_t(rp)*size_t(d.gamma)+size_t(g)];
            const size_t off = (size_t(l)*size_t(maxGamma) + size_t(g))*size_t(RP) + size_t(rp);
            Vt_buf[off] = val;
          }
      }

      // broadcast geometry
      mpi->bcast(&numI); mpi->bcast(&numJ); mpi->bcast(&numEta);
      mpi->bcast(&RE);   mpi->bcast(&RP);   mpi->bcast(&maxGamma);
      mpi->bcast(gammaLens.data(), int(gammaLens.size()));

      // broadcast payload as doubles (2x)
      mpi->bcast(reinterpret_cast<double*>(SY_buf.data()), int(SY_buf.size())*2);
      mpi->bcast(reinterpret_cast<double*>(Vt_buf.data()), int(Vt_buf.size())*2);

    } else {
      mpi->bcast(&numI); mpi->bcast(&numJ); mpi->bcast(&numEta);
      mpi->bcast(&RE);   mpi->bcast(&RP);   mpi->bcast(&maxGamma);

      const int L = numI * numJ * numEta;
      gammaLens.resize(L);
      mpi->bcast(gammaLens.data(), L);

      SY_buf.resize(size_t(L)*size_t(RE)*size_t(maxGamma));
      Vt_buf.resize(size_t(L)*size_t(maxGamma)*size_t(RP));
      mpi->bcast(reinterpret_cast<double*>(SY_buf.data()), int(SY_buf.size())*2);
      mpi->bcast(reinterpret_cast<double*>(Vt_buf.data()), int(Vt_buf.size())*2);
    }

    // ---- allocate device views (4D fused) ----
    Kokkos::realloc(SVD_SY_device, numI, numJ, numEta, RE*maxGamma);
    Kokkos::realloc(SVD_Vt_device, numI, numJ, numEta, maxGamma*RP);
    Kokkos::realloc(gammaLen_ijk,  numI, numJ, numEta);

    // ---- fill host mirrors by converting std::complex -> Kokkos::complex ----
    auto SY_h  = Kokkos::create_mirror_view(SVD_SY_device);
    auto Vt_h  = Kokkos::create_mirror_view(SVD_Vt_device);
    auto gLenH = Kokkos::create_mirror_view(gammaLen_ijk);

    const int L = numI * numJ * numEta;
    int idx = 0;
    for (int i=0;i<numI;++i)
      for (int j=0;j<numJ;++j)
        for (int e=0;e<numEta;++e)
          gLenH(i,j,e) = gammaLens[idx++];

    // SY_h(i,j,e, re*maxGamma + g)
    for (int i=0;i<numI;++i)
      for (int j=0;j<numJ;++j)
        for (int e=0;e<numEta;++e) {
          const int l = (i*numJ + j)*numEta + e;
          for (int re=0; re<RE; ++re)
            for (int g=0; g<maxGamma; ++g) {
              const size_t off = (size_t(l)*RE + size_t(re))*size_t(maxGamma) + size_t(g);
              const auto z = SY_buf[off];
              SY_h(i,j,e, re*maxGamma + g) = CD(z.real(), z.imag());
            }
        }

    // Vt_h(i,j,e, g*RP + rp)
    for (int i=0;i<numI;++i)
      for (int j=0;j<numJ;++j)
        for (int e=0;e<numEta;++e) {
          const int l = (i*numJ + j)*numEta + e;
          for (int g=0; g<maxGamma; ++g)
            for (int rp=0; rp<RP; ++rp) {
              const size_t off = (size_t(l)*size_t(maxGamma) + size_t(g))*size_t(RP) + size_t(rp);
              const auto z = Vt_buf[off];
              Vt_h(i,j,e, g*RP + rp) = CD(z.real(), z.imag());
            }
        }

    Kokkos::deep_copy(SVD_SY_device, SY_h);
    Kokkos::deep_copy(SVD_Vt_device, Vt_h);
    Kokkos::deep_copy(gammaLen_ijk,  gLenH);

    // finalize
    numWsR1Vectors = RE;
    numWsR2Vectors = RP;
    numWannierOrbitals = numElBands;

    if (mpi->mpiHead()) {
      std::cout << "SVD factors loaded: (I,J,η)=("
                << numI << "," << numJ << "," << numEta << "), "
                << "RE=" << RE << ", RP=" << RP
                << ", maxGamma=" << maxGamma << "\n";
    }

  } catch (const std::exception &e) {
    Error(std::string("parseSVDKokkos failed: ") + e.what());
  }

}

// void InteractionElPhSVD::printSVDInfo() const {
//   if (ElPh_Matrix.data() == nullptr) {
//     std::cout << "ElPh_Matrix is not initialized." << std::endl;
//     return;
//   }
//   std::cout << "--- Electron-Phonon Matrix Info ---" << std::endl;
//   std::cout << "ElPh_Matrix dimensions (i,j,eta,R_e,R_p): " << ElPh_Matrix.extent(0) << " x "
//             << ElPh_Matrix.extent(1) << " x " << ElPh_Matrix.extent(2) << " x "
//             << ElPh_Matrix.extent(3) << " x " << ElPh_Matrix.extent(4) << std::endl;
//   std::cout << "Total elements: " << ElPh_Matrix.span() << std::endl;
//   std::cout << "Total memory usage (approx): " << getDeviceMemoryUsage() / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
// }

// void InteractionElPhSVD::printSVDSample(size_t i, size_t j, size_t eta) const {
//   if (ElPh_Matrix.data() == nullptr) {
//     std::cout << "ElPh_Matrix is not initialized." << std::endl;
//     return;
//   }
//   auto ElPh_Matrix_h = Kokkos::create_mirror_view(ElPh_Matrix);
//   Kokkos::deep_copy(ElPh_Matrix_h, ElPh_Matrix);
//   std::cout << "--- ElPh Matrix Sample at (i,j,eta) = (" << i << "," << j << ","
//             << eta << ") ---" << std::endl;
//   std::cout << "ElPh_Matrix(i,j,eta,0,0) = " << ElPh_Matrix_h(i, j, eta, 0, 0) << std::endl;
//   if (ElPh_Matrix.extent(3) > 1 && ElPh_Matrix.extent(4) > 1) {
//     std::cout << "ElPh_Matrix(i,j,eta,1,1) = " << ElPh_Matrix_h(i, j, eta, 1, 1) << std::endl;
//   }
// }

void InteractionElPhSVD::cacheElPh(const Eigen::MatrixXcd &eigvec1,
                                   const Eigen::Vector3d  &k1C) {

                                       Kokkos::Profiling::pushRegion("cacheElPh_SVD");

  // REFACTOR 
  const int numGamma            = maxGamma;
  const int numWannierOrbitals  = numI;           // i- (and j-) dimension
  const int numWsR1Vectors_local= numWsR1Vectors; // R_e rows

  Kokkos::complex<double> complexI(0.0, 1.0);

  // Get dimensions (your names)
  auto nb1 = int(eigvec1.cols()); // number of bands at this kpoint
  auto elPhCached_SVD_SY = this->elPhCached_SVD_SY; // final container of output from this function, stored in the class 
  auto SVD_SY_device     = this->SVD_SY_device; // input data of the first part of the SVD 

  if (mpi->mpiHead()) {
    std::cout << "SVD cacheElPh: nb1=" << nb1
              << ", numGamma=" << numGamma
              << ", numPhBands=" << numPhBands
              << ", numWannierOrbitals=" << numWannierOrbitals << std::endl;
  }

  // Copy the eigenvector and wavevector to the accelerator
  ComplexView2D eigvec1_device("ev1", nb1, numWannierOrbitals);
  DoubleView1D  k1C_device("k", 3);
  {
    HostComplexView2D eigvec1_host((Kokkos::complex<double>*)eigvec1.data(), nb1, numWannierOrbitals);
    HostDoubleView1D  k1C_host((double*)k1C.data(), 3);
    Kokkos::deep_copy(eigvec1_device, eigvec1_host);
    Kokkos::deep_copy(k1C_device,     k1C_host);
  }

  // REFACTOR maybe we want constructor to do this instead of the function call 
  if (wsR1Vectors_device.data() == nullptr) {
    Kokkos::resize(wsR1Vectors_device,numWsR1Vectors_local, 3);
    Kokkos::resize(wsR1VectorsDegeneracies_device, numWsR1Vectors_local);
    HostDoubleView2D r1_h((double*)elBravaisVectors.data(), numWsR1Vectors_local, 3);
    HostDoubleView1D d1_h((double*)elBravaisVectorsDegeneracies.data(), numWsR1Vectors_local);
    Kokkos::deep_copy(wsR1Vectors_device, r1_h);
    Kokkos::deep_copy(wsR1VectorsDegeneracies_device, d1_h);
  }

  // ---------------- phases over R_e ----------------
  Kokkos::Profiling::pushRegion("precompute_phases_k1");
  ComplexView1D phases_device("phases", numWsR1Vectors_local);
  {
    auto r1 = wsR1Vectors_device;
    auto d1 = wsR1VectorsDegeneracies_device;
    Kokkos::parallel_for("phases_k1c", numWsR1Vectors_local, KOKKOS_LAMBDA(int irE) {
      double arg = 0.0;
      for (int j = 0; j < 3; ++j) arg += k1C_device(j) * r1(irE, j);
      phases_device(irE) = exp(complexI * arg) / d1(irE);
    });
    Kokkos::fence();
  }
  Kokkos::Profiling::popRegion();

  // ---------------- pack_SY(R_e, gamma*eta*i*j) ----------------
  Kokkos::Profiling::pushRegion("fourier_transform");
  
  const size_t flattened_size =
      size_t(numGamma) * size_t(numPhBands) *
      size_t(numWannierOrbitals) * size_t(numWannierOrbitals);

  // Create intermediate/"cache" container of size (gamma, eta, i, j)
  ComplexView4D g_FT_output(
    Kokkos::ViewAllocateWithoutInitializing("g1"),
    numGamma, numPhBands, numWannierOrbitals, numWannierOrbitals);
      
  {
    
  // FIXME is layoutRight ok or wrong??  
  // set up the empty container for output 
  // Create intermediate/"cache" container of size (gamma, eta, i, j)
  Kokkos::View<Kokkos::complex<double>*, Kokkos::LayoutRight> g_FT_output_1D(
    g_FT_output.data(), flattened_size);
  
  // layout right was used earlier because EIGEN has opposite storage system
  // FIXME is layoutRight ok or wrong??  -- I suspect it's ok 
  Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight> coupling_2D(
    SVD_SY_device.data(), numWsR1Vectors_local, flattened_size);

  KokkosBlas::gemv("T", Kokkos::complex<double>(1.0, 0.0),
                   coupling_2D, phases_device,
                   Kokkos::complex<double>(0.0, 0.0), g_FT_output_1D);

  if (mpi->mpiHead()) {
    std::cout << "Applied Fourier transform via gemv with flattened_size="
              << flattened_size << std::endl;
    std::cout << "g_FT_output_1D dimensions: "
              << g_FT_output_1D.extent(0) << std::endl;
    std::cout << "coupling_2D dimensions: ("
              << coupling_2D.extent(0) << ", " << coupling_2D.extent(1) << ")"
              << std::endl;
   }
  }
  Kokkos::Profiling::popRegion();

  // ------------- Merger rotation into SVD elements ----------------
  Kokkos::Profiling::pushRegion("rotation_to_band_basis");

  Kokkos::realloc(elPhCached_SVD_SY,
                  numGamma, numPhBands, nb1, numWannierOrbitals, numWannierOrbitals);

  //int numWannierOrbitals_copy = numWannierOrbitals;
  
  // x is an m-element vector
  // y is an n-element vector,
  // A is an m-by-n general matrix.
  
  // FIXME replace with ger
  Kokkos::parallel_for(
      "elPhCached_SVD_SY_device",
      Range5D({0, 0, 0, 0, 0},
              {numGamma, numPhBands, nb1, numWannierOrbitals, numWannierOrbitals}),
      KOKKOS_LAMBDA(int igamma, int ieta, int ib1, int iw1, int iw2) {
        Kokkos::complex<double> tmp(0.0);
        elPhCached_SVD_SY(igamma, ieta, ib1, iw1, iw2) = g_FT_output(igamma, ieta, iw1, iw2) * eigvec1_device(ib1, iw1);
      });
  Kokkos::fence();

  this->elPhCached_SVD_SY = elPhCached_SVD_SY;  // keep device ref

  double newMemory = getDeviceMemoryUsage();
  kokkosDeviceMemory->addDeviceMemoryUsage(newMemory);

  Kokkos::Profiling::popRegion(); // rotation_to_band_basis
  Kokkos::Profiling::popRegion(); // cacheElPh_SVD
}

void InteractionElPhSVD::calcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3,
    const std::vector<Eigen::Vector3d> &q3Cs,
    const std::vector<Eigen::VectorXcd> &polarData) {

  if (mpi->mpiHead()) {
    // std::cout << "\n--------------------------------------------------" << std::endl;
    // std::cout << "CHECKPOINT: SVD Interaction Module Triggered" << std::endl;
    // std::cout << "--------------------------------------------------" << std::endl;
  }

  cacheCoupling.resize(eigvecs2.size());
  for (size_t i = 0; i < eigvecs2.size(); ++i) {
    auto nb1 = eigvec1.cols();
    auto nb2 = eigvecs2[i].cols();
    cacheCoupling[i] = Eigen::Tensor<double, 3>(nb1, nb2, numPhBands);
    cacheCoupling[i].setZero();
  }
  if (mpi->mpiHead()) {
      // std::cout << "Populated coupling cache with zero-tensors to allow execution to continue." << std::endl;
      // std::cout << "--------------------------------------------------\n" << std::endl;
  }
}

void InteractionElPhSVD::resetK1() {
  cacheCoupling.clear();
}

const Eigen::Tensor<double, 3> &
InteractionElPhSVD::getCouplingSquared(const int &ik2) const {
  if (ik2 >= int(cacheCoupling.size())) {
    Error("Index ik2 is out of bounds for cached coupling in SVD class.");
  }
  return cacheCoupling[ik2];
}

const Eigen::VectorXi InteractionElPhSVD::getCouplingDimensions() const {
  Eigen::VectorXi elph_dims(5);
  // elph_dims << ElPh_Matrix.extent(0), ElPh_Matrix.extent(1), ElPh_Matrix.extent(2),
  //     ElPh_Matrix.extent(3), ElPh_Matrix.extent(4);
  return elph_dims;
}

const double InteractionElPhSVD::getDeviceMemoryUsage() const {
  double bytes = 0.;
  // if (ElPh_Matrix.data() != nullptr) {
  //   bytes += ElPh_Matrix.span() * sizeof(Kokkos::complex<double>);
  // }
  return bytes;
}

int InteractionElPhSVD::estimateNumBatches(const int &nk2,
                                           const int &nb1) const {
    // Placeholder for testing.
  return 1;
}

// void InteractionElPhSVD::exportElPhMatrixToHDF5(const std::string &filename) const {
//   if (!mpi->mpiHead()) return;

//   if (ElPh_Matrix.data() == nullptr) {
//     std::cout << "ElPh_Matrix is not initialized - cannot export." << std::endl;
//     return;
//   }

//   auto ElPh_Matrix_h = Kokkos::create_mirror_view(ElPh_Matrix);
//   Kokkos::deep_copy(ElPh_Matrix_h, ElPh_Matrix);

//   try {
//     HighFive::File file(filename, HighFive::File::Truncate);

//     // Save dimensions
//     std::vector<size_t> dims = {ElPh_Matrix.extent(0), ElPh_Matrix.extent(1), ElPh_Matrix.extent(2),
//                                 ElPh_Matrix.extent(3), ElPh_Matrix.extent(4)};
//     file.createDataSet("dimensions", dims);

//     // Save real and imaginary parts as separate datasets
//     size_t total_size = ElPh_Matrix.span();
//     std::vector<double> real_data(total_size), imag_data(total_size);

//     size_t idx = 0;
//     for (size_t i = 0; i < ElPh_Matrix.extent(0); ++i) {
//       for (size_t j = 0; j < ElPh_Matrix.extent(1); ++j) {
//         for (size_t eta = 0; eta < ElPh_Matrix.extent(2); ++eta) {
//           for (size_t R_e = 0; R_e < ElPh_Matrix.extent(3); ++R_e) {
//             for (size_t R_p = 0; R_p < ElPh_Matrix.extent(4); ++R_p) {
//               auto val = ElPh_Matrix_h(i, j, eta, R_e, R_p);
//               real_data[idx] = val.real();
//               imag_data[idx] = val.imag();
//               idx++;
//             }
//           }
//         }
//       }
//     }

//     file.createDataSet("real", real_data);
//     file.createDataSet("imag", imag_data);

//     std::cout << "Exported ElPh_Matrix to: " << filename << std::endl;

//   } catch (const std::exception &e) {
//     std::cout << "Error exporting to HDF5: " << e.what() << std::endl;
//   }
// }
