#include "interaction_elph_svd.h"
#include <iomanip>
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
InteractionElPhSVD::InteractionElPhSVD(Context& context, Crystal &crystal,
                                       PhononH0 &phononH0)
    : InteractionElPhBase(crystal, &phononH0) {



  std::string fileName = context.getElphFileName();



  Eigen::MatrixXd wsR1Vectors;
  Eigen::VectorXd wsR1VectorsDegeneracies;
  Eigen::MatrixXd wsR2Vectors;
  Eigen::VectorXd wsR2VectorsDegeneracies;

  int r1=0,c1=0,r2=0,c2=0,d1=0,d2=0;

  if (mpi->mpiHead()) {
    HighFive::File file(fileName, HighFive::File::ReadOnly);
    file.getDataSet("/numElBands").read(numElBands);
    file.getDataSet("/numPhModes").read(numPhBands);
    file.getDataSet("/elBravaisVectors").read(wsR1Vectors);
    file.getDataSet("/phBravaisVectors").read(wsR2Vectors);
    file.getDataSet("/elDegeneracies").read(wsR1VectorsDegeneracies);
    file.getDataSet("/phDegeneracies").read(wsR2VectorsDegeneracies);
    r1 = (int)wsR1Vectors.rows(); c1 = (int)wsR1Vectors.cols();
    r2 = (int)wsR2Vectors.rows(); c2 = (int)wsR2Vectors.cols();
    d1 = (int)wsR1VectorsDegeneracies.size();
    d2 = (int)wsR2VectorsDegeneracies.size();
  }

  // Broadcast sizes first

  mpi->bcast(&numElBands);
  mpi->bcast(&numPhBands);
  mpi->bcast(&r1); mpi->bcast(&c1);
  mpi->bcast(&r2); mpi->bcast(&c2);
  mpi->bcast(&d1); mpi->bcast(&d2);

  if (!mpi->mpiHead()) {
    wsR1Vectors.resize(r1, c1);
    wsR2Vectors.resize(r2, c2);
    wsR1VectorsDegeneracies.resize(d1);
    wsR2VectorsDegeneracies.resize(d2);
  }

  // Broadcast payload

  mpi->bcast(wsR1Vectors.data(), r1*c1);
  mpi->bcast(wsR2Vectors.data(), r2*c2);
  mpi->bcast(wsR1VectorsDegeneracies.data(), d1);
  mpi->bcast(wsR2VectorsDegeneracies.data(), d2);

  // Initialize SVD-specific variables
  numWsR1Vectors = (int)wsR1Vectors.cols();
  numWsR2Vectors = (int)wsR2Vectors.cols();
  numWannierOrbitals = numElBands; // readability



  // Copy the R vectors to the device (unchanged from your version)
  {

    Kokkos::realloc(wsR1VectorsDegeneracies_device, numWsR1Vectors);
    Kokkos::realloc(wsR2VectorsDegeneracies_device, numWsR2Vectors);
    Kokkos::realloc(wsR1Vectors_device, numWsR1Vectors, 3);
    Kokkos::realloc(wsR2Vectors_device, numWsR2Vectors, 3);

    // Set up host views of the data - need to transpose from (3,N) to (N,3)
    HostDoubleView1D wsR1VectorsDegeneracies_host((double*)wsR1VectorsDegeneracies.data(), numWsR1Vectors);
    HostDoubleView1D wsR2VectorsDegeneracies_host((double*)wsR2VectorsDegeneracies.data(), numWsR2Vectors);



    // Create temporary host mirrors and transpose the data
    auto wsR1_host_temp = Kokkos::create_mirror_view(wsR1Vectors_device);
    auto wsR2_host_temp = Kokkos::create_mirror_view(wsR2Vectors_device);

    // Transpose R vectors from (3, N) to (N, 3)
    for (int i = 0; i < numWsR1Vectors; ++i) {
      for (int j = 0; j < 3; ++j) {
        wsR1_host_temp(i, j) = wsR1Vectors(j, i);
      }
    }
    for (int i = 0; i < numWsR2Vectors; ++i) {
      for (int j = 0; j < 3; ++j) {
        wsR2_host_temp(i, j) = wsR2Vectors(j, i);
      }
    }

    // Copy to the device containers held by the class
    Kokkos::deep_copy(wsR1Vectors_device, wsR1_host_temp);
    Kokkos::deep_copy(wsR2Vectors_device, wsR2_host_temp);
    Kokkos::deep_copy(wsR1VectorsDegeneracies_device, wsR1VectorsDegeneracies_host);
    Kokkos::deep_copy(wsR2VectorsDegeneracies_device, wsR2VectorsDegeneracies_host);

    double memoryUsed = getDeviceMemoryUsage();
    kokkosDeviceMemory->addDeviceMemoryUsage(memoryUsed);


  }



  parseSVDKokkos(context);


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

// --------------------- Helpers for group name parsing ---------------------
static inline std::tuple<int,int,int> parseSliceName(const std::string &name) {
  static const std::regex r("slice_(\\d+)_(\\d+)_(\\d+)");
  std::smatch m;
  if (!std::regex_match(name, m, r))
    throw std::invalid_argument("Bad SVD group name: " + name);
  return {std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3])};
}

// -------------------------- Process all groups ---------------------------
std::vector<InteractionElPhSVD::SVDGroupData>
InteractionElPhSVD::processAllSVDGroups(const HighFive::Group &svdGroup,
                                        int &num_i, int &num_j, int &num_eta,
                                        int &RE, int &RP, int &maxGamma) {
  std::vector<std::string> names = svdGroup.listObjectNames();

  if (names.size() == 0) {
    throw std::runtime_error("No SVD slices found in HDF5 file!");
  }

  size_t mi=0, mj=0, me=0;
  RE = 0; RP = 0; maxGamma = 0;

  // First pass to determine geometry
  int processedSlices = 0;
  for (const auto &name : names) {
    auto [i,j,eta] = parseSliceName(name);
    mi=std::max(mi,size_t(i)); mj=std::max(mj,size_t(j)); me=std::max(me,size_t(eta));

    auto g = svdGroup.getGroup(name);
    Eigen::MatrixXcd Uc, Vc;
    Eigen::VectorXd Sr;

    g.getDataSet("U").read(Uc);
    // S may be stored as 1D or 2D
    try { g.getDataSet("S").read(Sr); }
    catch (...) {
      Eigen::MatrixXd S_matrix;
      g.getDataSet("S").read(S_matrix);
      if (S_matrix.cols()==1)       Sr = S_matrix.col(0);
      else if (S_matrix.rows()==1)  Sr = S_matrix.row(0).transpose();
      else if (S_matrix.rows()==S_matrix.cols() && S_matrix.rows()>1)
        Sr = S_matrix.diagonal();
      else
        throw std::runtime_error("Invalid S shape in "+name);
    }

    const int gamma = (int)Sr.size();

    g.getDataSet("V").read(Vc);

    Eigen::MatrixXcd V_RPbyGamma;
    if (Vc.cols() == gamma)        V_RPbyGamma = Vc.adjoint(); // V saved → convert to V^dagger
    else if (Vc.rows() == gamma)   V_RPbyGamma = Vc;
    else
      throw std::runtime_error("Unexpected V shape in "+name);

    RE = std::max(RE, (int)Uc.rows());
    RP = std::max(RP, (int)V_RPbyGamma.cols());
    maxGamma = std::max(maxGamma, gamma);

    processedSlices++;
  }


  num_i = int(mi+1); num_j = int(mj+1); num_eta = int(me+1);

  // Second pass: pack slices
  std::vector<SVDGroupData> slices;
  int sliceCount = 0;

  for (const auto &name : names) {
    auto [i,j,eta] = parseSliceName(name);
    auto g = svdGroup.getGroup(name);



    Eigen::MatrixXcd Uc, Vc;
    Eigen::VectorXd Sr;

    // Read U, V, S
    g.getDataSet("U").read(Uc);

    try { g.getDataSet("S").read(Sr); }
    catch (...) {
      Eigen::MatrixXd S_matrix;
      g.getDataSet("S").read(S_matrix);
      if (S_matrix.cols()==1)       Sr = S_matrix.col(0);
      else if (S_matrix.rows()==1)  Sr = S_matrix.row(0).transpose();
      else if (S_matrix.rows()==S_matrix.cols() && S_matrix.rows()>1)
        Sr = S_matrix.diagonal();
      else
        throw std::runtime_error("Invalid S shape in "+name);
    }

    g.getDataSet("V").read(Vc);
    const int gamma = (int)Sr.size();

    // Normalize V to RP×γ
    Eigen::MatrixXcd V_RPbyGamma;
    if (Vc.cols() == gamma)        V_RPbyGamma = Vc;
    else if (Vc.rows() == gamma)   V_RPbyGamma = Vc.adjoint();
    else
      throw std::runtime_error("Unexpected V shape in "+name);

    // Convert to desired layouts for packing (U row-major for re,g loop)
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_rowmajor = Uc;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Vh_colmajor = V_RPbyGamma;

    SVDGroupData d; d.i=i; d.j=j; d.eta=eta; d.RE=(int)Uc.rows(); d.RP=(int)V_RPbyGamma.cols(); d.gamma=gamma;
    d.U.resize(size_t(d.RE)*size_t(gamma));
    d.V.resize(size_t(d.RP)*size_t(gamma));
    d.S.resize(size_t(gamma));

    // Pack U, V, S
    for (int gidx=0; gidx<gamma; ++gidx) {
      d.S[size_t(gidx)] = Sr(gidx); // keep real
      for (int re=0; re<d.RE; ++re)
        d.U[size_t(re)*size_t(gamma)+size_t(gidx)] = U_rowmajor(re,gidx);
      for (int rp=0; rp<d.RP; ++rp)
        d.V[size_t(rp)*size_t(gamma)+size_t(gidx)] = Vh_colmajor(gidx,rp);
    }
    slices.push_back(std::move(d));
    sliceCount++;
  }

  return slices;
}


void InteractionElPhSVD::parseSVDKokkos(Context &context) {
  const std::string fileName = context.getElphFileName();
  if (mpi->mpiHead()) {
    std::cout << "Loading SVD slices from " << fileName << " ...\n";
  }

  int RE=0, RP=0;
  std::vector<std::complex<double>> SY_buf;
  std::vector<std::complex<double>> Vt_buf;
  std::vector<int> gammaLens;

  try {
    if (mpi->mpiHead()) {
      HighFive::File file(fileName, HighFive::File::ReadOnly);
      HighFive::Group svdGroup = file.getGroup("zSVD");

      int ni, nj, ne, mG;
      auto slices = processAllSVDGroups(svdGroup, ni, nj, ne, RE, RP, mG);
      numI=ni; numJ=nj; numPhBands=ne; maxGamma=mG;

      const int L = numI * numJ * numPhBands;

      SY_buf.assign(size_t(L)*size_t(RE)*size_t(maxGamma), {0,0});
      Vt_buf.assign(size_t(L)*size_t(maxGamma)*size_t(RP), {0,0});
      gammaLens.assign(L, 0);


      auto mapL = [J=numJ, E=numPhBands](int i,int j,int eta){ return i*(J*E) + j*E + eta; };


      int processedSlicesCount = 0;
      for (const auto &d : slices) {

        const int l = mapL(d.i, d.j, d.eta);
        gammaLens[l] = d.gamma;

        // SY(l,re,g) = U(re,g)*S(g)
        for (int re=0; re<RE; ++re) {
          for (int g=0; g<d.gamma; ++g) {
            const auto val = d.U[size_t(re)*size_t(d.gamma)+size_t(g)] * d.S[size_t(g)]; // S is double
            const size_t off = (size_t(l)*RE + size_t(re))*size_t(maxGamma) + size_t(g);
            SY_buf[off] = val;
          }
        }

        for (int g=0; g<d.gamma; ++g) {
          for (int rp=0; rp<RP; ++rp) {
            const auto val = d.V[size_t(rp)*size_t(d.gamma)+size_t(g)];
            const size_t off = (size_t(l)*size_t(maxGamma) + size_t(g))*size_t(RP) + size_t(rp);
            Vt_buf[off] = val;
          }
        }
        processedSlicesCount++;
      }


      // broadcast geometry
      mpi->bcast(&numI); mpi->bcast(&numJ); mpi->bcast(&numPhBands);
      mpi->bcast(&RE);   mpi->bcast(&RP);   mpi->bcast(&maxGamma);
      mpi->bcast(gammaLens.data(), int(gammaLens.size()));

      // broadcast payload as doubles (2x)
      mpi->bcast(reinterpret_cast<double*>(SY_buf.data()), int(SY_buf.size())*2);
      mpi->bcast(reinterpret_cast<double*>(Vt_buf.data()), int(Vt_buf.size())*2);

    } else {
      mpi->bcast(&numI); mpi->bcast(&numJ); mpi->bcast(&numPhBands);
      mpi->bcast(&RE);   mpi->bcast(&RP);   mpi->bcast(&maxGamma);

      const int L = numI * numJ * numPhBands;
      gammaLens.resize(L);
      mpi->bcast(gammaLens.data(), L);

      SY_buf.resize(size_t(L)*size_t(RE)*size_t(maxGamma));
      Vt_buf.resize(size_t(L)*size_t(maxGamma)*size_t(RP));
      mpi->bcast(reinterpret_cast<double*>(SY_buf.data()), int(SY_buf.size())*2);
      mpi->bcast(reinterpret_cast<double*>(Vt_buf.data()), int(Vt_buf.size())*2);
    }

    // ---- allocate device views----

    try {
      Kokkos::realloc(SVD_SY_device, numI, numJ, numPhBands, RE*maxGamma);
      Kokkos::realloc(SVD_Vt_device, numI, numJ, numPhBands, maxGamma*RP);
      Kokkos::realloc(gammaLen_ijk,  numI, numJ, numPhBands);

    } catch (const std::exception &e) {
      std::cerr << "CRITICAL ERROR in Kokkos realloc: " << e.what() << std::endl;
      throw;
    }

    // ---- fill host mirrors ----
    try {
      auto SY_h  = Kokkos::create_mirror_view(SVD_SY_device);
      auto Vt_h  = Kokkos::create_mirror_view(SVD_Vt_device);
      auto gLenH = Kokkos::create_mirror_view(gammaLen_ijk);


    int idx = 0;
    for (int i=0;i<numI;++i)
      for (int j=0;j<numJ;++j)
        for (int e=0;e<numPhBands;++e)
          gLenH(i,j,e) = gammaLens[idx++];

    // SY_h(i,j,e, re*maxGamma + g)
    for (int i=0;i<numI;++i)
      for (int j=0;j<numJ;++j)
        for (int e=0;e<numPhBands;++e) {
          const int l = (i*numJ + j)*numPhBands + e;
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
        for (int e=0;e<numPhBands;++e) {
          const int l = (i*numJ + j)*numPhBands + e;
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

    } catch (const std::exception &e) {
      std::cerr << "CRITICAL ERROR in mirror/deep_copy operations: " << e.what() << std::endl;
      throw;
    }
    numWsR1Vectors = RE;
    numWsR2Vectors = RP;
    numWannierOrbitals = numElBands;

    if (mpi->mpiHead()) {
      std::cout << "SVD factors loaded: (I,J,η)=("
                << numI << "," << numJ << "," << numPhBands << "), "
                << "RE=" << RE << ", RP=" << RP
                << ", maxGamma=" << maxGamma << "\n";
    }

  } catch (const std::exception &e) {
    Error(std::string("parseSVDKokkos failed: ") + e.what());
  }
}


void InteractionElPhSVD::cacheElPh(const Eigen::MatrixXcd &eigvec1,
                                   const Eigen::Vector3d  &k1C) {

  Kokkos::Profiling::pushRegion("cacheElPh_SVD");

  const int numGamma            = maxGamma;    // FIXME how is this set
  //const int numWannierOrbitals  = numI;           // i- (and j-) dimension
  //const int numWsR1Vectors_local= numWsR1Vectors; // R_e rows
  Kokkos::complex<double> complexI(0.0, 1.0);

  // Get dimensions
  auto nb1 = int(eigvec1.cols()); // number of bands at this kpoint

  auto elPhCached_SVD_SY = this->elPhCached_SVD_SY; // final container of output from this function, stored in the class
  auto SVD_SY_device     = this->SVD_SY_device; // input data of the first part of the SVD

  // Copy the eigenvector and wavevector to the accelerator
  ComplexView2D eigvec1_device("ev1", nb1, numWannierOrbitals);
  DoubleView1D  k1C_device("k", 3);

  {
    // Create a copy to avoid storage order issues
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigvec1_rowmajor = eigvec1;

    HostComplexView2D eigvec1_host((Kokkos::complex<double>*)eigvec1_rowmajor.data(), nb1, numWannierOrbitals);
    HostDoubleView1D  k1C_host((double*)k1C.data(), 3);

    Kokkos::deep_copy(eigvec1_device, eigvec1_host);
    Kokkos::deep_copy(k1C_device,     k1C_host);


  }

  // ---------------- phases for R_e ----------------
  Kokkos::Profiling::pushRegion("precompute_phases_k1");
  ComplexView1D phases_device("phases", numWsR1Vectors);
  {
    auto r1 = wsR1Vectors_device;
    auto d1 = wsR1VectorsDegeneracies_device;
    Kokkos::parallel_for("phases_k1c", numWsR1Vectors, KOKKOS_LAMBDA(int irE) {
      double arg = 0.0;
      for (int j = 0; j < 3; ++j) arg += k1C_device(j) * r1(irE, j);
      phases_device(irE) = exp(complexI * arg) / d1(irE);
    });
    Kokkos::fence();
  }
  Kokkos::Profiling::popRegion();

  // ---------------- pack_SY(R_e, gamma*eta*i*j) ----------------
  Kokkos::Profiling::pushRegion("fourier_transform");

  const size_t flattened_size = size_t(numGamma) * size_t(numPhBands) * size_t(numWannierOrbitals) * size_t(numWannierOrbitals);

  // Create intermediate container of size (gamma, eta, i, j)
  // used to store the output of the FT on K1
  ComplexView4D g_FT_output("g1",
    numGamma, numPhBands, numWannierOrbitals, numWannierOrbitals);

  // perform fourier transform using precomputed phases
  {
  // FIXME is layoutRight ok or wrong??
  // set up the empty container for output
  // Create intermediate/"cache" container of size (gamma, eta, i, j)
  Kokkos::View<Kokkos::complex<double>*, Kokkos::LayoutRight> g_FT_output_1D(
    g_FT_output.data(), flattened_size);

  // layout right was used earlier because EIGEN has opposite storage system
  // FIXME is layoutRight ok or wrong??  -- I suspect it's ok
  Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight> coupling_2D(
    SVD_SY_device.data(), numWsR1Vectors, flattened_size);

  KokkosBlas::gemv("T", Kokkos::complex<double>(1.0, 0.0),
                   coupling_2D, phases_device,
                   Kokkos::complex<double>(0.0, 0.0), g_FT_output_1D);


  }
  Kokkos::Profiling::popRegion();

  // ------------- Merger rotation into SVD elements ----------------
  Kokkos::Profiling::pushRegion("rotation_to_band_basis");

  Kokkos::realloc(elPhCached_SVD_SY,
                  numGamma, numPhBands, nb1, numWannierOrbitals, numWannierOrbitals);

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

// k1C is a dummy variable kept for legacy code consistency - not used in SVD implementation
void InteractionElPhSVD::calcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3,
    const std::vector<Eigen::Vector3d> &q3Cs, const Eigen::Vector3d &k1C,
    const std::vector<Eigen::VectorXcd> &polarData) {

  Kokkos::Profiling::pushRegion("calcCouplingSquared");

  //const int numWannierOrbitals = numI;
  //const int numWannier         = numElBands;
  const int nb1                = int(eigvec1.cols());
  const int numK2              = int(eigvecs2.size());

  auto elPhCached_SVD_SY = this->elPhCached_SVD_SY;

  const int numGamma        = maxGamma;
  //const int numPhBands      = this->numPhBands;
  const int numWsR2Vectors  = this->numWsR2Vectors;

  // R2 lattice (could be Rp or Re' in JDFTx case)
  DoubleView2D wsR2Vectors_device             = this->wsR2Vectors_device;
  DoubleView1D wsR2VectorsDegeneracies_device = this->wsR2VectorsDegeneracies_device;

  // each k2 may have a different number of bands: find max, pad to rectangular
  IntView1D nb2s_device("nb2s", numK2);
  int nb2max = 0;
  auto nb2s_h = Kokkos::create_mirror_view(nb2s_device);
  for (int ik = 0; ik < numK2; ++ik) {
    nb2s_h(ik) = int(eigvecs2[ik].cols());
    if (nb2s_h(ik) > nb2max) nb2max = nb2s_h(ik);
  }
  Kokkos::deep_copy(nb2s_device, nb2s_h);


  // -------------------- Polar corrections  --------------------
  // ================ REFACTOR : strip this out into polar helper functions, maybe in parent =========================
  IntView1D   usePolarCorrections_device("usePolarCorrections", numK2);
  ComplexView4D polarCorrections_device("polarCorrections",
      numK2, numPhBands, nb1, nb2max);

  auto usePolarCorrections_host = Kokkos::create_mirror_view(usePolarCorrections_device);
  auto polarCorrections_host    = Kokkos::create_mirror_view(polarCorrections_device);

  #pragma omp parallel for
  for (int ik = 0; ik < numK2; ++ik) {
    const Eigen::Vector3d q3C = q3Cs[ik];
    const Eigen::MatrixXcd &eigvec2 = eigvecs2[ik];

    usePolarCorrections_host(ik) = usePolarCorrection && std::abs(q3C.norm()) > 1.0e-8;

    if (usePolarCorrections_host(ik)) {
      // returns (b1, b2, nu)
      Eigen::Tensor<std::complex<double>, 3> single =
          polarCorrectionPart2(eigvec1, eigvec2, polarData[ik]);

      for (int nu = 0; nu < numPhBands; ++nu) {
        for (int ib1_ = 0; ib1_ < nb1; ++ib1_) {
          for (int ib2 = 0; ib2 < nb2s_h(ik); ++ib2) {
            polarCorrections_host(ik, nu, ib1_, ib2) = single(ib1_, ib2, nu);
          }
        }
      }
    } else {
      const Kokkos::complex<double> kZero(0., 0.);
      for (int nu = 0; nu < numPhBands; ++nu) {
        for (int ib1_ = 0; ib1_ < nb1; ++ib1_) {
          for (int ib2 = 0; ib2 < nb2s_h(ik); ++ib2) {
            polarCorrections_host(ik, nu, ib1_, ib2) = kZero;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(polarCorrections_device,    polarCorrections_host);
  Kokkos::deep_copy(usePolarCorrections_device, usePolarCorrections_host);
  // ================ REFACTOR : strip this out into polar helper functions, maybe in parent =========================

   // ================ REFACTOR : strip this out into parent class helper function =========================
  // -------------------- Copy to device --------------------
  DoubleView2D q3Cs_device("q3", numK2, 3);
  ComplexView3D eigvecs2Dagger_device("ev2Dagger", numK2, numWannierOrbitals, nb2max);
  ComplexView3D eigvecs3_device      ("ev3",       numK2, numPhBands,         numPhBands);

  {
    auto eigvecs2Dagger_host = Kokkos::create_mirror_view(eigvecs2Dagger_device);
    auto eigvecs3_host       = Kokkos::create_mirror_view(eigvecs3_device);
    auto q3Cs_host           = Kokkos::create_mirror_view(q3Cs_device);

    #pragma omp parallel for
    for (int ik = 0; ik < numK2; ++ik) {
      for (int j = 0; j < numWannierOrbitals; ++j) {
        for (int ib2 = 0; ib2 < nb2s_h(ik); ++ib2) {
          eigvecs2Dagger_host(ik, j, ib2) = std::conj(eigvecs2[ik](j, ib2));
        }
      }
      for (int nu2 = 0; nu2 < eigvecs3[ik].cols(); ++nu2) {
        for (int nu  = 0; nu  < eigvecs3[ik].rows(); ++nu) {
          if (phaseConvention == JdftxPhaseConvention) {
            // If JDFTx: caller should have adjusted phases; use conj transpose mapping
            eigvecs3_host(ik, nu2, nu) = std::conj(eigvecs3[ik](nu, nu2));
          } else {
            eigvecs3_host(ik, nu2, nu) = eigvecs3[ik](nu, nu2);
          }
        }
      }
      for (int a = 0; a < 3; ++a) {
        // If JDFTx variant needs k', assume the caller preloaded q3Cs accordingly
        q3Cs_host(ik, a) = q3Cs[ik](a);
      }
    }
    Kokkos::deep_copy(eigvecs2Dagger_device, eigvecs2Dagger_host);
    Kokkos::deep_copy(eigvecs3_device,       eigvecs3_host);
    Kokkos::deep_copy(q3Cs_device,           q3Cs_host);
  }
 // ================ REFACTOR : strip this out into parent class helper function =========================

  // -------------------- Phases over R2 --------------------
  ComplexView2D phases_device("phases", numK2, numWsR2Vectors);
  const Kokkos::complex<double> complexI(0.0, 1.0);
  Kokkos::parallel_for(
      "Interaction elph: calculate phase 2",
      Range2D({0, 0}, {numK2, numWsR2Vectors}),
      KOKKOS_LAMBDA(const int ik, const int irP) {
        double arg = 0.0;
        for (int j = 0; j < 3; ++j) arg += q3Cs_device(ik, j) * wsR2Vectors_device(irP, j);
        phases_device(ik, irP) = exp(complexI * arg) / wsR2VectorsDegeneracies_device(irP);
      });
    Kokkos::fence();



 // -------------------- Fourier transform for K2 --------------------
  const size_t flattened_size_vt =
      size_t(numGamma) * size_t(numPhBands) * size_t(numWannierOrbitals) * size_t(numWannierOrbitals);

  ComplexView5D v_FT_output("v_FT_output",
      numK2, numGamma, numPhBands, numWannierOrbitals, numWannierOrbitals);
  Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight>
      v_FT_output_2D(v_FT_output.data(), numK2, flattened_size_vt);

  Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight>
      coupling_2D(this->SVD_Vt_device.data(), numWsR2Vectors, flattened_size_vt);

  KokkosBlas::gemm("N", "N", Kokkos::complex<double>(1.0, 0.0),
        phases_device, coupling_2D, Kokkos::complex<double>(0.0, 0.0), v_FT_output_2D);

  // -------------------- Unitary rotation --------------------
  ComplexView6D elPhCached_SVD_Vt("elPhCached_SVD_Vt",
          numK2, numGamma, numPhBands, nb2max, numWannierOrbitals, numWannierOrbitals);

  Kokkos::parallel_for(
      "rotate using U^dagger eigenvectors. elPhCached_SVD_Vt_device",
      Range6D({0, 0, 0, 0, 0, 0},
              {numK2, numGamma, numPhBands, nb2max, numWannierOrbitals, numWannierOrbitals}),
      KOKKOS_LAMBDA(const int ik, const int g, const int eta,
                    const int ib2, const int i, const int j) {
        elPhCached_SVD_Vt(ik, g, eta, ib2, i, j) = v_FT_output(ik, g, eta, i, j) * eigvecs2Dagger_device(ik, j, ib2);
      });
  Kokkos::fence();

  // -------------------- Phonon rotation --------------------
  Kokkos::Profiling::pushRegion("phonon_rotation");

  ComplexView7D elPhCached_SVD_Vt_ph("elPhCached_SVD_Vt_ph",
      numK2, nb2max, numPhBands, numGamma, numPhBands, numWannierOrbitals, numWannierOrbitals);

  // IMPORTANT CHANGE, KEYNESH SHOULD READ THIS
  // Before this was actually performing the matrix product. As before, we are here just merging things in,
  // we will sum later.
   Kokkos::parallel_for(
      "phonon_rotation",
      Range6D({0, 0, 0, 0, 0, 0},
              {numK2, nb2max, numPhBands, numGamma, numPhBands, numWannierOrbitals}),
      KOKKOS_LAMBDA(const int iK2, const int iBand2, const int iPh, const int iGamma, const int iEta, const int iWannierI) {
        for (int iWannierJ = 0; iWannierJ < numWannierOrbitals; ++iWannierJ) {
          elPhCached_SVD_Vt_ph(iK2, iBand2, iPh, iGamma, iEta, iWannierI, iWannierJ) = elPhCached_SVD_Vt(iK2, iGamma, iEta, iBand2, iWannierI, iWannierJ)
                         * eigvecs3_device(iK2, iEta, iPh);
        }
    });
  Kokkos::fence();



  Kokkos::Profiling::popRegion(); // phonon rotation
  // done with elPhCached_SVD_Vt, deallocate
  Kokkos::realloc(elPhCached_SVD_Vt, 0, 0, 0, 0, 0, 0);

  // REFACTOR I think there is a possible alternative where we do these products and sum at the same time
  // basically, we take  [] Sum_{ij,eta,gamma} ( s_Y_ij,eta,gamma,m ) * ( Vt_ij,eta,gamma,n )) ] * u_ph
  // beause we don't really like allocating this giant 6D tensor ...

  // -------------------- Sum over ij,eta,gamma --------------------

  // here we need to loop over K2 (because we can't do a sum over this dimension,
  // so it doesn't really work to put it into a gemm)

  Kokkos::Profiling::pushRegion("sum_g_ij_eta_gamma");

  const size_t flattened_size = // i * j * eta * gamma
        size_t(numGamma) * size_t(numPhBands) * size_t(numWannierOrbitals) * size_t(numWannierOrbitals);

  // container for final output
  ComplexView4D gFinal("gFinal", numK2, nb1, nb2max, numPhBands);
  // full views for data used in gemm below
  ComplexView2D SVD_SY_2D(this->elPhCached_SVD_SY.data(), size_t(nb1), size_t(numGamma) * size_t(numPhBands) * size_t(numWannierOrbitals) * size_t(numWannierOrbitals));
  ComplexView3D SVD_Vt_3D(elPhCached_SVD_Vt_ph.data(), numK2,size_t(nb2max) * size_t(numPhBands), size_t(numGamma) * size_t(numPhBands) * size_t(numWannierOrbitals) * size_t(numWannierOrbitals));
  ComplexView3D gFinal_gemm_output_3D(gFinal.data(), numK2, size_t(nb1), size_t(nb2max) * size_t(numPhBands));



  // Feels like there must be a better way to do this!
  // NOTE : no OMP HERE because gemm will use threads!
  for (long ik2 = 0; ik2 < numK2; ++ik2) {

    // subviews of the SVD pieces for each k2 point
    ComplexView2D SVD_Vt_2D_k2 = Kokkos::subview(SVD_Vt_3D, ik2, Kokkos::ALL, Kokkos::ALL);
    ComplexView2D gFinal_gemm_output_2d_k2 = Kokkos::subview(gFinal_gemm_output_3D, ik2, Kokkos::ALL, Kokkos::ALL);

    // this is product of [ nBands, i * j * eta * gamma ] [ mBands, i * j * eta * gamma].transpose
    KokkosBlas::gemm("N","T", Kokkos::complex<double>(1.0, 0.0),
          SVD_SY_2D, SVD_Vt_2D_k2, Kokkos::complex<double>(0.0, 0.0), gFinal_gemm_output_2d_k2);
  }

  Kokkos::Profiling::popRegion(); // sum over g_ij eta gamma
  // now we are done with elPhCached_SVD_SY and elPhCached_SVD_Vt_ph
  // so we deallocate them
  Kokkos::realloc(elPhCached_SVD_SY, 0, 0, 0, 0, 0);
  Kokkos::realloc(elPhCached_SVD_Vt_ph, 0, 0, 0, 0, 0, 0, 0);

  // we now add the precomputed polar corrections, before taking the norm of g
  if (usePolarCorrection) {
    Kokkos::parallel_for(
        "re-add polar correction to g",
        Range4D({0, 0, 0, 0}, {numK2, numPhBands, nb1, nb2max}),
        KOKKOS_LAMBDA(const int ik, const int nu, const int ib1_, const int ib2) {
          gFinal(ik, ib1_, ib2, nu) += polarCorrections_device(ik, nu, ib1_, ib2);
        });
  }
  Kokkos::realloc(polarCorrections_device, 0, 0, 0, 0);

  // -------------------- |g|^2 --------------------
  DoubleView4D coupling_device("gSq", numK2, numPhBands, nb2max, nb1);

  Kokkos::parallel_for(
      "Interaction elph: modulus coupling",
      Range4D({0, 0, 0, 0}, {numK2, nb1, nb2max, numPhBands}),
      KOKKOS_LAMBDA(const int ik, const int ib1_, const int ib2, const int nu) {
        const auto z = gFinal(ik, ib1_, ib2, nu);
        coupling_device(ik, nu, ib2, ib1_) = z.real()*z.real() + z.imag()*z.imag();
      });

  Kokkos::fence();
  Kokkos::realloc(gFinal, 0, 0, 0, 0);

  //now, copy back to CPU cacheCoupling
  cacheCoupling.resize(numK2);
  auto coupling_host = Kokkos::create_mirror_view(coupling_device);
  Kokkos::deep_copy(coupling_host, coupling_device);
  Kokkos::fence();
  Kokkos::fence();

  #pragma omp parallel for default(none) shared(numK2, cacheCoupling, coupling_host, nb1, nb2s_h, numPhBands)
  for (int ik = 0; ik < numK2; ++ik) {
    Eigen::Tensor<double, 3> coupling(nb1, nb2s_h(ik), numPhBands);
    for (int nu = 0; nu < numPhBands; ++nu) {
      for (int ib2 = 0; ib2 < nb2s_h(ik); ++ib2) {
        for (int ib1 = 0; ib1 < nb1; ++ib1) {
          coupling(ib1, ib2, nu) = coupling_host(ik, nu, ib2, ib1);
        }
      }
    }
    cacheCoupling[ik] = coupling;
  }
  Kokkos::Profiling::popRegion(); // calcCouplingSquared
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

double InteractionElPhSVD::getDeviceMemoryUsage() const {
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
