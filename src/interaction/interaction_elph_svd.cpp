#include "interaction_elph_svd.h"
#include "common_kokkos.h"
#include "context.h"
#include "crystal.h"
#include "exceptions.h"
#include "harmonic/phonon_h0.h"
#include "mpi/mpiController.h"
#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <iomanip>
#include <iostream>
#include <regex>
#include <tuple>
#include <vector>


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
  numGamma = 1;  // Will be updated during parsing

  // Parsing function
  parseSVDKokkos(context);

  if (mpi->mpiHead()) {
     printSVDInfo();
   }
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

// Iterate over all HDF5 subgroups to read their U,S,V datasets
std::vector<InteractionElPhSVD::SVDGroupData>
InteractionElPhSVD::processAllSVDGroups(const HighFive::Group &svdGroup,
                                        size_t &num_i, size_t &num_j,
                                        size_t &num_eta) {
  std::vector<SVDGroupData> allData;
  auto groupNames = svdGroup.listObjectNames();

  size_t max_i = 0, max_j = 0, max_eta = 0;

  for (const auto &groupName : groupNames) {
    HighFive::Group subGroup = svdGroup.getGroup(groupName);
    auto [i, j, eta] = extractIndicesFromGroupName(groupName);

    if (size_t(i) > max_i) max_i = i;
    if (size_t(j) > max_j) max_j = j;
    if (size_t(eta) > max_eta) max_eta = eta;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_mat, s_mat;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> v_mat;

    subGroup.getDataSet("U").read(u_mat);
    subGroup.getDataSet("V").read(v_mat);
    subGroup.getDataSet("S").read(s_mat); // Read S as a 2D matrix, it's saved as a [x,1] matrix

    // Flatten all matrices into 1D vectors for storage
    auto s = std::make_unique<std::vector<double>>(s_mat.data(), s_mat.data() + s_mat.size());
    auto u = std::make_unique<std::vector<double>>(u_mat.data(), u_mat.data() + u_mat.size());
    auto v = std::make_unique<std::vector<double>>(v_mat.data(), v_mat.data() + v_mat.size());

    allData.emplace_back(s, v, u, i, j, eta); // construct an SVD data group 
  }
  num_i = max_i + 1;
  num_j = max_j + 1;
  num_eta = max_eta + 1;
  return allData;
}


// Core routine to read data
void InteractionElPhSVD::parseSVDKokkos(Context &context) {
  std::string fileName = context.getElphFileName();
  if (mpi->mpiHead()) {
      std::cout << "--------------------------------------------------" << std::endl;
      std::cout << "Parsing SVD data from HDF5 file: " << fileName << std::endl;
      std::cout << "--------------------------------------------------" << std::endl;
  }

  try {
    // MPI rank 0 reads all metadata from the HDF5 file.
    if (mpi->mpiHead()) {
      HighFive::File file(fileName, HighFive::File::ReadOnly);
      file.getDataSet("/numElBands").read(numElBands);
      file.getDataSet("/numPhModes").read(numPhBands);
      file.getDataSet("/elBravaisVectors").read(elBravaisVectors);
      file.getDataSet("/phBravaisVectors").read(phBravaisVectors);
      file.getDataSet("/elDegeneracies").read(elBravaisVectorsDegeneracies);
      file.getDataSet("/phDegeneracies").read(phBravaisVectorsDegeneracies);
    }

    // Broadcast header data to all other MPI processes using the .data() method.
    if (!mpi->mpiHead()) {
      int el_rows = 0, el_cols = 0, ph_rows = 0, ph_cols = 0;
      int el_degen_size = 0, ph_degen_size = 0;
      if(mpi->mpiHead()) { // only rank 0 knows the size
          el_rows = elBravaisVectors.rows(); el_cols = elBravaisVectors.cols();
          ph_rows = phBravaisVectors.rows(); ph_cols = phBravaisVectors.cols();
          el_degen_size = elBravaisVectorsDegeneracies.size();
          ph_degen_size = phBravaisVectorsDegeneracies.size();
      }
      mpi->bcast(&el_rows); mpi->bcast(&el_cols);
      mpi->bcast(&ph_rows); mpi->bcast(&ph_cols);
      mpi->bcast(&el_degen_size); mpi->bcast(&ph_degen_size);

      // now resize on other ranks
      elBravaisVectors.resize(el_rows, el_cols);
      phBravaisVectors.resize(ph_rows, ph_cols);
      elBravaisVectorsDegeneracies.resize(el_degen_size);
      phBravaisVectorsDegeneracies.resize(ph_degen_size);
    }

    mpi->bcast(&numElBands);
    mpi->bcast(&numPhBands);
    mpi->bcast(elBravaisVectors.data(), elBravaisVectors.size());
    mpi->bcast(phBravaisVectors.data(), phBravaisVectors.size());
    mpi->bcast(elBravaisVectorsDegeneracies.data(), elBravaisVectorsDegeneracies.size());
    mpi->bcast(phBravaisVectorsDegeneracies.data(), phBravaisVectorsDegeneracies.size());

    size_t numElBravaisVectors = elBravaisVectorsDegeneracies.size();
    size_t numPhBravaisVectors = phBravaisVectorsDegeneracies.size();

    if (mpi->mpiHead()) {
        HighFive::File file(fileName, HighFive::File::ReadOnly);
        HighFive::Group svdGroup = file.getGroup("zSVD");

        size_t num_i, num_j, num_eta;
        std::vector<SVDGroupData> allSVDData = processAllSVDGroups(svdGroup, num_i, num_j, num_eta);

        if (allSVDData.empty()) {
          throw std::runtime_error("No SVD groups found in the HDF5 file.");
        }

        size_t numSingularValues = allSVDData[0].singularVector->size();
        size_t leftMatrixCols = allSVDData[0].leftMatrix->size() / numSingularValues;   // R_e
        size_t rightMatrixRows = allSVDData[0].rightMatrix->size() / numSingularValues; // R_p

        // numGamma variable for cacheElPh later
        numGamma = numSingularValues;

        // Allocate the reconstructed 5D electron-phonon matrix: (i, j, eta, R_e, R_p)
        Kokkos::realloc(ElPh_Matrix, num_i, num_j, num_eta, leftMatrixCols, rightMatrixRows);

        auto ElPh_Matrix_h = Kokkos::create_mirror_view(ElPh_Matrix);

        // Initialize
        Kokkos::deep_copy(ElPh_Matrix_h, Kokkos::complex<double>(0.0, 0.0));

        // FIXME: check the dimensions of these containers, where is numGamma
        // these data sets are in format of a 1d std::Vector 
        HostComplexView5D SVD_SY_host(allSVDData[0].leftMatrix, numElBands, numElBands, numPhBands, numWsR1Vectors, numGamma);
        HostComplexView5D SVD_Vt_host(allSVDData[0].rightMatrix, numElBands, numElBands, numPhBands, numWsR2Vectors, numGamma);
        Kokkos::deep_copy(SVD_SY_device, SVD_SY_host);
        Kokkos::deep_copy(SVD_Vt_device, SVD_Vt_host);        
        
        // Reconstruct the matrices from SVD components
        for (const auto &data : allSVDData) {
          size_t i = data.idxX;
          size_t j = data.idxY;
          size_t eta = data.idxZ;
          const auto &sVec = *data.singularVector;
          const auto &uMat = *data.leftMatrix;
          const auto &vMat = *data.rightMatrix;

          for (size_t R_e = 0; R_e < leftMatrixCols; ++R_e) {
            for (size_t R_p = 0; R_p < rightMatrixRows; ++R_p) {
              Kokkos::complex<double> element(0.0, 0.0);
              for (size_t s = 0; s < numSingularValues; ++s) {
                double us_term = uMat[R_e * numSingularValues + s] * sVec[s];
                double vt_term = vMat[s * rightMatrixRows + R_p];
                element += Kokkos::complex<double>(us_term * vt_term, 0.0);
              }
              ElPh_Matrix_h(i, j, eta, R_e, R_p) = element;
            }
          }
        }

        Kokkos::deep_copy(ElPh_Matrix, ElPh_Matrix_h);
        std::cout << "Successfully reconstructed 5D electron-phonon matrix from SVD data." << std::endl;
        std::cout << "ElPh_Matrix dimensions: (" << ElPh_Matrix.extent(0) << ", "
                  << ElPh_Matrix.extent(1) << ", " << ElPh_Matrix.extent(2) << ", "
                  << ElPh_Matrix.extent(3) << ", " << ElPh_Matrix.extent(4) << ")" << std::endl;
        std::cout << "numGamma = " << numGamma << " (retained singular values)" << std::endl;

        // Export for Python analysis - uncomment when needed
        exportElPhMatrixToHDF5("elph_matrix_debug.h5");
    }

    mpi->barrier();

  } catch (const std::exception &e) {
    Error("Error while parsing SVD file: " + std::string(e.what()));
  }
}

// FIXME update this to not use ElPh_Matrix
void InteractionElPhSVD::printSVDInfo() const {
  if (ElPh_Matrix.data() == nullptr) {
    std::cout << "ElPh_Matrix is not initialized." << std::endl;
    return;
  }
  std::cout << "--- Electron-Phonon Matrix Info ---" << std::endl;
  std::cout << "ElPh_Matrix dimensions (i,j,eta,R_e,R_p): " << ElPh_Matrix.extent(0) << " x "
            << ElPh_Matrix.extent(1) << " x " << ElPh_Matrix.extent(2) << " x "
            << ElPh_Matrix.extent(3) << " x " << ElPh_Matrix.extent(4) << std::endl;
  std::cout << "Total elements: " << ElPh_Matrix.span() << std::endl;
  std::cout << "Total memory usage (approx): " << getDeviceMemoryUsage() / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
}

void InteractionElPhSVD::printSVDSample(size_t i, size_t j, size_t eta) const {
  if (ElPh_Matrix.data() == nullptr) {
    std::cout << "ElPh_Matrix is not initialized." << std::endl;
    return;
  }
  auto ElPh_Matrix_h = Kokkos::create_mirror_view(ElPh_Matrix);
  Kokkos::deep_copy(ElPh_Matrix_h, ElPh_Matrix);
  std::cout << "--- ElPh Matrix Sample at (i,j,eta) = (" << i << "," << j << ","
            << eta << ") ---" << std::endl;
  std::cout << "ElPh_Matrix(i,j,eta,0,0) = " << ElPh_Matrix_h(i, j, eta, 0, 0) << std::endl;
  if (ElPh_Matrix.extent(3) > 1 && ElPh_Matrix.extent(4) > 1) {
    std::cout << "ElPh_Matrix(i,j,eta,1,1) = " << ElPh_Matrix_h(i, j, eta, 1, 1) << std::endl;
  }
}

void InteractionElPhSVD::cacheElPh(const Eigen::MatrixXcd &eigvec1,
                                   const Eigen::Vector3d &k1C) {

  Kokkos::Profiling::pushRegion("cacheElPh_SVD");

  Kokkos::complex<double> complexI(0.0, 1.0);

  // Get dimensions
  auto nb1 = int(eigvec1.cols()); // number of bands at this kpoint
  auto elPhCached_SVD_SY = this->elPhCached_SVD_SY;
  auto SVD_SY_device = this->SVD_SY_device;

  if (mpi->mpiHead()) {
    std::cout << "SVD cacheElPh: nb1=" << nb1 << ", numGamma=" << numGamma
              << " (from SVD truncation), numPhBands=" << numPhBands
              << ", numWannierOrbitals=" << numWannierOrbitals << std::endl;
  }

  // Create intermediate/"cache" container of size (numGamma, numPhBands, numWannierOrbitals, numWannierOrbitals)
  ComplexView4D g_FT1_output(Kokkos::ViewAllocateWithoutInitializing("g1"),
                   numGamma, numPhBands, numWannierOrbitals, numWannierOrbitals);

  // Copy the eigenvector and wavevector to the accelerator
  ComplexView2D eigvec1_device("ev1", nb1, numWannierOrbitals);
  DoubleView1D k1C_device("k", 3);
  {
    // Make a "view" of the data stored on host
    HostComplexView2D eigvec1_host((Kokkos::complex<double> *)eigvec1.data(), nb1, numWannierOrbitals);
    HostDoubleView1D k1C_host((double*)k1C.data(), 3);
    // Copy the information under this view to the GPU
    Kokkos::deep_copy(eigvec1_device, eigvec1_host);
    Kokkos::deep_copy(k1C_device, k1C_host);
  }

  if (wsR1Vectors_device.data() == nullptr) {  //Introduce conditional allocation instead of always copying to device
    Kokkos::resize(wsR1Vectors_device, numWsR1Vectors, 3);
    Kokkos::resize(wsR1VectorsDegeneracies_device, numWsR1Vectors);

    // Copy from Eigen matrices to device Kokkos views
    {
      HostDoubleView2D wsR1Vectors_host((double*)elBravaisVectors.data(), numWsR1Vectors, 3);
      HostDoubleView1D wsR1VectorsDegeneracies_host((double*)elBravaisVectorsDegeneracies.data(), numWsR1Vectors);
      Kokkos::deep_copy(wsR1Vectors_device, wsR1Vectors_host);
      Kokkos::deep_copy(wsR1VectorsDegeneracies_device, wsR1VectorsDegeneracies_host);
    }
  }

  Kokkos::Profiling::pushRegion("precompute_phases");
  // Precompute phases
  ComplexView1D phases_device("phases", numWsR1Vectors);
  auto wsR1Vectors_copy = wsR1Vectors_device;
  auto wsR1VectorsDegeneracies_copy = wsR1VectorsDegeneracies_device;
  Kokkos::parallel_for(
      "phases_k1c", numWsR1Vectors, KOKKOS_LAMBDA(int irE) {
        double arg = 0.0;
        for (int j = 0; j < 3; j++) {
          arg += k1C_device(j) * wsR1Vectors_copy(irE, j);
        }
        phases_device(irE) = exp(complexI * arg) / wsR1VectorsDegeneracies_copy(irE);
      });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("fourier_transform");

  // Apply the Fourier transform to g using gemv
  size_t flattened_size = numGamma * numPhBands * numWannierOrbitals * numWannierOrbitals;

  // FIXME put back SVD_Y here 
  // We need to reshape to (R_e, gamma*eta*i*j) for gemv
  Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight> coupling_2D(
    SVD_SY_device.data(), numWsR1Vectors, flattened_size);

  // Make a view for the output of the FT
  Kokkos::View<Kokkos::complex<double> *> g_FT1_output_1D(
      g_FT1_output.data(), flattened_size);

  // Apply the gemv call: coupling_2D.T * phases = g_FT1_output
  KokkosBlas::gemv("T", Kokkos::complex<double>(1.0), coupling_2D, phases_device,
                   Kokkos::complex<double>(0.0), g_FT1_output_1D);

  if (mpi->mpiHead()) {
    std::cout << "Applied Fourier transform via gemv with flattened_size=" << flattened_size << std::endl;
    std::cout << "g_FT1_output_1D dimensions: " << g_FT1_output_1D.extent(0) << std::endl;
    std::cout << "coupling_2D dimensions: (" << coupling_2D.extent(0) << ", " << coupling_2D.extent(1) << ")" << std::endl;

    // Check if values are filled by computing norm
    auto g_FT1_output_1D_h = Kokkos::create_mirror_view(g_FT1_output_1D);
    Kokkos::deep_copy(g_FT1_output_1D_h, g_FT1_output_1D);
    double norm = 0.0;
    for (size_t i = 0; i < std::min(size_t(10), g_FT1_output_1D_h.extent(0)); ++i) {
      auto val = g_FT1_output_1D_h(i);
      norm += val.real() * val.real() + val.imag() * val.imag();
      if (i < 5) {
        std::cout << "g_FT1_output_1D[" << i << "] = " << val << std::endl;
      }
    }
    std::cout << "g_FT1_output_1D norm (first 10 elements): " << sqrt(norm) << std::endl;
  }
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("rotation_to_band_basis");
  // Rotation to Band basis - apply the eigenvector rotation U(k)_mi
  // Resize the elPhCached_SVD_SY container
  Kokkos::realloc(elPhCached_SVD_SY, numGamma, numPhBands, nb1, numWannierOrbitals);

  // Apply rotation: sum over Wannier index i
  int numWannierOrbitals_copy = numWannierOrbitals;
  Kokkos::parallel_for(
      "elPhCached_SVD_SY_device",
      Range4D({0, 0, 0, 0}, {numGamma, numPhBands, nb1, numWannierOrbitals}),
      KOKKOS_LAMBDA(int igamma, int ieta, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0.0);
        for (int iw1 = 0; iw1 < numWannierOrbitals_copy; iw1++) {
          tmp += g_FT1_output(igamma, ieta, iw1, iw2) * eigvec1_device(ib1, iw1);
        }
        elPhCached_SVD_SY(igamma, ieta, ib1, iw2) = tmp;
      });
  Kokkos::fence();

  this->elPhCached_SVD_SY = elPhCached_SVD_SY;  // Should always be on the device

  double newMemory = getDeviceMemoryUsage();
  kokkosDeviceMemory->addDeviceMemoryUsage(newMemory);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

//TODO: check below
//
// Print index 1,2,3,4 in cacheELPh
//
// implement first fourier transform (and then rotation second), but first import phases_k(check interaction.cpp)
//
// Try to set up views as previously done but matches nodes (flatten index i,j, eta).


// this function starts once elPhCached_SVD_SY is already constructed by cacheElph
// this is the first SVD matrix already transformed 
void InteractionElPhSVD::calcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3,
    const std::vector<Eigen::Vector3d> &q3Cs,
    const std::vector<Eigen::VectorXcd> &polarData) {
  
  Kokkos::Profiling::pushRegion("calcCouplingSquared");
  // set up all information used by this function 
  int numWannier = numElBands; // just used for clarity in naming 
  int nb1 = int(eigvec1.cols());
  int numK2 = int(eigvecs2.size()); // the number of k2 and q points
  
  auto elPhCached_SVD_SY = this->elPhCached_SVD_SY;  // cached coupling from last time
  int numPhBands = this->numPhBands;
  int numWsR2Vectors = this->numWsR2Vectors;
  
  // make a view of the R2 vectors (could be Rp or Re' in JDFTx case)
  DoubleView2D wsR2Vectors_device = this->wsR2Vectors_device;
  DoubleView1D wsR2VectorsDegeneracies_device = this->wsR2VectorsDegeneracies_device;

  // each k2 kpoint may have a different number of bands. 
  // therefore, we find the max number of bands at a given k2 point, 
  // and pad with zeros, because loops and views must be rectangular, not ragged
  IntView1D nb2s_device("nb2s", numK2);
  int nb2max = 0;
  auto nb2s_h = Kokkos::create_mirror_view(nb2s_device);
  // TODO put OMP on this loop 
  for (int ik = 0; ik < numK2; ik++) {
    nb2s_host(ik) = int(eigvecs2[ik].cols());
    if (nb2s_host(ik) > nb2max) {
      nb2max = nb2s_host(ik);
    }
  }
  Kokkos::deep_copy(nb2s_device, nb2s_host);
  
  // REFACTOR: should be a helper function in the parent class ==========================
  // Polar corrections are computed on the CPU and then transferred to GPU
  IntView1D usePolarCorrections_device("usePolarCorrections", numK2);
  ComplexView4D polarCorrections_device(
      Kokkos::ViewAllocateWithoutInitializing("polarCorrections"), numK2,
      numPhBands, nb1, nb2max);
  auto usePolarCorrections_host = Kokkos::create_mirror_view(usePolarCorrections);
  auto polarCorrections_host = Kokkos::create_mirror_view(polarCorrections);

  // precompute all needed polar corrections
  #pragma omp parallel for
  for (int ik = 0; ik < numK2; ik++) {

    Eigen::Vector3d q3C = q3Cs[ik];
    Eigen::MatrixXcd eigvec2 = eigvecs2[ik];
    usePolarCorrections_host(ik) = usePolarCorrection && abs(q3C.norm()) > 1.0e-8;
    if (usePolarCorrections_host(ik)) {
      Eigen::Tensor<std::complex<double>, 3> singleCorrection =
          polarCorrectionPart2(eigvec1, eigvec2, polarData[ik]);

      for (int nu = 0; nu < numPhBands; nu++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int ib2 = 0; ib2 < nb2s_host(ik); ib2++) {
            polarCorrections_host(ik, nu, ib1, ib2) =
                singleCorrection(ib1, ib2, nu);
          }
        }
      }
    } else {
      Kokkos::complex<double> kZero(0., 0.);
      for (int nu = 0; nu < numPhBands; nu++) {
        for (int ib1 = 0; ib1 < nb1; ib1++) {
          for (int ib2 = 0; ib2 < nb2s_host(ik); ib2++) {
            polarCorrections_host(ik, nu, ib1, ib2) = kZero;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(polarCorrections_device, polarCorrections_host);
  Kokkos::deep_copy(usePolarCorrections_device, usePolarCorrections_host);
  
  // copy eigenvectors etc. to device
  DoubleView2D q3Cs_k("q3", numLoops, 3);
  ComplexView3D eigvecs2Dagger_k("ev2Dagger", numLoops, numWannier, nb2max),
      eigvecs3_k("ev3", numLoops, numPhBands, numPhBands);

  {
    auto eigvecs2Dagger_h = Kokkos::create_mirror_view(eigvecs2Dagger_k);
    auto eigvecs3_h = Kokkos::create_mirror_view(eigvecs3_k);
    auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_k);

//FIXME convert these to _h -> host and _k -> device, numLoop -> numK2
// FIXME collapse(3)
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
          q3Cs_host(ik, i) =
              (q3Cs[ik](i) + k1C(i)); // k' wavevector stored here in this case
        } else {
          q3Cs_host(ik, i) = q3Cs[ik](i);
        }
      }
    }
    Kokkos::deep_copy(eigvecs2Dagger_device, eigvecs2Dagger_host);
    Kokkos::deep_copy(eigvecs3_device, eigvecs3_host);
    Kokkos::deep_copy(q3Cs_device, q3Cs_host); // kokkos, h = host is the cpu
  }
  // REFACTOR: should be a helper function in the parent class ==========================

  // REFACTOR: update this to use kokkoskernels dot, followed by kokkoskernels exp to compute a block of phases 

  // now we finish the Wannier transform. We have to do the Fourier transform
  // on the lattice degrees of freedom, and then do two rotations (at k2 and q)
  // -------------------------------------------------------------------------
  // set up the phases related to phonons
  ComplexView2D phases("phases", numK2, numWsR2Vectors);
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
  
  // perform the second Fourier transform of the data related to k2
  
  Kokkos::Profiling::pushRegion("fourier_transform");

  // Apply the Fourier transform to g using gemv
  size_t flattened_size = numGamma * numPhBands * numWannierOrbitals * numWannierOrbitals;

  // FIXME put back SVD_Vt here 
  // We need to reshape to (R_e, gamma*eta*i*j) for gemv
  Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight> coupling_2D(
    SVD_Vt_device.data(), numWsR1Vectors, flattened_size);

  // Make a view for the output of the FT
  Kokkos::View<Kokkos::complex<double> *> g_FT_output_1D(
      g_FT_output.data(), flattened_size);

  // Apply the gemv call: coupling_2D.T * phases = g_FT1_output
  KokkosBlas::gemv("T", Kokkos::complex<double>(1.0), coupling_2D, phases_device,
                   Kokkos::complex<double>(0.0), g_FT_output_1D);

  if (mpi->mpiHead()) {
    // Check if values are filled by computing norm
    auto g_FT_output_1D_h = Kokkos::create_mirror_view(g_FT1_output_1D);
    Kokkos::deep_copy(g_FT1_output_1D_h, g_FT1_output_1D);
    double norm = 0.0;
    for (size_t i = 0; i < std::min(size_t(10), g_FT_output_1D_h.extent(0)); ++i) {
      auto val = g_FT_output_1D_h(i);
      norm += val.real() * val.real() + val.imag() * val.imag();
      if (i < 5) {
        std::cout << "g_FT1_output_1D[" << i << "] = " << val << std::endl;
      }
    }
    std::cout << "g_FT1_output_1D norm (first 10 elements): " << sqrt(norm) << std::endl;
  }
  Kokkos::Profiling::popRegion();
  
  // perform the second rotation to Wannier basis of the data related to k2
  Kokkos::Profiling::pushRegion("rotation_to_band_basis");
  // Rotation to Band basis - apply the eigenvector rotation U(k)_mi
  
  // intermediate container which is the output of the transformation 
  // of the SVD_Vt object * U_wannier_nj 
  ComplexView4D elPhCached_SVD_Vt(numGamma, numPhBands, nb1, nb2max);

  // Apply rotation: sum over Wannier index j
  int numWannierOrbitals_copy = numWannierOrbitals;
  Kokkos::parallel_for(
      "wannier_transform_SVD_Vt",
      Range4D({0, 0, 0, 0}, {numGamma, numPhBands, nb1, numWannierOrbitals}),
      KOKKOS_LAMBDA(int igamma, int ieta, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0.0);
        for (int iw1 = 0; iw1 < numWannierOrbitals_copy; iw1++) {
          tmp += g_FT_output(igamma, ieta, iw1, iw2) * eigvec2_device(ib1, iw1);
        }
        elPhCached_SVD_SY(igamma, ieta, ib1, iw2) = tmp;
      });
  Kokkos::fence();
  
  // Apply rotation: sum over phonon index eta
  int numWannierOrbitals_copy = numWannierOrbitals;
  Kokkos::parallel_for(
      "phonon_transform",
      Range4D({0, 0, 0, 0}, {numGamma, numPhBands, nb1, numWannierOrbitals}),
      KOKKOS_LAMBDA(int igamma, int ieta, int ib1, int iw2) {
        Kokkos::complex<double> tmp(0.0);
        for (int iw1 = 0; iw1 < numWannierOrbitals_copy; iw1++) {
          tmp += g_FT_output(igamma, ieta, iw1, iw2) * eigvec3_device(ib1, iw1); 
        }
        elPhCached_SVD_SY(igamma, ieta, ib1, iw2) = tmp;
      });
  Kokkos::fence();
  

  // REFACTOR: Duplicate with standrd class, break it up into helper --------------------
  
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
        coupling_k(ik, nu, ib2, ib1) =
            tmp.real() * tmp.real() + tmp.imag() * tmp.imag();
      });
  Kokkos::realloc(gFinal, 0, 0, 0, 0);

  // now, copy results back to the CPU
  cacheCoupling.resize(0);
  cacheCoupling.resize(numK2);
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
  // REFACTOR: Duplicate with standrd class, break it up into helper --------------------
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
  elph_dims << ElPh_Matrix.extent(0), ElPh_Matrix.extent(1), ElPh_Matrix.extent(2),
      ElPh_Matrix.extent(3), ElPh_Matrix.extent(4);
  return elph_dims;
}

const double InteractionElPhSVD::getDeviceMemoryUsage() const {
  double bytes = 0.;
  if (ElPh_Matrix.data() != nullptr) {
    bytes += ElPh_Matrix.span() * sizeof(Kokkos::complex<double>);
  }
  return bytes;
}

int InteractionElPhSVD::estimateNumBatches(const int &nk2,
                                           const int &nb1) const {
    // Placeholder for testing.
  return 1;
}

void InteractionElPhSVD::exportElPhMatrixToHDF5(const std::string &filename) const {
  if (!mpi->mpiHead()) return;

  if (ElPh_Matrix.data() == nullptr) {
    std::cout << "ElPh_Matrix is not initialized - cannot export." << std::endl;
    return;
  }

  auto ElPh_Matrix_h = Kokkos::create_mirror_view(ElPh_Matrix);
  Kokkos::deep_copy(ElPh_Matrix_h, ElPh_Matrix);

  try {
    HighFive::File file(filename, HighFive::File::Truncate);

    // Save dimensions
    std::vector<size_t> dims = {ElPh_Matrix.extent(0), ElPh_Matrix.extent(1), ElPh_Matrix.extent(2),
                                ElPh_Matrix.extent(3), ElPh_Matrix.extent(4)};
    file.createDataSet("dimensions", dims);

    // Save real and imaginary parts as separate datasets
    size_t total_size = ElPh_Matrix.span();
    std::vector<double> real_data(total_size), imag_data(total_size);

    size_t idx = 0;
    for (size_t i = 0; i < ElPh_Matrix.extent(0); ++i) {
      for (size_t j = 0; j < ElPh_Matrix.extent(1); ++j) {
        for (size_t eta = 0; eta < ElPh_Matrix.extent(2); ++eta) {
          for (size_t R_e = 0; R_e < ElPh_Matrix.extent(3); ++R_e) {
            for (size_t R_p = 0; R_p < ElPh_Matrix.extent(4); ++R_p) {
              auto val = ElPh_Matrix_h(i, j, eta, R_e, R_p);
              real_data[idx] = val.real();
              imag_data[idx] = val.imag();
              idx++;
            }
          }
        }
      }
    }

    file.createDataSet("real", real_data);
    file.createDataSet("imag", imag_data);

    std::cout << "Exported ElPh_Matrix to: " << filename << std::endl;

  } catch (const std::exception &e) {
    std::cout << "Error exporting to HDF5: " << e.what() << std::endl;
  }
}
