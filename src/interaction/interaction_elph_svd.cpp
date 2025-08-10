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
#include <iomanip> // for std::setprecision
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

    allData.push_back(
        SVDGroupData{std::move(s), std::move(v), std::move(u), i, j, eta});
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

  // We need to reshape to (R_e, gamma*eta*i*j) for gemv
  Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight> coupling_2D(
      ElPh_Matrix.data(), numWsR1Vectors, flattened_size);

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
//
// eigen & kokkos bound checking



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

  if (ElPh_Matrix.data() == nullptr) {
    Error("ElPh_Matrix Kokkos container was NOT allocated. SVD parsing has failed.");
  } else {
    if (mpi->mpiHead()) {
      std::cout << "5D ElPh_Matrix Kokkos container is allocated." << std::endl;
    }
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
