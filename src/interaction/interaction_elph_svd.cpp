#include "interaction/interaction_elph_svd.h"
#include "common_kokkos.h"
#include "context.h"
#include "crystal.h"
#include "exceptions.h"
#include "harmonic/phonon_h0.h"
#include "mpi.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <iomanip> // for std::setprecision
#include <iostream>
#include <regex>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// Factory Function and Constructor
// ---------------------------------------------------------------------------

// This is the main public entry point for creating the SVD interaction object.
std::unique_ptr<InteractionElPhSVD>
InteractionElPhSVD::parse(Context &context, Crystal &crystal,
                          PhononH0 &phononH0) {

  // Check for the existence of the HDF5 file.
  std::string fileName = context.getElphFileName();
  if (mpi->mpiHead()) {
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
      Error("Required electron-phonon SVD HDF5 file \"" + fileName +
            "\" not found.");
    }
  }
  mpi->barrier();

  return std::make_unique<InteractionElPhSVD>(crystal, context, phononH0);
}

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

  // Parsing function
  parseSVDKokkos(context);
}

// ---------------------------------------------------------------------------
// Core HDF5 Parsing Logic
// ---------------------------------------------------------------------------

// Helper to parse the (i,j,eta) indices from a group name
inline std::tuple<int, int, int>
extractIndicesFromGroupName(const std::string &groupName) {
  static const std::regex r("slice_GrR_SVD_(\\d+)_(\\d+)_(\\d+)");
  std::smatch match;
  if (std::regex_match(groupName, match, r)) {
    return {std::stoi(match[1]), std::stoi(match[2]), std::stoi(match[3])};
  }
  throw std::invalid_argument("SVD group name has incorrect format: " +
                              groupName);
}

// Iterate over all HDF5 subgroups to read their U,S,V datasets
// TODO: confirm the structure of the HDF5 overall dataset. I think there is some inconsitency. CHECK!!!
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

    auto s = std::make_unique<std::vector<double>>();
    auto u = std::make_unique<std::vector<double>>();
    auto v = std::make_unique<std::vector<double>>();
    subGroup.getDataSet("S").read(*s);
    subGroup.getDataSet("U").read(*u);
    subGroup.getDataSet("V").read(*v);

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
  std::string hdf5FileName = context.getElphFileName();
  try {
    HighFive::File file(hdf5FileName, HighFive::File::ReadOnly);

    HighFive::Group svdGroup = file.getGroup("SVD");

    // Process all subgroups to determine dimensions and load raw data into RAM.
    size_t num_i, num_j, num_eta;
    std::vector<SVDGroupData> allSVDData =
        processAllSVDGroups(svdGroup, num_i, num_j, num_eta);

    if (allSVDData.empty()) {
      throw std::runtime_error("No SVD groups found in the HDF5 file.");
    }

    // Infer dimensions from the first group (assuming they are all consistent).
    size_t numSingularValues = allSVDData[0].singularVector->size();
    size_t leftMatrixCols =
        allSVDData[0].leftMatrix->size() / numSingularValues;
    size_t rightMatrixRows =
        allSVDData[0].rightMatrix->size() / numSingularValues;

    // Allocate the 5D Kokkos containers on the device.
    Kokkos::realloc(SVD_Y, num_i, num_j, num_eta, leftMatrixCols,
                    numSingularValues);
    Kokkos::realloc(SVD_Vt, num_i, num_j, num_eta, rightMatrixRows,
                    numSingularValues);

    // Create host-side mirrors to populate the data before copying to device.
    auto SVD_Y_h = Kokkos::create_mirror_view(SVD_Y);
    auto SVD_Vt_h = Kokkos::create_mirror_view(SVD_Vt);

    // Loop through the SVD data to fill the host-side Kokkos views.
    for (const auto &data : allSVDData) {
      size_t i = data.idxX;
      size_t j = data.idxY;
      size_t eta = data.idxZ;

      const auto &sVec = *data.singularVector;
      const auto &uMat = *data.leftMatrix;
      const auto &vMat = *data.rightMatrix;

      // Populate Y = U * S
      for (size_t l = 0; l < leftMatrixCols; ++l) {
        for (size_t s = 0; s < numSingularValues; ++s) {
          SVD_Y_h(i, j, eta, l, s) = Kokkos::complex<double>(
              uMat[l * numSingularValues + s] * sVec[s], 0.0);
        }
      }

      // Populate Vt = V^T
      for (size_t r = 0; r < rightMatrixRows; ++r) {
        for (size_t s = 0; s < numSingularValues; ++s) {
          SVD_Vt_h(i, j, eta, r, s) =
              Kokkos::complex<double>(vMat[r * numSingularValues + s], 0.0);
        }
      }
    }

    // Deep copy the populated host views to the device views.
    Kokkos::deep_copy(SVD_Y, SVD_Y_h);
    Kokkos::deep_copy(SVD_Vt, SVD_Vt_h);

  } catch (const HighFive::Exception &e) {
    Error("HighFive/HDF5 error while parsing SVD file: " +
          std::string(e.what()));
  }
}

// ---------------------------------------------------------------------------
// Public Inspection Methods
// ---------------------------------------------------------------------------

void InteractionElPhSVD::printSVDInfo() const {
  if (SVD_Y.data() == nullptr) {
    std::cout << "SVD containers are not initialized." << std::endl;
    return;
  }
  std::cout << "--- SVD Container Info ---" << std::endl;
  std::cout << "SVD_Y dimensions: " << SVD_Y.extent(0) << " x "
            << SVD_Y.extent(1) << " x " << SVD_Y.extent(2) << " x "
            << SVD_Y.extent(3) << " x " << SVD_Y.extent(4) << std::endl;
  std::cout << "SVD_Vt dimensions: " << SVD_Vt.extent(0) << " x "
            << SVD_Vt.extent(1) << " x " << SVD_Vt.extent(2) << " x "
            << SVD_Vt.extent(3) << " x " << SVD_Vt.extent(4) << std::endl;
  double mem = getDeviceMemoryUsage() / 1024.0 / 1024.0;
  std::cout << "Total Device Memory: " << std::fixed << std::setprecision(2)
            << mem << " MB" << std::endl;
}

void InteractionElPhSVD::printSVDSample(size_t i, size_t j, size_t eta) const {
  if (SVD_Y.data() == nullptr) {
    std::cout << "SVD containers are not initialized." << std::endl;
    return;
  }
  auto SVD_Y_h = Kokkos::create_mirror_view(SVD_Y);
  Kokkos::deep_copy(SVD_Y_h, SVD_Y);
  std::cout << "--- SVD Data Sample at (i,j,eta) = (" << i << "," << j << ","
            << eta << ") ---" << std::endl;
  std::cout << "SVD_Y(i,j,eta,0,0) = " << SVD_Y_h(i, j, eta, 0, 0) << std::endl;
}

// ---------------------------------------------------------------------------
// Virtual Function Implementations (Placeholders for Testing)
// ---------------------------------------------------------------------------

void InteractionElPhSVD::cacheElPh(const Eigen::MatrixXcd &eigvec1,
                                   const Eigen::Vector3d &k1C) {
  // Placeholder for testing.
  if (mpi->mpiHead()) {
    std::cout << "Checkpoint: InteractionElPhSVD::cacheElPh called." << std::endl;
  }
}

void InteractionElPhSVD::calcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3,
    const std::vector<Eigen::Vector3d> &q3Cs,
    const std::vector<Eigen::VectorXcd> &polarData) {

  if (mpi->mpiHead()) {
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "CHECKPOINT: SVD Interaction Module Triggered" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
  }

  if (SVD_Y.data() == nullptr || SVD_Vt.data() == nullptr) {
    Error("SVD Kokkos containers were NOT allocated. SVD parsing has failed.");
  } else {
    if (mpi->mpiHead()) {
      std::cout << "SUCCESS: 5D Kokkos containers are allocated." << std::endl;
      printSVDInfo();
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
      std::cout << "Populated coupling cache with zero-tensors to allow execution to continue." << std::endl;
      std::cout << "--------------------------------------------------\n" << std::endl;
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
  Eigen::VectorXi SVD_dims(5);
  SVD_dims << SVD_Y.extent(0), SVD_Y.extent(1), SVD_Y.extent(2),
      SVD_Y.extent(3), SVD_Y.extent(4);
  return SVD_dims;
}

const double InteractionElPhSVD::getDeviceMemoryUsage() const {
  double bytes = 0.;
  if (SVD_Y.data() != nullptr) {
    bytes += SVD_Y.span() * sizeof(Kokkos::complex<double>);
  }
  if (SVD_Vt.data() != nullptr) {
    bytes += SVD_Vt.span() * sizeof(Kokkos::complex<double>);
  }
  return bytes;
}

int InteractionElPhSVD::estimateNumBatches(const int &nk2,
                                           const int &nb1) const {
    // Placeholder for testing.
  return 1;
}
