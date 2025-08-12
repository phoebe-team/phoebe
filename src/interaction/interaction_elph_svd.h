#ifndef EL_PH_SVD_INTERACTION_H
#define EL_PH_SVD_INTERACTION_H

#include "interaction_elph_base.h"
#include "common_kokkos.h"
#include "constants.h"
#include "eigen.h"
#include <memory>
#include <string>
#include <vector>

// Forward-declare classes to reduce header dependencies
class Context;
class Crystal;
class PhononH0;
namespace HighFive {
class Group;
}

/**
 * @class InteractionElPhSVD
 * @brief An implementation of the electron-phonon interaction using
 *        a Singular Value Decomposition (SVD) of the coupling matrix.
 *
 * This class inherits from InteractionElPhBase and is responsible for
 * reading pre-computed SVD data from an HDF5 file and reconstructing
 * the electron-phonon coupling on the fly. It is created via its
 * static `parse` method.
 */
class InteractionElPhSVD : public InteractionElPhBase {
public:
  /**
   * @brief Factory function to parse an HDF5 file and create the SVD
   * interaction object.
   *
   * This is the main, standalone entry point for this class. It reads the HDF5
   * file containing the SVD data, populates the internal 5D Kokkos containers,
   * and returns an object ready for calculations.
   *
   * @param context The application context, containing the path to the HDF5 file.
   * @param crystal The crystal structure object.
   * @param phononH0 The phonon Hamiltonian object (passed by reference).
   * @return A unique_ptr to the created InteractionElPhSVD object.
   */
  // static std::unique_ptr<InteractionElPhSVD> parse(Context &context,
  //                                                  Crystal &crystal,
  //                                                  PhononH0 &phononH0);

  // Override the pure virtual functions from InteractionElPhBase
  void cacheElPh(const Eigen::MatrixXcd &eigvec1,
                 const Eigen::Vector3d &k1C) override;
  void calcCouplingSquared(
      const Eigen::MatrixXcd &eigvec1,
      const std::vector<Eigen::MatrixXcd> &eigvecs2,
      const std::vector<Eigen::MatrixXcd> &eigvecs3,
      const std::vector<Eigen::Vector3d> &q3Cs,
      const std::vector<Eigen::VectorXcd> &polarData) override;
  void resetK1() override;
  const Eigen::Tensor<double, 3> &
  getCouplingSquared(const int &ik2) const override;
  const Eigen::VectorXi getCouplingDimensions() const override;
  const double getDeviceMemoryUsage() const override;
  int estimateNumBatches(const int &nk2, const int &nb1) const override;

  // Public inspection methods to verify the container
  void printSVDInfo() const;
  void printSVDSample(size_t i = 0, size_t j = 0, size_t eta = 0) const;

  // Export function for Python analysis
  void exportElPhMatrixToHDF5(const std::string &filename) const;

public: // Public constructor to be accessible by std::make_unique
  /**
   * @brief Constructor for InteractionElPhSVD.
   * @note Users should prefer the static `parse()` factory function to create
   * instances of this class.
   */
  InteractionElPhSVD(Crystal &crystal, Context &context, PhononH0 &phononH0);


 // void parse(Context &context);

private:
  // The core parsing routine that populates the Kokkos views.
  void parseSVDKokkos(Context &context);

  // Helper struct for temporarily holding data from HDF5 groups
  struct SVDGroupData {
    std::unique_ptr<std::vector<double>> singularVector;
    std::unique_ptr<std::vector<double>> rightMatrix;
    std::unique_ptr<std::vector<double>> leftMatrix;
    int idxX, idxY, idxZ;
  };
  std::vector<SVDGroupData>
  processAllSVDGroups(const HighFive::Group &svdGroup, size_t &num_i,
                      size_t &num_j, size_t &num_eta);

  ComplexView5D ElPh_Matrix;  // Reconstructed from SVD: (i, j, eta, R_e, R_p)

  // Store Bravais vectors and degeneracies locally.
  Eigen::MatrixXd elBravaisVectors;
  Eigen::MatrixXd phBravaisVectors;
  Eigen::VectorXd elBravaisVectorsDegeneracies;
  Eigen::VectorXd phBravaisVectorsDegeneracies;

  //kokkos object
  ComplexView4D SVD_SY_device;
  ComplexView4D SVD_Vt_device;

  ComplexView4D elPhCached_SVD_SY;
  DoubleView2D wsR1Vectors_device;  // Electronic Bravais vectors
  DoubleView1D wsR1VectorsDegeneracies_device;  // Electronic degeneracies
  DoubleView2D wsR2Vectors_device;  // Phonon Bravais vectors
  DoubleView1D wsR2VectorsDegeneracies_device;  // Phonon degeneracies
  std::vector<ComplexView4D::HostMirror> elPhCached_host;

  // Variables needed for SVD cacheElPh
  int numGamma;  // Number of retained singular values from SVD truncation
  int numWsR1Vectors;  // Number of electronic Wannier vectors
  int numWsR2Vectors;  // Number of phonon Wannier vectors
  int numWannierOrbitals;  // Total number of Wannier orbitals
};


#endif // EL_PH_SVD_INTERACTION_H
