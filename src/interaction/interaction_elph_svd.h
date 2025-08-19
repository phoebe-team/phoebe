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

using CD = Kokkos::complex<double>;
using LR = Kokkos::LayoutRight;

// Managed device views
using ComplexView1D = Kokkos::View<CD*,    LR>;
using ComplexView2D = Kokkos::View<CD**,   LR>;
using ComplexView4D = Kokkos::View<CD****, LR>;
using DoubleView1D  = Kokkos::View<double*, LR>;
using DoubleView2D  = Kokkos::View<double**,LR>;
using IntView3D     = Kokkos::View<int***, LR>;

// Unmanaged (pointer alias) views
template <class T> using U1D = Kokkos::View<T*,    LR, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class T> using U2D = Kokkos::View<T**,   LR, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class T> using U4D = Kokkos::View<T****, LR, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Host unmanaged views
template <class T> using H2D = Kokkos::View<T**, LR, Kokkos::HostSpace,
                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class T> using H1D = Kokkos::View<T*,  LR, Kokkos::HostSpace,
                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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

    InteractionElPhSVD(Context& context, Crystal& crystal, PhononH0& phononH0);
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

  // Override the pure virtual functions from InteractionElPhBase
  void cacheElPh(const Eigen::MatrixXcd &eigvec1,
                 const Eigen::Vector3d &k1C) override;
  // Override base class method - k1C is dummy variable for legacy code consistency
  void calcCouplingSquared(
      const Eigen::MatrixXcd &eigvec1,
      const std::vector<Eigen::MatrixXcd> &eigvecs2,
      const std::vector<Eigen::MatrixXcd> &eigvecs3,
      const std::vector<Eigen::Vector3d> &q3Cs, const Eigen::Vector3d &k1C,
      const std::vector<Eigen::VectorXcd> &polarData) override;

  void resetK1() override;

  const Eigen::Tensor<double, 3>& getCouplingSquared(const int &ik2) const override;

  const Eigen::VectorXi getCouplingDimensions() const override;

  double getDeviceMemoryUsage() const override;

  int estimateNumBatches(const int &nk2, const int &nb1) const override;

  // Public inspection methods to verify the container
  // void printSVDInfo() const;
  // void printSVDSample(size_t i = 0, size_t j = 0, size_t eta = 0) const;

  // Export function for Python analysis
  void exportElPhMatrixToHDF5(const std::string &filename) const;

private:
  // The core parsing routine that populates the Kokkos views.
  void parseSVDKokkos(Context &context);

  // Helper struct for temporarily holding data from HDF5 groups
  struct SVDGroupData {
    std::vector<std::complex<double>> U; // U(re,g)
    std::vector<std::complex<double>> V; // V(rp,g)
    std::vector<std::complex<double>> S; // S(g)
    int i=0, j=0, eta=0;
    int RE=0, RP=0, gamma=0;
  };

  std::vector<InteractionElPhSVD::SVDGroupData>
 processAllSVDGroups(const HighFive::Group &svdGroup,
                                          int &num_i, int &num_j, int &num_eta,
                                          int &RE, int &RP, int &maxGamma);


  private:
    // ---------- Geometry for slices and Wannier lattices ----------
    int numI = 0;                 // no. of i-slices // REFACTOR redundant, it's nWannier_i
    int numJ = 0;                 // no. of j-slices // REFACTOR redundant, it's nWannier_j
    //int numEta = 0;               // no. of phonon branches // REFACTOR this is redudant against numPhbands in parent class
    int maxGamma = 0;

    int numWsR1Vectors = 0;       // Re
    int numWsR2Vectors = 0;       // Rp
    int numWannierOrbitals = 0;

    ComplexView4D SVD_SY_device; //(i, j, eta, RE, gamma)
    ComplexView4D SVD_Vt_device; //(i, j, eta, gamma, RP)

    IntView3D gammaLen_ijk;

    DoubleView2D wsR1Vectors_device;
    DoubleView1D wsR1VectorsDegeneracies_device;
    DoubleView2D wsR2Vectors_device;
    DoubleView1D wsR2VectorsDegeneracies_device;

    ComplexView5D elPhCached_SVD_SY;
    //ComplexView5D elPhCached_SVD_Vt; // can just be defined in the cache function

    std::vector<Eigen::Tensor<double, 3>> cacheCoupling;

    Eigen::MatrixXd elBravaisVectors;
    Eigen::MatrixXd phBravaisVectors;
    Eigen::VectorXd elBravaisVectorsDegeneracies;
    Eigen::VectorXd phBravaisVectorsDegeneracies;
};


#endif // EL_PH_SVD_INTERACTION_H
