#ifndef EL_PH_INTERACTION_H
#define EL_PH_INTERACTION_H


#include "interaction_elph_base.h" // Corrected base class header name
// #include <any> // Unnecessary include
#include <complex>

#include "constants.h"
#include "crystal.h"
#include "eigen.h"
// #include "interactionBase.h"
#include "phonon_h0.h"
#include "points.h"
#include "utilities.h"
#include "context.h"
#include "common_kokkos.h"
#include <Kokkos_Core.hpp>
#include <memory>

/** Class to handle the coupling between electron and phonons.
 * Currently implements the calculation of the diagram for the interaction
 * k+q -> k'.
 * Use the static method to initialize an instance of this class.
 * Then, use calc + get to compute and retrieve the values of the
 * electron-phonon interaction strength in Bloch space |g|^2.
 *
 * This class starts from the interaction matrix elements in real space Wannier
 * representation and mostly does:
 * 1) a double Fourier transform on phonon and electron coordinates;
 * 2) multiply by phonon/electron eigenvectors/rotation matrices;
 * 3) for polar materials, adds the long-range Frohlich interaction.
 */
// TODO: add flag to let user decide whether to use or not polar corrections
class InteractionElPhWan: public InteractionElPhBase {

private:

    ComplexView4D elPhCached;
    ComplexView5D couplingWannier_k;

public:

    // InteractionElPhWan() = default; // Default constructor
    InteractionElPhWan(Crystal &crystal_, PhononH0 *phononH0_ = nullptr)
            : InteractionElPhBase(crystal_, phononH0_) {}

    /** Add polar correction to the electron-phonon coupling.
    * @param q3: phonon wavevector, in cartesian coordinates
    * @param ev1: eigenvector (rotation matrix U) at k
    * @param ev2: eigenvector (rotation matrix U) at k'
    * @param ev3: phonon eigenvector at q = k'-k
    * @return g^L: the long-range (Frohlich) component of the el-ph interaction,
    * as a tensor of shape (nb1,nb2,numPhBands)
    */
/*     Eigen::Tensor<std::complex<double>, 3>
    getPolarCorrection(const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
                        const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3);
 */
    /** Estimate the memory in bytes, occupied by the kokkos Views containing
    * the coupling tensor to be interpolated.
    *
    * @return a memory estimate in bytes
    */
    double getDeviceMemoryUsage() const override; // Added const

    // Override for getCouplingSquared from base class
    const Eigen::Tensor<double, 3>& getCouplingSquared(const int &ik2) const override;

    // Override for resetK1 from base class
    void resetK1() override;

    std::vector<ComplexView4D::HostMirror> elPhCached_hs;
#ifdef MPI_AVAIL
    std::vector<MPI_Request> mpi_requests;
#endif

public:

  /** Default constructor
   * @param crystal_: object describing the crystal unit cell.
   * @param couplingWannier_: matrix elements of the electron phonon
   * interaction. A tensor of shape (iw1,iw2,imode,rPh,rEl), where iw1 iw2 are
   * indices on Wannier functions, imode is a phonon mode index in real space,
   * rPh is an index on phonon Bravais lattice vectors, and rEl is an index on
   * electronic Bravais Lattice vectors. Built such that the iw2 Wannier
   * functions are set in the origin (k2 doesn't contribute to the Fourier
   * transform).
   * @param elBravaisVectors_: list of Bravais lattice vectors used in the
   * electronic Fourier transform of the coupling.
   * @param elBravaisVectorsWeights_: weights (degeneracies) of the
   * lattice vectors used in the electronic Fourier transform of the coupling.
   * @param phBravaisVectors_: list of Bravais lattice vectors used in the
   * phonon Fourier transform of the coupling.
   * @param phBravaisVectorsWeights_: weights (degeneracies) of the
   * lattice vectors used in the phonon Fourier transform of the coupling.
   * @param phononH0_: pointer to the phonon dynamical matrix object. Used for
   * adding the polar interaction.
   */
  InteractionElPhWan(
      Crystal &crystal_,
      const Eigen::Tensor<std::complex<double>, 5> &couplingWannier_,
      const Eigen::MatrixXd &elBravaisVectors_,
      const Eigen::VectorXd &elBravaisVectorsDegeneracies_,
      const Eigen::MatrixXd &phBravaisVectors_,
      const Eigen::VectorXd &phBravaisVectorsDegeneracies_,
      PhononH0 *phononH0_ = nullptr);

  /** Almost empty constructor.
   * Used to fake the existence of a coupling with the constant relaxation time
   * approximation.
   * @param crystal_: a Crystal object
   */
  InteractionElPhWan(Crystal &crystal_);

  /** Copy constructor
   */
  InteractionElPhWan(const InteractionElPhWan &that);

  /** Assignment operator
   */
  InteractionElPhWan &operator=(const InteractionElPhWan &that);

  /** Destructor
   */
  ~InteractionElPhWan();

  /** Computes the values of the el-ph coupling strength for transitions of
   * type k1,q3 -> k2, where k1 is one fixed wavevector, and k2,q3 are
   * wavevectors running in lists of wavevectors.
   * It is assumed that the relation (k2 = k1 + q3) holds.
   *
   * The call to this function must be preceded by a call to cacheElPh(),
   * which does a precomputation at fixed value of k1.
   * If not, results will be wrong.
   * Hence, structure a code calling this functions as:
   * for k1:
   *   cacheElPh(k1)
   *   for k2:
   *     k3 = k2 - k1
   *     calcCouplingSquared(k2,k3)
   *
   * Note: this method must be used in conjunction with getCouplingSquared,
   * which is used to return the values computed here.
   * @param el1Eigenvec: electron eigenvector matrix U_{mb}(k1), where U is
   * obtained by diagonalizing the Wannier Hamiltonian.
   * @param el2Eigenvecs: vector of electron eigenvectors matrix U_{mb}(k2),
   * where U is obtained by diagonalizing the Wannier Hamiltonian at a bunch
   * of k2 wavevectors.
   * @param phEigvecs: phonon eigenvectors, in matrix form, for a bunch of
   * wavevectors q3
   * @param k1: value of first wavevector.
   * @param k2s: list of k2 wavevectors.
   * @param q3s: list of phonon wavevectors.
   */
  void calcCouplingSquared(
      const Eigen::MatrixXcd &eigvec1,
      const std::vector<Eigen::MatrixXcd> &eigvecs2,
      const std::vector<Eigen::MatrixXcd> &eigvecs3,
      const std::vector<Eigen::Vector3d> &q3Cs,
      const std::vector<Eigen::VectorXcd> &polarData) override;

  /** Computes a partial Fourier transform over the k1/R_el variables.
   * @param k1C: values of the k1 cartesian coordinates over which the Fourier
   * transform is computed.
   * @param eigvec1: Wannier rotation matrix U at point k1.
   */
  void cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C) override;

  /** Static method to initialize the class by parsing a file.
   * @param fileName: name of the file containing the coupling matrix elements
   * in real space, and the information on the lattice vectors and degeneracies.
   * @param crystal: object describing the crystal unit cell.
   * @return intElPh: an instance of InteractionElPh.
   */
   static std::shared_ptr<InteractionElPhWan> parse(Context &context, Crystal &crystal,
                                                 PhononH0 *phononH0_ = nullptr);

   /** This function creates an object of type InteractionElPhWan and wraps it in
   a std::shared_ptr in order to return to the base class.
   * @param crystal: object describing the crystal unit cell. (Necessary here to match with the
   * constructor of the class)
   * @return std::shared_ptr<InteractionElPhWan>: The caller can use this pointer
   to access the instance without worrying about memory management.
   */
   // static std::shared_ptr<InteractionElPhWan> createInstance( Crystal &crystal,
   //                                               PhononH0 *phononH0_ = nullptr) {
   //     return std::make_shared<InteractionElPhWan>(crystal,
   //                                                   phononH0_);
   // }

  /** Auxiliary function to return the shape of the electron-phonon tensor
   * @return (numWannier,numWannier,numPhModes,numElVectors,numPhVectors)
   */
  const Eigen::VectorXi getCouplingDimensions() const override;

  /** Estimate the number of batches that the list of k2 wavevectors must be
   * split into, in order to fit in memory.
   *
   * @param nk2: total number of k2 wavevectors to be split in batches.
   * @param nb1: number of bands at the k1 wavevector.
   */
   int estimateNumBatches(const int &nk2, const int &nb1) const override;

};

#endif
