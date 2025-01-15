#ifndef EL_PH_INTERACTION_H
#define EL_PH_INTERACTION_H

#include <complex>

#include "constants.h"
#include "crystal.h"
#include "eigen.h"
#include "phonon_h0.h"
#include "points.h"
#include "utilities.h"
#include "context.h"
#include "common_kokkos.h"
#include <Kokkos_Core.hpp>

/** Class to handle the coupling between electron and phonons.
 * Currently implements the calculation of the diagram for the interaction
 * k + q -> k'.
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
class InteractionElPhWan {

  Crystal crystal;
  PhononH0 phononH0;
  int numPhBands, numElBands, numWsR1Vectors, numWsR2Vectors;
  std::vector<Eigen::Tensor<double, 3>> cacheCoupling;
  bool usePolarCorrection = false;
  const int phaseConvention;

  // kokkos objects for GPU accelerated elph coupling
  ComplexView4D elPhCached;
  ComplexView5D couplingWannier_k;
  DoubleView2D wsR2Vectors_k;
  DoubleView1D wsR2VectorsDegeneracies_k;
  DoubleView2D wsR1Vectors_k;
  DoubleView1D wsR1VectorsDegeneracies_k;
  std::vector<ComplexView4D::HostMirror> elPhCached_hs;

  // TODO TEMPORARY, for old interpolation method
  Eigen::MatrixXd wsR1Vectors;
  Eigen::VectorXd wsR1VectorsDegeneracies;
  Eigen::MatrixXd wsR2Vectors;
  Eigen::VectorXd wsR2VectorsDegeneracies;
  Eigen::Tensor<std::complex<double>, 4> elPhCached_old;
  Eigen::Vector3d cachedK1;
  Eigen::Tensor<std::complex<double>, 5> couplingWannier;

#ifdef MPI_AVAIL
  std::vector<MPI_Request> mpi_requests;
#endif

public:

  // phase convention options -- TODO switch to ENUM
  // Giustino uses Re, Rp for R vectors
  // JDFTx uses Re, Re' R vectors 
  static const int GiustinoPhaseConvention = 0; 
  static const int JdftxPhaseConvention = 1; 
  
  // spin types -- TODO switch to ENUM
  static const int spinNonPolarized = 1; 
  static const int spinPolarizedOrSOC = 2; 

  /** Main constructor
   * @param crystal_: object describing the crystal unit cell.
   * @param couplingWannier_: matrix elements of the electron phonon
   * interaction. A tensor of shape (iw1,iw2,imode,rPh,rEl), where iw1 iw2 are
   * indices on Wannier functions, imode is a phonon mode index in real space,
   * rPh is an index on phonon Bravais lattice vectors, and rEl is an index on
   * electronic Bravais Lattice vectors. Built such that the iw2 Wannier
   * functions are set in the origin (k2 doesn't contribute to the Fourier
   * transform).
   * @param wsR1Vectors_: list of Bravais lattice vectors used in the
   * electronic Fourier transform of the coupling.
   * @param wsR1VectorsWeights_: weights (degeneracies) of the
   * lattice vectors used in the electronic Fourier transform of the coupling.
   * @param wsR2Vectors_: list of Bravais lattice vectors used in the
   * phonon Fourier transform of the coupling.
   * @param wsR2VectorsWeights_: weights (degeneracies) of the
   * lattice vectors used in the phonon Fourier transform of the coupling.
   * @param phaseConvention_: whether to use g(Re,Rp) = convention = 0 or
   * g(Re,Re') = convention = 1, as used by JDFTx
   * @param phononH0_: the phonon dynamical matrix object. Used for
   * adding the polar interaction.
   */
  InteractionElPhWan(
    const Crystal& crystal_,
    const Eigen::Tensor<std::complex<double>, 5> &couplingWannier_,
    const Eigen::MatrixXd &wsR1Vectors_,
    const Eigen::VectorXd &wsR1VectorsDegeneracies_,
    const Eigen::MatrixXd &wsR2Vectors_,
    const Eigen::VectorXd &wsR2VectorsDegeneracies_,
    const int& phaseConvention, const PhononH0& phononH0_);

    void resetK1(); // just resets the first transform for a new cycle

  /** Almost empty constructor.
   * Used to fake the existence of a coupling with the constant relaxation time
   * approximation.
   * @param crystal_: a Crystal object
   */
  //InteractionElPhWan(Crystal &crystal_);

  /** Copy constructor
   */
  //InteractionElPhWan(const InteractionElPhWan &that);

  /** Assignment operator
   */
  //InteractionElPhWan &operator=(const InteractionElPhWan &that);

  /** Destructor
   */
  ~InteractionElPhWan();

  // Core elph interpolation functions --------------------------------

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
   * @param q3s: list of phonon wavevectors.
   * @param k1C: the electron wavevector last used by cacheElPh (only used when phaseConvention)
   * = 1, as here we need to reconstruct k2
   */
  void calcCouplingSquared(
      const Eigen::MatrixXcd &eigvec1,
      const std::vector<Eigen::MatrixXcd> &eigvecs2,
      const std::vector<Eigen::MatrixXcd> &eigvecs3,
      const std::vector<Eigen::Vector3d> &q3Cs,
      const Eigen::Vector3d &k1C,
      const std::vector<Eigen::VectorXcd> &polarData);

  void oldCalcCouplingSquared(
    const Eigen::MatrixXcd &eigvec1,
    const std::vector<Eigen::MatrixXcd> &eigvecs2,
    const std::vector<Eigen::MatrixXcd> &eigvecs3, const Eigen::Vector3d &k1C,
    const std::vector<Eigen::Vector3d> &k2Cs,
    const std::vector<Eigen::Vector3d> &q3Cs);

  /** Computes a partial Fourier transform over the k1/R_el variables.
   * @param k1C: values of the k1 cartesian coordinates over which the Fourier
   * transform is computed.
   * @param eigvec1: Wannier rotation matrix U at point k1.
   */
  void cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C);

  /** Get the coupling for the values of the wavevectors triplet (k1,k2,q3),
   * where k1 is the wavevector used at calcCoupling Squared(),
   * k2 (at index ik2) is the wavevector of the scattered electron in the
   * final state, and q3 = k2 - k1 is the phonon wavevector.
   * Note: this method only works AFTER calcCouplingSquared has been called.
   * @param ik2: index of the 2nd wavevector, aligned with the list of
   * wavevectors passed to calcCouplingSquared().
   * @return g2: a tensor of shape (nb1,nb2,numPhBands=3*numAtoms) with the
   * values of the coupling squared |g(ik1,ik2,iq3)|^2 for the el-ph transition
   * k1,q3 -> k2
   */
  Eigen::Tensor<double, 3>& getCouplingSquared(const int &ik2);

  /** Static method to initialize the class by parsing a file.
   * @param fileName: name of the file containing the coupling matrix elements
   * in real space, and the information on the lattice vectors and degeneracies.
   * @param crystal: object describing the crystal unit cell.
   * @return intElPh: an instance of InteractionElPh.
   */
  static InteractionElPhWan parse(Context &context, Crystal &crystal,
                                  PhononH0& phononH0_);

  // Helper functions --------------------------------

  /** Add polar correction to the electron-phonon coupling.
   * @param q3: phonon wavevector, in cartesian coordinates
   * @param ev1: eigenvector (rotation matrix U) at k
   * @param ev2: eigenvector (rotation matrix U) at k'
   * @param ev3: phonon eigenvector at q = k'-k
   * @return g^L: the long-range (Frohlich) component of the el-ph interaction,
   * as a tensor of shape (nb1,nb2,numPhBands)
   */
  Eigen::Tensor<std::complex<double>, 3>
  getPolarCorrection(const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
                     const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3);

  /** Estimate the memory in bytes, occupied by the kokkos Views containing
   * the coupling tensor to be interpolated.
   *
   * @return a memory estimate in bytes
   */
  double getDeviceMemoryUsage();

  /** Auxiliary function to return the shape of the electron-phonon tensor
   * @return (numWannier,numWannier,numPhModes,numElVectors,numPhVectors)
   */
  Eigen::VectorXi getCouplingDimensions();

  /** Estimate the number of batches that the list of k2 wavevectors must be
   * split into, in order to fit in memory.
   *
   * @param nk2: total number of k2 wavevectors to be split in batches.
   * @param nb1: number of bands at the k1 wavevector.
   */
  int estimateNumBatches(const int &nk2, const int &nb1);

  // Internal polar correction functions --------------------------------

  // TODO these functions need full documentation

  // functions to help with the calculation of the polar correction
  // as described in doi:10.1103/physRevLett.115.176401, Eq. 4

  // This calculates the long range V_L component of g_L, to be used
  // in the qe->Wannier transformation
  static Eigen::Tensor<std::complex<double>, 3> getPolarCorrectionStatic(
      const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
      const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3,
      const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
      const Eigen::Matrix3d &epsilon,
      const Eigen::Tensor<double, 3> &bornCharges,
      const Eigen::MatrixXd &atomicPositions,
      const Eigen::Vector3i &qCoarseMesh, const int dimensionality);

  // this function calculates V_L for the "static" case used in the bloch->Wannier transform
  static Eigen::VectorXcd polarCorrectionPart1Static(
        const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3,
        const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
        const Eigen::Matrix3d &epsilon, const Eigen::Tensor<double, 3> &bornCharges,
        const Eigen::MatrixXd &atomicPositions,
        const Eigen::Vector3i &qCoarseMesh, const int dimensionality);

  // Sets up a call to polarCorrectionPart1Static for the calculation of V_L
  // during wannier interpolation of the matrix elements
  // TODO this function is also public because of elph coupling app.
  // seems like it should be private
  Eigen::VectorXcd polarCorrectionPart1(
        const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3);

  // adds the transformation of V_L to the wannier gauge
  // regardless of the use case, this calculates <psi|e^{i(G+q).r}|psi> part of polar correction
  static Eigen::Tensor<std::complex<double>, 3> polarCorrectionPart2(
        const Eigen::MatrixXcd &ev1, const Eigen::MatrixXcd &ev2, const Eigen::VectorXcd &x);

  /** precompute the q-dependent part of the polar correction
  * @param phbandstructure: the bandstructure object containing all q-points
  *     for which this precomputation should occur
  * @return polarData: the q-dependent part of the polar correction
  */
  Eigen::MatrixXcd precomputeQDependentPolar(BaseBandStructure &phBandStructure,
		  			     const bool useMinusQ = false);

};

#endif
