#ifndef PHONON_H0_H
#define PHONON_H0_H

#include <cmath>

#include "bandstructure.h"
#include "constants.h"
#include "crystal.h"
#include "eigen.h"
#include "harmonic.h"
#include "particle.h"
#include "points.h"
#include "common_kokkos.h"

/** class that computes phonon energies, velocities and eigenvectors.
 * First, it contains the force constants, i.e. the second derivative of the
 * total energy w.r.t. ionic displacements. Additionally, it has all the
 * infrastructure to Fourier transform this matrix, and diagonalize it to get
 * the harmonic phonon properties.
 */
class PhononH0 : public HarmonicHamiltonian {

 public:
  /** Constructor, which stores all input data.
   * @param crystal: the object with the information on the crystal structure
   * @param forceConstants: a tensor of doubles with the force constants
   * size is (3, 3, meshX, meshY, meshZ, numAtoms, numAtoms)
   * @param qCoarseGrid: the supercell/coarse qmesh of this calculation as a 3vector
   * @param bravaisVectors: the R vectors used in the harmonic phonon real space representation
   * @param weights: the R vector weights used in the Fourier transform to recip space
   */
  PhononH0(Crystal &crystal, 
           Eigen::Tensor<double, 5> &forceConstants_,
           Eigen::Vector3i& qCoarseGrid,
           const Eigen::MatrixXd& bravaisVectors_,
           const Eigen::VectorXd& weights_, const int fcRangeType);

  /** Copy constructor
   */
  PhononH0(const PhononH0 &that);

  /** Copy assignment operator
   */
  PhononH0 &operator=(const PhononH0 &that);

  /** Destructor
   */
  ~PhononH0();

  /** Returns the number of phonon bands for the crystal in consideration.
   */
  int getNumBands() override;

  /** Returns the underlying phonon-boson particle.
   */
  Particle getParticle() override;

  /** get the phonon energies (in Ry) at a single q-point.
   * @param q: a point object with the wavevector. Must know the cartesian
   * coordinates of the wavevector.
   * @return tuple(energies, eigenvectors): the energies are a double vector
   * of size (numBands=3*numAtoms). Eigenvectors are a complex tensor of
   * size (3,numAtoms,numBands). The eigenvector is rescaled by the
   * sqrt(masses) (masses in rydberg)
   */
  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> diagonalize(Point &point) override;

  /** Equivalent to diagonalize() computes phonon eigenValues/Vectors given the
   * wavevector, but the wavevector is passed by coordinates.
   * @param q: a 3d eigen vector with the cartesian coordinates of the
   * phonon wavevector.
   * @param withMassScaling: if true, rescales the eigenvectors by the
   * mass z -> z/sqrt(m)
   * @return eigenvalues: all values of phonon energies for this point.
   * @return eigenvectors: the phonon eigenvectors, in matrix form, for this
   * point.
   */
  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> diagonalizeFromCoordinates(
      Eigen::Vector3d &q, const bool &withMassScaling);
  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> diagonalizeFromCoordinates(
      Eigen::Vector3d &q) override;

  /** get the phonon velocities (in atomic units) at a single q-point.
   * @param q: a Point object with the wavevector coordinates.
   * @return velocity(numBands,numBands,3): values of the velocity operator
   * for this state, in atomic units.
   */
  Eigen::Tensor<std::complex<double>, 3> diagonalizeVelocity(Point &point) override;
  Eigen::Tensor<std::complex<double>, 3> diagonalizeVelocityFromCoordinates(
      Eigen::Vector3d &coordinates) override;

  /** This method constructs a phonon band structure.
   * @param points: the object with the list/mesh of wavevectors
   * @param withVelocities: if true, compute the phonon velocity operator.
   * @param withEigenvectors: if true, stores the phonon eigenvectors.
   * @return FullBandStructure: the band structure object containing the
   * complete phonon band structure.
   */
  FullBandStructure populate(Points &points, const bool &withVelocities,
                             const bool &withEigenvectors,
                             const bool isDistributed = false) override;
  FullBandStructure cpuPopulate(Points &points, const bool &withVelocities,
                                const bool &withEigenvectors, bool isDistributed = false);
  FullBandStructure kokkosPopulate(Points &points, const bool &withVelocities,
                                   const bool &withEigenvectors, const bool isDistributed = false);
  StridedComplexView3D kokkosBatchedBuildBlochHamiltonian(
      const DoubleView2D &cartesianCoordinates) override;
  std::tuple<DoubleView2D, StridedComplexView3D> kokkosBatchedDiagonalizeFromCoordinates(
      const DoubleView2D &cartesianCoordinates, const bool withMassScaling = true) override;
  std::tuple<DoubleView2D, StridedComplexView3D, ComplexView4D>
  kokkosBatchedDiagonalizeWithVelocities(
      const DoubleView2D &cartesianCoordinates) override;
  void kokkosBatchedScaleEigenvectors(StridedComplexView3D& eigenvectors);

  /** Returns the size of the q-point coarse grid on which the force constants
   * have been computed.
   * @return qCoarseGrid: an Eigen vector of 3 integers.
   */
  Eigen::Vector3i getCoarseGrid();

  /** Utility to convert the indices of atom basis and polarization into
   *  the index that has to be used with the eigenvector in matrix form.
   * @param iAt: atomic basis index.
   * @param iPol: polarization index (0,1,2).
   * @return k: index to be used in the phonon eigenvector.
   */
  int getIndexEigenvector(const int &iAt, const int &iPol) const;

  /** same as getIndexEigenvector, but as a static member
   * @param nAtoms: the number of atoms in the unit cell
   */
  static int getIndexEigenvector(const int &iAt, const int &iPol, const int &nAtoms);

  /** Get the static dielectric constant matrix.
   * @return dielectricMatrix: a 3x3 eigen matrix.
   */
  Eigen::Matrix3d getDielectricMatrix();

  /** Get the Born effective charges for the ions in the unit cell
   * @return bornCharges: a real tensor of shape (numAtoms,3,3) with the charges
   */
  Eigen::Tensor<double, 3> getBornCharges();

  /** Estimate how many k-points we can compute on the GPU in one batch.
   *
   * @param withVelocity: set to true if computing also the velocity operator,
   * which requires more memory
   * @return numBatches: an estimate on how many k-point we can compute in one
   * call of the kokkosBatched functions.
   */
  int estimateBatchSize(const bool& withVelocity) override;

  /** Helper function to print the dynamical matrix file to HDF5, for developer testing purposes.
   * @param qCrys: a 3d eigen vector in crystal coordinates of the phonon wavevector.
   * @param points: the points object with with q belongs to. Here, used only to convert q to cartesian internally.
  */
  void printDynToHDF5(Eigen::Vector3d& qCrys);

  // we want to designate these as medium or short range 
  // "medium" range are fcs that come from finite difference codes
  // "short" range are fcs coming from q2r
  static const int mediumRange = 0;
  static const int shortRange = 1;

protected:

  Particle particle;
  Crystal &crystal;

  bool hasDielectric = false;
  int numAtoms;
  int numBands;
  double volumeUnitCell;
  Eigen::MatrixXi atomicSpecies;
  Eigen::VectorXd speciesMasses;
  Eigen::MatrixXd atomicPositions;
  Eigen::Matrix3d dielectricMatrix;
  Eigen::Tensor<double, 3> bornCharges;
  Eigen::Vector3i qCoarseGrid;
  Eigen::Matrix3d directUnitCell;
  int dimensionality;
  int fcRangeType;  // medium or short 
  
    // TODO -- when long range correction for dim=2 has been really well checked,
  // uncomment this to activate it
  bool longRange2d = false;
  double alat;
  double alpha;

  // the R vectors on the WS cell used in the Fourier transform
  int numBravaisVectors = 0;
  Eigen::MatrixXd bravaisVectors;
  Eigen::VectorXd weights;

  // container to store the D(R) matrix/the harmonic force constants
  Eigen::Tensor<double,5> mat2R;

  Eigen::MatrixXd gVectors;
  Eigen::Tensor<double,3> longRangeCorrection1;
  const double gMax = 14.; // cutoff for ewald summation
  const double e2 = 4; 

  // kokkos members:
  DoubleView1D atomicMasses_d;
  DoubleView3D longRangeCorrection1_d;
  DoubleView2D gVectors_d;
  DoubleView2D dielectricMatrix_d;
  DoubleView3D bornCharges_d;
  DoubleView2D atomicPositions_d;
  DoubleView2D bravaisVectors_d;
  DoubleView1D weights_d;
  DoubleView3D mat2R_d;

  // private methods, used to diagonalize the Dyn matrix

  /** In wsInit, starting from the primitive crystal unit cell, we build the
   * list of bravais lattice vectors used for the phonon Fourier transform.
   */
  Eigen::Tensor<double, 5> wsInit(const Eigen::MatrixXd &unitCell,
                                  const Eigen::Matrix3d &directUnitCell,
                                  const int& nr1Big,
                                  const int& nr2Big,
                                  const int& nr3Big);


  /** Adds the long range correction to the dynamical matrix due to dipole-ion
   * interaction.
   * @param dyn : a container to which the short-range part of the dynamical matrix is added
   * @param q : the phonon wavevector at which to calculate this part of the dyn mat, cartesian coords
   */
  void addLongRangeTerm(Eigen::Tensor<std::complex<double>, 4> &dyn,
                        const Eigen::VectorXd &q);

  /** This part computes the short-range part of the dynamical matrix, which is
   * the Fourier transform of the force constants.
   * @param dyn : a container to which the short-range part of the dynamical matrix is added
   * @param q : the phonon wavevector at which to calculate this part of the dyn mat, cartesian coords
   */
  void shortRangeTerm(Eigen::Tensor<std::complex<double>, 4> &dyn,
                      const Eigen::VectorXd &q);

  /** This part computes the "medium"-range part of the dynamical matrix, which is 
   * needed for the polar correction to the dynamical matrix when finite differences are used.   
   * follows nonanal from rigid.f90 and the paper of Wang 2012
   * @param dyn : a container to which the short-range part of the dynamical matrix is added
   * @param q : the phonon wavevector at which to calculate this part of the dyn mat, cartesian coords
   */
  void addMediumRangeTerm(Eigen::Tensor<std::complex<double>, 4> &dyn,
                            const Eigen::VectorXd &q); 

  /** dynDiagonalize diagonalizes the dynamical matrix and returns eigenvalues and
   * eigenvectors.
   * @param dyn: the dynamical matrix in the shape 3,3,natoms,natoms
   */
  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> dynDiagonalize(
      Eigen::Tensor<std::complex<double>, 4> &dyn);

  /** Function analogous to nonanal_ifc in QE's matdyn -- 
  * returns f_of_q, which is a correction to be applied to the 
  * "medium range" fcs before performing the fourier transform
  * @param q: the phonon wavevector associated with the correction. 
  * @return correction to be applied to the IFCS before fourier transforming
  */
  void correctMediumRangeIFCs(Eigen::Tensor<double, 4> &fq, const Eigen::VectorXd &q); 

  /** Checks the size of Device-allocated views
   *
   * @return size occupied by Kokkos views, in bytes.
   */
  double getDeviceMemoryUsage();

};

// utility functions which are related but not necessarily connected to the class

/** Impose the acoustic sum rule on force constants and Born charges
 * @param sumRule: name of the sum rule to be used
 * @param crystal: crystal associated with the force constants
 * @param qCoarseGrid: q grid of the phonon calculation
 * @param forceConstants: force constants to be ASR'd
 * Currently supported values are akin to those from Quantum ESPRESSO
 * i.e. "simple" (for a rescaling of the diagonal elements) or "crystal"
 * (to find the closest matrix which satisfies the sum rule)
 */
void setAcousticSumRule(const std::string &sumRule, Crystal& crystal,
                        const Eigen::Vector3i& qCoarseGrid,
                        Eigen::Tensor<double, 7>& forceConstants);

/** If we read a set of second order force constants which did not come with the corresponding
 * R vectors, we need to generate the R vectors and weights, the
 * reorder these force constants into the format FC2(3,3,iAtom,jAtom,iR)
 * @param crystal: the crystal associated with the force constants
 * @param forceConstants: the force constants in the format 3,3,iRx,iRy,iRz,iAtom,jAtom
 * @param qCoarseGrid: the q grid from which the force constants were generated
 * @return: a tuple containing:
 *          1) the matrix of force constants in the format 3,3,iAtom,jAtom,iR
 *          2) the list of R vectors
 *          3) the weights of these R vectors
*/
std::tuple<Eigen::Tensor<double, 5>,Eigen::MatrixXd,Eigen::VectorXd>
          reorderHarmonicForceConstants(Crystal& crystal,
                                const Eigen::Tensor<double, 7>& forceConstants,
                                Eigen::Vector3i& qCoarseGrid);

  /** wsWeight computes the `weights`, i.e. the number of symmetry-equivalent
   * Bravais lattice vectors, that are used in the phonon Fourier transform.
   */
double wsWeight(const Eigen::VectorXd &r, const Eigen::MatrixXd &rws);

#endif

