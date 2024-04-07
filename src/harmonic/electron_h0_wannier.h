#ifndef WANNIER_H0_H
#define WANNIER_H0_H

#include <cmath>

#include "bandstructure.h"
#include "constants.h"
#include "eigen.h"
#include "harmonic.h"
#include "points.h"
#include "common_kokkos.h"

/** Class for diagonalizing electronic energies with the Wannier interpolation
 * The object is built passing the information produced by the file _tb.dat of
 * Wannier90, and can be used to compute the interpolated band structure.
 */
class ElectronH0Wannier : public HarmonicHamiltonian {
 public:

 /** Constructor for the electronic Hamiltonian in the Wannier case. 
  * @param directUnitCell_: the real space unit cell of the crystal in Bohr
  * @param bravaisVectors_: the R vectors for the Fourier transform, dimension (3, nRVectors)
  * @param vectorsDegeneracies_: the weights of the R vectors for the Fourier transform
  * @param h0R_: the Wannier Hamiltonian in real space 
  * @param rMatrix_:  position matrix elements <0m|r|nR>
 */
  ElectronH0Wannier(
      const Eigen::Matrix3d &directUnitCell_,
      const Eigen::MatrixXd &bravaisVectors_,
      const Eigen::VectorXd &vectorsDegeneracies_,
      const Eigen::Tensor<std::complex<double>, 3> &h0R_,
      const Eigen::Tensor<std::complex<double>, 4>* rMatrix_=nullptr);

  /** Copy constructor
   */
  ElectronH0Wannier(const ElectronH0Wannier &that);

  /** Copy assignment
   */
  ElectronH0Wannier &operator=(const ElectronH0Wannier &that);

  /** Class destructor
   */
  ~ElectronH0Wannier();

  /** Method to return that the underlying is that of an electronic Fermion.
   */
  Particle getParticle() override;

  /** get the total number of bands.
   * This is a constant for all wavevectors.
   */
  int getNumBands() override;

  /** get the electronic energies (in Ry) at a single k-point.
   * Energies don't have any reference value, and must be used in connection
   * with a chemical potential.
   * @param k: a point object with the wavevector. Must have the cartesian
   * coordinates of the wavevector.
   * @return tuple(energies, eigenvectors): the energies are a double vector
   * of size (numBands). Eigenvectors of size (numBands,numBands) are the
   * complex unitary transformation matrix U, connecting the Wannier gauge
   * with the Bloch gauge.
   */
  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> diagonalize(Point &point) override;

  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> diagonalizeFromCoordinates(
      Eigen::Vector3d &k) override;

  /** Computes energies and eigenvectors of electrons for a batch of nk
   * wavevectors
   *
   * @param cartesianWavevectors: a std::vector containing the cartesian
   * coordinates of nk wavevectors.
   * @return tuple with vectors of energies(nk,nb) and eigenvectors(nk,nb,nb).
   */
  std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXcd>>
  batchedDiagonalizeFromCoordinates(std::vector<Eigen::Vector3d>& cartesianWavevectors);

  /** Computes the Fourier transform of the Wannier Hamiltonian at a batch of
   * wavevectors.
   *
   * @param cartesianWavevectors: a std::vector containing the cartesian
   * coordinates of nk wavevectors.
   * @return a vector of hamiltonian matrices in the Bloch representation.
   */
  std::vector<Eigen::MatrixXcd> batchedBuildHamiltonians(
    std::vector<Eigen::Vector3d>& cartesianWavevectors);

  /** Computes the Fourier transform of the Wannier Hamiltonian at a batch of
   * wavevectors.
   *
   * @param cartesianCoordinates: a ComplexView2D object of size (nk,3)
   * (must already be on the GPU), containing the cartesian coordinates of a
   * batch of nk wavevectors.
   * @return a View object of size nk,nb,nb containing the nk Hamiltonian
   * matrices.
   */
  StridedComplexView3D kokkosBatchedBuildBlochHamiltonian(
    const DoubleView2D &cartesianCoordinates) override;

  /** Computes energies and eigenvectors of electrons for a batch of nk
   * wavevectors.
   *
   * @param cartesianCoordinates: a ComplexView2D object of size (nk,3)
   * (must already be on the GPU), containing the cartesian coordinates of a
   * batch of nk wavevectors.
   * @return a tuple of views of energy(nk,nb) and eigenvectors(nk,nb,nb) at
   * each wavevector.
   */
  std::tuple<DoubleView2D, StridedComplexView3D> kokkosBatchedDiagonalizeFromCoordinates(
      const DoubleView2D &cartesianCoordinates, const bool withMassScaling=true) override;
  /** Using kokkos, computes the electronic properties of a batch of wavevectors
   *
   * @param cartesianCoordinates: a ComplexView2D object of size (nk,3)
   * (must already be on the GPU), containing the cartesian coordinates of a
   * batch of nk wavevectors.
   * @return a tuple with energies(nk,nb), eigenvectors(nk,nb,nb) and
   * velocities(nk,nb,nb,3) at each wavevector.
   */
  std::tuple<DoubleView2D, StridedComplexView3D, ComplexView4D>
  kokkosBatchedDiagonalizeWithVelocities(
      const DoubleView2D &cartesianCoordinates) override;

  /** get the electron velocities (in atomic units) at a single k-point.
   * @param k: a Point object with the wavevector coordinates.
   * @return velocity(numBands,numBands,3): values of the velocity operator
   * for this state, in atomic units.
   */
  Eigen::Tensor<std::complex<double>, 3> diagonalizeVelocity(Point &point) override;
  Eigen::Tensor<std::complex<double>, 3> diagonalizeVelocityFromCoordinates(
      Eigen::Vector3d &coordinates) override;

  /** Same as diagonalizeVelocityFromCoordinates, but constructs the velocity
   * operator for a batch of wavevectors.
   *
   * @param cartesianCoordinates: of size(nK,3), contains the cartesian
   * coordinates of the wavevectors at which we want the electronic properties.
   * @return tuple, containing vectors of energies, eigenvectors and velocities
   * at each input wavevector.
   */
  std::tuple<std::vector<Eigen::VectorXd>,
             std::vector<Eigen::MatrixXcd>,
             std::vector<Eigen::Tensor<std::complex<double>, 3>>>
    batchedDiagonalizeWithVelocities(
      std::vector<Eigen::Vector3d> cartesianCoordinates);

  /** This method constructs an electron band structure.
   * @param points: the object with the list/mesh of wavevectors
   * @param withVelocities: if true, compute the electron velocity operator.
   * @param withEigenvectors: if true, stores the Wannier eigenvectors.
   * @return FullBandStructure: the band structure object containing the
   * complete electronic band structure.
   */
  FullBandStructure populate(Points &fullPoints, const bool &withVelocities,
                             const bool &withEigenvectors,
                             const bool isDistributed=false) override;

  /** Runs populate on a points list, without creating a new bandstructure object
   * this is necessary if we need energies for some non-uniform grid of points
   * currently this only used by the phonon electron scattering calculation
   * @param cartesianCoordinates: the vector with the list of wavevectors in cartesianCoords
   * @param withVelocities: if true, compute the electron velocity operator.
   * @param withEigenvectors: if true, stores the Wannier eigenvectors.
   * @return std::tuple: A tuple of std::vectors of eigen objs containing energies, velocities,
   * and eigenvectors of the electronic band structure on the provided points
   */
  std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXcd>,
           std::vector<Eigen::Tensor<std::complex<double>,3>>>
                    populate(const std::vector<Eigen::Vector3d>& cartesianCoordinates,
                             const bool &withVelocities,
                             const bool &withEigenvectors);

  /** Internal helper function for cpu diag of electron H0 */
  FullBandStructure cpuPopulate(Points &fullPoints, const bool &withVelocities,
                                const bool &withEigenvectors,
                                const bool isDistributed=false);

  /** Internal helper function for kokkos diag of electron H0 */
  std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXcd>,
        std::vector<Eigen::Tensor<std::complex<double>,3>>>
                    kokkosPopulate(const std::vector<Eigen::Vector3d>& cartesianCoordinates,
                                   const bool &withVelocities,
                                   const bool &withEigenvectors, const std::vector<int>& iks);

  /** compute the Berry connection <u_mk| nabla_k |u_nk> at arb. wavevectors.
   * @param point: the Point coordinates of the wavevector.
   * @return Berry connection: a generalized Berry connection in the form of
   * a matrix <u_mk| nabla_k |u_nk> for a fixed wavevector. The Berry
   * connection is actually just the diagonal matrix elements.
   */
  std::vector<Eigen::MatrixXcd> getBerryConnection(Point &point);

  /** Function used during parsing, to add the shifts used in the phases for
   * the Fourier interpolation. Results will be slightly more expensive but more
   * accurate, especially for very small mesh sizes.
   *
   * @param degeneracyShifts_: vector with the degeneracy of each shifted
   * bravais lattice vector
   * @param vectorsShifts_: the shifts to be added to the bravais lattice
   * vectors.
   */
  void addShiftedVectors(Eigen::Tensor<double,3> degeneracyShifts_,
                         Eigen::Tensor<double,5> vectorsShifts_);

  /** Estimate how many k-points we can compute on the GPU in one batch.
   *
   * @param withVelocity: set to true if computing also the velocity operator,
   * which requires more memory
   * @return numBatches: an estimate on how many k-point we can compute in one
   * call of the kokkosBatched functions.
   */
  int estimateBatchSize(const bool& withVelocity) override;

 protected:
  Particle particle;

  // list of lattice vectors, used for the Fourier transform from real
  // to reciprocal space
  Eigen::MatrixXd bravaisVectors;
  // count the vector degeneracies, to use symmetries
  Eigen::VectorXd vectorsDegeneracies;
  Eigen::Matrix3d directUnitCell;
  // hamiltonian matrix in real space <0m|H|nR>
  Eigen::Tensor<std::complex<double>, 3> h0R;
  // position matrix elements <0m|r|nR>
  Eigen::Tensor<std::complex<double>, 4> rMatrix;

  int numWannier;
  int numVectors;

  Eigen::Tensor<double,3> degeneracyShifts;
  Eigen::Tensor<double,5> vectorsShifts;
  bool hasShiftedVectors = false;

  ComplexView3D h0R_d;
  DoubleView3D degeneracyShifts_d;
  DoubleView5D vectorsShifts_d;
  DoubleView1D vectorsDegeneracies_d;
  DoubleView2D bravaisVectors_d;

  /** Checks the size of Device-allocated views
   *
   * @return size occupied by Kokkos views, in bytes.
   */
  double getDeviceMemoryUsage();

};

#endif
