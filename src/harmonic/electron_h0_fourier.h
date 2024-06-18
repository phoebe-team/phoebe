#ifndef H0FOURIER_H
#define H0FOURIER_H

#include <string>

#include "bandstructure.h"
#include "harmonic.h"
#include "points.h"
#include "context.h"
#include "common_kokkos.h"

/** Class for a Fourier-like interpolation of an electronic band structure.
 * Takes the information on the band structure computed on a uniform coarse
 * grid of k-points, and interpolates it to finer grids with a plane wave based
 * method.
 */
class ElectronH0Fourier : public HarmonicHamiltonian {
 public:

  /** Constructor of the Fourier interpolation
   * This class stores a copy of the electronic band structure on the coarse
   * grid.
   * @param crystal: the crystal used in the band structure calculation
   * @param coarseBandStructure: values of the electronic bands over a full
   * grid of k-points, provided by an external (DFT) code.
   * @param cutoff: a parameter used to define the number of coefficients
   * in the plane-wave interpolation. It should be an integer >1.
   * This parameter controls that the interpolation is generated using a grid
   * of lattice vectors which is of size 2*Grid(i)*cutoff, where i is the
   * direction index and grid(i) is the size of the wavevector coarse grid.
   */
  ElectronH0Fourier(Crystal &crystal_, const Points& coarsePoints_,
                    const FullBandStructure& coarseBandStructure_, double cutoff);

  /** Method to return that the underlying is that of an electronic Fermion.
   */
  Particle getParticle() override;

  /** Get the total number of bands available at ech wavevector.
   *
   */
  int getNumBands() override;

  /** get the electronic energies (in Ry) at a single k-point.
   * Energies don't have any reference value, and must be used in connection
   * with a chemical potential.
   * @param k: a point object with the wavevector. Must have the cartesian
   * coordinates of the wavevector.
   * @return tuple(energies, eigenvectors): the energies are a double vector
   * of size (numBands). Eigenvectors of size (numBands,numBands), but are
   * simply set to zero, since there is no diagonalization happening here.
   */
  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> diagonalize(Point &point) override;

  std::tuple<Eigen::VectorXd, Eigen::MatrixXcd> diagonalizeFromCoordinates(
      Eigen::Vector3d &wavevector) override;

  /** get the electron velocities (in atomic units) at a single k-point.
   * @param k: a Point object with the wavevector coordinates.
   * @return velocity(numBands,numBands,3): values of the velocity operator
   * for this state, in atomic units. Note that the off-diagonal matrix
   * elements are set to zero, because this kind of interpolation, at the
   * moment, doesn't have any information on the off-diagonal elements.
   */
  Eigen::Tensor<std::complex<double>, 3> diagonalizeVelocity(Point &point) override;
  Eigen::Tensor<std::complex<double>, 3> diagonalizeVelocityFromCoordinates(
      Eigen::Vector3d &coordinates) override;

  /** This method constructs an electron band structure.
   * @param points: the object with the list/mesh of wavevectors
   * @param withVelocities: if true, compute the electron velocity operator.
   * @param withEigenvectors: can only be false, as there are no eigenvectors
   * with this kind of interpolation.
   * @return FullBandStructure: the band structure object containing the
   * complete electronic band structure.
   */
  FullBandStructure populate(Points &fullPoints, const bool &withVelocities,
                             const bool &withEigenvectors,
                             const bool isDistributed=false) override;

  void trimBands(Context &context, const double &minEn, const double &maxEn);
  virtual StridedComplexView3D kokkosBatchedBuildBlochHamiltonian(
      const DoubleView2D &cartesianCoordinates) override;
  virtual std::tuple<DoubleView2D, StridedComplexView3D, ComplexView4D>
  kokkosBatchedDiagonalizeWithVelocities(
      const DoubleView2D &cartesianCoordinates) override;
  std::tuple<DoubleView2D, StridedComplexView3D>
  kokkosBatchedDiagonalizeFromCoordinates(
      const DoubleView2D &cartesianCoordinates, const bool withMassScaling=true) override;
 protected:
  Crystal &crystal;
  FullBandStructure coarseBandStructure;
  Points coarsePoints;
  Particle particle;

  Eigen::MatrixXcd expansionCoefficients;

  int numBands;
  double cutoff;
  int numDataPoints;
  int numPositionVectors = 0;
  double minDistance = 10.;
  Eigen::VectorXd positionDegeneracies;
  Eigen::MatrixXd positionVectors;
  Eigen::Vector3d refWavevector;

  void setPositionVectors();
  Eigen::VectorXcd getLagrangeMultipliers(Eigen::VectorXd energies);
  Eigen::VectorXcd getCoefficients(Eigen::VectorXd energies);
  std::complex<double> getStarFunction(Eigen::Vector3d &wavevector, int &iR);
  Eigen::Vector3cd getDerivativeStarFunction(Eigen::Vector3d &wavevector,
                                             int &iR);
  double getRoughnessFunction(const Eigen::Vector3d &position) const;
  const double coefficient1 = 0.75;  // 3/4
  const double coefficient2 = 0.75;
  double getEnergyFromCoordinates(Eigen::Vector3d &wavevector, int &bandIndex);
  Eigen::Vector3d getGroupVelocityFromCoordinates(Eigen::Vector3d &wavevector,
                                             int &bandIndex);
};

#endif
