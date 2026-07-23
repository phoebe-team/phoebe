#ifndef RELAXONS_H
#define RELAXONS_H
#include "scattering_matrix.h"

// REFACTOR use requires on the bandstructures to be el first and ph second, do this more elegantly.
// This would also be nicely resolved by the "coupledBS" object.

/** Function to output relaxons eigenvector data to HDF5 for further analysis.
 * @param eigenvectors: eigenvectors of the scattering matrix
 * @param eigenvalues: eigenvalues of the scattering matrix
 * @param bandstructures: vector of first el, then ph bandstructures. For el or ph only, just supply one.
 * @param theta0: energy eigenvector
 * @param theta_e: charge eigenvector
 * @param phi: momentum eigenvector
 * @param numRelaxonsToOutput: top N relaxons to output to file
 * @param isCoupled: flag to say if the relaxons are coupled
 */
void outputRelaxonsToHDF5(ParallelMatrix<double>& eigenvectors,
                      const Eigen::VectorXd& eigenvalues,
                      std::vector<BaseBandStructure*>& bandStructures,
                      const Eigen::VectorXd& theta0,
                      const Eigen::VectorXd& theta_e,
                      const Eigen::MatrixXd& phi,
                      int numRelaxonsToOutput = 50,
                      bool isCoupled = false
                    );

/** Helper function to print information about the scalar products with the
 * special eigenvectors.
 * @param eigenvectors: eigenvectors of the scattering matrix
 * @param specialEigenvector: the special eigenvector we are checking the overlap with
 * @param eigenvectorName: the name of the special eigenvector we are printing
 */
int relaxonEigenvectorOverlap(ParallelMatrix<double>& eigenvectors,
                                      const Eigen::VectorXd& specialEigenvector,
                                      std::string eigenvectorName);


/** Helper function to pre-calculate the special eigenvectors theta0,
  * theta_e, phi as well as A, C
  * @param bandStructure: bandstructure for either phonons or electrons
  * @param spinFactor: to account for band degeneracy
  * @param statisticsSweep: object with temperatures, chemical potentials, etc
  * @param theta0: energy conservation eigenvector
  * @param thetae: charge conservation eigenvector
  * @param phi: momentum conservation eigenvectors
  * @param C: specific heat
  * @param U:
  * @param A: specific momentum
  */
  void genericCalcSpecialEigenvectors(Context& context, BaseBandStructure& bandStructure,
                            StatisticsSweep& statisticsSweep,
                            double spinFactor,
                            Eigen::VectorXd& theta0,
                            Eigen::VectorXd& theta_e,
                            Eigen::MatrixXd& phi,
                            double& C, double& U, Eigen::Vector3d& A);

  /** Outputs the viscosity to a json file.
   * @param outFileName: string representing the name of the json file
   * @param bandStructure: bandstructure for either phonons or electrons
   * @param statisticsSweep: object with temperatures, chemical potentials, etc
   * @param theta0: energy conservation eigenvector
   * @param thetae: charge conservation eigenvector
   * @param phi: momentum conservation eigenvectors
   * @param C: specific heat
   * @param A: specific momentum
   */
   void genericOutputRealSpaceToJSON(Context& context, ScatteringMatrix& scatteringMatrix,
                                  BaseBandStructure& bandStructure,
                                  StatisticsSweep& statisticsSweep,
                                  Eigen::VectorXd& theta0,
                                  Eigen::VectorXd& theta_e,
                                  Eigen::MatrixXd& phi,
                                  double C, Eigen::Vector3d& A);

  /** Outputs the relaxon velocities to file
   * @param eigenvalues: eigenvalues of the scattering matrix
   * @param V0: energy relaxon velocity
   * @param Ve: charge relaxon velocity
   * @param Vphi: momentum relaxon velocity
   * @param particle: particle type of output relaxons
   * @param numRelaxon: maximum number of top relaxons to output
   */
  void outputRelaxonVelocitiesToHDF5(const Eigen::VectorXd& eigenvalues,
                                      const Eigen::MatrixXd& V0,
                                      const Eigen::MatrixXd& Ve,
                                      const Eigen::Tensor<double, 3>& Vphi,
                                      const Particle& particle,
                                      int numRelaxons);

  /** Outputs the out of eq population to file
   * @param eigenvectors: eigenvectors of the scattering matrix
   * @param eigenvalues: eigenvalues of the scattering matrix
   * @param bandStructure: bandstructure for either phonons or electrons
   * @param V: a relaxon velocity associated with this out-of-eq distribution
   * @param coeff: the coefficient prefactor of the specified out-of-eq distribution type. For example for phonons, it's sqrt(C/KbT^2)
   * @param kBT: temperature used in calculations of population factors
   * @param stateOffset: in the coupled case, we need to note the number of states offsetting the BTE index used in the eigenvectors
   *    and the bandstructure in the phonon case. In all cases except for the phonon bandstructure, coupled case, this is zero.
   * @param keyname: what to save the data in hdf5 as
   */
  void outputRelaxonDeltaPopToHDF5(ParallelMatrix<double>& eigenvectors,
                        const Eigen::VectorXd& eigenvalues,
                        BaseBandStructure& bandStructure,
                        const Eigen::MatrixXd& V,
                        double coeff,
                        size_t stateOffset,
                        const std::string& keyname, bool append,
                        // TODO These ones should all be separated somehow so we aren't passing them constantly
                        int dimensionality, double kBT, double mu, int numRelaxons,
                        int alpha0, int alpha_e);

#endif