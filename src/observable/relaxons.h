#ifndef RELAXONS_H
#define RELAXONS_H
#include "scattering_matrix.h"

// TODO use requires on the bandstructures to be el first and ph second, do this more elegantly. 
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
  * @param A: specific momentum
  */
  void genericCalcSpecialEigenvectors(Context& context, BaseBandStructure& bandStructure,
                            StatisticsSweep& statisticsSweep,
                            double spinFactor,
                            Eigen::VectorXd& theta0,
                            Eigen::VectorXd& theta_e,
                            Eigen::MatrixXd& phi,
                            double C, Eigen::Vector3d& A);         
                            
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

#endif 