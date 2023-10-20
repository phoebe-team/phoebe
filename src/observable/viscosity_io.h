#ifndef VISCOSITY_IO_H
#define VISCOSITY_IO_H
#include "scattering.h"

  /** Prints the viscosity tensor to std out
   * @param viscosityName: string which is used to print the name of the viscosity tensor
   */
  void printViscosity(std::string& viscosityName,
        Eigen::Tensor<double, 5>& viscosityTensor, StatisticsSweep& statisticsSweep, int& dimensionality);

  /** Outputs the viscosity to a json file.
   * @param outFileName: string representing the name of the json file
   * @param viscosityName: string which is used to as the key in the json file for viscosity tensor
   * @param viscosityTensor: a tensor with indices (iCalc,i,j,k,l)
   * @param append : boolean to tell if this should be appended to an existing file
   * @param statisticsSweep: object containing temperature, chemPot, etc info
   * @param dimensionality: the dimension of the crystal
   */
   void outputViscosityToJSON(const std::string& outFileName, const std::string& viscosityName,
                Eigen::Tensor<double, 5>& viscosityTensor, const bool& append,
                StatisticsSweep& statisticsSweep, int& dimensionality);

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
   void genericOutputRealSpaceToJSON(ScatteringMatrix& scatteringMatrix,
                                BaseBandStructure& bandStructure,
                                StatisticsSweep& statisticsSweep,
                                Eigen::VectorXd& theta0,
                                Eigen::VectorXd& theta_e,
                                Eigen::MatrixXd& phi,
                                double& C, Eigen::Vector3d& A,
                                Context& context);

  /** Helper function to print information about the scalar products with the
   * special eigenvectors.
   * @param eigenvectors: eigenvectors of the scattering matrix
   * @param numRelaxons: the number of relaxons which have been calculated
   * @param particle: particle type, ph or el
   * @param theta0: energy conservation eigenvector
   * @param thetae: charge conservation eigenvector
   * @param alpha0: eigenvalue index of theta0 eigenvector
   * @param alphae: eigenvalue index of thetae eigenvector
   */
  void genericRelaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors,
                                int& numRelaxons, Particle& particle,
                                Eigen::VectorXd& theta0,
                                Eigen::VectorXd& theta_e,
                                int& alpha0, int& alpha_e);

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
   void genericCalcSpecialEigenvectors(BaseBandStructure& bandStructure,
                              StatisticsSweep& statisticsSweep,
                              double& spinFactor,
                              Eigen::VectorXd& theta0,
                              Eigen::VectorXd& theta_e,
                              Eigen::MatrixXd& phi,
                              double& C, Eigen::Vector3d& A);

#endif
