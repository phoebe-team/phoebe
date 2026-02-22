#ifndef TRANSPORT_COEFFS_H
#define TRANSPORT_COEFFS_H

#include "scattering_matrix.h"

/** Class for computing and storing the specific heat of a crystal.
 * for either electrons or phonons, it returns:
 * C = 1/(V*N) * sum(states) * dn/dT * energy
 *   = 1/(V*N) * sum(states) * n(n+/-1) * energy^2 / kB T^2
 */
class TransportCoefficients {
public:
  /** Constructor method
   * @param statisticsSweep: a StatisticsSweep object containing information
   * on the temperature loop
   * @param bandStructure: bandStructure to use for computing specific heat
   */
  TransportCoefficients(Context &context,  StatisticsSweep &statisticsSweep,  
            Crystal& crystal,  BaseBandStructure &bandStructure);
  
  void prepareRelaxons(ScatteringMatrix& scatteringMatrix); 
  
  /* Calc relaxons transport coefficients for electron or phonon only case */
  void calcFromRelaxons(const Eigen::VectorXd &eigenvalues, ParallelMatrix<double> &eigenvectors); 
      
  /** Prints the coefficients to screen for the user.
   */
  void print();
      
  /** Outputs the viscosity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON();

  void outputRelaxonContributionsToJSON(StatisticsSweep& statisticsSweep, const Particle &particle, 
      int dimensionality, const Eigen::Tensor<double, 3> sigmaContrib, const Eigen::Tensor<double, 3> kappaContrib, 
      const Eigen::Tensor<double, 3> sigmaSContrib, std::vector<double> iiiiContrib);

protected:

  // basic characteristics 
  Context &context;
  StatisticsSweep &statisticsSweep;
  int dimensionality;
  int spinFactor;
  Crystal &crystal; // used for volume
  BaseBandStructure &bandStructure;
  Particle particle; 

  // matrix had to be in memory for this calculation.
  // therefore, we can only ever have one numCalc
  int numCalculations;

  int alpha0 = -1; // the index of the energy eigenvector, to skip it
  int alpha_e = -1; // the index of the charge eigenvector, to skip it

  // here, the first dimension will always be one, as we'll
  // only ever do this kind of calculation one T and mu value at a time
  Eigen::Tensor<double, 3> sigma, mobility, seebeck, kappa;
  //Eigen::Tensor<double, 3> alpha;
  // viscosity tensors
  Eigen::Tensor<double, 5> viscosity; //, totalViscosity;

  // theta^0 - energy conservation eigenvector
  //   electronic states = ds * g-1 * (hE - mu) * 1/(kbT^2 * V * Nkq * Ctot)
  //   phonon states = ds * g-1 * h*omega * 1/(kbT^2 * V * Nkq * Ctot)
  Eigen::VectorXd theta0;

  // theta^e -- the charge conservation eigenvector
  //   electronic states = ds * g-1 * 1/(kbT * U)
  //   phonon state = 0
  Eigen::VectorXd theta_e;

  // phi -- the three momentum conservation eigenvectors
  //     phi = sqrt(1/(kbT*volume*Nkq*M)) * g-1 * ds * hbar * wavevector;
  Eigen::MatrixXd phi;

  // normalization coeff U 
  // U = D/(V*Nk) * (1/kT) sum_km F(1-F)
  double U = 0.;

  // normalization coeff A ("specific momentum")
  // A = 1/(V*Nq) * (1/kT) sum_qs (hbar*q)^2 * N(1+N)
  Eigen::Vector3d A;

  Eigen::VectorXd specificHeat; // specific heat, indexed by numCalc
  
};

#endif
