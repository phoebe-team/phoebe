#ifndef COUPLED_COEFFS_H
#define COUPLED_COEFFS_H

#include "bandstructure.h"
#include "context.h"
#include "coupled_scattering_matrix.h"
#include "statistics_sweep.h"
#include "specific_heat.h"
#include <eigen.h>

/** Calc transport coefficients from coupled scattering matrix  */
class CoupledCoefficients {
public:

  /** Constructor method
   */
  CoupledCoefficients(StatisticsSweep& statisticsSweep_,
                                       Crystal &crystal_,
                                       Context &context_);

  /** Prints to screen the thermal conductivity at various temperatures
   * in a a nicely formatted way.
   */
  virtual void print();

  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON(const std::string &outFileName);

  /** This function solves the BTE in the relaxons basis, rotates the population
   * in the electron basis, and calls calcFromPopulation to compute the
   * transport coefficients and viscosity.
   *
   * @param eigenvalues: eigenvalues of $\tilde{\Omega}$
   * @param eigenvectors : eigenvectors of $\tilde{\Omega}$
   * @param scatteringMatrix: $\tilde{\Omega}$
   */
  void calcFromRelaxons(CoupledScatteringMatrix& scatteringMatrix,
                      SpecificHeat& phSpecificHeat,
                      SpecificHeat& elSpecificHeat,
                      Eigen::VectorXd& eigenvalues,
                      ParallelMatrix<double>& eigenvectors);

  void relaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors,
                        int& numRelaxons, int& numPhStates,
                        Eigen::VectorXd& theta0, Eigen::VectorXd& theta_e);

  /** Calculate Du(i,j) and output it to JSON for real space solvers
  * @param coupledScatteringMatrix: the scattering matrix, before diagonalization
  * @param context: contains all info about calculation inputs
  */
  void outputDuToJSON(CoupledScatteringMatrix& coupledScatteringMatrix, Context& context);

  /** Calculates the special eigenvectors and saves them to the class objects for
   * theta0, thetae, phi, as well as norm coeffs U, G, and A
   * @param phononBandStructure: the phonon bandstructure from the coupled calculation
   * @param electronBandStructure: the electron bandstructure of the coupled calculation
   */
  void calcSpecialEigenvectors(StatisticsSweep& statisticsSweep,
        BaseBandStructure* phBandStructure, BaseBandStructure* elBandStructure);


protected:

  StatisticsSweep &statisticsSweep;
  Crystal &crystal;
  Context &context;

  int dimensionality;
  double spinFactor;
  // matrix had to be in memory for this calculation.
  // therefore, we can only ever have one numCalc
  int numCalculations;

  int alpha0 = -1; // the index of the energy eigenvector, to skip it
  int alpha_e = -1; // the index of the charge eigenvector, to skip it

  // here, the first dimension will always be one, as we'll
  // only ever do this kind of calculation one T and mu value at a time
  Eigen::Tensor<double, 3> sigma, mobility, sigmaTotal, mobilityTotal;
  Eigen::Tensor<double, 3> seebeckSelf, seebeckDrag, seebeck, seebeckTotal;
  Eigen::Tensor<double, 3> alphaEl, alphaPh, alpha;
  Eigen::Tensor<double, 3> kappaEl, kappaPh, kappaDrag, kappa, kappaTotal;
  // viscosity tensors
  Eigen::Tensor<double, 5> phViscosity, elViscosity, dragViscosity, totalViscosity;

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

  // normalization coeff U (summed up below)
  // U = D/(V*Nk) * (1/kT) sum_km F(1-F)
  double U = 0;
  // normalization coeff G ("electron specific momentum")
  // G = D/(V*Nk) * (1/kT) sum_km (hbar*k)^2 * F(1-F)
  Eigen::Vector3d G;
  // normalization coeff A ("phonon specific momentum")
  // A = 1/(V*Nq) * (1/kT) sum_qs (hbar*q)^2 * N(1+N)
  Eigen::Vector3d A;
  // M = G + A
  Eigen::Vector3d M; // = G + A;

  double Ctot; //the total electron and phonon specific heats, Cph + Cel
  double Cph, Cel; // phonon and electron specific heats

  /** Helper function to simplify outputing 3x3 transport tensors to json
  */
  void appendTransportTensorForOutput(Eigen::Tensor<double, 3>& tensor,
                        double& unitConv, int& iCalc,
                        std::vector<std::vector<std::vector<double>>>& outFormat);

};

#endif
