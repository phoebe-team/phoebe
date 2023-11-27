#ifndef ONSAGER_H
#define ONSAGER_H

#include "bandstructure.h"
#include "context.h"
#include "crystal.h"
#include "el_scattering_matrix.h"
#include "statistics_sweep.h"
#include "vector_bte.h"
#include "vector_epa.h"
#include <nlohmann/json.hpp>

/** Class to compute the electronic transport coefficients.
 */
class OnsagerCoefficients {
public:
  /** Constructor method
   */
  OnsagerCoefficients(StatisticsSweep &statisticsSweep_, Crystal &crystal_,
                      BaseBandStructure &bandStructure_, Context &context);

  /** Copy constructor
   */
  OnsagerCoefficients(const OnsagerCoefficients &that);

  /** Copy assignment operator
   */
  OnsagerCoefficients &operator=(const OnsagerCoefficients &that);

  /** Compute the transport coefficients from the electron populations
   * @param n: the electron population out-of-equilibrium.
   */
  virtual void calcFromPopulation(VectorBTE &nE, VectorBTE &nT);

  /** Compute the transport coefficients from the symmetrized electron
   * populations $\delta \tilde{n} = (f(1-f))^{1/2}) \delta n $.
   * In practice, it just does a rescaling and calls calcFromPopulation.
   * @param n: the electron population out-of-equilibrium.
   */
  virtual void calcFromSymmetricPopulation(VectorBTE &nE, VectorBTE &nT);

  /** Compute the transport coefficients from the canonical electron populations
   * where the canonical population f is related to the population n as
   * n = f0 + f0 (1-f0) f, with f0 being the Fermi--Dirac population.
   *
   * @param f: the canonical electron population out-of-equilibrium.
   */
  void calcFromCanonicalPopulation(VectorBTE &fE, VectorBTE &fT);

  /** Prints to screen the thermal conductivity at various temperatures
   * in a a nicely formatted way.
   */
  virtual void print();

  /** Short format for printing the electrical conductivity. To be used
   * for quickly evaluate the convergence of an iterative BTE solver.
   */
  void print(const int &iter);

  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON(const std::string &outFileName);

  /** After the Onsager coefficients L_EE, L_TT, L_ET, L_TE have been computed
   * this function evaluates the transport coefficients such as electrical
   * conductivity, Seebeck and thermal conductivity.
   */
  //void calcTransportCoefficients();

  /** Evaluation of the Onsager coefficients within the EPA approximation.
   *
   * @param scatteringRates: lifetimes vs energy.
   * @param energyProjVelocity: $v^2 \rho$ as a function of energy.
   * @param energies: energies at which the previous quantities are evaluated.
   */
  void calcFromEPA(VectorEPA &scatteringRates,
                   Eigen::Tensor<double, 3> &energyProjVelocity,
                   Eigen::VectorXd &energies);

  /** This function solves the BTE in the relaxons basis, rotates the population
   * in the electron basis, and calls calcFromPopulation to compute the
   * transport coefficients.
   *
   * @param eigenvalues: eigenvalues of $\tilde{\Omega}$
   * @param eigenvectors : eigenvectors of $\tilde{\Omega}$
   * @param scatteringMatrix: $\tilde{\Omega}$
   */
  void calcFromRelaxons(Eigen::VectorXd &eigenvalues,
                        ParallelMatrix<double> &eigenvectors,
                        ElScatteringMatrix &scatteringMatrix);

  /** This function computes the electrical and thermal conductivity using the
   * variational functional (f \Omega f - b f).
   *
   * @param afE: product of scattering matrix with populations (E perturbation)
   * @param afT: product of scattering matrix with populations ($\nabla T$
   * perturbation)
   * @param fE: populations (E perturbation)
   * @param fT: populations ($\nabla T$ perturbation)
   * @param nE: diffusion vector (E perturbation)
   * @param nT: diffusion vector ($\nabla T$ perturbation)
   * @param scalingCG: to use if the conjugate gradient has a rescaling (not atm).
   */
  void calcVariational(VectorBTE &afE, VectorBTE &afT, VectorBTE &fE,
                       VectorBTE &fT, VectorBTE &nE, VectorBTE &nT,
                       VectorBTE &scalingCG);

  Eigen::Tensor<double, 3> getElectricalConductivity();
  Eigen::Tensor<double, 3> getThermalConductivity();

  /** For developer purposes, writing the df/dE * (E-mu) style terms to file */
  void writeIntegralContributions();

protected:
  StatisticsSweep &statisticsSweep;
  Crystal &crystal;
  BaseBandStructure &bandStructure;
  Context &context;

  int dimensionality;
  double spinFactor;
  int numCalculations;

  Eigen::Tensor<double, 3> sigma, seebeck, kappa, mobility;
  Eigen::Tensor<double, 3> LEE, LET, LTE, LTT;

};

#endif
