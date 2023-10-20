#ifndef PHONON_VISCOSITY_H
#define PHONON_VISCOSITY_H

#include "drift.h"
#include "observable.h"
#include "ph_scattering_matrix.h"

/** Object to compute and store the phonon viscosity.
 */
class PhononViscosity : public Observable {
public:
  /** Object constructor. Simply stores references to the inputs
   * @param statisticsSweep: object with info on the temperature and chemical
   * potential loops
   * @param crystal: object with the crystal structure
   * @param BaseBandStructure: object with the quasiparticle energies and
   * velocities computed on a mesh of wavevectors.
   */
  PhononViscosity(Context &context_, StatisticsSweep &statisticsSweep_,
                  Crystal &crystal_, BaseBandStructure &bandStructure_);

  /** Compute the viscosity within the relaxation time approximation
   * Stores it internally.
   * @param n: the absolute phonon population.
   */
  virtual void calcRTA(VectorBTE &n);

  /** Prints the viscosity to screen for the user.
   */
  void print();

  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON(const std::string& outFileName);

  /** Outputs the quantities needed for a real space solution
   *  in hydrodynamic materials.
   */
  void outputRealSpaceToJSON(ScatteringMatrix& scatteringMatrix);

  /** Computes the viscosity from the scattering matrix eigenvectors.
   * Following Simoncelli PRX 2020.
   * @param eigenvalues: the VectorBTE object with relaxon eigenvalues.
   * @param eigenvectors: the eigenvectors of the scattering matrix above.
   */
  void calcFromRelaxons(Eigen::VectorXd &eigenvalues,
                        ParallelMatrix<double> &eigenvectors);

  /** Helper function to print information about the scalar products with the
   * special eigenvectors.
   * @param eigenvectors: eigenvectors of the scattering matrix
   * @param numRelaxons: the number of relaxons which have been calculated
   */
  void relaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors, int& numRelaxons);

  /** Helper function to pre-calculate the special eigenvectors theta0 + phi,
   * as well as A, C
   */
  void calcSpecialEigenvectors();

protected:

  int whichType() override;
  BaseBandStructure& bandStructure;
  double spinFactor = 1;
  int alpha0 = -1; // the index of the energy eigenvector, to skip it
  int alpha_e = -1; // UNUSED, no charge eigenvector here

  // theta^0 - energy conservation eigenvector
  //   electronic states = ds * g-1 * (hE - mu) * 1/(kbT^2 * V * Nkq * Ctot)
  //   phonon states = ds * g-1 * h*omega * 1/(kbT^2 * V * Nkq * Ctot)
  Eigen::VectorXd theta0;

  // UNUSED: theta^e -- the charge conservation eigenvector
  //   electronic states = ds * g-1 * 1/(kbT * U)
  Eigen::VectorXd theta_e;

  // phi -- the three momentum conservation eigenvectors
  //     phi = sqrt(1/(kbT*volume*Nkq*M)) * g-1 * ds * hbar * wavevector;
  Eigen::MatrixXd phi;

  // normalization coeff A ("phonon specific momentum")
  // A = 1/(V*Nq) * (1/kT) sum_qs (hbar*q)^2 * N(1+N)
  Eigen::Vector3d A;

  double C; // phonon specific heat

};

#endif
