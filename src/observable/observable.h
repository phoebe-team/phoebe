#ifndef OBSERVABLE_H
#define OBSERVABLE_H

#include "context.h"
#include "eigen.h"
#include "vector_bte.h"

/** Base class for computing properties depending on quasiparticle occupation
 * numbers.
 * Supports the calculation of scalar (specific heat) observables, vector
 * (polarization), rank-2 tensors (conductivity) and rank-4 (viscosity)
 * observables.
 * The dimensionality of the observables is set by Crystal -> e.g. conductivity
 * becomes a DxD tensor with D the dimensionality taken from Crystal.
 */
class Observable {
public:
  /** Object constructor.
   * Note: the object doesn't initialize the underlying tensors, which must
   * be done in every subclass.
   * @param context: object with user input
   * @param statisticsSweep: object with the loops on temperature and
   * chemical potential
   * @param crystal: object with the crystal structure.
   */
  Observable(Context &context_, StatisticsSweep &statisticsSweep_,
             Crystal &crystal_);

  /** copy constructor operator
   */
  Observable(const Observable &that);

  /** copy assignment operator
   */
  Observable &operator=(const Observable &that);

  /** Overload of the difference operator. To be used e.g. to check e.g. if a
   * property changes over iterations.
   */
  Observable operator-(const Observable &that);

  /** Computes the (Frobenius=Euclidean=L2) norm of the property.
   * @return Eigen::vectorXd: a vector with the norm, of size numCalculations. Where
   * numCalculations is the number of different temperatures/chemical potentials.
   */
  Eigen::VectorXd getNorm();

  /** Use crystal symmetries to reinforce symmetry on a 3x3 transport tensor 
   * @param transportCoeffs: the 3x3 tensor to symmetrize, which will be updated
   * NOTE: duplicate code with the one in observables... really should be merged
   **/
  void symmetrize(Eigen::Tensor<double, 3>& allTransportCoeffs);

protected:
  // save input parameters
  Context &context;
  StatisticsSweep &statisticsSweep;
  Crystal &crystal;

  // auxiliary variables, here to simplify the code
  int numChemPots;
  int numTemps;
  int dimensionality = 3;
  int numCalculations;

  // whichType, to be reimplemented in the subclasses, decides what kind of
  // property must be stored. Returns one of the const int defined below
  virtual int whichType();
  const int isScalar = 0;
  const int isVector = 1;
  const int is2Tensor = 2;
  const int is4Tensor = 3;

  // These are the actual objects used to store the observables.
  // uninitialized to avoid unnecessary memory usage.
  Eigen::VectorXd scalar;                 // e.g. specific heat
  Eigen::MatrixXd vectord;                // e.g. sound speed
  Eigen::Tensor<double, 3> tensordxd;     // e.g. conductivity
  Eigen::Tensor<double, 5> tensordxdxdxd; // e.g. viscosity

  // (de)combine the indices on chemical potentials and temperature.
  // used to reduce the number of indices of observables stored in memory.
  // We use strong typing to avoid confusing indices.
  int glob2Loc(const ChemPotIndex &imu, const TempIndex &it) const;
  std::tuple<ChemPotIndex, TempIndex> loc2Glob(const int &i) const;

  // base method for the - operator. It is separated because each subclass
  // should instantiate a dedicated object for the difference.
  void baseOperatorMinus(Observable &newObservable, const Observable &that);
};

#endif
