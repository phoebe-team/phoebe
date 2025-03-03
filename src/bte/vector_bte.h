#ifndef VECTOR_BTE_H
#define VECTOR_BTE_H

#include "Matrix.h"
#include "PMatrix.h"
#include "active_bandstructure.h"
#include "context.h"
#include "eigen.h"

/** Class used to store the "vector" of out-of-equilibrium populations.
 * The vector indices are over the Bloch state. Additionally, there is one of
 * these vector for each pair of (temperatures,chemicalPotentials).
 * Can be used to store both scalar quantities (linewidths) or vectors
 * (like velocities).
 */
class VectorBTE {
public:
  /** Constructor method, initializes raw buffer data and saves helper
   * variables.
   * @param statisticsSweep: saves the info on how many temperatures/chemical
   * potentials we are evaluating.
   * @param bandStructure: saves the underlying band structure for which we
   * compute the out-of-equilibrium populations.
   * @param dimensionality: determines the size of the vector on cartesian
   * indices. 1 for scalar quantities like linewidths Gamma(BlochIndices), 3
   * for vector quantities like phonon populations f(blochIndices,cartesian).
   */
  VectorBTE(StatisticsSweep &statisticsSweep_,
            BaseBandStructure &bandStructure_, const int &dimensionality_ = 3);

  /** Copy constructor
   */
  VectorBTE(const VectorBTE &that);

  /** Copy assignment operator
   */
  VectorBTE &operator=(const VectorBTE &that);

  /** Get and set operator
   */
  double &operator()(const int &iCalc, const int &iDim, const int &iState);

  /** Const get and set operator
   */
  const double &operator()(const int &iCalc, const int &iDim,
                           const int &iState) const;

  /** Computes the scalar product between two VectorBTE objects.
   * The scalar product of x and y, is defined such as
   * z(iCalc) = sum_i x(iCalc,i) y(iCalc,i), where i is an index over Bloch
   * states, and iCalc is an index over temperatures and chemical potentials.
   * @param that: the second vector used in the scalar product
   */
  Eigen::MatrixXd dot(const VectorBTE &that);

  /** element wise product between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x*y.
   * If y has dimensionality 1, we compute x(every dim)*y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this*y
   */
  VectorBTE operator*(VectorBTE &that);

  /** Computes the product of a VectorBTE with a scalar, i.e. all elements
   * of vectorBTE x -> x * scalar.
   * @param scalar: a double with the constant factor to be used in the
   * element-wise multiplication.
   */
  VectorBTE operator*(const double &scalar);

  /** Computes the product of a VectorBTE with a vector. The vector has
   * size equal to the number of calculations (i.e. number of temperatures
   * times the number of chemical potentials) used in the run. Given a
   * calculation index iCalc, the result is an element-wise x(it)*vector(it).
   * @param vector: a double vector to be used in the product, of size
   * equal to numCalculations.
   */
  VectorBTE operator*(const Eigen::MatrixXd &vector);

  /** Computes the product of a VectorBTE with a parallel matrix. Only works
   * if the number of temperatures/chemical potentials (numCalculations) is equal
   * to one. At fixed calculation index iCalc, the result is an matrix-vector
   * multiplication x(it,i)*pMatrix(i,j).
   * @param pMatrix: a parallel distributed double matrix to be used in the
   * product, of size equal to numStates x numStates.
   */
  VectorBTE operator*(ParallelMatrix<double> &matrix);

  /** element wise sum between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x+y.
   * If y has dimensionality 1, we compute x(every dim)+y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this+y
   */
  VectorBTE operator+(VectorBTE &that);

  /** element wise difference between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x-y.
   * If y has dimensionality 1, we compute x(every dim)-y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this-y
   */
  VectorBTE operator-(VectorBTE &that);

  /** Invert the sign of the VectorBTE content i.e. x -> -x
   */
  VectorBTE operator-();

  /** element wise division between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x/y.
   * If y has dimensionality 1, we compute x(every dim)/y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this/y
   */
  VectorBTE operator/(VectorBTE &that);

  /** Replace the content of VectorBTE with its square root
   * (element-wise x -> sqrt(x) ).
   */
  VectorBTE sqrt();
  /** Replace the content of VectorBTE with its reciprocal
   * (element-wise x -> 1/x).
   */
  VectorBTE reciprocal();

  /** Convert an out-of-equilibrium population from the canonical form f to
   * the absolute value n, such that n = bose(bose+1)f or n=fermi(1-fermi)f.
   */
  void canonical2Population();

  /** Convert an out-of-equilibrium population from the absolute value n to
   * the canonical value n, such that n = bose(bose+1)f or n=fermi(1-fermi)f.
   */
  void population2Canonical();

  /** raw buffer containing the values of the vector
 *  The matrix has size (numCalculations, numStates), where numCalculations is the number
 *  of pairs of temperature and chemical potentials, and numStates is the
 *  number of Bloch states used in the Boltzmann equation.
 */
  Eigen::MatrixXd data;

  /** glob2Loc and loc2Glob compress/decompress the indices on temperature,
   * chemical potential, and cartesian direction into/from a single index.
   * TODO: these indices, and how they are used elsewhere, is rather messy
   * That's because we have to work both with quantities such as line-widths,
   * which are a scalar over the Bloch states, and phonon populations, which
   * are cartesian vectors over the Bloch states.
   * I should probably create two different classes for these.
   */
  int glob2Loc(const ChemPotIndex &imu, const TempIndex &it,
               const CartIndex &iDim) const;
  std::tuple<ChemPotIndex, TempIndex, CartIndex> loc2Glob(const int &i) const;

  /** List of Bloch states to be excluded from the calculation (i.e. for
   * which vectorBTE values are 0), for example, the acoustic modes at the
   * gamma point, whose zero frequencies may cause problems.
   */
  std::vector<int> excludeIndices;
  int dimensionality;

  /** Set the whole content (raw buffer) of BaseVectorBTE to a scalar value.
   * @param constant: the value to be used in the set.
   */
  void setConst(const double &constant);

  /** return the number of states in this vector BTE object 
  * @return numStates
  */
  int getNumStates();

 protected:
  // we store auxiliary objects and parameters
  StatisticsSweep &statisticsSweep;
  int numCalculations;
  int numStates;
  int numChemPots;
  int numTemps;

  // we store auxiliary objects and parameters
  BaseBandStructure &bandStructure;

  friend class ScatteringMatrix; // this is also to remember that
  // if we change the index order of this class, we should check the
  // ScatteringMatrix implementations: they are high efficiency methods
  // so they need low-level access to the raw buffer

  /** base class to implement +, -, / and * operations.
   * It's split separately so that subclasses can create the correct output
   * class, and also because operations are rather similar.
   */
  VectorBTE baseOperator(VectorBTE &that, const int &operatorType);
  const int operatorSums = 0;
  const int operatorDivs = 1;
  const int operatorProd = 2;
  const int operatorDiff = 3;
};

#endif
