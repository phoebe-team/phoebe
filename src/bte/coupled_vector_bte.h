#ifndef COUPLED_VECTOR_BTE_H
#define COUPLED_VECTOR_BTE_H

#include "Matrix.h"
#include "PMatrix.h"
#include "vector_bte.h"
#include "active_bandstructure.h"
#include "context.h"
#include "eigen.h"

/**
 * A version of the vectorBTE used to store rta linewidths in the CBTE case.
 * For now only to be used with relaxons, so some functionality outside that purpose
 * is blocked, though it could be opened in the future if needed.
 */
class CoupledVectorBTE : public VectorBTE {
public:
  /** Constructor method, initializes raw buffer data and saves helper
   * variables.
   * @param statisticsSweep: saves the info on how many temperatures/chemical
   * potentials we are evaluating.
   * @param phBandStructure: phonon bandstructure
   * @param elBandStructure: electron bandstructure (saved to underlying
   *   VBTE bandstructure object)
   * @param dimensionality: determines the size of the vector on cartesian
   * indices. 1 for scalar quantities like linewidths Gamma(BlochIndices), 3
   * for vector quantities like phonon populations f(blochIndices,cartesian).
   */
  CoupledVectorBTE(StatisticsSweep &statisticsSweep_, BaseBandStructure &phBandStructure_,
            BaseBandStructure &elBandStructure_, const int &dimensionality_ = 3);

  //CoupledVectorBTE(const VectorBTE &that);

  /* Note: Get and set operators taken from underlying vectorBTE object */

  /** Copy constructor */
  //CoupledVectorBTE(const CoupledVectorBTE &that);

  /** Copy assignment operator */
  //CoupledVectorBTE &operator=(const CoupledVectorBTE &that);

  /** Get and set operator */
  //double &operator()(const int &iCalc, const int &iDim, const int &iState);

  /** Const get and set operator */
  //const double &operator()(const int &iCalc, const int &iDim, const int &iState) const;

  /** Computes the scalar product between two VectorBTE objects.
   * The scalar product of x and y, is defined such as
   * z(iCalc) = sum_i x(iCalc,i) y(iCalc,i), where i is an index over Bloch
   * states, and iCalc is an index over temperatures and chemical potentials.
   * @param that: the second vector used in the scalar product
   */
  Eigen::MatrixXd dot(const CoupledVectorBTE &that);

  /** element wise product between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x*y.
   * If y has dimensionality 1, we compute x(every dim)*y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this*y
   */
  CoupledVectorBTE operator*(CoupledVectorBTE &that);

  /** Computes the product of a VectorBTE with a scalar, i.e. all elements
   * of vectorBTE x -> x * scalar.
   * @param scalar: a double with the constant factor to be used in the
   * element-wise multiplication.
   */
  CoupledVectorBTE operator*(const double &scalar);

  /** Computes the product of a VectorBTE with a vector. The vector has
   * size equal to the number of calculations (i.e. number of temperatures
   * times the number of chemical potentials) used in the run. Given a
   * calculation index iCalc, the result is an element-wise x(it)*vector(it).
   * @param vector: a double vector to be used in the product, of size
   * equal to numCalculations.
   */
  CoupledVectorBTE operator*(const Eigen::MatrixXd &vector);

  /** Computes the product of a VectorBTE with a parallel matrix. Only works
   * if the number of temperatures/chemical potentials (numCalculations) is equal
   * to one. At fixed calculation index iCalc, the result is an matrix-vector
   * multiplication x(it,i)*pMatrix(i,j).
   * @param pMatrix: a parallel distributed double matrix to be used in the
   * product, of size equal to numStates x numStates.
   */
  CoupledVectorBTE operator*(ParallelMatrix<double> &matrix);

  /** element wise sum between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x+y.
   * If y has dimensionality 1, we compute x(every dim)+y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this+y
   */
  CoupledVectorBTE operator+(CoupledVectorBTE &that);

  /** element wise difference between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x-y.
   * If y has dimensionality 1, we compute x(every dim)-y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this-y
   */
  CoupledVectorBTE operator-(CoupledVectorBTE &that);

  /** Invert the sign of the VectorBTE content i.e. x -> -x
   */
  CoupledVectorBTE operator-();

  /** element wise division between two VectorBTE objects x and y.
   * If the dimensionality of the two objects is the same, we compute
   * element-wise result = x/y.
   * If y has dimensionality 1, we compute x(every dim)/y(0), and the result
   * has the dimensionality of x.
   * @param that: the second VectorBTE object y, such that result = *this/y
   */
  CoupledVectorBTE operator/(CoupledVectorBTE &that);

  /** Replace the content of VectorBTE with its square root
   * (element-wise x -> sqrt(x) ).
   */
  CoupledVectorBTE sqrt();
  /** Replace the content of VectorBTE with its reciprocal
   * (element-wise x -> 1/x).
   */
  CoupledVectorBTE reciprocal();

  /** Convert an out-of-equilibrium population from the canonical form f to
   * the absolute value n, such that n = bose(bose+1)f or n=fermi(1-fermi)f.
   */
  void canonical2Population();

  /** Convert an out-of-equilibrium population from the absolute value n to
   * the canonical value n, such that n = bose(bose+1)f or n=fermi(1-fermi)f.
   */
  void population2Canonical();

  // TODO why isn't this private?
  /** raw buffer containing the values of the vector
 *  The matrix has size (numCalculations, numStates), where numCalculations is the number
 *  of pairs of temperature and chemical potentials, and numStates is the
 *  number of Bloch states used in the Boltzmann equation.
 */
//  Eigen::MatrixXd data;

  /** glob2Loc and loc2Glob compress/decompress the indices on temperature,
   * chemical potential, and cartesian direction into/from a single index.
   * Indexing functions inhereted directly from VectorBTE parent */
  //int glob2Loc(const ChemPotIndex &imu, const TempIndex &it, const CartIndex &iDim) const;
  //std::tuple<ChemPotIndex, TempIndex, CartIndex> loc2Glob(const int &i) const;

  /** List of Bloch states to be excluded from the calculation (i.e. for
   * which vectorBTE values are 0), for example, the acoustic modes at the
   * gamma point, whose zero frequencies may cause problems.
   */
  // Inherited
  //std::vector<int> excludeIndices;
  //int dimensionality;

  /** Set the whole content (raw buffer) of BaseVectorBTE to a scalar value.
   * @param constant: the value to be used in the set.
   */
  void setConst(const double &constant);

 protected:

  // Note: we use the elBandStructure to be
  // equal to the bandstructure in the underlying parent
  // vectorBTE object
  //BaseBandStructure &elBandStructure;
  BaseBandStructure &phBandStructure;

  friend class ScatteringMatrix; // this is also to remember that
  // if we change the index order of this class, we should check the
  // ScatteringMatrix implementations: they are high efficiency methods
  // so they need low-level access to the raw buffer

  /** base class to implement +, -, / and * operations.
   * It's split separately so that subclasses can create the correct output
   * class, and also because operations are rather similar.
   */
  CoupledVectorBTE baseOperator(CoupledVectorBTE &that, const int &operatorType);

};

#endif
