#ifndef MATRIX_H
#define MATRIX_H

#include "PMatrix.h"
#include "SMatrix.h"

/** Container class which wraps an underlying serial or parallel matrix
 *
 * The class hierarchy is set up using this container rather than an inheritance
 * structure with specific intent. These objects cannot be set up using
 * an abstract Matrix parent class because because cpp does not allow for
 * virtual operators, as it's not possible to overload something like
 * +=, for which the virtual base class function would have to return
 * an instance of the abstract base class (not possible), and which
 * cannot be overloaded with a covariant return type by the child classes.
 * c++ does this intentionally -- otherwise, it would be possible to
 * write things like SMatrix + PMatrix in the code!
 * Similar issues with () and the assignment operator make a inheritance
 * structure impractical.
 *
 */
template <typename T>
class Matrix {

  /// Index from a 1D array to a position in a 2D array (matrix)
  int global2Local(const int& row, const int& col);
  std::tuple<int, int> local2Global(const int& k);

 /** Boolean variable which tells us if the underlying matrix is parallel
 * will be defaulted to false if no value is provided in constructor
 */
 bool isDistributed;

 /** Underlying ParallelMatrix instantiated only if isDistributed = true
 */
 ParallelMatrix<T>* pmat = nullptr;

 /** Underlying SerialMatrix instantiated only if isDistributed = false
 */
 SerialMatrix<T>* mat = nullptr;

 public:
  /** Default Matrix constructor.
   * Matrix elements are set to zero upon initialization.
   *
   * @param numRows: number of rows of the matrix
   * @param numCols: number of columns of the matrix.
   * @param numBlocksRows, numBlocksCols: these parameters are ignored and are
   * put here for mirroring the interface of ParallelMatrix.
   */
  Matrix(const int& numRows, const int& numCols, const int& numBlocksRows = 0,
         const int& numBlocksCols = 0, bool isDistributed_ = false);

  /** Default constructor
   */
  Matrix();

  /** Destructor, to delete raw buffer
   */
  ~Matrix();

  /** Copy constructor
   */
  Matrix(const Matrix<T>& that);

  /** Copy constructor
   */
  Matrix<T>& operator=(const Matrix<T>& that);

  /** Find the global indices of the matrix elements that are stored locally
   * by the current MPI process.
   */
  std::vector<std::tuple<int, int>> getAllLocalStates();

  /** Returns true if the global indices (row,col) identify a matrix element
   * stored by the MPI process.
   */
  bool indicesAreLocal(const int& row, const int& col);

  /** Find global number of rows
   */
  int rows() const;
  /** Return local number of rows */
  int localRows() const;
  /** Find global number of columns */
  int cols() const;
  /** Return local number of rows */
  int localCols() const;
  /** Find global number of matrix elements*/
  int size() const;
  /** Return the size of the matrix in GB */
  double getMemory() const;

  /** Get and set operator
   */
  T& operator()(const int &row, const int &col);

  /** Const get and set operator
   */
  const T& operator()(const int &row, const int &col) const;

  /** Matrix-matrix multiplication.
   * Computes result = trans1(*this) * trans2(that)
   * where trans(1/2( can be "N" (matrix as is), "T" (transpose) or "C" adjoint
   * (these flags are used so that the transposition/adjoint operation is never
   * directly operated on the stored values)
   *
   * @param that: the matrix to multiply "this" with
   * @param trans2: controls transposition of "this" matrix
   * @param trans1: controls transposition of "that" matrix
   * @return result: a ParallelMatrix object.
   */
  Matrix<T> prod(const Matrix<T>& that, const char& trans1,
                 const char& trans2);

  /** Matrix-matrix addition.
   */
  Matrix<T> operator+=(const Matrix<T>& m1) {
    if(isDistributed) (*pmat) += (*m1.pmat);
    else (*mat) += (*m1.mat);
    return *this;
  }

  /** Matrix-matrix subtraction.
   */
  Matrix<T> operator-=(const Matrix<T>& m1) {
    if(isDistributed) (*pmat) -= (*m1.pmat);
    else (*mat) -= (*m1.mat);
    return *this;
  }

  /** Matrix-scalar multiplication.
   */
  Matrix<T> operator*=(const T& that) {
    if(isDistributed) (*pmat) *= that;
    else (*mat) *= that;
    return *this;
  }

  /** Matrix-scalar division.
   */
  Matrix<T> operator/=(const T& that) {
    if(isDistributed) (*pmat) /= that;
    else (*mat) /= that;
    return *this;
  }

  /** Sets this matrix as the identity.
   */
  void eye();

  /** Diagonalize a complex-hermitian / real-symmetric matrix.
   * Nota bene: we don't check if it's hermitian/symmetric.
   */
  std::tuple<std::vector<double>, Matrix<T>> diagonalize();

  /** Diagonalize a complex-hermitian / real-symmetric matrix
   * for only some of it's eigenvalues.
   * Nota bene: we don't check if it's hermitian/symmetric.
   */
  std::tuple<std::vector<double>, Matrix<T>> diagonalize(int numEigenvalues,
                                                bool checkNegativeEigenvalues = true);

  /** Computes the squared Frobenius norm of the matrix
   * (or Euclidean norm, or L2 norm of the matrix)
   */
  double squaredNorm();

  /** Computes the Frobenius norm of the matrix
   * (or Euclidean norm, or L2 norm of the matrix).
   */
  double norm();

  /** Computes a "scalar product" between two matrices A and B,
   * defined as \sum_ij A_ij * B_ij. For vectors, this reduces to the standard
   * scalar product.
   */
  T dot(const Matrix<T>& that);

  /** Unary negation
   */
  Matrix<T> operator-() const;

  /** A function to write the matrix to HDF5. Only
   * for debugging, and only written for PMatrix. */
  void outputToHDF5();

  /** Symmetrize the matrix
  */
  void symmetrize();

};

/* ------------------ constructor implementations -------------- */

// A default constructor to build a dense matrix of zeros to be filled
template <typename T>
Matrix<T>::Matrix(const int& numRows, const int& numCols,
                  const int& numBlocksRows, const int& numBlocksCols, bool isDistributed_) {

  isDistributed = isDistributed_; // default to false if no value supplied

  if(isDistributed){
    pmat = new ParallelMatrix<T>(numRows,numCols,numBlocksRows,numBlocksCols);
  }
  else {
    mat = new SerialMatrix<T>(numRows,numCols);
  }
}

// default constructor
template <typename T>
Matrix<T>::Matrix() {
  isDistributed = false;
  if (pmat!=nullptr) delete pmat;
  if (mat!=nullptr) delete mat;
  mat = new SerialMatrix<T>();
}

// copy constructor
template <typename T>
Matrix<T>::Matrix(const Matrix<T>& that) {
  isDistributed = that.isDistributed;

  // call SMatrix or PMatrix copy constructor
  if(isDistributed) {
    pmat = that.pmat;
  }
  else {
    mat = that.mat;
  }
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& that) {
  if (this != &that) {
    isDistributed = that.isDistributed;
    // call SMatrix or PMatrix copy constructor
    if(isDistributed) {
      if (pmat!=nullptr) delete pmat;
      pmat = new ParallelMatrix<T>(*that.pmat);
    }
    else {
      if (mat!=nullptr) delete mat;
      mat = new SerialMatrix<T>(*that.mat);
    }
  }
  return *this;
}

// destructor
template <typename T>
Matrix<T>::~Matrix() {
  if (pmat!=nullptr) {
    delete pmat;
  }
  if (mat!=nullptr) {
    delete mat;
  }
}

/* ------------- Very basic operations -------------- */
template <typename T>
int Matrix<T>::rows() const {
  if(isDistributed) return pmat->rows();
  else{ return mat->rows(); }
}
template <typename T>
int Matrix<T>::localRows() const {
  if(isDistributed) return pmat->localRows();
  else{ return mat->rows(); }
}
template <typename T> int Matrix<T>::cols() const {
  if(isDistributed) return pmat->cols();
  else{ return mat->cols(); }
}
template <typename T>
int Matrix<T>::localCols() const {
  if(isDistributed) return pmat->localCols();
  else{ return mat->cols(); }
}
template <typename T>
int Matrix<T>::size() const {
  if(isDistributed) return pmat->size();
  else{ return mat->size(); }
}

/* ------------- get-set operations -------------- */
template <typename T>
double Matrix<T>::getMemory() const{
  // this is done in parts to avoid overflow;
  // size in GB is size of type*rows()*cols()/(1024)^3
  double temp = sizeof(T)*rows()/1024;
  temp = temp*cols()/1024;
  return temp/(1024);
}

template <typename T>
T& Matrix<T>::operator()(const int &row, const int &col) {
  if(isDistributed) return (*pmat)(row,col);
  else { return (*mat)(row,col); }
}

template <typename T>
const T& Matrix<T>::operator()(const int &row, const int &col) const {
  if(isDistributed) return (*pmat)(row,col);
  else{ return (*mat)(row,col); }
}

template <typename T>
bool Matrix<T>::indicesAreLocal(const int& row, const int& col) {
  if(isDistributed) return pmat->indicesAreLocal(row,col);
  else{ return true; }
}

template <typename T>
std::tuple<int, int> Matrix<T>::local2Global(const int& k) {
  if(isDistributed) return pmat->local2Global(k);
  else{ return mat->local2Global(k); }
}

// Indexing to set up the matrix in col major format
template <typename T>
int Matrix<T>::global2Local(const int& row, const int& col) {
  if(isDistributed) return pmat->global2Local(row,col);
  else{ return mat->global2Local(row,col); }
}

template <typename T>
std::vector<std::tuple<int, int>> Matrix<T>::getAllLocalStates() {
  if(isDistributed) return pmat->getAllLocalStates();
  else{ return mat->getAllLocalStates(); }
}

/* ------------- basic linear algebra ops -------------- */
// General unary negation
template <typename T>
Matrix<T> Matrix<T>::operator-() const {
  Matrix<T> c(*this); // copy this matrix
  if(isDistributed) c.pmat = -c.pmat;
  else{ c.mat = -c.mat; }
  return c;
}

// Sets the matrix to the identity matrix
template <typename T>
void Matrix<T>::eye() {
  if(isDistributed) pmat->eye();
  else{ mat->eye(); }
}

template <typename T>
double Matrix<T>::norm() {
  if(isDistributed) return pmat->norm();
  else{ return mat->norm(); }
}

template <typename T>
double Matrix<T>::squaredNorm() {
  if(isDistributed) return pmat->squaredNorm();
  else{ return mat->squaredNorm(); }
}

template <typename T>
T Matrix<T>::dot(const Matrix<T>& that) {
  if(isDistributed) return pmat->dot(that.pmat);
  else{ return mat->dot(that.mat); }
}

template <typename T>
void Matrix<T>::outputToHDF5() {
  if(isDistributed) pmat->outputToHDF5();
  else{ Error("Write to HDF5 not implemented for SMatrix."); }
}

#endif  // MATRIX_H
