#ifndef S_MATRIX_H
#define S_MATRIX_H

#include <tuple>
#include <vector>

#include "Blas.h"
#include "exceptions.h"

/** Class for managing a (serial) matrix stored in memory.
 *
 * It mirrors the functionality of ParallelMatrix, only that in this case it
 * doesn't use the scalapack library and instead relies on BLAS.
 * Data is NOT distributed across different MPI processes.
 * At the moment is used for making the code compile without MPI.
 *
 * Templated matrix class with specialization for double and complex<double>.
 */
template <typename T>
class SerialMatrix {
  /// Class variables
  int numRows_;
  int numCols_;
  int numElements_;

  T* mat = nullptr;  // pointer to the internal array structure.

  /// Index from a 1D array to a position in a 2D array (matrix)
  std::tuple<int, int> local2Global(const int& k);

 public:
  int global2Local(const int& row, const int& col);

  /** Indicates that the matrix A is not modified: transN(A) = A
   */
  static const char transN = 'N';
  /** Indicates that the matrix A is taken as its transpose: transT(A) = A^T
   */
  static const char transT = 'T';
  /** Indicates that the matrix A is taken as its adjoint: transC(A) = A^+
   */
  static const char transC = 'C';

  /** Default SMatrix constructor.
   * SerialMatrix elements are set to zero upon initialization.
   *
   * @param numRows: number of rows of the matrix
   * @param numCols: number of columns of the matrix.
   * @param numBlocksRows, numBlocksCols: these parameters are ignored and are
   * put here for mirroring the interface of ParallelMatrix.
   */
  SerialMatrix(const int& numRows, const int& numCols,
                 const int& numBlasRows = 0, const int& numBlasCols = 0,
                 const int& numBlocksRows = 0, const int& numBlocksCols = 0);

  /** Default constructor
   */
  SerialMatrix();

  /** Destructor, to delete raw buffer
   */
  ~SerialMatrix();

  /** Copy constructor
   */
  SerialMatrix(const SerialMatrix<T>& that);

  /** Copy constructor
   */
  SerialMatrix<T>& operator=(const SerialMatrix<T>& that);

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
  /** Return local number of rows
  */
  int localRows() const;
  /** Find global number of columns
   */
  int cols() const;
  /** Return local number of rows
  */
  int localCols() const;
  /** Find global number of matrix elements
   */
  int size() const;
  /** Get and set operator
   */
  T& operator()(const int &row, const int &col);
  /** Const get and set operator
   */
  const T& operator()(const int &row, const int &col) const;

  /** A dummy function not implemented, only for PMatrix currently
   * @param outFileName: string naming the output file
   * @param dataSetName: keyname for data in the output file
   */
  void outputToHDF5(const std::string &outFileName,
                                   const std::string &dataSetName) {}

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
  SerialMatrix<T> prod(const SerialMatrix<T>& that, const char& trans1 = transN,
                 const char& trans2 = transN);

  /** Matrix-matrix addition.
   */
  SerialMatrix<T> operator+=(const SerialMatrix<T>& m1) {
    if(numRows_ != m1.rows() || numCols_ != m1.cols()) {
      Error("Cannot add matrices of different sizes.");
    }
    for (int s = 0; s < size(); s++) mat[s] += m1.mat[s];
    return *this;
  }

  /** Matrix-matrix subtraction.
   */
  SerialMatrix<T> operator-=(const SerialMatrix<T>& m1) {
    if(numRows_ != m1.rows() || numCols_ != m1.cols()) {
      Error("Cannot subtract matrices of different sizes.");
    }
    for (int s = 0; s < size(); s++) mat[s] -= m1.mat[s];
    return *this;
  }

  /** Matrix-scalar multiplication.
   */
  SerialMatrix<T> operator*=(const T& that) {
    for (int s = 0; s < size(); s++) mat[s] *= that;
    return *this;
  }

  /** Matrix-scalar division.
   */
  SerialMatrix<T> operator/=(const T& that) {
    for (int s = 0; s < size(); s++) mat[s] /= that;
    return *this;
  }

  /** Sets this matrix as the identity.
   * Deletes any previous content.
   */
  void eye();

  /** Diagonalize a complex-hermitian / real-symmetric matrix.
   * Nota bene: we don't check if it's hermitian/symmetric.
   */
  std::tuple<std::vector<double>, SerialMatrix<T>> diagonalize();

  /** Diagonalize a complex-hermitian / real sym matrix
   * for only some of its eigenvalues.
   * Note: this isn't implemented for now and just calls the diagonalize() function
   */
  std::tuple<std::vector<double>, SerialMatrix<T>> diagonalize(int numEigenvalues,
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
  T dot(const SerialMatrix<T>& that);

  /** Unary negation
   */
  SerialMatrix<T> operator-() const;

  /** Symmetrize the matrix */
  void symmetrize();

};

// A default constructor to build a dense matrix of zeros to be filled
template <typename T>
SerialMatrix<T>::SerialMatrix(const int& numRows, const int& numCols,
                  const int& numBlasRows, const int& numBlasCols,
                  const int& numBlocksRows, const int& numBlocksCols) {
  // these are used only the in the pmatrix case
  (void) numBlocksRows;
  (void) numBlocksCols;
  (void) numBlasRows;
  (void) numBlasCols;

  numRows_ = numRows;
  numCols_ = numCols;
  numElements_ = numRows_ * numCols_;
  mat = new T[numRows_ * numCols_];
  for (int i = 0; i < numElements_; i++) mat[i] = 0;  // fill with zeroes
  assert(mat != nullptr);  // Memory could not be allocated, end program
}

// default constructor
template <typename T>
SerialMatrix<T>::SerialMatrix() {
  mat = nullptr;
  numRows_ = 0;
  numCols_ = 0;
  numElements_ = 0;
}

// copy constructor
template <typename T>
SerialMatrix<T>::SerialMatrix(const SerialMatrix<T>& that) {
  numRows_ = that.numRows_;
  numCols_ = that.numCols_;
  numElements_ = that.numElements_;
  if (mat != nullptr) {
    delete[] mat;
    mat = nullptr;
  }
  mat = new T[numElements_];
  assert(mat != nullptr);
  for (int i = 0; i < numElements_; i++) {
    mat[i] = that.mat[i];
  }
}

template <typename T>
SerialMatrix<T>& SerialMatrix<T>::operator=(const SerialMatrix<T>& that) {
  if (this != &that) {
    numRows_ = that.numRows_;
    numCols_ = that.numCols_;
    numElements_ = that.numElements_;
    // matrix allocation
    if (mat != nullptr) {
      delete[] mat;
    }
    mat = new T[numElements_];
    assert(mat != nullptr);
    for (int i = 0; i < numElements_; i++) {
      mat[i] = that.mat[i];
    }
  }
  return *this;
}

// destructor
template <typename T>
SerialMatrix<T>::~SerialMatrix() {
  if (mat != nullptr) delete[] mat;
}

/* ------------- Very basic operations -------------- */
template <typename T>
int SerialMatrix<T>::rows() const {
  return numRows_;
}
template <typename T>
int SerialMatrix<T>::localRows() const {
  return numRows_;
}
template <typename T>
int SerialMatrix<T>::cols() const {
  return numCols_;
}
template <typename T>
int SerialMatrix<T>::localCols() const {
  return numCols_;
}
template <typename T>
int SerialMatrix<T>::size() const {
  return numElements_;
}

// Get/set element
template <typename T>
T& SerialMatrix<T>::operator()(const int &row, const int &col) {
  if(row >= numRows_ || col >= numCols_ || row < 0 || col < 0) {
    Error("Developer Error: Attempted to reference a matrix element out of bounds.");
  }
  return mat[global2Local(row, col)];
}

template <typename T>
const T& SerialMatrix<T>::operator()(const int &row, const int &col) const {
  if(row >= numRows_ || col >= numCols_ || row < 0 || col < 0) {
    Error("Developer Error: Attempted to reference a matrix element out of bounds.");
  }
  return mat[global2Local(row, col)];
}

template <typename T>
bool SerialMatrix<T>::indicesAreLocal(const int& row, const int& col) {
  if(row >= numRows_ || col >= numCols_ || row < 0 || col < 0) {
    Error("Developer Error: Indices are out of bounds.");
  }
  //(void) row;
  //(void) col;
  return true;
}

template <typename T>
std::tuple<int, int> SerialMatrix<T>::local2Global(const int& k) {
  // we convert this combined local index k into row / col indices
  // k = j * nRows + i
  if(numRows_ == 0) Error("attempted to div by zero in l2g");
  int j = k / numRows_;
  int i = k - j * numRows_;
  return std::make_tuple(i, j);
}

// Indexing to set up the matrix in col major format
template <typename T>
int SerialMatrix<T>::global2Local(const int& row, const int& col) {
  return numRows_ * col + row;
}

template <typename T>
std::vector<std::tuple<int, int>> SerialMatrix<T>::getAllLocalStates() {
  std::vector<std::tuple<int, int>> x;
  for (int k = 0; k < numElements_; k++) {
    std::tuple<int, int> t = local2Global(k);  // bloch indices
    x.push_back(t);
  }
  return x;
}

// General unary negation
template <typename T>
SerialMatrix<T> SerialMatrix<T>::operator-() const {
  SerialMatrix<T> ret(numRows_, numCols_);
  for (int row = 0; row < numRows_; row++) {
    for (int col = 0; col < numCols_; col++) ret(row, col) = -(*this)(row, col);
  }
  return ret;
}

// Sets the matrix to the identity matrix
template <typename T>
void SerialMatrix<T>::eye() {
  if(numRows_ != numCols_) {
    Error("Cannot build an identity matrix with non-square matrix");
  }
  for (int row = 0; row < numRows_; row++) (*this)(row, row) = (T)1.0;
}

// forward declaration of explicit specialization
// is needed because the generic implementation does not
// work for complex types
template<> double SerialMatrix<std::complex<double>>::norm();

template <typename T>
double SerialMatrix<T>::norm() {
  T sumSq = 0;
  for (int row = 0; row < numRows_; row++) {
    for (int col = 0; col < numCols_; col++) {
      sumSq += ((*this)(row, col) * (*this)(row, col));
    }
  }
  return sqrt(sumSq);
}

template <typename T>
double SerialMatrix<T>::squaredNorm() {
  double x = norm();
  return x * x;
}

template <typename T>
T SerialMatrix<T>::dot(const SerialMatrix<T>& that) {
  T scalar = (T)0.;
  for (int i = 0; i < numElements_; i++) {
    scalar += (*(mat + i)) * (*(that.mat + i));
  }
  return scalar;
}

#endif  // S_MATRIX_H
