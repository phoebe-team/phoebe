#ifndef P_MATRIX_H
#define P_MATRIX_H

#include <tuple>
#include <vector>
#include "Blacs.h"
#include "exceptions.h"
#include "constants.h"
#include "mpiHelper.h"
#include "io.h"

#ifdef HDF5_AVAIL
#include <highfive/H5Easy.hpp>
#endif

#include "SMatrix.h"

// fall back to SerialMatrix if no MPI
#ifndef MPI_AVAIL
//alias template
template<typename T>
using ParallelMatrix = SerialMatrix<T>;
#else

#include <utility>
#include <set>

// https://www.ibm.com/docs/en/pessl/5.5?topic=programs-application-program-outline
// TODO we could remove the sq proc grid requirement as done here

/** Class for managing a matrix MPI-distributed in memory.
 *
 * This class uses the Scalapack library for matrix-matrix multiplication and
 * matrix diagonalization. For the time being we don't use other scalapack
 * functionalities.
 *
 * If the code is compiled without MPI, ParallelMatrix reduces to its serial
 * version Matrix.
 *
 * Template specialization only valid for double or complex<double>.
 */
template <typename T>
class ParallelMatrix {
 private:

  /// Class variables
  int numRows_ = 0;
  int numCols_ = 0;
  int numLocalRows_ = 0;
  int numLocalCols_ = 0;
  int numLocalElements_ = 0;

  // BLACS variables
  // numBlocksRows/Cols -- the number of units we divide nrows/ncols into
  int numBlocksRows_ = 0;
  int numBlocksCols_ = 0;
  // blockSizeRows/Cols -- the size of each unit we divide nrows/ncols into
  int blockSizeRows_ = 0;
  int blockSizeCols_ = 0;
  // numBlasRows/Cols - the number of rows/cols in the process grid
  int numBlasRows_ = 0;
  int numBlasCols_ = 0;
  // myBlasRow/Col - this process's row/col in the process grid
  int myBlasRow_ = 0;
  int myBlasCol_ = 0;
  int descMat_[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int blasRank_ = 0;
  int blacsContext_ = 0;
  char blacsLayout_ = 'R';  // block cyclic, row major processor mapping

  // dummy values to return when accessing elements not available locally
  T dummyZero = 0;
  T const dummyConstZero = 0;

  T* mat = nullptr; // raw buffer

  /** Converts a local one-dimensional storage index (MPI-dependent) into the
   * row/column index of the global matrix.
   */
  std::tuple<int, int> local2Global(const int& k) const;
  std::tuple<int, int> local2Global(const int& i, const int& j) const;

  /** Set the blacsContext for cases where two descriptors must share the same one */
  void setBlacsContext(int blacsContext);

 public:
  /** Converts a global row/column index of the global matrix into a local
   * one-dimensional storage index (MPI-dependent),
   * with value ranging from  0 to numLocalElements_-1.
   * Returns -1 if the matrix element is not stored on the current MPI process.
   */
  int global2Local(const int& row, const int& col) const;

  static const char transN = 'N';  // no transpose nor adjoint
  static const char transT = 'T';  // transpose
  static const char transC = 'C';  // adjoint (for complex numbers)

  /** Default constructor of the matrix class.
   * Matrix elements are set to zero in the initialization.
   * @param numRows: global number of matrix rows
   * @param numCols: global number of matrix columns
   * @param numBLocksRows: row size of the block for Blacs distribution
   * @param numBLocksCols: column size of the block for Blacs distribution
   */
  ParallelMatrix(const int& numRows, const int& numCols,
                 const int& numBlasRows = 0, const int& numBlasCols = 0,
                 const int& numBlocksRows = 0, const int& numBlocksCols = 0,
                 const int& blacsContext = -1);

  /** Empty constructor
   */
  ParallelMatrix();

  /** Destructor, to deallocate pointers
   */
  ~ParallelMatrix();

  /** Copy constructor
   */
  ParallelMatrix(const ParallelMatrix<T>& that);

  /** Copy assignment
   */
  ParallelMatrix& operator=(const ParallelMatrix<T>& that);

  /** A method to initialize blacs parameters, if needed.
  * @param numBlasRows -- number of rows requested for blacs grid
  * @param numBlasCols -- number of cosl requested for blacs grid
  * If both are zero, this function falls back to create a square
  * blacs process grid.
  */
  void initBlacs(const int& numBlasRows = 0, const int& numBlasCols = 0,
                                                const int& initBlacsContext = -1);

  /** Find the global indices of the matrix elements that are stored locally
   * by the current MPI process.
   */
  std::vector<std::tuple<int, int>> getAllLocalStates();

  /** Find the global indices of the rows that are stored locally
   * by the current MPI process.
   */
  std::vector<int> getAllLocalRows();

  /** Find the global indices of the cols that are stored locally
   * by the current MPI process.
   */
  std::vector<int> getAllLocalCols();

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
  /** Return local number of cols
  */
  int localCols() const;
  /** Find global number of matrix elements
   */
  int size() const;

  /** Get and set operator.
   * Returns the stored value if the matrix element (row,col) is stored in
   * memory by the MPI process, otherwise returns zero.
   */
  T& operator()(const int &row, const int &col);

  /** Const get and set operator
   * Returns the stored value if the matrix element (row,col) is stored in
   * memory by the MPI process, otherwise returns zero.
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
  ParallelMatrix<T> prod(const ParallelMatrix<T>& that,
                         const char& trans1 = transN,
                         const char& trans2 = transN);

  /** Matrix-matrix addition.
   */
  ParallelMatrix<T>& operator+=(const ParallelMatrix<T>& that);

  /** Matrix-matrix subtraction.
   */
  ParallelMatrix<T>& operator-=(const ParallelMatrix<T>& that);

  /** Matrix-scalar multiplication.
   */
  ParallelMatrix<T>& operator*=(const T& that);

  /** Matrix-scalar division.
   */
  ParallelMatrix<T>& operator/=(const T& that);

  /** Sets this matrix as the identity.
   * Deletes any previous content.
   */
  void eye();

  /** Sets this matrix to all zeros
   */
  void zeros();

  /** Diagonalize a complex-hermitian or real-symmetric matrix.
   * Nota bene: we don't check if the matrix is hermitian/symmetric or not.
   * By default, it operates on the upper-triangular part of the matrix.
   */
  std::tuple<std::vector<double>, ParallelMatrix<T>> diagonalize();
  std::tuple<std::vector<double>, ParallelMatrix<T>> diagonalize(int numEigenvalues,
                                                bool checkNegativeEigenvalues = true);

  /** Write contents of the matrix to file. Note, this is for testing because
   * it's very slow and in a production case could make an unreasonably large file.
   * @param outFileName: string naming the output file
   * @param dataSetName: keyname for data in the output file
   */
  void outputToHDF5(const std::string &outFileName,
                                   const std::string &dataSetName);

  /** Functions which actually diagonalize the matrix, which one is
   * called is determined by the main diagonalize call, and if ELPA_AVAIL
   * is defined.
   */
  std::tuple<std::vector<double>, ParallelMatrix<T>> elpaDiagonalize();
  std::tuple<std::vector<double>, ParallelMatrix<T>> scalapackDiagonalize();

  /** Computes the squared Frobenius norm of the matrix
   * (or Euclidean norm, or L2 norm of the matrix)
   */
  T squaredNorm();

  /** Computes the Frobenius norm of the matrix
   * (or Euclidean norm, or L2 norm of the matrix)
   */
  T norm();

  /** Computes a "scalar product" between two matrices A and B,
   * defined as \sum_ij A_ij * B_ij. For vectors, this reduces to the standard
   * scalar product.
   */
  T dot(const ParallelMatrix<T>& that);

  /** Unary negation
   */
  ParallelMatrix<T> operator-() const;

  /** Symmetrize the matrix with pdtran for transpose, (A + A^T)/2
  */
  void symmetrize();

  /** A function to fix up a matrix with a bit of noise to be positive semi-definite
  */
  void enforcePositiveSemiDefinite();

};

template <typename T>
ParallelMatrix<T>::ParallelMatrix(const int& numRows, const int& numCols,
                                  const int& numBlasRows,
                                  const int& numBlasCols,
                                  const int& numBlocksRows,
                                  const int& numBlocksCols,
                                  const int& blacsContext) {

  // call initBlacs to set all blacs related variables and contruct
  // the blacs context and process grid setup.
  // If numBlocksRows or numBlocksCols is not zero, this will
  // initialize blacs with a square process grid
  // (and number of processors must be a square number if we're doing lin alg ops)
  initBlacs(numBlasRows, numBlasCols, blacsContext);

  // initialize number of rows and columns of the global matrix
  numRows_ = numRows;
  numCols_ = numCols;

  // determine the number of blocks (for parallel distribution) along rows/cols
  // numBlocksRows_/Cols_ -- the number of blocks we divide r/c of the matrix into
  // For example, if numBlocksRows = 1, there's one "block" of rows.
  // If numBlocksRows = numRows, there's numRows of "blocks" of elements.
  //
  // If block size values are not supplied, the default is to make the
  // block sizes the same as the blacs grid divisions
  if(numBlocksRows == 0) { numBlocksRows_ = numBlasRows_; }
  else { numBlocksRows_ = numBlocksRows; }
  if(numBlocksCols == 0) { numBlocksCols_ = numBlasCols_; }
  else { numBlocksCols_ = numBlocksCols; }

  // compute the block size (chunks of rows/cols over which matrix is distributed)
  // if blockSizeRows_ = 1, there's each row element is a block.
  blockSizeRows_ = numRows_ / numBlocksRows_;
  if (numRows_ % numBlocksRows_ != 0) blockSizeRows_ += 1;
  blockSizeCols_ = numCols_ / numBlocksCols_;
  if (numCols_ % numBlocksCols_ != 0) blockSizeCols_ += 1;

  // determine the number of local rows and columns
  int iZero = 0;  // helper variable

  // numroc function takes information about the process grid and returns the number of
  // rows and cols which are local to this process
  numLocalRows_ = numroc_(&numRows_, &blockSizeRows_, &myBlasRow_, &iZero, &numBlasRows_);
  numLocalCols_ = numroc_(&numCols_, &blockSizeCols_, &myBlasCol_, &iZero, &numBlasCols_);
  numLocalElements_ = numLocalRows_ * numLocalCols_;

  //if(numLocalElements_ < 0) { // probably overflowed
  //  Error("The number of matrix elements local to this process overflows int\n"
  //		    "increase the number of MPI processes.");
  //} 

  // allocate the matrix
  mat = new T[numLocalElements_];

  // Memory could not be allocated, end program
  assert(mat != nullptr);

  // fill the matrix with zeroes
  for (int i = 0; i < numLocalElements_; ++i) *(mat + i) = 0.;

  // Create descriptor for block cyclic distribution of matrix
  int info;  // error code
  int lddA =
      numLocalRows_ > 1 ? numLocalRows_ : 1;  // if mpA>1, ldda=mpA, else 1

  descinit_(descMat_, &numRows_, &numCols_, &blockSizeRows_, &blockSizeCols_,
            &iZero, &iZero, &blacsContext_, &lddA, &info);

  if (info != 0) {
    DeveloperError("Something wrong calling descinit", info);
  }
}

template <typename T>
ParallelMatrix<T>::ParallelMatrix()  {

}

template <typename T>
ParallelMatrix<T>::ParallelMatrix(const ParallelMatrix<T>& that) {
  numRows_ = that.numRows_;
  numCols_ = that.numCols_;
  numLocalRows_ = that.numLocalRows_;
  numLocalCols_ = that.numLocalCols_;
  numLocalElements_ = that.numLocalElements_;
  numBlocksRows_ = that.numBlocksRows_;
  numBlocksCols_ = that.numBlocksCols_;
  blockSizeRows_ = that.blockSizeRows_;
  blockSizeCols_ = that.blockSizeCols_;
  numBlasRows_ = that.numBlasRows_;
  numBlasCols_ = that.numBlasCols_;
  myBlasRow_ = that.myBlasRow_;
  myBlasCol_ = that.myBlasCol_;
  blasRank_ = that.blasRank_;
  blacsContext_ = that.blacsContext_;

  for (int i = 0; i < 9; i++) {
    descMat_[i] = that.descMat_[i];
  }
  if ( mat != nullptr ) {
    delete[] mat;
  }
  // matrix allocation
  mat = new T[numLocalElements_];
  // Memory could not be allocated, end program
  assert(mat != nullptr);
  for (int i = 0; i < numLocalElements_; i++) {
    mat[i] = that.mat[i];
  }
}

template <typename T>
ParallelMatrix<T>& ParallelMatrix<T>::operator=(const ParallelMatrix<T>& that) {
  if (this != &that) {
    numRows_ = that.numRows_;
    numCols_ = that.numCols_;
    numLocalRows_ = that.numLocalRows_;
    numLocalCols_ = that.numLocalCols_;
    numLocalElements_ = that.numLocalElements_;
    numBlocksRows_ = that.numBlocksRows_;
    numBlocksCols_ = that.numBlocksCols_;
    blockSizeRows_ = that.blockSizeRows_;
    blockSizeCols_ = that.blockSizeCols_;
    numBlasRows_ = that.numBlasRows_;
    numBlasCols_ = that.numBlasCols_;
    myBlasRow_ = that.myBlasRow_;
    myBlasCol_ = that.myBlasCol_;
    blasRank_ = that.blasRank_;
    blacsContext_ = that.blacsContext_;

    for (int i = 0; i < 9; i++) {
      descMat_[i] = that.descMat_[i];
    }
    if ( mat != nullptr ) {
      delete[] mat;
    }
    // matrix allocation
    mat = new T[numLocalElements_];
    // Memory could not be allocated, end program
    assert(mat != nullptr);
    for (int i = 0; i < numLocalElements_; i++) {
      mat[i] = that.mat[i];
    }
  }
  return *this;
}

template <typename T>
ParallelMatrix<T>::~ParallelMatrix() {
  if ( mat != nullptr ) {
    delete[] mat;
  }
}

template <typename T>
void ParallelMatrix<T>::initBlacs(const int& numBlasRows, const int& numBlasCols,
                                                        const int& inputBlacsContext) {

  int size = mpi->getSize(); // temp variable for mpi world size, used in setup

  // TODO if we only give this the nearest square number of processors,
  // will it disregard the others for us? Could this fix the
  // sq process thing?
  //       NPROW = INT(SQRT(REAL(NNODES)))
  //    NPCOL = NNODES/NPROW
  //  IF (MYROW .LT. NPROW .AND. MYCOL .LT. NPCOL) THEN
  //  https://www.ibm.com/docs/en/pessl/5.5?topic=programs-application-program-outline
  blacs_pinfo_(&blasRank_, &size);
  int iZero = 0;
  if( inputBlacsContext == -1) { // no context has been created/supplied
    blacs_get_(&iZero, &iZero, &blacsContext_);  // -> get default system context
  }

  // kill the code if we asked for more blas rows/cols than there are procs
  if (mpi->getSize() < numBlasRows * numBlasCols) {
     Error("Developer error: initBlacs requested too many MPI processes.");
  }

  // Cases for a blacs grid where we specified rows, cols, both,
  // or the default, neither, which results in a square proc grid
  if(numBlasRows != 0 && numBlasCols == 0) {
    numBlasRows_ = numBlasRows;
    numBlasCols_ = mpi->getSize()/numBlasRows;
  }
  else if(numBlasRows == 0 && numBlasCols != 0) {
    numBlasRows_ = mpi->getSize()/numBlasCols;
    numBlasCols_ = numBlasCols;
  }
  else if(numBlasRows !=0 && numBlasCols !=0 ) {
    numBlasRows_ = numBlasRows;
    numBlasCols_ = numBlasCols;
  }
  else {
    // set up a square procs grid, as the default
    numBlasRows_ = (int)(sqrt(size)); // int does rounding down (intentional!)
    numBlasCols_ = numBlasRows_;

    // TODO will this just work ( not using all procs for matrix )
    // if we remove this error???
    //
    // Throw an error if we tried to set up a square proc grid with
    // a non-square number of processors
    if (mpi->getSize() > numBlasRows_ * numBlasCols_) {
      Error("Phoebe needs a square number of MPI processes");
    }
  }

  // if no context is given, create one.
  // Otherwise, use the one supplied
  if( inputBlacsContext == -1) { // no context has been created/supplied
    blacs_gridinit_(&blacsContext_, &blacsLayout_, &numBlasRows_, &numBlasCols_);
  } else {
    blacsContext_ = inputBlacsContext;
  }
  // Create the Blacs context
  // Context -> Context grid info (# procs row/col, current procs row/col)
  blacs_gridinfo_(&blacsContext_, &numBlasRows_, &numBlasCols_, &myBlasRow_,&myBlasCol_);
}

template <typename T>
int ParallelMatrix<T>::rows() const {
  return numRows_;
}

template <typename T>
int ParallelMatrix<T>::localRows() const {
  return numLocalRows_;
}

template <typename T>
int ParallelMatrix<T>::cols() const {
  return numCols_;
}

template <typename T>
int ParallelMatrix<T>::localCols() const {
  return numLocalCols_;
}

template <typename T>
int ParallelMatrix<T>::size() const {
  return cols() * rows();
}

// Get/set element
template <typename T>
T& ParallelMatrix<T>::operator()(const int &row, const int &col) {
  if(row > numRows_ || col > numCols_) Error("Tried to fill a PMatrix state out of bounds: " + std::to_string(row) + " " + std::to_string(col));
  int localIndex = global2Local(row, col);
  if (localIndex == -1) {
    dummyZero = 0.;
    return dummyZero;
  } else {
    return mat[localIndex];
  }
}

template <typename T>
const T& ParallelMatrix<T>::operator()(const int &row, const int &col) const {
  int localIndex = global2Local(row, col);
  if (localIndex == -1) {
    return dummyConstZero;
  } else {
    return mat[localIndex];
  }
}

template <typename T>
bool ParallelMatrix<T>::indicesAreLocal(const int& row, const int& col) {
  int localIndex = global2Local(row, col);
  if (localIndex == -1) {
    return false;
  } else {
    return true;
  }
}

template <typename T>
std::tuple<int,int> ParallelMatrix<T>::local2Global(const int& i, const int& j) const {
  int il = (int)i;
  int jl = (int)j;
  int iZero = 0;
  int ig = indxl2g_( &il, &blockSizeRows_, &myBlasRow_, &iZero, &numBlasRows_ );
  int jg = indxl2g_( &jl, &blockSizeCols_, &myBlasCol_, &iZero, &numBlasCols_ );
  return std::make_tuple(ig,jg);
}

template <typename T>
std::tuple<int, int> ParallelMatrix<T>::local2Global(const int& k) const {
  // first, we convert this combined local index k
  // into local row / col indices
  // k = j * numLocalRows_ + i
  int j = k / numLocalRows_;
  int i = k - j * numLocalRows_;

  // TODO this might need to be converted into a indxl2g version!
  // should just be able to uncomment the below two lines and comment the rest over
  // however, we should be careful to think that the above two conversions to i,j
  // are safe for a rectangular matrix.

  //int ig = indxl2g_( &il, &blockSizeRows_, &myBlasRow_, 0, &numBlasRows_ );
  //int jg = indxl2g_( &jl, &blockSizeCols_, &myBlasCol_, 0, &numBlasCols_ );
  //return std::make_tuple(ig,jg);

  // now we can convert local row/col indices into global indices

  int l_i = i / blockSizeRows_;  // which block
  int x_i = i % blockSizeRows_;  // where within that block
  // global row
  int I = (l_i * numBlasRows_ + myBlasRow_) * blockSizeRows_ + x_i;

  int l_j = j / blockSizeCols_;  // which block
  int x_j = j % blockSizeCols_;  // where within that block
  // global col
  int J = (l_j * numBlasCols_ + myBlasCol_) * blockSizeCols_ + x_j;
  return std::make_tuple(I, J);
}

template <typename T>
int ParallelMatrix<T>::global2Local(const int& row, const int& col) const {
  // note: row and col indices use the c++ convention of running from 0 to N-1
  // fortran (infog2l_) wants indices from 1 to N.
  int row_ = int(row) + 1;
  int col_ = int(col) + 1;

  // use infog2l_ to check that the current process owns this matrix element
  int iia, jja, iarow, iacol;
  infog2l_(&row_, &col_, &descMat_[0], &numBlasRows_, &numBlasCols_,
           &myBlasRow_, &myBlasCol_, &iia, &jja, &iarow, &iacol);

  // return -1 to signify the element is not local to this process
  if (myBlasRow_ != iarow || myBlasCol_ != iacol) {
    return -1;
  } else {
    // get the local indices, (il,jl) of the globally indexed element
    int iZero = 0;
    int il = indxg2l_( &row_, &blockSizeRows_, &myBlasRow_, &iZero, &numBlasRows_ );
    int jl = indxg2l_( &col_, &blockSizeCols_, &myBlasCol_, &iZero, &numBlasCols_ );
    return il + (jl - 1) * descMat_[8] - 1;
  }
}

template <typename T>
std::vector<std::tuple<int, int>> ParallelMatrix<T>::getAllLocalStates() {
  std::vector<std::tuple<int, int>> x;
  for (int k = 0; k < numLocalElements_; k++) {
    std::tuple<int, int> t = local2Global(k);  // bloch indices
    x.push_back(t);
  }
  return x;
}

template <typename T>
std::vector<int> ParallelMatrix<T>::getAllLocalRows() {
  int iZero = 0;
  std::vector<int> x;
  for (int k = 0; k < numLocalRows_; k++) {
    int gr = indxl2g_( &k, &blockSizeRows_, &myBlasRow_, &iZero, &numBlasRows_ );
    x.push_back(gr);
  }
  return x;
}

template <typename T>
std::vector<int> ParallelMatrix<T>::getAllLocalCols() {
  std::vector<int> x;
  int iZero = 0;
  for (int k = 0; k < numLocalCols_; k++) {
    int gc = indxl2g_( &k, &blockSizeCols_, &myBlasCol_, &iZero, &numBlasCols_ );
    x.push_back(gc);
  }
  return x;
}

template <typename T>
ParallelMatrix<T>& ParallelMatrix<T>::operator*=(const T& that) {
  for (int i = 0; i < numLocalElements_; i++) {
    *(mat + i) *= that;
  }
  return *this;
}

template <typename T>
ParallelMatrix<T>& ParallelMatrix<T>::operator/=(const T& that) {
  for (int i = 0; i < numLocalElements_; i++) {
    *(mat + i) /= that;
  }
  return *this;
}

template <typename T>
ParallelMatrix<T>& ParallelMatrix<T>::operator+=(const ParallelMatrix<T>& that) {
  if(numRows_ != that.rows() || numCols_ != that.cols()) {
    Error("Cannot adds matrices of different sizes.");
  }
  for (int i = 0; i < numLocalElements_; i++) {
    *(mat + i) += *(that.mat + i);
  }
  return *this;
}

template <typename T>
ParallelMatrix<T>& ParallelMatrix<T>::operator-=(const ParallelMatrix<T>& that) {
  if(numRows_ != that.rows() || numCols_ != that.cols()) {
    Error("Cannot subtract matrices of different sizes.");
  }
  for (int i = 0; i < numLocalElements_; i++) {
    *(mat + i) -= *(that.mat + i);
  }
  return *this;
}

template <typename T>
void ParallelMatrix<T>::eye() {
  if (numRows_ != numCols_) {
    Error("Cannot build an identity matrix with non-square matrix");
  }
  for (int i = 0; i < numLocalElements_; i++) {
    *(mat + i) = 0.;
  }
  for (int i = 0; i < numRows_; i++) {
    operator()(i, i) = 1.;
  }
}

template <typename T>
void ParallelMatrix<T>::zeros() {
  for (int i = 0; i < numLocalElements_; ++i) *(mat + i) = 0.;
}

template <typename T>
T ParallelMatrix<T>::squaredNorm() {
  return dot(*this);
}

template <typename T>
T ParallelMatrix<T>::norm() {
  return sqrt(squaredNorm());
}

template <typename T>
T ParallelMatrix<T>::dot(const ParallelMatrix<T>& that) {
  T scalar = dummyConstZero;
  T scalarOut = dummyConstZero;
  for (int i = 0; i < numLocalElements_; i++) {
    scalar += (*(mat + i)) * (*(that.mat + i));
  }
  mpi->allReduceSum(&scalar, &scalarOut);
  return scalarOut;
}

template <typename T>
ParallelMatrix<T> ParallelMatrix<T>::operator-() const {
  ParallelMatrix<T> result = *this;
  for (int i = 0; i < numLocalElements_; i++) {
    *(result.mat + i) = -*(result.mat + i);
  }
  return result;
}
// function to make sure two matrices share a blacs context...
template <typename T>
void ParallelMatrix<T>::setBlacsContext(int blacsContext) {
  blacsContext_ = blacsContext;
}

template <typename T>
void ParallelMatrix<T>::outputToHDF5(const std::string &outFileName,
                                   const std::string &dataSetName) {

  // output the matrix elements owned by this process to file.
  // The only tricky part here is that we need to sort them
  // Appropriately in the file without all reducing. This is nearly
  // impossible to do efficiently because of how HighFive is written,
  // wherein it expects chunks of data -- which we cannot expect to have
  // in a block-cyclic distribution of the matrix. Therefore, we do something
  // terrible and we write single elements at a time.

  #ifndef HDF5_AVAIL
    Error("Need Phoebe compiled with parallel HDF5 to write parallel matrix to file.");
  #elif HDF5_SERIAL
    Error("Need Phoebe compiled with parallel HDF5 to write parallel matrix to file.");
  #else

   try {

      // Create the file to write to, first removing if it exists already
      std::remove(&outFileName[0]);
      HighFive::FileAccessProps fapl;
      fapl.add(HighFive::MPIOFileAccess<MPI_Comm, MPI_Info>(MPI_COMM_WORLD, MPI_INFO_NULL));
      HighFive::File file(outFileName, HighFive::File::Overwrite, fapl);

      // setup the dataset
      unsigned int globalSize = size();

      // Create the data-space to write the elements to
      std::vector<size_t> dims(2);
      dims[0] = 1;
      dims[1] = size_t(globalSize);
      HighFive::DataSet dsMatrix= file.createDataSet<double>(
                dataSetName, HighFive::DataSpace(dims));

      LoopPrint loopPrint("writing the " + dataSetName + " to HDF5",
             "matrix elements", getAllLocalStates().size());

      // now we loop over the matrix elements and write them to file
      for(auto matEl : getAllLocalStates()) {

        loopPrint.update();

        // get matrix row and col indices
        size_t iMat1 = std::get<0>(matEl);
        size_t iMat2 = std::get<1>(matEl);
        // convert to 1d index, used on the underlying dataset
        size_t iGlobal = iMat1*cols() + iMat2;
        // get the actual value of the matrix element
        double value = this->operator()(iMat1,iMat2);

        // write the single element to the dataset
        // here, the offset is the global index, and the count is of course 1
        dsMatrix.select({0, iGlobal}, {1, 1}).write(value);
      }

      loopPrint.close();

      // ensure everything has been written
      file.flush();

    } catch (std::exception &error) {
      // catch and print any HDF5 error
      std::cerr << error.what() << std::endl;
      Error("An HDF5 error occurred while trying to write a distributed matrix to HDF5.");
    }
  #endif
}


/* generic function to enforce positive semi def */

template <typename T>
void ParallelMatrix<T>::enforcePositiveSemiDefinite() { 

  if(rows() != cols()) {
    DeveloperError("Can only enforce PSD on a square matrix."); 
  }

  // first, we need to grab the inverse sqrt of the diagonal of the matrix
  // we could do this by passing in the diagonal, which in the case of 
  // the scattering matrix is already stored in the linewidths vector
  std::vector<T> invSqrtDiagonal(numRows_); 
  std::vector<T> diagonal(numRows_); 
  for(auto matEl : getAllLocalStates()) {

    // get matrix row and col indices
    size_t iMat1 = std::get<0>(matEl);
    size_t iMat2 = std::get<1>(matEl);   
    if(iMat1 != iMat2) continue; 

    if(this->operator()(iMat1,iMat2) <= 1.e-16) continue; // avoid divide 1/~0
    invSqrtDiagonal[iMat1] = sqrt(1./this->operator()(iMat1,iMat2));
    diagonal[iMat1] = this->operator()(iMat1,iMat2);

  }

  // Now we make a container for a new matrix, G, with the same size as this one
  ParallelMatrix<T> G(numRows_, numCols_, 0, 0,
                              numBlocksRows_, numBlocksCols_, blacsContext_);

  // we construct G by performing D^-1/2 * C * D^-1/2 (where D are matrices)
  // with the diagonal elements set to sqrt(diagonal) of this matrix
  for(auto matEl : getAllLocalStates()) {

    // get matrix row and col indices
    size_t iMat1 = std::get<0>(matEl);
    size_t iMat2 = std::get<1>(matEl);   

    G(iMat1,iMat2) = invSqrtDiagonal[iMat1] * this->operator()(iMat1,iMat2) * invSqrtDiagonal[iMat2];

  }

  // Diagonalize G and store it's most negative eigenvalue 
  auto diagResult = G.diagonalize(); 
  T lowestEigenvalue = std::get<0>(diagResult)[0]; // zeroth element is the most negative

  if(lowestEigenvalue >= 0) return; // all eigenvalues are positive, no need to correct 

  if(mpi->mpiHead()) {
    std::cout << "Correcting scattering matrix for loweset eigenvalue " 
    << lowestEigenvalue << std::endl;
  }

  // we need the abs of this below, no sense in repeating it in the loop
  lowestEigenvalue = abs(lowestEigenvalue); // unsure if this would work on a complex matrix 

  // finally use this result to correct the real matrix 
  for(auto matEl : getAllLocalStates()) {

    // get matrix row and col indices
    size_t iMat1 = std::get<0>(matEl);
    size_t iMat2 = std::get<1>(matEl);  
    if(iMat1 != iMat2) continue; // we only fix the diagonal 

    this->operator()(iMat1,iMat2) = this->operator()(iMat1,iMat2) + 1.05*lowestEigenvalue * diagonal[iMat1]; 

  }
}

#endif  // include safeguard

#endif  // mpi_avail
