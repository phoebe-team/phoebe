#ifdef MPI_AVAIL

#include "PMatrix.h"

#include "Blacs.h"
#include "constants.h"
#include "mpiHelper.h"
#include "utilities.h"

template <>
ParallelMatrix<double> ParallelMatrix<double>::prod(
    const ParallelMatrix<double>& that, const char& trans1,
    const char& trans2) {

  if(cols() != that.rows()) {
    Error("Cannot multiply matrices for which lhs.cols != rhs.rows.");
  }
  auto result = that;
  result.zeros();

  int m;
  if (trans1 == transN) {
    m = numRows_;
  } else {
    m = numCols_;
  }
  int n;
  if (trans2 == transN) {
    n = that.numCols_;
  } else {
    n = that.numRows_;
  }
  int k;
  if (trans1 == transN) {
    k = numCols_;
  } else {
    k = numRows_;
  }
  // check on k being consistent
  if (trans2 == transN) {
    assert(k == that.numRows_);
  } else {
    assert(k == that.numCols_);
  }
  double alpha = 1.;
  double beta = 0.;
  int one = 1;
  pdgemm_(&trans1, &trans2, &m, &n, &k, &alpha, mat, &one, &one, &descMat_[0],
          that.mat, &one, &one, &that.descMat_[0], &beta, result.mat, &one,
          &one, &result.descMat_[0]);
  return result;
}

template <>
ParallelMatrix<std::complex<double>> ParallelMatrix<std::complex<double>>::prod(
    const ParallelMatrix<std::complex<double>>& that, const char& trans1,
    const char& trans2) {

  ParallelMatrix<std::complex<double>> result(numRows_, numCols_, 0, 0,
                                              numBlocksRows_, numBlocksCols_);
  if(cols() != that.rows()) {
    Error("Cannot multiply matrices for which lhs.cols != rhs.rows.");
  }
  int m;
  if (trans1 == transN) {
    m = numRows_;
  } else {
    m = numCols_;
  }
  int n;
  if (trans2 == transN) {
    n = that.numCols_;
  } else {
    n = that.numRows_;
  }
  int k;
  if (trans1 == transN) {
    k = numCols_;
  } else {
    k = numRows_;
  }
  // check on k being consistent
  if (trans2 == transN) {
    assert(k == that.numRows_);
  } else {
    assert(k == that.numCols_);
  }
  std::complex<double> alpha = complexOne;
  std::complex<double> beta = complexZero;
  int one = 1;
  pzgemm_(&trans1, &trans2, &m, &n, &k, &alpha, mat, &one, &one, &descMat_[0],
          that.mat, &one, &one, &that.descMat_[0], &beta, result.mat, &one,
          &one, &result.descMat_[0]);
  return result;
}

template <>
std::tuple<std::vector<double>, ParallelMatrix<double>>
ParallelMatrix<double>::diagonalize() {

  if (numRows_ != numCols_) {
    Error("Cannot diagonalize non-square matrix");
  }
  if ( numBlasRows_ != numBlasCols_ ) {
    Error("Cannot diagonalize via scalapack with a non-square process grid!");
  }

  double *eigenvalues;
  allocate(eigenvalues, numRows_);

  // Make a new PMatrix to receive the output
  // zeros here trigger a the default blacs process grid (a square one)
  ParallelMatrix<double> eigenvectors(numRows_,numCols_, 0, 0,
                              numBlocksRows_,numBlocksCols_, blacsContext_);

  char jobz = 'V';  // also eigenvectors
  char uplo = 'U';  // upper triangular
  int ia = 1;       // row index from which we diagonalize
  int ja = 1;       // row index from which we diagonalize

  int info = 0;

  // we will let pdseyv determine lwork for us. if we run it with
  // lwork = -1 and work of length 1, it will fill work with
  // an appropriate lwork number
  double *work;
  int lwork = -1;
  allocate(work, 1);
  // somehow autodetermination never works for liwork, so we compute this manually
  // liwork ≥ 7n + 8npcol + 2
  int liwork = 7*numRows_ + 8* numBlasCols_ + 2;
  int *iwork;
  allocate(iwork, liwork);
  // calculate lwork
  pdsyevd_(&jobz, &uplo, &numRows_, mat, &ia, &ja, &descMat_[0], eigenvalues,
          eigenvectors.mat, &ia, &ja, &eigenvectors.descMat_[0],
          work, &lwork, iwork, &liwork, &info);

  //size_t tempWork = int(work[0]);
  // automatic detection finds the minimum needed,
  // we actually pretty much always need way more than this!
  //if(tempWork > 2147483640) { lwork = 2147483640; } // check for overflow
  //else { lwork = tempWork; }

  lwork = int(work[0]);
  delete[] work;
  try{ allocate(work, lwork); }
  catch (std::bad_alloc& ba) {
    Error("PDSYEVD lwork array allocation failed.");
  }

  if(mpi->mpiHead()) {
     std::cout << "Starting matrix diagonalization." << std::endl;
     mpi->time();
  }

  // call the function to now diagonalize
  pdsyevd_(&jobz, &uplo, &numRows_, mat, &ia, &ja, &descMat_[0], eigenvalues,
          eigenvectors.mat, &ia, &ja, &eigenvectors.descMat_[0],
          work, &lwork, iwork, &liwork, &info);

  if(mpi->mpiHead()) {
     std::cout << "Matrix diagonalization completed." << std::endl;
     mpi->time();
  }

  if(info != 0) {
    if (mpi->mpiHead()) {
      std::cout << "Developer Error: "
                "One of the input params to PDSYEVD is wrong!" << std::endl;
    }
    Error("PDSYEVD failed.", info);
  }

  if( eigenvalues[0] < 0 ) { // negative modes were found
    if(mpi->mpiHead()) {
      Warning("Relaxons diagonalization found negative eigenvalues."
                "\n\tThis can happen when there's a bit of numerical noise on the scattering matrix,"
                "\n\tand finding them may indicate the calculation is unconverged."
                "\n\tWhile we simply do not include these when computing transport, "
                "\n\tyou may want to consider using with more wavevectors or an improved DFT calculation."
                "\n\tAdditionally, setting symmetrizeMatrix = true in your input file will help.");
      std::cout << "These eigenvalues are (in atomic units):" << std::endl;
      for (int i = 0; eigenvalues[i] <= 0; i++) {
        std::cout << i << " " << eigenvalues[i] << std::endl;
        if(i+1 == numRows_) break; // avoid out of bounds
      }
      std::cout << std::endl;
    }
  }
  // copy things into output containers
  std::vector<double> eigenvalues_(numRows_);
  for (int i = 0; i < numRows_; i++) {
    eigenvalues_[i] = *(eigenvalues + i);
  }
  delete[] eigenvalues;
  delete[] work;
  // note that the scattering matrix now has different values
  return std::make_tuple(eigenvalues_, eigenvectors);
}

template <>
std::tuple<std::vector<double>, ParallelMatrix<std::complex<double>>>
ParallelMatrix<std::complex<double>>::diagonalize() {

  if (numRows_ != numCols_) {
    Error("Can not diagonalize non-square matrix");
  }
  if ( numBlasRows_ != numBlasCols_ ) {
    Error("Cannot diagonalize via scalapack with a non-square process grid!");
  }
  double* eigenvalues = nullptr;
  eigenvalues = new double[numRows_];

  // the two zeros here trigger the default square blacs process grid in initBlacs
  ParallelMatrix<std::complex<double>> eigenvectors(
      numRows_, numCols_, 0, 0, numBlocksRows_, numBlocksCols_);

  // find the value of lwork and lrwork. These are internal "scratch" arrays
  int NB = descMat_[5];
  int aZero = 0;
  int NN = std::max(std::max(numRows_, NB), 2);
  int NP0 = numroc_(&NN, &NB, &aZero, &aZero, &numBlasRows_);
  int NQ0 = numroc_(&NN, &NB, &aZero, &aZero, &numBlasCols_);
  int lwork = (NP0 + NQ0 + NB) * NB + 3 * numRows_ + numRows_ * numRows_;
  int lrwork = 2 * numRows_ + 2 * numRows_ - 2;

  std::complex<double>* work = nullptr;
  work = new std::complex<double>[lwork];
  std::complex<double>* rwork = nullptr;
  rwork = new std::complex<double>[lrwork];

  char jobz = 'V';  // also eigenvectors
  char uplo = 'U';  // upper triangular
  int ia = 1;       // row index from which we diagonalize
  int ja = 1;       // row index from which we diagonalize
  int info = 0;
  pzheev_(&jobz, &uplo, &numRows_, mat, &ia, &ja, &descMat_[0], eigenvalues,
          eigenvectors.mat, &ia, &ja, &eigenvectors.descMat_[0], work, &lwork,
          rwork, &lrwork, &info);

  if (info != 0) {
    Error("PZHEEV failed", info);
  }

  std::vector<double> eigenvalues_(numRows_);
  for (int i = 0; i < numRows_; i++) {
    eigenvalues_[i] = *(eigenvalues + i);
  }
  delete[] eigenvalues;
  delete[] work;
  delete[] rwork;

  // note that the scattering matrix now has different values
  return std::make_tuple(eigenvalues_, eigenvectors);
}

// function to only compute some eigenvectors/values
template <>
std::tuple<std::vector<double>, ParallelMatrix<double>>
                ParallelMatrix<double>::diagonalize(int numEigenvalues_,
                                                    bool checkNegativeEigenvalues) {

  int numEigenvalues = numEigenvalues_;

  if (numRows_ != numCols_) {
    Error("Cannot diagonalize non-square matrix");
  }
  if ( numBlasRows_ != numBlasCols_ ) {
    Error("Cannot diagonalize via scalapack with a non-square process grid!");
  }
  if ( numRows_ < numEigenvalues) {
    numEigenvalues = numRows_;
  }

  // eigenvalue return container
  double *eigenvalues;
  allocate(eigenvalues, numRows_);

  // NOTE even though we only need numEigenvalues + numEigenvalues worth of z matrix,
  // scalapack absolutely requires this matrix be the same size as the original.
  // It's a huge waste of memory, and we should check to see if another code
  // can get around this.
  // Make a new PMatrix to receive the output
  ParallelMatrix<double> eigenvectors(numRows_, numCols_,
                        0, 0, numBlocksRows_,numBlocksCols_, blacsContext_);

  char jobz = 'V';  // also eigenvectors
  char uplo = 'U';  // upper triangular
  int ia = 1;       // row index of start of A
  int ja = 1;       // col index of start of A
  int iz = 1;       // row index of start of Z
  int jz = 1;       // row index of start of Z

  int info = 0;     // error code on return
  int m = 0;        // filled on return with number of eigenvalues found
  int nz = 0;       // filled on return with number of eigenvectors found

  char range = 'I'; // compute a range (from smallest to largest) of the eigenvalues
  int il = 1;       // lower eigenvalue index (indexed from 1)
  int iu = numEigenvalues;              // higher eigenvalue index (indexed from 1)
  double vl = -1;                 // not used unless range = V
  double vu = 0;                  // not used unless range = V

  // we will let pdseyv determine lwork for us. if we run it with
  // lwork = -1 and work of length 1, it will fill work with
  // an appropriate lwork number
  double *work;
  int *iwork;
  int lwork = -1;
  int liwork = -1; // liwork seems not to be determined this way
  allocate(work, 1);
  allocate(iwork, 1);

  pdsyevr_(&jobz, &range, &uplo,  &numRows_, mat, &ia, &ja, &descMat_[0],
        &vl, &vu, &il, &iu, &m, &nz, eigenvalues,
        eigenvectors.mat, &iz, &jz, &eigenvectors.descMat_[0],
        work, &lwork, iwork, &liwork, &info);

  lwork=int(work[0]);
  delete[] work;
  delete[] iwork;
  allocate(work, lwork);

  // for some reason scalapack won't fill liwork automatically:
  //Let nnp = max( n, nprow*npcol + 1, 4 ). Then:
  //liwork≥ 12*nnp + 2*n when the eigenvectors are desired
  int nnp = std::max(std::max(numRows_, numBlasRows_*numBlasCols_ + 1), 4);
  liwork = 12*nnp + 2*numRows_;
  allocate(iwork, liwork);

  // first, call and report any negative eigenvalues ------------------------------
  // we want to note these for convergence reasons
  if(checkNegativeEigenvalues) {

    if(mpi->mpiHead()) mpi->time();
    if(mpi->mpiHead())
      std::cout << "Checking scattering matrix for negative eigenvalues." << std::endl;

    eigenvectors = *this; // use the space of this matrix first to
                // placehold the actual matrix, as it's changed by pdsyevr
    range = 'V';
    jobz = 'N';
    vl = -1.;
    vu = 1e-16;
    pdsyevr_(&jobz, &range, &uplo,  &numRows_, eigenvectors.mat, &ia, &ja, &descMat_[0],
          &vl, &vu, &il, &iu, &m, &nz, eigenvalues,
          eigenvectors.mat, &iz, &jz, &descMat_[0],
          work, &lwork, iwork, &liwork, &info);

    if( m > 3 ) { // more than just the zero eigenmode was found
      if(mpi->mpiHead()) {
        Warning("Relaxons diagonalization found " + std::to_string(m) +
               " in the range -1 < eigenvalues <= 0."
                  "\n\tThis can happen when there's a bit of numerical noise on the scattering matrix,"
                  "\n\tand finding them may indicate the calculation is unconverged."
                  "\n\tWhile we simply do not include these when computing transport, "
                  "\n\tyou may want to consider using with more wavevectors or an improved DFT calculation."
                  "\n\tAdditionally, setting symmetrizeMatrix = true in your input file will help.");
        std::cout << "These eigenvalues are (in atomic units):" << std::endl;

        for (int i = 0; i < m; i++) {
          std::cout << i << " " << eigenvalues[i] << std::endl;
        }
        std::cout << std::endl;
      }
    }
  }

  // now we perform the regular call to get the largest ones ---------------------
  if(mpi->mpiHead()) mpi->time();
  if(mpi->mpiHead()) {
    std::cout << "Now computing first " << numEigenvalues <<
        " eigenvalues and vectors of the scattering matrix." << std::endl;
  }

  range = 'I';
  jobz = 'V';
  eigenvectors.zeros();

  // We could make sure these two matrices have identical blacsContexts.
  // However, as dim(Z) must = dim(A), we can just pass A's desc twice.
  pdsyevr_(&jobz, &range, &uplo,  &numRows_, mat, &ia, &ja, &descMat_[0],
        &vl, &vu, &il, &iu, &m, &nz, eigenvalues,
        eigenvectors.mat, &iz, &jz, &eigenvectors.descMat_[0],
        work, &lwork, iwork, &liwork, &info);

  if(info != 0) {
    if (mpi->mpiHead()) {
      std::cout << "Developer Error: "
                "One of the input params to PDSYEVR is wrong!" << std::endl;
    }
    Error("PDSYEVR failed.", info);
  }
  if(mpi->mpiHead()) mpi->time();


  // copy to return containers and free the no longer used containers.
  std::vector<double> eigenvalues_(numEigenvalues);
  for (int i = 0; i < numEigenvalues; i++) {
    eigenvalues_[i] = *(eigenvalues + i);
  }
  delete[] eigenvalues;
  delete[] work; delete[] iwork;

  // note that the scattering matrix now has different values
  // it's going to be the upper triangle and diagonal of A
  return std::make_tuple(eigenvalues_, eigenvectors);
}

#endif  // MPI_AVAIL
