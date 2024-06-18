#include "common_kokkos.h"
#include "eigen.h"
#include "mpiHelper.h"
#include <stdexcept>

#ifdef KOKKOS_ENABLE_CUDA
#include <cusolver_common.h>
#include <cusolverDn.h>
#endif

void kokkosZHEEV(StridedComplexView3D &A, DoubleView2D &W) {
  // kokkos people didn't implement the diagonalization of matrices.
  // So, we have to do a couple of dirty tricks

  int M = A.extent(0);// number of matrices
  int N = A.extent(1);// matrix size is NxN

#ifndef KOKKOS_ENABLE_CUDA
  // general case, copy to CPU (if necessary) and
  // use Eigen to solve the diagonalization problems

  auto A_h = Kokkos::create_mirror_view(A);
  auto W_h = Kokkos::create_mirror_view(W);
  Kokkos::deep_copy(A_h, A);

 #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    // this is a pointer to the storage, N values, viewing it as std::complex
    // note: don't use kokkos::subview to locate the matrix.
    // In fact, Kokkos::Views don't play well within the #pragme loop.
    // hence, I take advantage of A having a right-layout and place the offset
    // i*N*N that locates the start of the i-th matrix.
    auto *storage = reinterpret_cast<std::complex<double> *>(A_h.data()) + i*N*N;

    // now, I feed the data to Eigen
    // BEWARE: this statement doesn't do data copy, but points directly to the
    // memory array. No out-of-bounds checks are made!
    // Each matrix is column-major due to the strided layout
    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> thisH(storage, N, N);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(thisH);
    Eigen::VectorXd energies = eigenSolver.eigenvalues();
    Eigen::MatrixXcd eigenvectors = eigenSolver.eigenvectors();
    for (int m = 0; m < N; ++m) {
      W_h(i, m) = energies(m);
      for (int n = 0; n < N; ++n) {
        //A_h(i, m, n) *= 3; // TODO: undo
        A_h(i, m, n) = eigenvectors(m, n);
      }
    }
  }
  Kokkos::deep_copy(A, A_h);
  Kokkos::deep_copy(W, W_h);
#else
  // CUDA special case, call the cuSOLVER library directly

  // cuSOLVER setup
  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);
  syevjInfo_t params;
  cusolverDnCreateSyevjInfo(&params);
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  int lda = N;
  int lwork;

  cuDoubleComplex* Aptr = (cuDoubleComplex*) A.data();
  double *Wptr = W.data();

  // determine work array size
  cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, N, Aptr, lda, Wptr, &lwork, params, M);
  //printf("lwork = %d\n", lwork);

  ComplexView1D work("work", lwork);
  cuDoubleComplex* workptr = (cuDoubleComplex*) work.data();
  IntView1D info("info", M);

  // call diagonalization
  cusolverDnZheevjBatched(handle, jobz, uplo, N, Aptr, lda, Wptr, workptr, lwork, info.data(), params, M);

  // check if any diagonalizations went wrong
  int sum_infos = 0;
  Kokkos::parallel_reduce(M, KOKKOS_LAMBDA(int i, int &sum){
      sum += info(i)!=0;
  }, sum_infos);

  if(sum_infos > 0){
    // check info for each diagonalization
    auto info_h = Kokkos::create_mirror_view(info);
    Kokkos::deep_copy(info_h, info);

    for(int i = 0; i < M; i++){
      if(info_h(i) != 0) printf("info[%d] = %d\n", i, info_h(i));
    }

    throw std::runtime_error("Error in cusolverDnZheevjBatched!");
  }
#endif
}

DeviceManager *kokkosDeviceMemory = nullptr;

DeviceManager::DeviceManager() {

  memoryUsed = 0.;
  char *memStr = std::getenv("MAXMEM");
  if (memStr != nullptr) {
    memoryTotal = std::atof(memStr) * 1.0e9;
  } else {
    memoryTotal = 16.0e9; // 16 Gb is our educated guess for available memory
  }
}

void DeviceManager::addDeviceMemoryUsage(const double& memoryBytes) {
  this->memoryUsed += memoryBytes;
  //printf("rank %d added %g MB, using %g MB\n", mpi->getRank(), memoryBytes/1e6, this->memoryUsed/1e6);
  if (this->memoryUsed > this->memoryTotal) {
    Warning("DeviceManager: running low on device memory.");
  }
}

void DeviceManager::removeDeviceMemoryUsage(const double& memoryBytes) {
  this->memoryUsed -= memoryBytes;
  //printf("rank %d removed %g MB, using %g MB\n", mpi->getRank(), memoryBytes/1e6, this->memoryUsed/1e6);
}

double DeviceManager::getAvailableMemory() {
  return this->memoryTotal - this->memoryUsed;
}

double DeviceManager::getTotalMemory() {
  return this->memoryTotal;
}

std::vector<std::vector<int>> DeviceManager::splitToBatches(
    const std::vector<int>& iterator, int& batchSize) {

  batchSize = std::min(batchSize, int(iterator.size()));

  // decide how many chunks we want to do
  // we want to be conservative and have 1 more chunk if necessary,
  // as this likely leads to a smaller memory requirement per batch
  int iteratorSize = iterator.size();
  int numBatches = iteratorSize / batchSize; // 4/3 = 1
  if (iteratorSize % batchSize != 0) ++numBatches; // 2 numBatches

  int start = 0;
  int end = batchSize;
  std::vector<std::vector<int>> result(numBatches);
  int iBatch = 0;
  while (start < end) {
    int thisBatchSize = end - start;
    std::vector<int> batch(thisBatchSize);
    for (int j=0; j<thisBatchSize; ++j) {
      batch[j] = iterator[start + j];
    }
    result[iBatch] = batch;
    start += batchSize;
    end += batchSize;
    end = std::min(end, iteratorSize);
    ++iBatch;
  }

//  std::vector<std::vector<int>> result(numBatches);
//  for (int iBatch = 0; iBatch < numBatches; iBatch++) {
//    // start and end point for current batch
//    int start = (iteratorSize * iBatch) / numBatches;
//    // the min avoids possible out-of-bounds when the last chunk is slightly
//    // smaller than the iteratorSize/numBatches
//    int end = std::min((iteratorSize * (iBatch + 1)) / numBatches, iteratorSize);
//    int thisBatchSize = end - start;
//    std::vector<int> batch(thisBatchSize);
//    for (int j=0; j<thisBatchSize; ++j) {
//      batch[j] = iterator[start + j];
//    }
//    result[iBatch] = batch;
//  }
  return result;
}

void initKokkos(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  kokkosDeviceMemory = new DeviceManager();
}

void deleteKokkos() {
  delete kokkosDeviceMemory;
  Kokkos::finalize();
}

void kokkosInfo() {
  if (mpi->mpiHead()) {
    printf("The maximal memory used by the device (Kokkos) will be %g "
           "GB,\nset the MAXMEM environment variable to the preferred memory "
           "usage in GB.\n",
           kokkosDeviceMemory->getAvailableMemory() / 1.0e9);
  }
}
