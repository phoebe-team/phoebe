// #include "interaction_elph_SVD.h"
// #include "interaction_elph_parsing.h"
// #include "Kokkos_View.hpp"
// #include "common_kokkos.h"
// #include "highfive/H5DataSet.hpp"
// #include "impl/Kokkos_Profiling.hpp"
// #include "interaction_elph.h"
// #include "context.h"
// #include "traits/Kokkos_IterationPatternTrait.hpp"
// #include <Kokkos_Core.hpp>
// #include <KokkosBlas2_gemv.hpp>
// #include <cstddef>
// #include <stdexcept>

// #ifdef HDF5_AVAIL
// #include <Kokkos_ScatterView.hpp>
// #include <highfive/H5Easy.hpp>
// #endif

// // constructors ==================================

// InteractionElPhSVD(Crystal& crystal, const Context& context,
//                 //const Eigen::Tensor<std::complex<double>, 5>& couplingWannier,
//                 //const Eigen::MatrixXd& elBraviasVectors,
//                 //const Eigen::VectorXd& elBraviasVectorsDegeneracies,
//                 //const Eigen::MatrixXd& phBravaisVectors,
//                 //const Eigen::VectorXd& phBraviasVectorsDegeneracies,
//                 PhononH0* phononH0 = nullptr)
//                 : InteractionElPhBase(crystal, phononH0) {
//                         //elBraviasVectors, elBraviasVectorsDegeneracies,
//                         //phBravaisVectors, phBraviasVectorsDegeneracies) {

//     // parse the header data from HDF5 file
//     auto t = parseHeaderHDF5(context);
//     numElBands = std::get<0>(t);
//     numPhBands = std::get<1>(t);
//     elBravaisVectors = std::get<3>(t);
//     phBravaisVectors = std::get<4>(t);
//     elBravaisVectorsDegeneracies = std::get<6>(t);
//     phBravaisVectorsDegeneracies = std::get<7>(t);
//     numElBravaisVectors = elBravaisVectorsDegeneracies.size();
//     numPhBravaisVectors = phBravaisVectorsDegeneracies.size();

//     // parse the SVD data from the HDF5 file
//     parseSVDKokkos(context);

//   }

// // parsing functions ========================================================

// //Function to extraxt indices from group name
// inline std::tuple<int, int, int> extractIndicesFromGroup(const std::string& groupName){
//     std::regex indexRegex("slice_GrR_SVD_(\\d+)_(\\d+)_(\\d+)");
//     std::smatch match;
//     if (std::regex_match(groupName, match, indexRegex)) {
//         int idxX = std::stoi(match[1].str());
//         int idxY = std::stoi(match[2].str());
//         int idxZ = std::stoi(match[3].str());

//         return{idxX,idxY,idxZ};
//     }
//     throw std::invalid_argument("Name format doesnt match, please check importinng file's name: " + groupName);
// }




// /**
// * Function to process all the SVD groups, by taking a HighFive:Group (svdGroup)
// * as impuit and returning a vector of SVDGroupData in which each SVDGroupData
// * contains smart pointers to the singularMatrix, rightMatrix, leftMatrix,
// * idxX, idxY, idxZ.
// * @param svdGroup HighFive::Group where SVD datasets are stored
// * @return std::vector<SVDGroupData> container of SVDGroupData containting pointers
// * to singularMatrix, rightMatrix, leftMatrix, idxX, idxY, idxZ.
// */
// std::vector<InteractionElPhSVD::SVDGroupData>
// InteractionElPhSVD::processAllSVDGroups(const HighFive::Group& svdGroup,
//                                         size_t& num_i, size_t& num_j, size_t& num_eta)
// {

//     // Container to store all the SVD data
//     std::vector<InteractionElPhSVD::SVDGroupData> allData;

//     auto groupNames = svdGroup.listObjectNames();

//     size_t max_X{}, max_Y{}, max_Z{};

//     // read in each group
//     for (const auto& groupName : groupNames) {
//         HighFive::Group subGroup = svdGroup.getGroup(groupName);

//         auto [idxX, idxY, idxZ] = extractIndicesFromGroup(groupName);

//         if (idxX != max_X){max_X = idxX;}
//         if (idxY != max_Y){max_Y = idxY;}
//         if (idxZ != max_Z){max_Z = idxZ;}

//         auto singularMatrix = std::make_unique<std::vector<double>>();
//         auto rightMatrix = std::make_unique<std::vector<double>>();
//         auto leftMatrix = std::make_unique<std::vector<double>>();

//         subGroup.getDataSet("S").read(*singularMatrix);
//         subGroup.getDataSet("V").read(*rightMatrix);
//         subGroup.getDataSet("U").read(*leftMatrix);

//         allData.emplace_back(
//             std::move(singularMatrix),
//             std::move(rightMatrix),
//             std::move(leftMatrix),
//             idxX,
//             idxY,
//             idxZ
//         );
//     }


//     num_i = max_X + 1;
//     num_j = max_Y + 1;
//     num_eta = max_Z + 1;
//     return allData;
// }

// // Function to process the SVD data and populate the Kokkos::View with dimensions [totalLayers, leftMatrix, rightMatrix]
// void InteractionElPhSVD::parseSVDKokkos(const Context& context) {
//     try {
//         // Open the HDF5 file
//         HighFive::File file(context.getElPhFileName(), HighFive::File::ReadOnly);

//         // Get the specified SVD group
//         HighFive::Group svdGroup = file.getGroup(svdGroupName);


//         size_t num_i, num_j, num_eta;
//         // Parse all SVD data from the group using processAllSVDGroups
//         std::vector<InteractionElPhSVD::SVDGroupData> svdData = processAllSVDGroups(svdGroup, num_i, num_j, num_eta);

//         // Extract dimensions from the parsed data
//         size_t num_ijn = numElBands * numElBands * numPhBands;
//         size_t numSingularValues = svdData[0].singularVector->size();
//         size_t leftMatrixCols = svdData[0].leftMatrix->size() / numSingularValues;
//         size_t rightMatrixRows = svdData[0].rightMatrix->size();



//         // Allocate the Kokkos containers
//         //kokkosContainer = Kokkos::View<double***>("SVD_KokkosView", totalLayers, leftDim, rightDim);
//         Kokkos::realloc(SVD_Y, num_i, num_j, num_eta, leftMatrixCols, numSingularValues);
//         Kokkos::realloc(SVD_Vt, num_i, num_j, num_eta, rightMatrixRows, numSingularValues);



//         // Fill the Kokkos container with the parsed data
//         auto svdRawData = svdData.data();  // Get a raw pointer to the vector for Kokkos compatibility
//         Kokkos::parallel_for("FillKokkosView", num_ijn, KOKKOS_LAMBDA(const size_t idx) {
//             // Compute (i, j, eta) from linear index
//             size_t i = idx / (num_j * num_eta);
//             size_t j = (idx / num_eta) % num_j;
//             size_t eta = idx % num_eta;

//             auto* singularValues = svdRawData[idx].singularValues->data();
//             auto* leftMatrix = svdRawData[idx].leftMatrix->data();
//             auto* rightMatrix = svdRawData[idx].rightMatrix->data();

//             // Populate SVD_Y
//             for (size_t l = 0; l < leftMatrixCols; ++l) {
//                 for (size_t s = 0; s < numSingularValues; ++s) {
//                     SVD_Y(i, j, eta, l, s) = singularValues[s] * leftMatrix[l * numSingularValues + s];
//                 }
//             }

//             // Populate SVD_Vt
//             for (size_t r = 0; r < rightMatrixRows; ++r) {
//                 for (size_t s = 0; s < numSingularValues; ++s) {
//                     SVD_Vt(i, j, eta, r, s) = rightMatrix[r * numSingularValues + s];
//                 }
//             }
//         });

//         std::cout << "Kokkos container has been successfully populated!" << std::endl;

//     } catch (const HighFive::Exception& e) {
//         throw std::runtime_error("Error reading HDF5 file or parsing SVD data: " + std::string(e.what()));
//     }
// }

// // KEYNESH edits here -- transform using Uk, e^{ik1 . Re}
// void InteractionElPhBase::cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C) {
//     Kokkos::Profiling::pushRegion("cacheElPh");  //This starts the profiling
//     //  int numWannier = numElBands;
//     auto nb1 = int(eigvec1.cols());
//     Kokkos::complex<double> complexI(0.0, 1.0);
//     auto elPhCached = this->elPhCached;
//     int numPhBands = this->numPhBands;
//     int numElBands = this->numElBands;
//     int numElBravaisVectors = this->numElBravaisVectors;
//     int numPhBravaisVectors = this->numPhBravaisVectors;

//     int totalSlices = numPhBands * numElBands * numElBands;
//     Kokkos::View<double**> leftMatrix;
//     Kokkos::View<double**> rightMatrix;

//     double memory = InteractionBase::getDeviceMemoryUsage();
//     kokkosDeviceMemory->removeDeviceMemoryUsage(memory);

//     int pool_rank = mpi->getRank(mpi->intraPoolComm);
//     int pool_size = mpi->getSize(mpi->intraPoolComm);

// #ifdef MPI_AVAIL
//     mpi_requests.resize(pool_size);
//     elPhCached_hs.resize(pool_size);
// #endif
//     ComplexView4D g1(Kokkos::ViewAllocateWithoutInitializing("g1"),
//         numPhBravaisVectors, numPhBands, numElBands, numElBands);

//     // ComplexView3D truncSlices(Kokkos::ViewAllocateWithoutInitializing("truncSlices"),
//     //     totalSlices, leftMatrix, rightMatrix);
//     // note: this loop is a parallelization over the group (Pool) of MPI
//     // processes, which together contain all the el-ph coupling tensor
//     // First, loop over the MPI processes in the pool
//     for (int iPool = 0; iPool < pool_size; iPool++) {
//       Kokkos::Profiling::pushRegion("cacheElPh setup");

//       // the current MPI process must first broadcast the k-point and the
//       // eigenvector that will be computed now.
//       // So, first broadcast the number of bands of the iPool-th process
//       int poolNb1 = 0;
//       if (iPool == pool_rank) {
//         poolNb1 = nb1;
//       }
//       mpi->bcast(&poolNb1, mpi->intraPoolComm, iPool);

//       // broadcast also the wavevector and the eigenvector at k for process iPool
//       Eigen::Vector3d poolK1C = Eigen::Vector3d::Zero(); //Just a vector of ZEROS
//       Eigen::MatrixXcd poolEigvec1 = Eigen::MatrixXcd::Zero(poolNb1, numElBands);
//       if (iPool == pool_rank) {
//         poolK1C = k1C;
//         poolEigvec1 = eigvec1;
//       }
//       // broadcast to other processes on this pool
//       mpi->bcast(&poolK1C, mpi->intraPoolComm, iPool);
//       mpi->bcast(&poolEigvec1, mpi->intraPoolComm, iPool);

//       // now, copy the eigenvector and wavevector to the accelerator
//       ComplexView2D eigvec1_k("ev1", poolNb1, numElBands);
//       DoubleView1D poolK1C_k("k", 3);
//       {
//         HostComplexView2D eigvec1_h((Kokkos::complex<double> *) poolEigvec1.data(), poolNb1, numElBands);
//         HostDoubleView1D poolK1C_h(poolK1C.data(), 3);
//         Kokkos::deep_copy(eigvec1_k, eigvec1_h);
//         Kokkos::deep_copy(poolK1C_k, poolK1C_h);
//       }

//       // now compute the Fourier transform on electronic coordinates.
//       ComplexView5D couplingWannier_k = this->couplingWannier_k;
//       DoubleView2D elBravaisVectors_k = this->elBravaisVectors_k;
//       DoubleView2D phBravaisVectors_k = this->phBravaisVectors_k;
//       DoubleView1D elBravaisVectorsDegeneracies_k = this->elBravaisVectorsDegeneracies_k;
//       DoubleView1D phBravaisVectorsDegeneracies_k = this->phBravaisVectorsDegeneracies_k;
//       Kokkos::Profiling::popRegion();

//       // first we precompute the phases
//       ComplexView1D phases_k("phases", numElBravaisVectors);
//       Kokkos::parallel_for("phases_k", numElBravaisVectors,
//           KOKKOS_LAMBDA(int irE) {
//             double arg = 0.0;
//             for (int j = 0; j < 3; j++) {
//               arg += poolK1C_k(j) * elBravaisVectors_k(irE, j);
//             }
//             phases_k(irE) = exp(complexI * arg) / elBravaisVectorsDegeneracies_k(irE);
//           });
//       Kokkos::fence();

//       // // first we precompute the phases
//       // ComplexView1D phases_q("phasesForq", numPhBravaisVectors);
//       // Kokkos::parallel_for("phases_q", numPhBravaisVectors,
//       //     KOKKOS_LAMBDA(int iRp) {
//       //       double arg = 0.0;
//       //       for (int j = 0; j < 3; j++) {
//       //         arg += poolK1C_k(j) * phBravaisVectors_k(iRp, j);
//       //       }
//       //       phases_q(iRp) = exp(complexI * arg) / phBravaisVectorsDegeneracies_k(iRp);
//       //     });
//       // Kokkos::fence();
//       // Kokkos::Profiling::popRegion();

//       Kokkos::Profiling::popRegion();

//       // now we complete the Fourier transform
//       // We have to write two codes: one for when the GPU runs on CUDA,
//       // the other for when we compile the code without GPU support
//   #ifdef KOKKOS_ENABLE_CUDA
//       Kokkos::parallel_for(
//           "g1",
//           Range4D({0, 0, 0, 0},
//                   {numPhBravaisVectors, numPhBands, numElBands, numElBands}),
//           KOKKOS_LAMBDA(int irP, int nu, int iw1, int iw2) {
//             Kokkos::complex<double> tmp(0.0);
//             for (int irE = 0; irE < numElBravaisVectors; irE++) {
//               // important note: the first index iw2 runs over the k+q transform
//               // while iw1 runs over k
//               tmp += couplingWannier_k(irE, irP, nu, iw1, iw2) * phases_k(irE);
//             }
//             g1(irP, nu, iw1, iw2) = tmp;
//           });
//      Kokkos::fence();
//   #else

//     // Here we create a view to the elph matrix elements which represents it
//     // in 2D, so that we can use a matrix-vector product with the phases to accelerate
//     // an otherwise very expensive loop
//     //
//     // tutorial description of this gemv function:
//     // https://youtu.be/_qD4X66MQF8?t=2434
//     // read me about gemv https://github.com/kokkos/kokkos-kernels/wiki/BLAS-2%3A%3Agemv

//     // product of phase factor with g
//     Kokkos::View<Kokkos::complex<double>*> g1_1D(g1.data(), numPhBravaisVectors*numPhBands*numElBands*numElBands);
//     Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutRight> coupling_2D(couplingWannier_k.data(), numElBravaisVectors, numPhBravaisVectors*numPhBands*numElBands*numElBands);
//     KokkosBlas::gemv("T", Kokkos::complex<double>(1.0), coupling_2D, phases_k, Kokkos::complex<double>(0.0), g1_1D);

//     // // [indices, LeftMatrix, rightMatrix]
//     // Kokkos::View<Kokkos::complex<double>***, Kokkos::LayoutRight> threeLeggedMatrix("matrix_3D",
//     //                                                                         totalSlices,
//     //                                                                         leftMatrix,
//     //                                                                         rightMatrix);



//     // // Parallel loop over all indices
//     // Kokkos::parallel_for("MatrixVectorMultiplication", totalSlices, KOKKOS_LAMBDA(const int idx) {
//     //     // Extract the leftMatrix (2D slice) from the 3D container
//     //     auto leftMatrix = Kokkos::subview(threeLeggedMatrix, idx, Kokkos::make_pair(0, leftMatrix), Kokkos::ALL());

//     //     // Extract the result slice for this index (same shape as leftMatrixCols)
//     //     auto resultVector = Kokkos::subview(threeLeggedMatrix, idx, Kokkos::make_pair(0, leftMatrix), Kokkos::ALL());

//     //     // Perform the GEMV operation (transpose or no transpose as needed)
//     //     KokkosBlas::gemv("T", Kokkos::complex<double>(1.0), leftMatrix, phases_k, Kokkos::complex<double>(0.0), resultVector);
//     // });

//   /*
//     // Previous method -- slower than the gemv call
//       Kokkos::deep_copy(g1, Kokkos::complex<double>(0.0, 0.0));
//       Kokkos::Experimental::ScatterView<Kokkos::complex<double> ****> g1scatter(g1);
//       Kokkos::parallel_for(
//           "g1",
//           Range5D({0, 0, 0, 0, 0},
//                   {numElBravaisVectors, numPhBravaisVectors, numPhBands, numElBands, numElBands}),
//           KOKKOS_LAMBDA(int irE, int irP, int nu, int iw1, int iw2) {
//             auto g1 = g1scatter.access();
//             g1(irP, nu, iw1, iw2) += couplingWannier_k(irE, irP, nu, iw1, iw2) * phases_k(irE);
//           });
//       Kokkos::Experimental::contribute(g1, g1scatter);
//   */
//   #endif

//       // now we need to add the rotation on the electronic coordinates
//       // and finish the transformation on electronic coordinates
//       // we distinguish two cases. If each MPI process has the whole el-ph
//       // tensor, we don't need communication and directly store results in
//       // elPhCached. Otherwise, we need to do an MPI reduction

//       if (pool_size == 1) {
//         Kokkos::realloc(elPhCached, numPhBravaisVectors, numPhBands, poolNb1, numElBands);

//         Kokkos::parallel_for(
//             "elPhCached",
//             Range4D({0, 0, 0, 0},
//                     {numPhBravaisVectors, numPhBands, poolNb1, numElBands}),
//             KOKKOS_LAMBDA(int irP, int nu, int ib1, int iw2) {
//               Kokkos::complex<double> tmp(0.0);
//               for (int iw1 = 0; iw1 < numElBands; iw1++) {
//                 tmp += g1(irP, nu, iw1, iw2) * eigvec1_k(ib1, iw1);
//               }
//               elPhCached(irP, nu, ib1, iw2) = tmp;
//             });
//         Kokkos::fence();

//       } else {

//         ComplexView4D poolElPhCached_k(Kokkos::ViewAllocateWithoutInitializing("poolElPhCached"),
//                                      numPhBravaisVectors, numPhBands, poolNb1, numElBands);

//         Kokkos::parallel_for(
//             "elPhCached",
//             Range4D({0, 0, 0, 0},
//                     {numPhBravaisVectors, numPhBands, poolNb1, numElBands}),
//             KOKKOS_LAMBDA(int irP, int nu, int ib1, int iw2) {
//               Kokkos::complex<double> tmp(0.0);
//               for (int iw1 = 0; iw1 < numElBands; iw1++) {
//                 tmp += g1(irP, nu, iw1, iw2) * eigvec1_k(ib1, iw1);
//               }
//               poolElPhCached_k(irP, nu, ib1, iw2) = tmp;
//             });

//         // note: we do the reduction after the rotation, so that the tensor
//         // may be a little smaller when windows are applied (nb1<numWannier)

//         // do a mpi->allReduce across the pool
//         //mpi->allReduceSum(&poolElPhCached_h, mpi->intraPoolComm);

//         Kokkos::Profiling::pushRegion("copy elPhCached to CPU");
//         // copy from accelerator to CPU
//         auto poolElPhCached_h = Kokkos::create_mirror_view(poolElPhCached_k);
//         Kokkos::deep_copy(poolElPhCached_h, poolElPhCached_k);
//         Kokkos::Profiling::popRegion();

//         elPhCached_hs[iPool] = poolElPhCached_h;

//   #ifdef MPI_AVAIL
//         // start reduction for current iteration
//         Kokkos::Profiling::pushRegion("call MPI_reduce");
//         // previously, we had tried non-blocking collectives here.
//         // However, this resulted in some segfaults, so we fell back to standard reduce.
//         if (pool_rank == iPool) {
//           //MPI_Ireduce(MPI_IN_PLACE, poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm), &mpi_requests[iPool]);
//           MPI_Reduce(MPI_IN_PLACE, poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm)); //, &mpi_requests[iPool]);
//         }
//         else{
//           MPI_Reduce(poolElPhCached_h.data(), poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm)); //, &mpi_requests[iPool]);
//           //MPI_Ireduce(poolElPhCached_h.data(), poolElPhCached_h.data(), poolElPhCached_h.size(), MPI_COMPLEX16, MPI_SUM, iPool, mpi->getComm(mpi->intraPoolComm), &mpi_requests[iPool]);
//         }
//         Kokkos::Profiling::popRegion();
//   #endif

//       }
//     }
//     this->elPhCached = elPhCached;
//     double newMemory = InteractionBase::getDeviceMemoryUsage();
//     kokkosDeviceMemory->addDeviceMemoryUsage(newMemory);
//     Kokkos::Profiling::popRegion();

// }


// // KEYNESH edits here
// // second part of the wannier transform, the transform using phase(i q.Rp), and uq, Uk2
// void InteractionElPhWan::calcCouplingSquared(
//     const Eigen::MatrixXcd &eigvec1,
//     const std::vector<Eigen::MatrixXcd> &eigvecs2,
//     const std::vector<Eigen::MatrixXcd> &eigvecs3,
//     const std::vector<Eigen::Vector3d> &q3Cs,
//     const std::vector<Eigen::VectorXcd> &polarData) {
//   Kokkos::Profiling::pushRegion("calcCouplingSquared");
//   int numWannier = numElBands;
//   auto nb1 = int(eigvec1.cols());
//   auto numLoops = int(eigvecs2.size());

// #ifdef MPI_AVAIL
//   int pool_rank = mpi->getRank(mpi->intraPoolComm);
//   int pool_size = mpi->getSize(mpi->intraPoolComm);
//   if(pool_size > 1 && mpi_requests[0] != MPI_REQUEST_NULL){
//       Kokkos::Profiling::pushRegion("wait for reductions");
//       // wait for MPI_Ireduces from cacheCoupling
//       //MPI_Waitall(pool_size, mpi_requests.data(), MPI_STATUSES_IGNORE);
//       Kokkos::Profiling::popRegion();

//       Kokkos::Profiling::pushRegion("copy to GPU");
//       this->elPhCached = Kokkos::create_mirror_view_and_copy(
//           Kokkos::DefaultExecutionSpace(), elPhCached_hs[pool_rank]
//       );
//       Kokkos::Profiling::popRegion();
//   }
// #endif

//   auto elPhCached = this->elPhCached;
//   int numPhBands = this->numPhBands;
//   int numPhBravaisVectors = this->numPhBravaisVectors;
//   DoubleView2D phBravaisVectors_k = this->phBravaisVectors_k;
//   DoubleView1D phBravaisVectorsDegeneracies_k = this->phBravaisVectorsDegeneracies_k;

//   // get nb2 for each ik and find the max
//   // since loops and views must be rectangular, not ragged
//   IntView1D nb2s_k("nb2s", numLoops);
//   int nb2max = 0;
//   auto nb2s_h = Kokkos::create_mirror_view(nb2s_k);
//   for (int ik = 0; ik < numLoops; ik++) {
//     nb2s_h(ik) = int(eigvecs2[ik].cols());
//     if (nb2s_h(ik) > nb2max) {
//       nb2max = nb2s_h(ik);
//     }
//   }
//   Kokkos::deep_copy(nb2s_k, nb2s_h);

//   // Polar corrections are computed on the CPU and then transferred to GPU

//   IntView1D usePolarCorrections("usePolarCorrections", numLoops);
//   ComplexView4D polarCorrections(Kokkos::ViewAllocateWithoutInitializing("polarCorrections"),
//                                  numLoops, numPhBands, nb1, nb2max);
//   auto usePolarCorrections_h = Kokkos::create_mirror_view(usePolarCorrections);
//   auto polarCorrections_h = Kokkos::create_mirror_view(polarCorrections);

//   // precompute all needed polar corrections
// #pragma omp parallel for
//   for (int ik = 0; ik < numLoops; ik++) {
//     Eigen::Vector3d q3C = q3Cs[ik];
//     Eigen::MatrixXcd eigvec2 = eigvecs2[ik];
//     Eigen::MatrixXcd eigvec3 = eigvecs3[ik];
//     usePolarCorrections_h(ik) = usePolarCorrection && q3C.norm() > 1.0e-8;
//     if (usePolarCorrections_h(ik)) {
//       Eigen::Tensor<std::complex<double>, 3> singleCorrection =
//           polarCorrectionPart2(eigvec1, eigvec2, polarData[ik]);
//       for (int nu = 0; nu < numPhBands; nu++) {
//         for (int ib1 = 0; ib1 < nb1; ib1++) {
//           for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
//             polarCorrections_h(ik, nu, ib1, ib2) =
//                 singleCorrection(ib1, ib2, nu);
//           }
//         }
//       }
//     } else {
//       Kokkos::complex<double> kZero(0., 0.);
//       for (int nu = 0; nu < numPhBands; nu++) {
//         for (int ib1 = 0; ib1 < nb1; ib1++) {
//           for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
//             polarCorrections_h(ik, nu, ib1, ib2) = kZero;
//           }
//         }
//       }
//     }
//   }

//   Kokkos::deep_copy(polarCorrections, polarCorrections_h);
//   Kokkos::deep_copy(usePolarCorrections, usePolarCorrections_h);

//   // copy eigenvectors etc. to device
//   DoubleView2D q3Cs_k("q3", numLoops, 3);
//   ComplexView3D eigvecs2Dagger_k("ev2Dagger", numLoops, numWannier, nb2max),
//       eigvecs3_k("ev3", numLoops, numPhBands, numPhBands);
//   {
//     auto eigvecs2Dagger_h = Kokkos::create_mirror_view(eigvecs2Dagger_k);
//     auto eigvecs3_h = Kokkos::create_mirror_view(eigvecs3_k);
//     auto q3Cs_h = Kokkos::create_mirror_view(q3Cs_k);

// #pragma omp parallel for default(none) shared(eigvecs3_h, eigvecs2Dagger_h, nb2s_h, q3Cs_h, q3Cs_k, q3Cs, numLoops, numWannier, numPhBands, eigvecs2Dagger_k, eigvecs3_k, eigvecs2, eigvecs3)
//     for (int ik = 0; ik < numLoops; ik++) {
//       for (int i = 0; i < numWannier; i++) {
//         for (int j = 0; j < nb2s_h(ik); j++) {
//           eigvecs2Dagger_h(ik, i, j) = std::conj(eigvecs2[ik](i, j));
//         }
//       }
//       for (int i = 0; i < numPhBands; i++) {
//         for (int j = 0; j < numPhBands; j++) {
//           eigvecs3_h(ik, i, j) = eigvecs3[ik](j, i);
//         }
//       }
//       for (int i = 0; i < numPhBands; i++) {
//         for (int j = 0; j < numPhBands; j++) {
//           eigvecs3_h(ik, i, j) = eigvecs3[ik](j, i);
//         }
//       }

//       for (int i = 0; i < 3; i++) {
//         q3Cs_h(ik, i) = q3Cs[ik](i);
//       }
//     }
//     Kokkos::deep_copy(eigvecs2Dagger_k, eigvecs2Dagger_h);
//     Kokkos::deep_copy(eigvecs3_k, eigvecs3_h);
//     Kokkos::deep_copy(q3Cs_k, q3Cs_h);
//   }

//   // now we finish the Wannier transform. We have to do the Fourier transform
//   // on the lattice degrees of freedom, and then do two rotations (at k2 and q)
//   ComplexView2D phases("phases", numLoops, numPhBravaisVectors);
//   Kokkos::complex<double> complexI(0.0, 1.0);
//   Kokkos::parallel_for(
//       "phases", Range2D({0, 0}, {numLoops, numPhBravaisVectors}),
//       KOKKOS_LAMBDA(int ik, int irP) {
//         double arg = 0.0;
//         for (int j = 0; j < 3; j++) {
//           arg += q3Cs_k(ik, j) * phBravaisVectors_k(irP, j);
//         }
//         phases(ik, irP) =
//             exp(complexI * arg) / phBravaisVectorsDegeneracies_k(irP);
//      });
//    Kokkos::fence();

//   ComplexView4D g3(Kokkos::ViewAllocateWithoutInitializing("g3"), numLoops, numPhBands, nb1, numWannier);
//   Kokkos::parallel_for(
//       "g3", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, numWannier}),
//       KOKKOS_LAMBDA(int ik, int nu, int ib1, int iw2) {
//         Kokkos::complex<double> tmp(0., 0.);
//         for (int irP = 0; irP < numPhBravaisVectors; irP++) {
//           tmp += phases(ik, irP) * elPhCached(irP, nu, ib1, iw2);
//         }
//         g3(ik, nu, ib1, iw2) = tmp;
//       });
//   Kokkos::realloc(phases, 0, 0);

//   ComplexView4D g4(Kokkos::ViewAllocateWithoutInitializing("g4"), numLoops, numPhBands, nb1, numWannier);
//   Kokkos::parallel_for(
//       "g4", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, numWannier}),
//       KOKKOS_LAMBDA(int ik, int nu2, int ib1, int iw2) {
//         Kokkos::complex<double> tmp(0., 0.);
//         for (int nu = 0; nu < numPhBands; nu++) {
//           tmp += g3(ik, nu, ib1, iw2) * eigvecs3_k(ik, nu2, nu);
//         }
//         g4(ik, nu2, ib1, iw2) = tmp;
//       });
//   Kokkos::realloc(g3, 0, 0, 0, 0);

//   ComplexView4D gFinal(Kokkos::ViewAllocateWithoutInitializing("gFinal"), numLoops, numPhBands, nb1, nb2max);
//   Kokkos::parallel_for(
//       "gFinal", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, nb2max}),
//       KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
//         Kokkos::complex<double> tmp(0., 0.);
//         for (int iw2 = 0; iw2 < numWannier; iw2++) {
//           tmp += eigvecs2Dagger_k(ik, iw2, ib2) * g4(ik, nu, ib1, iw2);
//         }
//         gFinal(ik, nu, ib1, ib2) = tmp;
//       });
//   Kokkos::realloc(g4, 0, 0, 0, 0);

//   // we now add the precomputed polar corrections, before taking the norm of g
//   if (usePolarCorrection) {
//     Kokkos::parallel_for(
//         "correction",
//         Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb1, nb2max}),
//         KOKKOS_LAMBDA(int ik, int nu, int ib1, int ib2) {
//           gFinal(ik, nu, ib1, ib2) += polarCorrections(ik, nu, ib1, ib2);
//         });
//   }
//   Kokkos::realloc(polarCorrections, 0, 0, 0, 0);

//   // finally, compute |g|^2 from g
//   DoubleView4D coupling_k(Kokkos::ViewAllocateWithoutInitializing("coupling"), numLoops, numPhBands, nb2max, nb1);
//   Kokkos::parallel_for(
//       "coupling", Range4D({0, 0, 0, 0}, {numLoops, numPhBands, nb2max, nb1}),
//       KOKKOS_LAMBDA(int ik, int nu, int ib2, int ib1) {
//         // notice the flip of 1 and 2 indices is intentional
//         // coupling is |<k+q,ib2 | dV_nu | k,ib1>|^2
//         auto tmp = gFinal(ik, nu, ib1, ib2);
//         coupling_k(ik, nu, ib2, ib1) =
//             tmp.real() * tmp.real() + tmp.imag() * tmp.imag();
//       });
//   Kokkos::realloc(gFinal, 0, 0, 0, 0);

//   // now, copy results back to the CPU
//   cacheCoupling.resize(0);
//   cacheCoupling.resize(numLoops);
//   auto coupling_h = Kokkos::create_mirror_view(coupling_k);
//   Kokkos::deep_copy(coupling_h, coupling_k);
// #pragma omp parallel for default(none) shared(numLoops, cacheCoupling, coupling_h, nb1, nb2s_h, numPhBands)
//   for (int ik = 0; ik < numLoops; ik++) {
//     Eigen::Tensor<double, 3> coupling(nb1, nb2s_h(ik), numPhBands);
//     for (int nu = 0; nu < numPhBands; nu++) {
//       for (int ib2 = 0; ib2 < nb2s_h(ik); ib2++) {
//         for (int ib1 = 0; ib1 < nb1; ib1++) {
//           coupling(ib1, ib2, nu) = coupling_h(ik, nu, ib2, ib1);
//         }
//       }
//     }
//     // and we save the coupling |g|^2 it for later
//     cacheCoupling[ik] = coupling;
//   }
//   Kokkos::Profiling::popRegion();
// }

// ```

// // void InteractionElPhSVD::parseHDF5_SVD( Context& context){
// //     const std::string fileName = context.getElphFileName();

// //     auto metadata = parseHeaderHDF5_SVD(context);
// //     int numElBands = std::get<0>(metadata);
// //     int numPhBands = std::get<1>(metadata);
// //     int totalNumElBravaisVectors = std::get<2>(metadata);
// //     Eigen::MatrixXd elBravaisVectors = std::get<3>(metadata);
// //     Eigen::MatrixXd phBravaisVectors = std::get<4>(metadata);
// //     std::vector<size_t> localElVectors = std::get<5>(metadata);
// //     Eigen::VectorXd elBravaisVectorsDegeneracies = std::get<6>(metadata);
// //     Eigen::VectorXd phBravaisVectorsDegeneracies = std::get<7>(metadata);

// //     // Calculate useful derived quantities
// //     int numElBravaisVectors = elBravaisVectorsDegeneracies.size();
// //     int numPhBravaisVectors = phBravaisVectorsDegeneracies.size();

// //     // Define Kokkos Views for SVD data
// //     Kokkos::View<double**> singularValues, leftSingularVectors, rightSingularVectors;


// //     parseHDF5_to_Kokkos(context, "U_Dataset", leftSingularVectors);
// //     parseHDF5_to_Kokkos(context, "S_Dataset", singularValues);
// //     parseHDF5_to_Kokkos(context, "V_Dataset", rightSingularVectors);
// // }

// // Eigen::Tensor<double, 3>& InteractionBase::getCouplingSquared(const int &ik2) {
// //   return cacheCoupling[ik2];
// // }



// // std::tuple<int, int, int, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<size_t>,
// //     Eigen::VectorXd, Eigen::VectorXd> InteractionElPhSVD::parseHeaderHDF5_SVD(Context &context) {
// //   std::string fileName = context.getElphFileName();

// //   int numElectrons, numSpin;
// //   int numElBands, numElBravaisVectors, totalNumElBravaisVectors, numPhBands, numPhBravaisVectors;
// //   // suppress initialization warning
// //   numElBravaisVectors = 0; totalNumElBravaisVectors = 0; numPhBravaisVectors = 0;
// //   Eigen::MatrixXd phBravaisVectors_, elBravaisVectors_;
// //   Eigen::VectorXd phBravaisVectorsDegeneracies_, elBravaisVectorsDegeneracies_;
// //   Eigen::Tensor<std::complex<double>, 5> couplingWannier_;
// //   std::vector<size_t> localElVectors;

// //   try {
// //     // Use MPI head only to read in the small data structures
// //     // then distribute them below this
// //     if (mpi->mpiHeadPool()) {
// //       // need to open the files differently if MPI is available or not
// //       // NOTE: do not remove the braces inside this if -- the file must
// //       // go out of scope, so that it can be reopened for parallel
// //       // read in the next block.
// //       {
// //         // Open the HDF5 ElPh file
// //         HighFive::File file(fileName, HighFive::File::ReadOnly);

// //         // read in the number of electrons and the spin
// //         HighFive::DataSet dnelec = file.getDataSet("/numElectrons");
// //         HighFive::DataSet dnspin = file.getDataSet("/numSpin");
// //         dnelec.read(numElectrons);
// //         dnspin.read(numSpin);

// //         // read in the number of phonon and electron bands
// //         HighFive::DataSet dnElBands = file.getDataSet("/numElBands");
// //         HighFive::DataSet dnModes = file.getDataSet("/numPhModes");
// //         dnElBands.read(numElBands);
// //         dnModes.read(numPhBands);

// //         // read phonon bravais lattice vectors and degeneracies
// //         HighFive::DataSet dphbravais = file.getDataSet("/phBravaisVectors");
// //         HighFive::DataSet dphDegeneracies = file.getDataSet("/phDegeneracies");
// //         dphbravais.read(phBravaisVectors_);
// //         dphDegeneracies.read(phBravaisVectorsDegeneracies_);
// //         numPhBravaisVectors = int(phBravaisVectors_.cols());

// //         // read electron Bravais lattice vectors and degeneracies
// //         HighFive::DataSet delDegeneracies = file.getDataSet("/elDegeneracies");
// //         delDegeneracies.read(elBravaisVectorsDegeneracies_);
// //         totalNumElBravaisVectors = int(elBravaisVectorsDegeneracies_.size());
// //         numElBravaisVectors = int(elBravaisVectorsDegeneracies_.size());
// //         HighFive::DataSet delbravais = file.getDataSet("/elBravaisVectors");
// //         delbravais.read(elBravaisVectors_);
// //         // redistribute in case of pools are present
// //         if (mpi->getSize(mpi->intraPoolComm) > 1) {
// //           localElVectors = mpi->divideWorkIter(totalNumElBravaisVectors, mpi->intraPoolComm);
// //           numElBravaisVectors = int(localElVectors.size());
// //           // copy a subset of elBravaisVectors
// //           Eigen::VectorXd tmp1 = elBravaisVectorsDegeneracies_;
// //           Eigen::MatrixXd tmp2 = elBravaisVectors_;
// //           elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
// //           elBravaisVectors_.resize(3, numElBravaisVectors);
// //           int i = 0;
// //           for (auto irE : localElVectors) {
// //             elBravaisVectorsDegeneracies_(i) = tmp1(irE);
// //             elBravaisVectors_.col(i) = tmp2.col(irE);
// //             i++;
// //           }
// //         }
// //       }
// //     }
// //     // broadcast to all MPI processes
// //     mpi->bcast(&numElectrons);
// //     mpi->bcast(&numSpin);
// //     mpi->bcast(&numPhBands);
// //     mpi->bcast(&numPhBravaisVectors);
// //     mpi->bcast(&numElBands);
// //     mpi->bcast(&numElBravaisVectors, mpi->interPoolComm);
// //     mpi->bcast(&totalNumElBravaisVectors, mpi->interPoolComm);
// //     mpi->bcast(&numElBravaisVectors, mpi->interPoolComm);

// //     if (numSpin == 2) {
// //       Error("Spin is not currently supported");
// //     }
// //     context.setNumOccupiedStates(numElectrons);

// //     if (!mpi->mpiHeadPool()) {// head already allocated these
// //       localElVectors = mpi->divideWorkIter(totalNumElBravaisVectors,
// //                                            mpi->intraPoolComm);
// //       phBravaisVectors_.resize(3, numPhBravaisVectors);
// //       phBravaisVectorsDegeneracies_.resize(numPhBravaisVectors);
// //       elBravaisVectors_.resize(3, numElBravaisVectors);
// //       elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
// //       couplingWannier_.resize(numElBands, numElBands, numPhBands,
// //                               numPhBravaisVectors, numElBravaisVectors);
// //     }
// //     mpi->bcast(&elBravaisVectors_, mpi->interPoolComm);
// //     mpi->bcast(&elBravaisVectorsDegeneracies_, mpi->interPoolComm);
// //     mpi->bcast(&phBravaisVectors_, mpi->interPoolComm);
// //     mpi->bcast(&phBravaisVectorsDegeneracies_, mpi->interPoolComm);
// //   } catch (std::exception &error) {
// //     Error("Issue reading elph Wannier representation header data from hdf5.");
// //   }

// //   return std::make_tuple(numElBands, numPhBands, totalNumElBravaisVectors, elBravaisVectors_,
// //           phBravaisVectors_, localElVectors, elBravaisVectorsDegeneracies_,
// //           phBravaisVectorsDegeneracies_);
// // }



// /*


// // KEYNESH edits here -- transform using Uk, e^{ik1 . Re}
// void InteractionElPhWanSVD::cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C) {}

// // KEYNESH edits here
// // second part of the wannier transform, the transform using phase(i q.Rp), and uq, Uk2
// void InteractionElPhWanSVD::calcCouplingSquared(

//     const Eigen::MatrixXcd &eigvec1,
//     const std::vector<Eigen::MatrixXcd> &eigvecs2,
//     const std::vector<Eigen::MatrixXcd> &eigvecs3,
//     const std::vector<Eigen::Vector3d> &q3Cs,
//     const std::vector<Eigen::VectorXcd> &polarData) {

//   int numWannier = numElBands;
//   auto nb1 = int(eigvec1.cols());
//   auto nk2 = int(eigvecs2.size());

//   auto elPhCached = this->elPhCached;
//   int numPhBands = this->numPhBands;
//   int numPhBravaisVectors = this->numPhBravaisVectors;

//   // TODO this needs to store the coupling in cacheCoupling to later be accesses

// }

// Eigen::Tensor<double, 3>&
// InteractionElPhWanSVD::getCouplingSquared(const int &ik2) {
//   return cacheCoupling[ik2];
// }

// void parseHDF5_SVD(const Context& context) {

//   std::string fileName = context.getElphFileName();

//   auto t = parseHeaderHDF5(context);
//   int numElBands = std::get<0>(t);
//   int numPhBands = std::get<1>(t);
//   int totalNumElBravaisVectors = std::get<2>(t);
//   Eigen::MatrixXd elBravaisVectors_ = std::get<3>(t);
//   Eigen::MatrixXd phBravaisVectors_ = std::get<4>(t);
//   std::vector<size_t> localElVectors = std::get<5>(t);
//   Eigen::VectorXd elBravaisVectorsDegeneracies_ = std::get<6>(t);
//   Eigen::VectorXd phBravaisVectorsDegeneracies_ = std::get<7>(t);
//   int numElBravaisVectors = elBravaisVectorsDegeneracies_.size();
//   int numPhBravaisVectors = phBravaisVectorsDegeneracies_.size();

//   parseHDF5_to_Kokkos(hdf5U, "U_Dataset", U_kokkos);
//   parseHDF5_to_Kokkos(hdf5S, "S_Dataset", S_kokkos);
//   parseHDF5_to_Kokkos(hdf5V, "V_Dataset", V_kokkos);

// }

// // Method to read HDF5 files and load data into Kokkos Views
// template <typename T>
// void parseHDF5_to_Kokkos(const Context& context, const std::string& datasetName, Kokkos::View<T**>& kokkosView){

//   const std::string fileName = context.getElphFileName();

//   // Open the HDF5 file using HighFive
//     HighFive::File file(hdf5File, HighFive::File::ReadOnly);

//     // Open the specified dataset
//     HighFive::DataSet dataset = file.getDataSet(datasetName);

//     // Get dataset dimensions
//     std::vector<size_t> dims = dataset.getDimensions();

//     // Ensure Kokkos View dimensions match the dataset
//     if (kokkosView.extent(0) != dims[0] || kokkosView.extent(1) != dims[1]) {
//         kokkosView = Kokkos::View<T**>("kokkosView", dims[0], dims[1]);
//     }

//     // Create a buffer and read data into it
//     std::vector<T> buffer(dims[0] * dims[1]);
//     dataset.read(buffer);

//     // Copy data from buffer to Kokkos View
//     Kokkos::parallel_for("copy_to_kokkos", dims[0], KOKKOS_LAMBDA(const int i) {
//         for (size_t j = 0; j < dims[1]; ++j) {
//             kokkosView(i, j) = buffer[i * dims[1] + j];
//         }
//     });

// }

// #ifdef HDF5_AVAIL

// std::tuple<int, int, int, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<size_t>,
//     Eigen::VectorXd, Eigen::VectorXd> parseHeaderHDF5(Context &context) {
//   std::string fileName = context.getElphFileName();

//   int numElectrons, numSpin;
//   int numElBands, numElBravaisVectors, totalNumElBravaisVectors, numPhBands, numPhBravaisVectors;
//   // suppress initialization warning
//   numElBravaisVectors = 0; totalNumElBravaisVectors = 0; numPhBravaisVectors = 0;
//   Eigen::MatrixXd phBravaisVectors_, elBravaisVectors_;
//   Eigen::VectorXd phBravaisVectorsDegeneracies_, elBravaisVectorsDegeneracies_;
//   Eigen::Tensor<std::complex<double>, 5> couplingWannier_;
//   std::vector<size_t> localElVectors;

//   try {
//     // Use MPI head only to read in the small data structures
//     // then distribute them below this
//     if (mpi->mpiHeadPool()) {
//       // need to open the files differently if MPI is available or not
//       // NOTE: do not remove the braces inside this if -- the file must
//       // go out of scope, so that it can be reopened for parallel
//       // read in the next block.
//       {
//         // Open the HDF5 ElPh file
//         HighFive::File file(fileName, HighFive::File::ReadOnly);

//         // read in the number of electrons and the spin
//         HighFive::DataSet dnelec = file.getDataSet("/numElectrons");
//         HighFive::DataSet dnspin = file.getDataSet("/numSpin");
//         dnelec.read(numElectrons);
//         dnspin.read(numSpin);

//         // read in the number of phonon and electron bands
//         HighFive::DataSet dnElBands = file.getDataSet("/numElBands");
//         HighFive::DataSet dnModes = file.getDataSet("/numPhModes");
//         dnElBands.read(numElBands);
//         dnModes.read(numPhBands);

//         // read phonon bravais lattice vectors and degeneracies
//         HighFive::DataSet dphbravais = file.getDataSet("/phBravaisVectors");
//         HighFive::DataSet dphDegeneracies = file.getDataSet("/phDegeneracies");
//         dphbravais.read(phBravaisVectors_);
//         dphDegeneracies.read(phBravaisVectorsDegeneracies_);
//         numPhBravaisVectors = int(phBravaisVectors_.cols());

//         // read electron Bravais lattice vectors and degeneracies
//         HighFive::DataSet delDegeneracies = file.getDataSet("/elDegeneracies");
//         delDegeneracies.read(elBravaisVectorsDegeneracies_);
//         totalNumElBravaisVectors = int(elBravaisVectorsDegeneracies_.size());
//         numElBravaisVectors = int(elBravaisVectorsDegeneracies_.size());
//         HighFive::DataSet delbravais = file.getDataSet("/elBravaisVectors");
//         delbravais.read(elBravaisVectors_);
//         // redistribute in case of pools are present
//         if (mpi->getSize(mpi->intraPoolComm) > 1) {
//           localElVectors = mpi->divideWorkIter(totalNumElBravaisVectors, mpi->intraPoolComm);
//           numElBravaisVectors = int(localElVectors.size());
//           // copy a subset of elBravaisVectors
//           Eigen::VectorXd tmp1 = elBravaisVectorsDegeneracies_;
//           Eigen::MatrixXd tmp2 = elBravaisVectors_;
//           elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
//           elBravaisVectors_.resize(3, numElBravaisVectors);
//           int i = 0;
//           for (auto irE : localElVectors) {
//             elBravaisVectorsDegeneracies_(i) = tmp1(irE);
//             elBravaisVectors_.col(i) = tmp2.col(irE);
//             i++;
//           }
//         }
//       }
//     }
//     // broadcast to all MPI processes
//     mpi->bcast(&numElectrons);
//     mpi->bcast(&numSpin);
//     mpi->bcast(&numPhBands);
//     mpi->bcast(&numPhBravaisVectors);
//     mpi->bcast(&numElBands);
//     mpi->bcast(&numElBravaisVectors, mpi->interPoolComm);
//     mpi->bcast(&totalNumElBravaisVectors, mpi->interPoolComm);
//     mpi->bcast(&numElBravaisVectors, mpi->interPoolComm);

//     if (numSpin == 2) {
//       Error("Spin is not currently supported");
//     }
//     context.setNumOccupiedStates(numElectrons);

//     if (!mpi->mpiHeadPool()) {// head already allocated these
//       localElVectors = mpi->divideWorkIter(totalNumElBravaisVectors,
//                                            mpi->intraPoolComm);
//       phBravaisVectors_.resize(3, numPhBravaisVectors);
//       phBravaisVectorsDegeneracies_.resize(numPhBravaisVectors);
//       elBravaisVectors_.resize(3, numElBravaisVectors);
//       elBravaisVectorsDegeneracies_.resize(numElBravaisVectors);
//       couplingWannier_.resize(numElBands, numElBands, numPhBands,
//                               numPhBravaisVectors, numElBravaisVectors);
//     }
//     mpi->bcast(&elBravaisVectors_, mpi->interPoolComm);
//     mpi->bcast(&elBravaisVectorsDegeneracies_, mpi->interPoolComm);
//     mpi->bcast(&phBravaisVectors_, mpi->interPoolComm);
//     mpi->bcast(&phBravaisVectorsDegeneracies_, mpi->interPoolComm);
//   } catch (std::exception &error) {
//     Error("Issue reading elph Wannier representation header data from hdf5.");
//   }

//   return std::make_tuple(numElBands, numPhBands, totalNumElBravaisVectors, elBravaisVectors_,
//           phBravaisVectors_, localElVectors, elBravaisVectorsDegeneracies_,
//           phBravaisVectorsDegeneracies_);
// }
// #endif

// */

// // Eigen::Tensor<std::complex<double>, 3> InteractionElPhWanSVD::getPolarCorrection(
// //     const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
// //     const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3) {
// //   // doi:10.1103/physrevlett.115.176401, Eq. 4, is implemented here

// //   Eigen::VectorXcd x = polarCorrectionPart1(q3, ev3);
// //   return polarCorrectionPart2(ev1, ev2, x);
// // }

// // Eigen::Tensor<std::complex<double>, 3>
// // InteractionElPhSVD::getPolarCorrectionStatic(
// //     const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
// //     const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3,
// //     const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
// //     const Eigen::Matrix3d &epsilon,
// //     const Eigen::Tensor<double, 3> &bornCharges,
// //     const Eigen::MatrixXd &atomicPositions,
// //     const Eigen::Vector3i &qCoarseMesh) {
// //   Eigen::VectorXcd x = polarCorrectionPart1Static(q3, ev3, volume, reciprocalUnitCell,
// //                                                   epsilon, bornCharges, atomicPositions, qCoarseMesh);
// //   return polarCorrectionPart2(ev1, ev2, x);
// // }

// // Eigen::VectorXcd
// // InteractionElPhSVD::polarCorrectionPart1(const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3) {
// //   // gather variables
// //   double volume = crystal.getVolumeUnitCell();
// //   Eigen::Matrix3d reciprocalUnitCell = crystal.getReciprocalUnitCell();
// //   Eigen::Matrix3d epsilon = phononH0->getDielectricMatrix();
// //   Eigen::Tensor<double, 3> bornCharges = phononH0->getBornCharges();
// //   // must be in Bohr
// //   Eigen::MatrixXd atomicPositions = crystal.getAtomicPositions();
// //   Eigen::Vector3i qCoarseMesh = phononH0->getCoarseGrid();

// //   return polarCorrectionPart1Static(q3, ev3, volume, reciprocalUnitCell,
// //                                     epsilon, bornCharges, atomicPositions, qCoarseMesh);
// // }

// // Eigen::VectorXcd InteractionElPhSVD::polarCorrectionPart1Static(
// //     const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3,
// //     const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
// //     const Eigen::Matrix3d &epsilon, const Eigen::Tensor<double, 3> &bornCharges,
// //     const Eigen::MatrixXd &atomicPositions, const Eigen::Vector3i &qCoarseMesh) {
// //   // doi:10.1103/physRevLett.115.176401, Eq. 4, is implemented here

// //   auto numAtoms = int(atomicPositions.rows());

// //   // auxiliary terms
// //   double gMax = 14.;
// //   double chargeSquare = 2.;// = e^2/4/Pi/eps_0 in atomic units
// //   std::complex<double> factor = chargeSquare * fourPi / volume * complexI;

// //   // build a list of (q+G) vectors
// //   std::vector<Eigen::Vector3d> gVectors;// here we insert all (q+G)
// //   for (int m1 = -qCoarseMesh(0); m1 <= qCoarseMesh(0); m1++) {
// //     for (int m2 = -qCoarseMesh(1); m2 <= qCoarseMesh(1); m2++) {
// //       for (int m3 = -qCoarseMesh(2); m3 <= qCoarseMesh(2); m3++) {
// //         Eigen::Vector3d gVector;
// //         gVector << m1, m2, m3;
// //         gVector = reciprocalUnitCell * gVector;
// //         gVector += q3;
// //         gVectors.push_back(gVector);
// //       }
// //     }
// //   }

// //   auto numPhBands = int(ev3.rows());
// //   Eigen::VectorXcd x(numPhBands);
// //   x.setZero();
// //   for (Eigen::Vector3d gVector : gVectors) {
// //     double qEq = gVector.transpose() * epsilon * gVector;
// //     if (qEq > 0. && qEq / 4. < gMax) {
// //       std::complex<double> factor2 = factor * exp(-qEq / 4.) / qEq;
// //       for (int iAt = 0; iAt < numAtoms; iAt++) {
// //         double arg = -gVector.dot(atomicPositions.row(iAt));
// //         std::complex<double> phase = {cos(arg), sin(arg)};
// //         std::complex<double> factor3 = factor2 * phase;
// //         for (int iPol : {0, 1, 2}) {
// //           double gqDotZ = gVector(0) * bornCharges(iAt, 0, iPol) + gVector(1) * bornCharges(iAt, 1, iPol) + gVector(2) * bornCharges(iAt, 2, iPol);
// //           int k = PhononH0::getIndexEigenvector(iAt, iPol, numAtoms);
// //           for (int ib3 = 0; ib3 < numPhBands; ib3++) {
// //             x(ib3) += factor3 * gqDotZ * ev3(k, ib3);
// //           }
// //         }
// //       }
// //     }
// //   }
// //   return x;
// // }

// // Eigen::Tensor<std::complex<double>, 3>
// // InteractionElPhSVD::polarCorrectionPart2(const Eigen::MatrixXcd &ev1, const Eigen::MatrixXcd &ev2, const Eigen::VectorXcd &x) {
// //   // overlap = <U^+_{b2 k+q}|U_{b1 k}>
// //   //         = <psi_{b2 k+q}|e^{i(q+G)r}|psi_{b1 k}>
// //   Eigen::MatrixXcd overlap = ev2.adjoint() * ev1;// matrix size (nb2,nb1)
// //   overlap = overlap.transpose();                 // matrix size (nb1,nb2)

// //   int numPhBands = x.rows();
// //   Eigen::Tensor<std::complex<double>, 3> v(overlap.rows(), overlap.cols(),
// //                                            numPhBands);
// //   v.setZero();
// //   for (int ib3 = 0; ib3 < numPhBands; ib3++) {
// //     for (int i = 0; i < overlap.rows(); i++) {
// //       for (int j = 0; j < overlap.cols(); j++) {
// //         v(i, j, ib3) += x(ib3) * overlap(i, j);
// //       }
// //     }
// //   }
// //   return v;
// // }

// // // TODO needs to return an eigen array of dimensions
// // Eigen::VectorXi InteractionElPhSVD::getCouplingDimensions() {
// //   Eigen::VectorXi xx(5);
// //   //for (int i : {0, 1, 2, 3, 4}) {
// //   //  xx(i) = couplingWannier_k.extent(i);
// //   //}
// //   return xx;
// // }

// // int InteractionElPhSVD::estimateNumBatches(const int &nk2, const int &nb1) {
// //   return 1;
// // }

// // void InteractionElPhSVD::getDeviceMemoryUsage() {
// //   return 1;
// // }






// /*

// This code works with the general structure:

// for k1 in k1List:

//   interactionElph.cacheElPh(k1, Uk1);

//   for q in qList: // if you know q, then you know k2 = k1 + q

//     interactionElph.calcCouplingSquared(Uk1, Uk2, uq, qBatchList, polarData);
//     ...
//     // calculate scattering rate
//     rate += <some constants> * interactionElph.getCouplingSq(k2);

// */



// // // #include "interaction_elph_svd.h"
// // #include <HighFive/HighFive.hpp>

// // // // Constructor
// // // InteractionElPhSVD::InteractionElPhSVD(Crystal &crystal_,
// // //                                        const Eigen::Tensor<std::complex<double>, 5> &couplingWannier_,
// // //                                        const Eigen::MatrixXd &elBravaisVectors_,
// // //                                        const Eigen::VectorXd &elBravaisVectorsDegeneracies_,
// // //                                        const Eigen::MatrixXd &phBravaisVectors_,
// // //                                        const Eigen::VectorXd &phBravaisVectorsDegeneracies_,
// // //                                        PhononH0 *phononH0_)
// // //     : InteractionElPhWan(crystal_, couplingWannier_, elBravaisVectors_, elBravaisVectorsDegeneracies_,
// // //                          phBravaisVectors_, phBravaisVectorsDegeneracies_, phononH0_) {}

// // // Eigen::Tensor<double, 3>& InteractionElPhSVD::getCouplingSq(const int &k2) {
// // //     return InteractionElPhWan::getCouplingSquared(k2);
// // // }


// // // Eigen::Tensor<double, 3>& InteractionElPhSVD::getCouplingSq(const Eigen::MatrixXd &U, const Eigen::MatrixXd &S, const Eigen::MatrixXd &V) {

// // //     return 1;
// // // }


// // // void InteractionElPhSVD::loadSVDFromFile(const std::string &filePath, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V) {
// // //     try {
// // //         HighFive::File file(filePath, HighFive::File::ReadOnly);
// // //         file.getDataSet("U").read(U);
// // //         file.getDataSet("S").read(S);
// // //         file.getDataSet("V").read(V);
// // //     } catch (const HighFive::Exception &e) {
// // //         std::cerr << "Error reading HDF5 file: " << e.what() << std::endl;
// // //         throw;
// // //     }
// // // }

// // // /*

// // // So it seem by doing this, we have 3 effective functions to work on

// // // 1. InteractionElPhSVD::calcCouplingSquared
// // // 2. InteractionElPhSVD::cacheElPh
// // // 3. InteractionElPhSVD::loadSVDFromFile

// // // Let's quickly go through what's happening.

// // // Recall that the psedo code is essentially written below:

// // // ```
// // //   for k1 in k1List:

// // //     interactionElph.cacheElPh(k1, Uk1);

// // //     for q in qList: // if you know q, then you know k2 = k1 + q

// // //       interactionElph.calcCouplingSquared(Uk1, Uk2, uq, qBatchList, polarData);
// // //       ...
// // //       // calculate scattering rate
// // //       rate += <some constants> * interactionElph.getCouplingSq(k2);

// // // ```

// // // -> So for cacheElPh,
// // //     Purpose: Precomputes G(k,q) for a specific k (in this case, k1). This is said to be important since
// // //      it reduces computation overhead since we dont have to repeatedly calculate k1

// // //     How it works is, it uses k1C and eigVec1 to perform transformation on G(k,q)
// // //     So simply put, the base class already does this, we jsut need to extend this to interactionElPhSVD since we now work with different data strcutures

// // // -> FOr calcCouplingSquared,
// // //     This essentially calcualtes the |G(k1,k2,g1)|^2. Obviously, we want to calculate this since it gives back
// // //     back a real value necessary for furture calculations

// // //     How it works is, it WILL LOAD the already computed SVD matrices from the HDF5 file from the other function, to which then we simply just call for it

// // // */




// // '''
