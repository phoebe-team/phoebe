// #ifndef EL_PH_SVD_INTERACTION_H
// #define EL_PH_SVD_INTERACTION_H

// #include "highfive/H5File.hpp"
// #include "interaction_base_elph.h"

// #include <complex>

// #include "constants.h"
// #include "crystal.h"
// #include "eigen.h"
// #include "phonon_h0.h"
// #include "points.h"
// #include "utilities.h"
// #include "context.h"
// #include "common_kokkos.h"
// #include <Kokkos_Core.hpp>
// #include <memory>
// #include <string>
// #include <vector>
// #include <regex>

// class InteractionElPhSVD: public InteractionElPhBase {

// public:

//     InteractionElPhSVD(Crystal& crystal, Context& context, // TODO put ph H0 up here
//                 //const Eigen::Tensor<std::complex<double>, 5>& couplingWannier,
//                 //const Eigen::MatrixXd& elBraviasVectors,
//                 //const Eigen::VectorXd& elBraviasVectorsDegeneracies,
//                 //const Eigen::MatrixXd& phBravaisVectors,
//                 //const Eigen::VectorXd& phBraviasVectorsDegeneracies,
//                 PhononH0* phononH0 = nullptr);

//     //~InteractionElPhSVD() = default;

//     void cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C) override ;

//     void calcCouplingSquared(const Eigen::MatrixXcd &eigvec1,
//                             const std::vector<Eigen::MatrixXcd> &eigvecs2,
//                             const std::vector<Eigen::MatrixXcd> &eigvecs3,
//                             const std::vector<Eigen::Vector3d> &q3Cs,
//                             const std::vector<Eigen::VectorXcd> &polarData) override;

//     struct SVDGroupData {
//         std::unique_ptr<std::vector<double>> singularVector;
//         std::unique_ptr<std::vector<double>> rightMatrix;
//         std::unique_ptr<std::vector<double>> leftMatrix;
//         int idxX; // Extracted index X
//         int idxY; // Extracted index Y
//         int idxZ; // Extracted index Z

//         // Constructor to initialize all members
//         SVDGroupData(std::unique_ptr<std::vector<double>> singularVector,
//                         std::unique_ptr<std::vector<double>> rightMatrix_,
//                         std::unique_ptr<std::vector<double>> leftMatrix_,
//                         int idxX_, int idxY_, int idxZ_)
//             : singularVector(std::move(singularVector)),
//                 rightMatrix(std::move(rightMatrix_)),
//                 leftMatrix(std::move(leftMatrix_)),
//                 idxX(idxX_),
//                 idxY(idxY_),
//                 idxZ(idxZ_) {}

//         // Move constructor
//         SVDGroupData(SVDGroupData&& other) noexcept
//             : singularVector(std::move(other.singularVector)),
//                 rightMatrix(std::move(other.rightMatrix)),
//                 leftMatrix(std::move(other.leftMatrix)),
//                 idxX(other.idxX),
//                 idxY(other.idxY),
//                 idxZ(other.idxZ) {}

//         // Move assignment operator
//         SVDGroupData& operator=(SVDGroupData&& other) noexcept {
//             if (this != &other) {
//                 singularVector = std::move(other.singularVector);
//                 rightMatrix = std::move(other.rightMatrix);
//                 leftMatrix = std::move(other.leftMatrix);
//                 idxX = other.idxX;
//                 idxY = other.idxY;
//                 idxZ = other.idxZ;
//             }
//             return *this;
//         }

//         // Disable copying
//         SVDGroupData(const SVDGroupData&) = delete;
//         SVDGroupData& operator=(const SVDGroupData&) = delete;
//     };

//     // Add additional functions HERE
//     //     // TODO for Keynesh, write this docstring
//     //     /** Method to read HDF5 files and load data into Kokkos Views */
//     //
//     void parseSVDKokkos(const std::string& hdf5FilePath, const std::string& svdGroupName);

//     std::vector<SVDGroupData> processAllSVDGroups(const HighFive::Group& svdGroupsize_t,size_t& num_i, size_t& num_j, size_t& num_eta);

//     // Disable other inherited functions to restrcit theri uses
//     //InteractionElPhWan& operator=(const InteractionElPhWan&) = delete;
//     //InteractionElPhSVD(const InteractionElPhSVD&) = delete;

// private:
//     //Kokkos::View<double***> kokkosContainer;
//     // the SVD data
//     ComplexView5D SVD_Y; // i,j,n,Re,gamma
//     ComplexView5D SVD_Vt; // i,j,n,Rp,gamma
//     // size_t numSingularValues;

// };

// #endif
