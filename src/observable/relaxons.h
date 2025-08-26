#ifndef RELAXONS_H
#define RELAXONS_H
#include "scattering_matrix.h"

void outputRelaxonsToHDF5(ParallelMatrix<double>& eigenvectors,
                      const Eigen::VectorXd& eigenvalues,
                      std::vector<BaseBandStructure*>& bandStructures,
                      const Eigen::VectorXd& theta0,
                      const Eigen::VectorXd& theta_e,
                      const Eigen::MatrixXd& phi,
                      int numRelaxonsToOutput = 50, 
                      bool isCoupled = false 
                    );
#endif 