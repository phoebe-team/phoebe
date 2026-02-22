#ifndef ONSAGER_UTILITIES_H
#define ONSAGER_UTILITIES_H

#include "statistics_sweep.h"

  /** After the Onsager coefficients L_EE, L_TT, L_ET, L_TE have been computed
   * this function evaluates the transport coefficients such as electrical
   * conductivity, Seebeck and thermal conductivity.
   * @param statisticsSweep: object containing temperature, chemPot, etc info
   * @param dimensionality: the dimension of the crystal
   * @param LEE,LET,LTE,LTT: Onsager coefficient tensors used in the calculation.
   * @param kappa: thermal conductivity tensor, to be filled in by function
   * @param sigma: electrical conductivity tensor, to be filled in
   * @param mobility: mobility tensor, to be filled in
   * @param seebeck: seebeck coefficient tensor, to be filled in
   */
   void onsagerToTransportCoeffs(StatisticsSweep& statisticsSweep, int& dimensionality,
                        Eigen::Tensor<double, 3>& LEE, Eigen::Tensor<double, 3>& LTE,
                        Eigen::Tensor<double, 3>& LET, Eigen::Tensor<double, 3>& LTT,
                        Eigen::Tensor<double, 3>& kappa, Eigen::Tensor<double, 3>& sigma,
                        Eigen::Tensor<double, 3>& mobility, Eigen::Tensor<double, 3>& seebeck);

#endif
