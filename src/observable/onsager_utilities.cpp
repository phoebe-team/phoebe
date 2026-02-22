#include "onsager_utilities.h"
#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include <nlohmann/json.hpp>

void onsagerToTransportCoeffs(StatisticsSweep& statisticsSweep, int& dimensionality,
                        Eigen::Tensor<double, 3>& LEE, Eigen::Tensor<double, 3>& LTE,
                        Eigen::Tensor<double, 3>& LET, Eigen::Tensor<double, 3>& LTT,
                        Eigen::Tensor<double, 3>& kappa, Eigen::Tensor<double, 3>& sigma,
                        Eigen::Tensor<double, 3>& mobility, Eigen::Tensor<double, 3>& seebeck) {

  Kokkos::Profiling::pushRegion("onsagerToTransportCoeffs");

  int numCalculations = statisticsSweep.getNumCalculations();

  sigma = LEE;
  mobility = sigma;

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
    Eigen::MatrixXd thisLEE(dimensionality, dimensionality);
    Eigen::MatrixXd thisLET(dimensionality, dimensionality);
    Eigen::MatrixXd thisLTE(dimensionality, dimensionality);
    Eigen::MatrixXd thisLTT(dimensionality, dimensionality);
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        thisLEE(i, j) = LEE(iCalc, i, j);
        thisLET(i, j) = LET(iCalc, i, j);
        thisLTE(i, j) = LTE(iCalc, i, j);
        thisLTT(i, j) = LTT(iCalc, i, j);
      }
    }

    // seebeck = matmul(L_EE_inv, L_ET)
    Eigen::MatrixXd thisSeebeck = (thisLEE.inverse()) * thisLET;

    // k = L_tt - L_TE L_EE^-1 L_ET
    Eigen::MatrixXd thisKappa = thisLTE * (thisLEE.inverse());
    thisKappa = -(thisLTT - thisKappa * thisLET);
    double doping = abs(statisticsSweep.getCalcStatistics(iCalc).doping);
    doping *= pow(distanceBohrToCm, dimensionality); // from cm^-3 to bohr^-3
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        seebeck(iCalc, i, j) = thisSeebeck(i, j);
        kappa(iCalc, i, j) = thisKappa(i, j);
        if (doping > 0.) {
          mobility(iCalc, i, j) /= doping;
        }
      }
    }
  }
  Kokkos::Profiling::popRegion();
}
