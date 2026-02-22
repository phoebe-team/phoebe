#ifndef TRANSPORT_IO_H
#define TRANSPORT_IO_H

#include "statistics_sweep.h"
        
  /** Prints to screen the transport coefficients at various temperatures
   * in a a nicely formatted way.
   * @param statisticsSweep: object containing temperature, chemPot, etc info
   * @param dimensionality: the dimension of the crystal
   * @param kappa: thermal conductivity tensor, to be filled in by function
   * @param sigma: electrical conductivity tensor, to be filled iin
   * @param mobility: mobility tensor, to be filled in
   * @param seebeck: seebeck coefficient tensor, to be filled in
   */
void printHelper(StatisticsSweep& statisticsSweep, int dimensionality,
                                const Eigen::Tensor<double, 3>& kappa,
                                const Eigen::Tensor<double, 3>& sigma,
                                const Eigen::Tensor<double, 3>& mobility,
                                const Eigen::Tensor<double, 3>& seebeck); 

  /** Short format for printing transport coeffs, to be used
   * for quickly evaluate the convergence of an iterative BTE solver.
   * @param iter: iteration number of the conductivities, as in iterative bte solves
   * @param statisticsSweep: object containing temperature, chemPot, etc info
   * @param dimensionality: the dimension of the crystal
   * @param kappa: thermal conductivity tensor
   * @param sigma: electrical conductivity tensor
   */
   void printHelper(const int iter, StatisticsSweep& statisticsSweep,
                                int dimensionality,
                                const Eigen::Tensor<double, 3>& kappa,
                                const Eigen::Tensor<double, 3>& sigma);
                                
  /** Return tuple of strings and units for sigma, kappa, nu (which all depend on dimension)
   */
  std::tuple<std::string, std::string, std::string, double, double, double> 
        getTransportUnitsWithDimensions(int dimensionality);     
        
  /** Helper function to simplify outputing 3x3 transport tensors to json
  * @param tensor: transport tensor to output
  * @param unitConv: conversion from AU to SI to fro this transport property
  * @param iCalc: calculation index
  * @param outFormat: container to write the end tensor to, using std::vectors for JSON
  */
  void appendTransportTensorForOutput(const Eigen::Tensor<double, 3>& tensor, int dimensionality,
                        double unitConv, int iCalc,
                        std::vector<std::vector<std::vector<double>>>& outFormat);   
                                                     
  /** Outputs the electrical transport coefficients to a json file. 
   * @param outFileName: string representing the name of the json file
   * @param statisticsSweep: object containing temperature, chemPot, etc info
   * @param dimensionality: the dimension of the crystal
   * @param kappa: thermal conductivity tensor
   * @param sigma: electrical conductivity tensor
   * @param mobility: mobility tensor
   * @param seebeck: seebeck coefficient tensor
   */
void outputElectronicCoeffsToJSON(const std::string &outFileName,
                                StatisticsSweep& statisticsSweep,
                                int dimensionality,
                                const Eigen::Tensor<double, 3>& kappa,
                                const Eigen::Tensor<double, 3>& sigma,
                                const Eigen::Tensor<double, 3>& mobility,
                                const Eigen::Tensor<double, 3>& seebeck); 
          
  /** Outputs the thermal transport coefficients to a json file. 
   * @param outFileName: string representing the name of the json file
   * @param statisticsSweep: object containing temperature, chemPot, etc info
   * @param dimensionality: the dimension of the crystal
   * @param kappa: thermal conductivity tensor
   */
  void outputPhononThermalCondToJSON(const std::string &outFileName, 
                              StatisticsSweep& statisticsSweep, 
                              int dimensionality,
                              const Eigen::Tensor<double, 3>& kappa); 
                    
#endif 