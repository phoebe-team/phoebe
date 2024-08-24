#include "viscosity_io.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

// TODO potentially this should be a viscosity parent object...

// here alpha0 and alpha e are set through passing by reference
void genericRelaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors,
                                    int& numRelaxons, Particle& particle,
                                    Eigen::VectorXd& theta0,
                                    Eigen::VectorXd& theta_e,
                                    int& alpha0, int& alpha_e, bool print) {

  // calculate the overlaps with special eigenvectors
  Eigen::VectorXd prodTheta0(numRelaxons); prodTheta0.setZero();
  Eigen::VectorXd prodThetae(numRelaxons); prodThetae.setZero();
  for (auto tup : eigenvectors.getAllLocalStates()) {

    auto is = std::get<0>(tup);
    auto gamma = std::get<1>(tup);
    //if(std::isnan(eigenvectors(is,gamma)) || std::isnan(theta0(is))) std::cout << is << " " << gamma << " " << eigenvectors(is,gamma) << " " << theta0(is) << std::endl;
    prodTheta0(gamma) += eigenvectors(is,gamma) * theta0(is);
    prodThetae(gamma) += eigenvectors(is,gamma) * theta_e(is);

  }
  mpi->allReduceSum(&prodThetae); mpi->allReduceSum(&prodTheta0);

  // find the element with the maximum product
  prodTheta0 = prodTheta0.cwiseAbs();
  prodThetae = prodThetae.cwiseAbs();
  Eigen::Index maxCol0, idxAlpha0;
  Eigen::Index maxCol_e, idxAlpha_e;
  float maxTheta0 = prodTheta0.maxCoeff(&idxAlpha0, &maxCol0);
  float maxThetae = prodThetae.maxCoeff(&idxAlpha_e, &maxCol_e);

  if(mpi->mpiHead() && print) {
    // avoid a segfault in an edge case of few el states
    int maxPrint = 10; 
    if(numRelaxons < 10) { maxPrint = numRelaxons; } 

    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << "Maximum scalar product theta_0.theta_alpha = " << maxTheta0 << " at index " << idxAlpha0 << "." << std::endl;
    std::cout << "First ten products with theta_0:";
    for(int gamma = 0; gamma < maxPrint; gamma++) { std::cout << " " << prodTheta0(gamma); }
    if(particle.isElectron()) {
      std::cout << "\n\nMaximum scalar product theta_e.theta_alpha = " << maxThetae << " at index " << idxAlpha_e << "." << std::endl;
      std::cout << "First ten products with theta_e:";
      for(int gamma = 0; gamma < maxPrint; gamma++) { std::cout << " " << prodThetae(gamma); }
    }
    std::cout << "\n" << std::endl;
    //std::cout << "Eigenvector norm check: " << theta0.dot(theta0) << " " << theta_e.dot(theta_e);
  }

  // save these indices to the class objects
  // if they weren't really found, we leave these indices
  // as -1 so that no relaxons are skipped
  if(maxTheta0 >= 0.75) alpha0 = idxAlpha0;
  if(maxThetae >= 0.75) alpha_e = idxAlpha_e;

}

/*
std::tuple<int,int> relaxonEigenvectorsCheck(ParallelMatrix<double>& eigenvectors,
                              int& numRelaxons, Particle& particle, 
                              Eigen::VectorXd& theta0, Eigen::VectorXd& theta_e) {

  Eigen::VectorXd prod0(numRelaxons);
  Eigen::VectorXd prod_e(numRelaxons);
  prod0.setZero(); prod_e.setZero();
  //Eigen::Vector3d vecprodphi1(numRelaxons);
  //Eigen::Vector3d vecprodphi2(numRelaxons);
  //Eigen::Vector3d vecprodphi3(numRelaxons);

  // sum over the alpha and v states that this process owns
  for (auto tup : eigenvectors.getAllLocalStates()) {

    auto is = std::get<0>(tup);
    auto gamma = std::get<1>(tup);

    prod0(gamma) += eigenvectors(is,gamma) * theta0(is);
    prod_e(gamma) += eigenvectors(is,gamma) * theta_e(is);
    //vecprodphi1[gamma] += eigenvectors(is,gamma) * phi(0,is);
    //vecprodphi2[gamma] += eigenvectors(is,gamma) * phi(1,is);
    //vecprodphi3[gamma] += eigenvectors(is,gamma) * phi(2,is);

  }
  // scalar products with vectors
  mpi->allReduceSum(&prod0); mpi->allReduceSum(&prod_e);
  //mpi->allReduceSum(&vecprodphi1); mpi->allReduceSum(&vecprodphi2); mpi->allReduceSum(&vecprodphi3);

  // find the element with the maximum product
  prod0 = prod0.cwiseAbs();
  prod_e = prod_e.cwiseAbs();
  Eigen::Index maxCol0, idxAlpha0;
  Eigen::Index maxCol_e, idxAlpha_e;
  float maxTheta0 = prod0.maxCoeff(&idxAlpha0, &maxCol0);
  float maxThetae = prod_e.maxCoeff(&idxAlpha_e, &maxCol_e);

  if(mpi->mpiHead()) {

    // avoid a segfault in an edge case of few el states
    int maxPrint = 10; 
    if(numRelaxons < 10) { maxPrint = numRelaxons; } 

    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << "Maximum scalar product theta_0.theta_alpha = " << maxTheta0 << " at index " << idxAlpha0 << "." << std::endl;
    std::cout << "First ten scalar products with theta_0:";
    for(int gamma = 0; gamma < maxPrint; gamma++) { std::cout << " " << prod0(gamma); }
    std::cout << "\n\nMaximum scalar product theta_e.theta_alpha = " << maxThetae << " at index " << idxAlpha_e << "." << std::endl;
    std::cout << "First ten scalar products with theta_e:";
    for(int gamma = 0; gamma < maxPrint; gamma++) { std::cout << " " << prod_e(gamma); }
    std::cout << std::endl;
  }

  // save these indices to the class objects
  // if they weren't really found, we leave these indices
  // as -1 so that no relaxons are skipped
  int alpha0, alpha_e; 
  if(maxTheta0 >= 0.75) {
    if(mpi->mpiHead()) std::cout << "Identified energy eigenvector, it will be discarded from viscosity." << std::endl;
    alpha0 = idxAlpha0;
  }
  if(maxThetae >= 0.75) {
    if(mpi->mpiHead()) std::cout << "Identified charge eigenvector, it will be discarded from viscosity." << std::endl;
    alpha_e = idxAlpha_e;
  } 
  return std::make_tuple(alpha0,alpha_e);
}*/

// calculate special eigenvectors
void genericCalcSpecialEigenvectors(BaseBandStructure& bandStructure,
                                    StatisticsSweep& statisticsSweep,
                                    double& spinFactor,
                                    Eigen::VectorXd& theta0,
                                    Eigen::VectorXd& theta_e,
                                    Eigen::MatrixXd& phi,
                                    double& C, Eigen::Vector3d& A) {

  int dimensionality = bandStructure.getPoints().getCrystal().getDimensionality();
  double volume = bandStructure.getPoints().getCrystal().getVolumeUnitCell(dimensionality);
  auto particle = bandStructure.getParticle();
  int numStates = bandStructure.getNumStates();

  int iCalc = 0; // set to zero because of relaxons
  auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
  double kBT = calcStat.temperature;
  double T = calcStat.temperature / kBoltzmannRy;
  double chemPot = 0; // has to be zero for phonons,
                      // don't use the stat sweep one which may have
                      // finite values if phel scattering is used
  double Npts = bandStructure.getPoints().getNumPoints();

  // set particle specific quantities
  if(particle.isElectron()) {
    chemPot = calcStat.chemicalPotential;
  }

  // Precalculate theta_e, theta0, phi  ----------------------------------

  // theta^0 - energy conservation eigenvector
  //   electronic states = ds * g-1 * (hE - mu) * 1/(kbT^2 * V * Nkq * Ctot)
  //   phonon states = ds * g-1 * h*omega * 1/(kbT^2 * V * Nkq * Ctot)
  theta0 = Eigen::VectorXd::Zero(numStates);

  // theta^e -- the charge conservation eigenvector
  //   electronic states = ds * g-1 * 1/(kbT * U)
  // for the phonons, this is unused
  theta_e = Eigen::VectorXd::Zero(numStates);

  // phi -- the three momentum conservation eigenvectors
  //     phi = sqrt(1/(kbT*volume*Npts*M)) * g-1 * ds * hbar * wavevector;
  phi = Eigen::MatrixXd::Zero(3, numStates);

  // spin degen vector
  Eigen::VectorXd ds = Eigen::VectorXd::Zero(numStates);

  // normalization for theta_e
  double U = 0;

  // specific heat
  C = 0.;

  // calculate the special eigenvectors ----------------
  for (int is : bandStructure.parallelStateIterator()) {

    ds(is) = sqrt(spinFactor);
    auto isIdx = StateIndex(is);
    double en = bandStructure.getEnergy(isIdx);
    if(particle.isPhonon() && en < phEnergyCutoff) { continue; }
    double pop = particle.getPopPopPm1(en, kBT, chemPot);

    theta0(is) = sqrt(pop) * (en - chemPot) * ds(is);
    if(particle.isElectron()) {
      theta_e(is) = sqrt(pop) * ds(is);
      U += pop;
    }
    auto popCont = pop * (en - chemPot) * (en - chemPot);
    C += pop * (en - chemPot) * (en - chemPot);
  }
  mpi->allReduceSum(&theta0);
  mpi->allReduceSum(&theta_e);
  mpi->allReduceSum(&C);
  mpi->allReduceSum(&U);

  // apply normalizations
  C *= spinFactor / (volume * size_t(Npts) * kBT * T);
  theta0 *= 1./sqrt(kBT * T * volume * size_t(Npts) * C);
  U *= spinFactor / (volume * Npts * kBT);
  if(particle.isPhonon()) U = 1.; // avoid making theta_e nan instead of zero
  theta_e *= 1./sqrt(kBT * U * Npts * volume);

  // calculate A_i ----------------------------------------

  // normalization coeff A ("phonon specific momentum")
  // A = 1/(V*N) * (1/kT) sum_qs (hbar*q)^2 * N(1+N)
  A = Eigen::Vector3d::Zero();

  for (int is : bandStructure.parallelStateIterator()) {
    auto isIdx = StateIndex(is);
    auto en = bandStructure.getEnergy(isIdx);

    if(particle.isPhonon() && en < phEnergyCutoff) { continue; }

    double pop = particle.getPopPopPm1(en, kBT, chemPot); // = n(n+1)
    auto q = bandStructure.getWavevector(isIdx);
    q = bandStructure.getPoints().bzToWs(q,Points::cartesianCoordinates);

    Eigen::Vector3d contrib; contrib.setZero();
    for (int iDim = 0; iDim < dimensionality; iDim++) {
      A(iDim) += pop * q(iDim) * q(iDim);
      contrib(iDim) += pop * q(iDim) * q(iDim);
    }
  }
  mpi->allReduceSum(&A);
  A *= spinFactor / (kBT * Npts * volume);

  // then calculate the drift eigenvectors, phi (eq A12 of PRX Simoncelli)
  // -----------------------------------------------------------------
  for (int is : bandStructure.parallelStateIterator()) {

    auto isIdx = StateIndex(is);
    auto en = bandStructure.getEnergy(isIdx);
    if(particle.isPhonon() && en < phEnergyCutoff) { continue; }

    double pop = particle.getPopPopPm1(en, kBT, chemPot); // = n(n+1)
    auto q = bandStructure.getWavevector(isIdx);
    q = bandStructure.getPoints().bzToWs(q,Points::cartesianCoordinates);
    for (int i = 0; i < dimensionality; i++) {
      phi(i, is) = q(i) * sqrt(pop) * ds(is);
    }
  }
  mpi->allReduceSum(&phi);
  // apply normalization to phi
  for(int is = 0; is < numStates; is++) {
    for (int i = 0; i < dimensionality; i++) phi(i,is) *= 1./sqrt(kBT * volume * Npts * A(i));
  }
/*
  // print phi overlap
  if(mpi->mpiHead()) {
    for(int i : {0,1,2} ) {
      double phiTot = 0;
      for (int is : bandStructure.irrStateIterator()) {
        phiTot += phi(i, is)*phi(i,is);
      }
     std::cout << "phi norm " << i << " " << phiTot << std::endl;;
    }
  }
*/
}

void printViscosity(std::string& viscosityName, Eigen::Tensor<double, 5>& viscosityTensor,
                                StatisticsSweep& statisticsSweep, int& dimensionality) {

  if (!mpi->mpiHead()) return;

  int numCalculations = statisticsSweep.getNumCalculations();

  if(numCalculations > 20) {
    std::cout << "\nBecause there are more than 20 calculations in this run,\n"
            << "the transport tensors will not be printed to output, but can\n"
            << "still be found in the corresponding output json file.\n"
            << std::endl;
    return;
  }

  // TODO Very important, do we have to multiply this by height / thickness?
  std::string units, printHeader;
  if (dimensionality == 1)      {
    units = "Pa s / m^2";
    printHeader = "i, eta[i,0]\n";
  } // 1d
  else if (dimensionality == 2) {
    units = "Pa s / m";
    printHeader = "i, j, eta[i,j,0], eta[i,j,1]\n";
  } // 2d
  else {
    units = "Pa s";
    printHeader = "i, j, k, eta[i,j,k,0], eta[i,j,k,1], eta[i,j,k,2]\n";
  } // 3d

  std::cout << "\n";
  std::cout << viscosityName << " viscosity (" << units << ")\n";
  std::cout << printHeader;

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;

    std::cout << std::fixed;
    std::cout.precision(2);
    std::cout << "Temperature: " << temp * temperatureAuToSi << " (K)\n";
    std::cout.precision(5);
    std::cout << std::scientific;
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        for (int k = 0; k < dimensionality; k++) {
          if(dimensionality == 1) std::cout << i;
          if(dimensionality == 2) std::cout << i << " " << j;
          if(dimensionality == 3) std::cout << i << " " << j << " " << k;
          for (int l = 0; l < dimensionality; l++) {
            std::cout << " " << std::setw(12) << std::right
                      << viscosityTensor(iCalc, i, j, k, l) * viscosityAuToSi;
          }
          std::cout << "\n";
        }
      }
    }
    std::cout << std::endl;
  }
}

void outputViscosityToJSON(const std::string& outFileName, const std::string& viscosityName,
                                Eigen::Tensor<double, 5>& viscosityTensor,
                                const bool& append,
                                StatisticsSweep& statisticsSweep, int& dimensionality) {

  if (!mpi->mpiHead()) return;

  int numCalculations = statisticsSweep.getNumCalculations();

  std::string units; // TODO CHECK THIS!
  if (dimensionality == 1) {      units = "Pa s / m^2"; } // 3d
  else if (dimensionality == 2) { units = "Pa s / m";   } // 2d
  else {                          units = "Pa s";       } // 1d

  std::vector<double> temps;
  // this vector mess is of shape (iCalculations, iRows, iColumns, k, l)
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> viscosity;

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    // store temperatures
    auto calcStat = statisticsSweep.getCalcStatistics(iCalc);
    double temp = calcStat.temperature;
    temps.push_back(temp * temperatureAuToSi);

    // store viscosity
    std::vector<std::vector<std::vector<std::vector<double>>>> rows;
    for (int i = 0; i < dimensionality; i++) {
      std::vector<std::vector<std::vector<double>>> cols;
      for (int j = 0; j < dimensionality; j++) {
        std::vector<std::vector<double>> ijk;
        for (int k = 0; k < dimensionality; k++) {
          std::vector<double> ijkl;
          for (int l = 0; l < dimensionality; l++) {
            ijkl.push_back(viscosityTensor(iCalc, i, j, k, l) * viscosityAuToSi);
          }
          ijk.push_back(ijkl);
        }
        cols.push_back(ijk);
      }
      rows.push_back(cols);
    }
    viscosity.push_back(rows);
  }

  // output to json
  nlohmann::json output;
  if(append) { // we're going to add this to an existing one (as in the coupled case)
    std::ifstream f(outFileName);
    output = nlohmann::json::parse(f);
  }
  output["temperatures"] = temps;
  output[viscosityName] = viscosity;
  output["temperatureUnit"] = "K";
  output["viscosityUnit"] = units;
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

// TODO we need to fix the dimensionality to work for
// low dim materials in all the coefficients!
void genericOutputRealSpaceToJSON(ScatteringMatrix& scatteringMatrix,
                                BaseBandStructure& bandStructure,
                                StatisticsSweep& statisticsSweep,
                                Eigen::VectorXd& theta0,
                                Eigen::VectorXd& theta_e,
                                Eigen::MatrixXd& phi,
                                double& C, Eigen::Vector3d& A,
                                Context& context) {

  // write D to file before diagonalizing, as the scattering matrix
  // will be destroyed by scalapack

  if(mpi->mpiHead()) std::cout << "\nWriting real-space solver quantities to file.\n" << std::endl;

  bool isPhonon = bandStructure.getParticle().isPhonon();
  int dimensionality = bandStructure.getPoints().getCrystal().getDimensionality();

  auto calcStat = statisticsSweep.getCalcStatistics(0); // only one calc for relaxons
  double kBT = calcStat.temperature;

  Eigen::MatrixXd Du(dimensionality,dimensionality); Du.setZero();
  Eigen::MatrixXd Wji0(dimensionality,dimensionality); Wji0.setZero();
  // below used only for electrons
  Eigen::MatrixXd Wjie(dimensionality,dimensionality); Wjie.setZero();

  // sum over the alpha and v states that this process owns
  for (auto tup : scatteringMatrix.getAllLocalStates()) {

    auto is1 = std::get<0>(tup);
    auto is2 = std::get<1>(tup);
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        if(context.getUseUpperTriangle()) {
          if( i == j ) {
            Du(i,j) += phi(i,is1) * scatteringMatrix(is1,is2) * phi(j,is2);
          } else {
            Du(i,j) += 2. * phi(i,is1) * scatteringMatrix(is1,is2) * phi(j,is2);
          }
        } else {
          Du(i,j) += phi(i,is1) * scatteringMatrix(is1,is2) * phi(j,is2);
        }
      }
    }
  }
  mpi->allReduceSum(&Du);

  for (int is : bandStructure.parallelStateIterator()) {
    auto isIdx = StateIndex(is);
    double en = bandStructure.getEnergy(isIdx);
    // discard acoustic phonon modes
    if (isPhonon && en < phEnergyCutoff) { continue; }
    auto v = bandStructure.getGroupVelocity(isIdx);
    for (int i = 0; i < dimensionality; i++) {
      for (int j = 0; j < dimensionality; j++) {
        // note: phi and theta here are elStates long, so we need to shift the state
        // index to account for the fact that we summed over the electronic part above
        // calculate qunatities for the real-space solve
        Wji0(j,i) += phi(i,is) * v(j) * theta0(is);
        Wjie(j,i) += phi(i,is) * v(j) * theta_e(is);
      }
    }
  }
  mpi->allReduceSum(&Wji0); mpi->allReduceSum(&Wjie);

  // NOTE we cannot use nested vectors from the start, as
  // vector<vector> is not necessarily contiguous and MPI
  // cannot all reduce on it
  std::vector<std::vector<double>> vecDu;
  std::vector<std::vector<double>> vecWji0;
  std::vector<std::vector<double>> vecWjie;
  for (int i = 0; i < dimensionality; i++) {
    std::vector<double> temp1,temp2,temp3;
    for (int j = 0; j < dimensionality; j++) {
      temp1.push_back(Du(i,j) / (energyRyToFs / twoPi));
      temp2.push_back(Wji0(i,j) * velocityRyToSi);
      temp3.push_back(Wjie(i,j) * velocityRyToSi);
    }
    vecDu.push_back(temp1);
    vecWji0.push_back(temp2);
    vecWjie.push_back(temp3);
  }

  // convert Ai to SI, in units of picograms/(mu m^3)
  double Aconversion = electronMassSi /
                       std::pow(distanceBohrToMum,dimensionality) * // convert AU mass / V -> SI
                       2. *   // factor of two is a Ry->Ha conversion required here
                       1.e15; // convert electronMassSi in kg to pico g

                       // Michele's version of this, gives thes same answer
                       // double altConv =  1./rydbergSi * // convert kBT
                       // std::pow(hBarSi/bohrRadiusSi,2) * // convert (hbar * q)^2
                       // 1./std::pow(bohrRadiusSi, dimensionality) * // convert 1/V
                       // 1e-3; //convert from kg->pg, 1/m^3 -> 1/mum^3; // converting to pico and mu

  std::string specificHeatUnits;
  std::string AiUnits;
  if (dimensionality == 1) {
    specificHeatUnits = "J / K / m";
    AiUnits = "pg/(mum)";
  } else if (dimensionality == 2) {
    specificHeatUnits = "J / K / m^2";
    AiUnits = "pg/(mum)^2";
  } else {
    specificHeatUnits = "J / K / m^3";
    AiUnits = "pg/(mum)^3";
  }

  // this extra kBoltzmannRy is required when we calculate specific heat ...
  // TODO need to keep track of this and figure out where it's coming from
  double specificHeatConversion = kBoltzmannSi / pow(bohrRadiusSi, 3) / kBoltzmannRy;
  auto particle = bandStructure.getParticle();

  if(mpi->mpiHead()) {
    // output to json
    std::string outFileName = "el_relaxons_real_space_coeffs.json";
    if(isPhonon) outFileName = "ph_relaxons_real_space_coeffs.json";
    nlohmann::json output;
    output["temperature"] = kBT * temperatureAuToSi;
    output["Wji0"] = vecWji0;
    if(!isPhonon) output["Wjie"] = vecWjie;
    output["Du"] = vecDu;
    output["temperatureUnit"] = "K";
    output["wUnit"] = "m/s";
    output["DuUnit"] = "fs^{-1}";
    output["specificHeat"] = C * specificHeatConversion;
    output["specificHeatUnit"] = specificHeatUnits;
    output["particleType"] = particle.isPhonon() ? "phonon" : "electron";
    std::vector<double> Atemp;
    for(int i = 0; i < dimensionality; i++) {
      Atemp.push_back(A(i) * Aconversion );
    }
    output["Ai"] = Atemp;
    output["AiUnit"] = AiUnits;
    std::ofstream o(outFileName);
    o << std::setw(3) << output << std::endl;
    o.close();
  }
}

