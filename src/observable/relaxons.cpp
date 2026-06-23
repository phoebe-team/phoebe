#include "relaxons.h"
#include <nlohmann/json.hpp>
#include "constants.h"

// returns the index of largest overlap with a special eigenvector
int relaxonEigenvectorOverlap(ParallelMatrix<double>& eigenvectors,
                              const Eigen::VectorXd& specialEigenvector,
                              std::string eigenvectorName) {

  // calculate the overlaps with special eigenvectors
  int numRelaxons = specialEigenvector.size();
  Eigen::VectorXd overlaps(numRelaxons); overlaps.setZero();

  // TODO need to update this for useUpperTriangle and case of less numRelaxons
  for (auto tup : eigenvectors.getAllLocalStates()) {
    auto is = std::get<0>(tup);
    auto gamma = std::get<1>(tup);
    overlaps(gamma) += eigenvectors(is,gamma) * specialEigenvector(is);
  }
  mpi->allReduceSum(&overlaps);

  // find the element with the maximum product
  overlaps = overlaps.cwiseAbs();
  Eigen::Index maxCol, idxMaxOverlap;
  float maxOverlap = overlaps.maxCoeff(&idxMaxOverlap, &maxCol);

  if(mpi->mpiHead()) {

    // avoid a segfault in an edge case of few states
    int maxPrint = 10;
    if(numRelaxons < 10) { maxPrint = numRelaxons; }

    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << "\nMaximum scalar product " << eigenvectorName << ".theta_alpha = " << maxOverlap << " at alpha = " << idxMaxOverlap << "." << std::endl;
    std::cout << "First ten products with " << eigenvectorName << ":";
    for(int gamma = 0; gamma < maxPrint; gamma++) { std::cout << " " << overlaps(gamma); }
    std::cout << std::endl;
  }

  // If the best overlap isn't very good, we return -1 so nothing is skipped
  if(maxOverlap >= 0.75) return idxMaxOverlap;
  else { return -1; }
}

// TODO change this maybe so it directly takes the transportCoeffs object?
// calculate special eigenvectors
void genericCalcSpecialEigenvectors(Context& context, BaseBandStructure& bandStructure,
                            StatisticsSweep& statisticsSweep,
                            double spinFactor,
                            Eigen::VectorXd& theta0,
                            Eigen::VectorXd& theta_e,
                            Eigen::MatrixXd& phi,
                            double& C, double& U, Eigen::Vector3d& A){ // note C is by ref because it needs to be filled in!

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
  //double Npts = bandStructure.getPoints().getNumPoints();
  double Npts;
  if(particle.isPhonon()) Npts = context.getQMesh().prod();
  else { Npts = context.getKMesh().prod(); }

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

  // zero normalization for theta_e, specific heat
  U = 0;
  C = 0;

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
    // auto popCont = pop * (en - chemPot) * (en - chemPot);
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
}

// TODO use requires on the bandstructures to be el first and ph second, do this more elegantly
void outputRelaxonsToHDF5(ParallelMatrix<double>& eigenvectors,
                          const Eigen::VectorXd& eigenvalues,
                          std::vector<BaseBandStructure*>& bandStructures,
                          const Eigen::VectorXd& theta0,
                          const Eigen::VectorXd& theta_e,
                          const Eigen::MatrixXd& phi,
                          int numRelaxonsToOutput,
                          bool isCoupled) {

  if(bandStructures.size() == 0)
    DeveloperError("Cannot output relaxons with no specified bandstructure.");

  if(isCoupled) {
    if(bandStructures.size() != 2) {
      DeveloperError("Need both bandstructures to output coupled relaxons.");
    }
    else if(!(bandStructures[0]->getParticle().isElectron() && bandStructures[1]->getParticle().isPhonon())) {
      DeveloperError("First bandstructure must be electron, second must be phonon, when outputing relaxons to HDF5.");
    }
  }

  // make a lambda to handle indexing if it's coupled -- return phonon state index
  std::function<int(int)> shiftedStateIdx;
  if(isCoupled) {
    int numElStates = int(bandStructures[0]->irrStateIterator().size());
    shiftedStateIdx = [numElStates](int stateIndex) {
      if(stateIndex < numElStates) { return stateIndex; }
      else { return stateIndex-numElStates; }
    };
  } else {
    shiftedStateIdx = [](int stateIndex) {  return stateIndex; };
  }

  for (auto bandStructure : bandStructures) {

    size_t numPoints = bandStructure->getPoints().getNumPoints();
    int numBands = bandStructure->getFullNumBands();
    Particle particle = bandStructure->getParticle();
    // convertion for time units
    double energyToTime = particle.isPhonon() ? energyRyToFs * 1e-3 / twoPi : energyRyToFs;

    // cannot use vector<vector> as this is not contiguous
    Eigen::Tensor<double,3> relaxon(numPoints, numBands, numRelaxonsToOutput);
    relaxon.setZero();
    for (auto [iBte,iRelaxon] : eigenvectors.getAllLocalStates()) {

      // get the band state associated with this state
      BteIndex BTEidx(shiftedStateIdx(iBte));
      auto is = bandStructure->bteToState(BTEidx);
      auto [ik,ib] = bandStructure->getIndex(is);

      if(iRelaxon >= numRelaxonsToOutput) continue;
      relaxon(ik.get(),ib.get(), iRelaxon) = eigenvectors(iBte,iRelaxon);
    }
    mpi->allReduceSum(&relaxon);

    // write the analytical special eigenvectors ---------------------------
    Eigen::MatrixXd theta_e_kn(numPoints, numBands), theta0_kn(numPoints, numBands);
    Eigen::MatrixXd phi_kn1(numPoints, numBands), phi_kn2(numPoints, numBands), phi_kn3(numPoints, numBands);
    theta_e_kn.setZero();  theta0_kn.setZero();
    phi_kn1.setZero(); phi_kn2.setZero(); phi_kn3.setZero();

    for (int is : bandStructure->parallelStateIterator()) {
      auto [ik,ib] = bandStructure->getIndex(is);
      theta_e_kn(ik.get(), ib.get()) = theta_e(is);
      theta0_kn(ik.get(), ib.get()) = theta0(is);
      phi_kn1(ik.get(), ib.get()) = phi(0,is);
      phi_kn2(ik.get(), ib.get()) = phi(1,is);
      phi_kn3(ik.get(), ib.get()) = phi(2,is);
    }
    mpi->allReduceSum(&theta_e_kn);
    mpi->allReduceSum(&theta0_kn);
    mpi->allReduceSum(&phi_kn1); mpi->allReduceSum(&phi_kn2); mpi->allReduceSum(&phi_kn3);

    // output crystal coords mesh
    Eigen::MatrixXd wavevectors(numPoints,3); wavevectors.setZero();
    for (int ik : bandStructure->parallelIrrPointsIterator()) {
      WavevectorIndex ikIdx(ik);
      Eigen::Vector3d k = bandStructure->getWavevector(ikIdx);
      // // bandStructure->getPoints().cartesianToCrystal(k);
      wavevectors(ik,Eigen::placeholders::all) = bandStructure->getPoints().bzToWs(k, Points::cartesianCoordinates) / distanceBohrToAng;
    }
    mpi->allReduceSum(&wavevectors);

    // for now, the head process writes to file --------------------------
    if(mpi->mpiHead()) {

      std::string filename = particle.isPhonon() ? "relaxons_ph_eigenvectors.hdf5" : "relaxons_el_eigenvectors.hdf5";
      H5Easy::File file(filename, H5Easy::File::Overwrite);

      std::vector<double> tau;
      for(int alpha = 0; alpha < numRelaxonsToOutput; alpha++) {
        tau.push_back(1./eigenvalues(alpha) * energyToTime);
        // write to an eigen matrix so I can save to hdf5
        Eigen::MatrixXd relaxonCopy(numPoints,numBands);
        for (size_t ik = 0; ik < numPoints; ik++) {
          std::vector<double> tmp;
          for ( int ib = 0; ib < numBands; ib++) {
            relaxonCopy(ik,ib) = relaxon(ik,ib,alpha);
          }
        }
        H5Easy::dump(file, "/relaxonEigenvectors_"+std::to_string(alpha), relaxonCopy);
      }
      H5Easy::dump(file, "/relaxonRelaxationTimes", tau);
      if(bandStructure->getParticle().isElectron()) H5Easy::dump(file, "/theta_e", theta_e_kn);
      H5Easy::dump(file, "/theta0", theta0_kn);
      H5Easy::dump(file, "/phi_x", phi_kn1);
      H5Easy::dump(file, "/phi_y", phi_kn2);
      H5Easy::dump(file, "/phi_z", phi_kn3);
      H5Easy::dump(file, "/wavevectorCoordinatesCartesianWS", wavevectors);
      //H5Easy::dump(file, "/wavevectorCoordinatesType", "cartesian"); // cannot write fixed length string...
      //H5Easy::dump(file, "/relaxationTimeUnit", "fs"); // cannot write fixed length string...
      H5Easy::dump(file, "/numPoints", numPoints);
      H5Easy::dump(file, "/numRelaxons", numRelaxonsToOutput);
    }
  }
}


// TODO we need to fix the dimensionality to work for
// low dim materials in all the coefficients!
void genericOutputRealSpaceToJSON(Context& context, ScatteringMatrix& scatteringMatrix,
                                BaseBandStructure& bandStructure,
                                StatisticsSweep& statisticsSweep,
                                Eigen::VectorXd& theta0,
                                Eigen::VectorXd& theta_e,
                                Eigen::MatrixXd& phi,
                                double C, Eigen::Vector3d& A) {

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
    std::string outFileName = "relaxons_el_real_space_coefficients.json";
    if(isPhonon) outFileName = "relaxons_ph_real_space_coefficients.json";
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


// TODO use requires on the bandstructures to be el first and ph second, do this more elegantly
void outputRelaxonContributionsToHDF5(const Eigen::VectorXd& eigenvalues,
                                      const Eigen::MatrixXd& V0,
                                      const Eigen::MatrixXd& Ve,
                                      const Eigen::Tensor<double, 3>& Vphi,
                                      const Particle& particle,
                                      const int numRelaxons) {

  double energyToTime = particle.isPhonon() ? energyRyToFs * 1e-3 / twoPi : energyRyToFs;

  Eigen::VectorXd tau = energyToTime * eigenvalues.array().inverse();
  Eigen::MatrixXd Vphi_x(numRelaxons, 3);  Vphi_x.setZero();
  Eigen::MatrixXd Vphi_y(numRelaxons, 3);  Vphi_y.setZero();
  Eigen::MatrixXd Vphi_z(numRelaxons, 3);  Vphi_z.setZero();
  Eigen::MatrixXd V0_out = V0; // copy because we will add a unit conversion
  Eigen::MatrixXd Ve_out = Ve;
  // Seems there is not a clear way to slice this, so I will loop to copy it
  for(int alpha = 0; alpha < numRelaxons; alpha++) {
    for(auto i : {0,1,2}) {
      Vphi_x(alpha, i) = Vphi(alpha, 0, i);
      Vphi_y(alpha, i) = Vphi(alpha, 1, i);
      Vphi_z(alpha, i) = Vphi(alpha, 2, i);
    }
  }

  for(auto V : {&Vphi_x, &Vphi_y, &Vphi_z, &V0_out, &Ve_out}) {
    *V *= velocityRyToSi;
  }

  // for now, the head process writes to file --------------------------
  if(mpi->mpiHead()) {

    std::string filename = particle.isPhonon() ? "relaxons_ph_velocities.hdf5" : "relaxons_el_velocities.hdf5";
    H5Easy::File file(filename, H5Easy::File::Overwrite);
    if(particle.isElectron()) H5Easy::dump(file, "/relaxonRelaxationTimes_in_fs", tau);
    if(particle.isPhonon()) H5Easy::dump(file, "/relaxonRelaxationTimes_in_ps", tau);
    H5Easy::dump(file, "/V0", V0_out);
    if(particle.isElectron()) H5Easy::dump(file, "/Ve", Ve_out);
    H5Easy::dump(file, "/Vphi_x", Vphi_x);
    H5Easy::dump(file, "/Vphi_y", Vphi_y);
    H5Easy::dump(file, "/Vphi_z", Vphi_z);
    H5Easy::dump(file, "/numRelaxons", numRelaxons);
  }
}

