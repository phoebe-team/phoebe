#include "relaxons.h"

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
    double energyToTime = particle.isPhonon() ? energyRyToFs * 1e-3 : energyRyToFs;

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
      wavevectors(ik,Eigen::all) = bandStructure->getPoints().bzToWs(k, Points::cartesianCoordinates) / distanceBohrToAng; 
    }
    mpi->allReduceSum(&wavevectors);

    // for now, the head process writes to file --------------------------
    if(mpi->mpiHead()) {

      std::string filename = particle.isPhonon() ? "ph_relaxons_eigenvectors.hdf5" : "el_relaxons_eigenvectors.hdf5";
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
  }

  // If the best overlap isn't very good, we return -1 so nothing is skipped 
  if(maxOverlap >= 0.75) return idxMaxOverlap;
  else { return -1; }
}

 
// calculate special eigenvectors
void genericCalcSpecialEigenvectors(Context& context, BaseBandStructure& bandStructure,
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