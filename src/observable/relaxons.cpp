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
      wavevectors(ik,Eigen::indexing::all) = bandStructure->getPoints().bzToWs(k, Points::cartesianCoordinates) / distanceBohrToAng;
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
